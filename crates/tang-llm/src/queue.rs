//! The order requests reach the model in: two lanes, interactive and background (the
//! `x-frog-priority` header), first in first out within each. The worker always takes the
//! interactive lane first, and a background request that's prefilling stops between prefill
//! chunks while an interactive one waits (see [`crate::engine::Control`]); it then goes back to
//! the front of its lane and picks up from the blocks it already prefilled.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::Instant;

/// How urgent a request is: someone waiting on it, or work that can wait (warming a cache,
/// compaction, subagents, benches).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Priority {
    #[default]
    Interactive,
    Background,
}

impl Priority {
    /// The `x-frog-priority` header's value (absent: interactive).
    pub fn parse(header: Option<&str>) -> Result<Self, String> {
        match header.map(str::trim) {
            None | Some("") => Ok(Self::Interactive),
            Some(v) if v.eq_ignore_ascii_case("interactive") => Ok(Self::Interactive),
            Some(v) if v.eq_ignore_ascii_case("background") => Ok(Self::Background),
            Some(v) => Err(format!(
                "x-frog-priority must be interactive or background, not {v:?}"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Background => "background",
        }
    }
}

/// What the queue knows about a waiting item.
#[derive(Debug, Clone)]
pub struct Ticket {
    pub id: u64,
    pub priority: Priority,
    /// The prompt's length in tokens, when it was counted.
    pub prompt_tokens: Option<usize>,
    /// When it was first queued (a yielded request keeps its place in time).
    pub since: Instant,
    /// Times it gave way to interactive work part way through its prefill.
    pub yields: u32,
}

pub struct Queue<T> {
    lanes: Mutex<Lanes<T>>,
    ready: Condvar,
}

struct Lanes<T> {
    interactive: VecDeque<(Ticket, T)>,
    background: VecDeque<(Ticket, T)>,
    next: u64,
}

impl<T> Lanes<T> {
    fn lane(&mut self, p: Priority) -> &mut VecDeque<(Ticket, T)> {
        match p {
            Priority::Interactive => &mut self.interactive,
            Priority::Background => &mut self.background,
        }
    }
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Self {
            lanes: Mutex::new(Lanes {
                interactive: VecDeque::new(),
                background: VecDeque::new(),
                next: 1,
            }),
            ready: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Lanes<T>> {
        self.lanes.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Queue `item` at the back of its lane; returns its ticket.
    pub fn push(&self, priority: Priority, prompt_tokens: Option<usize>, item: T) -> Ticket {
        let mut l = self.lock();
        let t = Ticket {
            id: l.next,
            priority,
            prompt_tokens,
            since: Instant::now(),
            yields: 0,
        };
        l.next += 1;
        l.lane(priority).push_back((t.clone(), item));
        drop(l);
        self.ready.notify_one();
        t
    }

    /// Put an item that was taken back at the front of its lane (it gave way part way
    /// through, and goes on before anything queued after it).
    pub fn push_front(&self, ticket: Ticket, item: T) {
        self.lock().lane(ticket.priority).push_front((ticket, item));
        self.ready.notify_one();
    }

    /// The next item: interactive first. Blocks until there is one.
    pub fn pop(&self) -> (Ticket, T) {
        let mut l = self.lock();
        loop {
            if let Some(x) = l
                .interactive
                .pop_front()
                .or_else(|| l.background.pop_front())
            {
                return x;
            }
            l = self.ready.wait(l).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The next item, if any (interactive first).
    pub fn try_pop(&self) -> Option<(Ticket, T)> {
        let mut l = self.lock();
        l.interactive
            .pop_front()
            .or_else(|| l.background.pop_front())
    }

    /// Whether interactive work is waiting (so background prefill should give way).
    pub fn interactive_waiting(&self) -> bool {
        !self.lock().interactive.is_empty()
    }

    /// Everything waiting, in the order it would run.
    pub fn waiting(&self) -> Vec<Ticket> {
        let l = self.lock();
        l.interactive
            .iter()
            .chain(&l.background)
            .map(|(t, _)| t.clone())
            .collect()
    }

    pub fn len(&self) -> usize {
        let l = self.lock();
        l.interactive.len() + l.background.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(q: &Queue<&'static str>) -> Vec<&'static str> {
        std::iter::from_fn(|| q.try_pop().map(|(_, x)| x)).collect()
    }

    #[test]
    fn priority_header() {
        assert_eq!(Priority::parse(None), Ok(Priority::Interactive));
        assert_eq!(
            Priority::parse(Some("background")),
            Ok(Priority::Background)
        );
        assert_eq!(
            Priority::parse(Some(" Interactive ")),
            Ok(Priority::Interactive)
        );
        assert!(Priority::parse(Some("urgent")).is_err());
    }

    #[test]
    fn interactive_runs_before_waiting_background() {
        let q = Queue::new();
        q.push(Priority::Background, Some(10_000), "warm-a");
        q.push(Priority::Background, None, "warm-b");
        q.push(Priority::Interactive, Some(50), "turn-1");
        q.push(Priority::Interactive, Some(60), "turn-2");
        let waiting: Vec<_> = q.waiting().iter().map(|t| t.priority).collect();
        assert_eq!(
            waiting,
            [
                Priority::Interactive,
                Priority::Interactive,
                Priority::Background,
                Priority::Background
            ]
        );
        assert_eq!(order(&q), ["turn-1", "turn-2", "warm-a", "warm-b"]);
    }

    #[test]
    fn a_yielded_request_goes_on_before_later_background_work() {
        let q = Queue::new();
        q.push(Priority::Background, None, "warm-a");
        q.push(Priority::Background, None, "warm-b");
        let (mut t, a) = q.pop();
        assert_eq!(a, "warm-a");
        // An interactive request arrives while warm-a prefills: warm-a gives way.
        q.push(Priority::Interactive, None, "turn");
        assert!(q.interactive_waiting());
        t.yields += 1;
        q.push_front(t, a);
        assert_eq!(order(&q), ["turn", "warm-a", "warm-b"]);
        assert!(!q.interactive_waiting());
    }

    #[test]
    fn tickets_are_numbered_in_arrival_order() {
        let q = Queue::new();
        let a = q.push(Priority::Background, None, ());
        let b = q.push(Priority::Interactive, None, ());
        assert!(b.id > a.id);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn pop_waits_for_work() {
        let q = std::sync::Arc::new(Queue::new());
        let q2 = q.clone();
        let h = std::thread::spawn(move || q2.pop().1);
        std::thread::sleep(std::time::Duration::from_millis(20));
        q.push(Priority::Background, None, 7);
        assert_eq!(h.join().unwrap(), 7);
    }
}
