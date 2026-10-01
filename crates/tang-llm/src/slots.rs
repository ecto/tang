//! Which KV cache a request runs in, when the engine keeps several.
//!
//! Each slot holds one conversation's keys and values. A request names its conversation with
//! `prompt_cache_key` (OpenAI's field for this); one without a key goes wherever the most of
//! its prompt is already cached. A conversation that has no slot takes an empty one, else the
//! least recently used.

/// What [`pick`] needs to know about a slot.
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    pub key: Option<&'a str>,
    pub tokens: &'a [u32],
    /// When it last ran a request (larger is later).
    pub used: u64,
}

/// Where a request runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pick {
    /// The slot that ran the last request.
    Active,
    /// A parked slot, by index.
    Parked(usize),
    /// A new slot (there's room for one).
    Fresh,
}

/// Choose a slot for a request with `key` and prompt `ids`. `room`: another slot may be
/// allocated.
pub fn pick(active: View, parked: &[View], key: Option<&str>, ids: &[u32], room: bool) -> Pick {
    let all = || {
        std::iter::once((Pick::Active, active)).chain(
            parked
                .iter()
                .enumerate()
                .map(|(i, v)| (Pick::Parked(i), *v)),
        )
    };
    if let Some(k) = key {
        if let Some((p, _)) = all().find(|(_, v)| v.key == Some(k)) {
            return p;
        }
    } else {
        let shared = |v: &View| v.tokens.iter().zip(ids).take_while(|(a, b)| a == b).count();
        // The first of the longest, so the active slot wins ties.
        let best = all()
            .map(|(p, v)| (p, shared(&v)))
            .reduce(|b, c| if c.1 > b.1 { c } else { b });
        if let Some((p, n)) = best {
            if n > 0 {
                return p;
            }
        }
    }
    if let Some((p, _)) = all().find(|(_, v)| v.tokens.is_empty() && v.key.is_none()) {
        return p;
    }
    if room {
        return Pick::Fresh;
    }
    all()
        .min_by_key(|(_, v)| v.used)
        .map_or(Pick::Active, |(p, _)| p)
}

#[cfg(test)]
mod tests {
    use super::{pick, Pick, View};

    fn v<'a>(key: Option<&'a str>, tokens: &'a [u32], used: u64) -> View<'a> {
        View { key, tokens, used }
    }

    #[test]
    fn one_slot_is_always_the_active_one() {
        let a = v(Some("a"), &[1, 2, 3], 5);
        assert_eq!(pick(a, &[], Some("b"), &[9], false), Pick::Active);
        assert_eq!(pick(a, &[], None, &[9], false), Pick::Active);
    }

    #[test]
    fn a_key_finds_its_own_slot() {
        let a = v(Some("a"), &[1, 2], 9);
        let parked = [v(Some("b"), &[1, 2], 3), v(Some("c"), &[7], 4)];
        assert_eq!(
            pick(a, &parked, Some("c"), &[1, 2, 3], false),
            Pick::Parked(1)
        );
        assert_eq!(pick(a, &parked, Some("a"), &[5], false), Pick::Active);
    }

    #[test]
    fn a_new_key_takes_room_then_the_least_recently_used() {
        let a = v(Some("a"), &[1, 2], 9);
        let parked = [v(Some("b"), &[3], 3), v(Some("c"), &[4], 4)];
        assert_eq!(pick(a, &parked, Some("d"), &[1, 2, 3], true), Pick::Fresh);
        assert_eq!(
            pick(a, &parked, Some("d"), &[1, 2, 3], false),
            Pick::Parked(0)
        );
    }

    #[test]
    fn a_new_key_takes_an_empty_slot_first() {
        let a = v(Some("a"), &[1], 9);
        let parked = [v(None, &[], 0)];
        assert_eq!(pick(a, &parked, Some("d"), &[1], true), Pick::Parked(0));
    }

    #[test]
    fn no_key_goes_where_the_prompt_is_cached() {
        let a = v(Some("a"), &[1, 2], 9);
        let parked = [v(Some("b"), &[1, 2, 3, 4], 3), v(None, &[5], 1)];
        assert_eq!(
            pick(a, &parked, None, &[1, 2, 3, 9], false),
            Pick::Parked(0)
        );
        assert_eq!(pick(a, &parked, None, &[1, 2], false), Pick::Active);
        // Nothing cached: the least recently used.
        assert_eq!(pick(a, &parked, None, &[8], false), Pick::Parked(1));
    }
}
