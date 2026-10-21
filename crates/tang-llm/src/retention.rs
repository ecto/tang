//! How long prompt-derived state is kept: `tang-llm serve --kv-ttl <secs>` (or `TANG_KV_TTL`).
//!
//! Off by default. When set, every cache that holds state computed from a prompt (KV blocks in
//! the pool, parked conversations, the on-disk KV tier, the Flash backend's saved states, the
//! suffix-draft store) drops an entry once it has gone unused for the TTL: lookups skip it at
//! once, and the server's periodic [`sweep`](crate::server::Backend::sweep) frees it (and
//! deletes its file). Use refreshes the clock. See `docs/retention.md`.
//!
//! Times are wall-clock ([`now`]), so files' modification times compare with them; tests move
//! the clock with [`advance`] (per thread).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// The process's TTL in seconds (0: off), set once at startup.
static TTL: AtomicU64 = AtomicU64::new(0);

/// Set the process's TTL (`None`: keep entries until their room is needed).
pub fn set_ttl(ttl: Option<Duration>) {
    TTL.store(ttl.map_or(0, |d| d.as_secs().max(1)), Ordering::Relaxed);
}

/// The process's TTL, if one is set. Caches take it when they're made.
pub fn ttl() -> Option<Duration> {
    match TTL.load(Ordering::Relaxed) {
        0 => None,
        s => Some(Duration::from_secs(s)),
    }
}

/// `--kv-ttl` / `TANG_KV_TTL`: seconds, 0 or `off` for none.
pub fn parse(v: &str) -> anyhow::Result<Option<Duration>> {
    match v.trim() {
        "" | "0" | "off" => Ok(None),
        s => {
            let secs: u64 = s
                .parse()
                .map_err(|_| anyhow::anyhow!("--kv-ttl <secs>: not a number: {s}"))?;
            Ok(Some(Duration::from_secs(secs)))
        }
    }
}

/// How often the server sweeps for expired entries: a tenth of the TTL, between 1 s and a
/// minute.
pub fn sweep_every(ttl: Duration) -> Duration {
    (ttl / 10).clamp(Duration::from_secs(1), Duration::from_secs(60))
}

#[cfg(test)]
thread_local! {
    static SKEW: std::cell::Cell<Duration> = const { std::cell::Cell::new(Duration::ZERO) };
}

/// The time now (moved by [`advance`] in tests).
pub fn now() -> SystemTime {
    #[cfg(test)]
    return SystemTime::now() + SKEW.with(|s| s.get());
    #[cfg(not(test))]
    SystemTime::now()
}

/// Move this thread's clock forward (tests).
#[cfg(test)]
pub fn advance(d: Duration) {
    SKEW.with(|s| s.set(s.get() + d));
}

/// Mark the file at `path` as just used (its modification time is when the TTL counts from).
pub fn touch(path: &std::path::Path) {
    if let Ok(f) = std::fs::File::options().append(true).open(path) {
        let _ = f.set_modified(now());
    }
}

/// Whether the file at `path` has gone unused past `ttl`.
pub fn stale_file(path: &std::path::Path, ttl: Option<Duration>) -> bool {
    ttl.is_some()
        && std::fs::metadata(path)
            .and_then(|m| m.modified())
            .is_ok_and(|t| expired(t, ttl))
}

/// Whether something last used `at` has outlived `ttl`.
pub fn expired(at: SystemTime, ttl: Option<Duration>) -> bool {
    ttl.is_some_and(|ttl| now().duration_since(at).is_ok_and(|age| age > ttl))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_expires() {
        assert_eq!(parse("off").unwrap(), None);
        assert_eq!(parse("0").unwrap(), None);
        assert_eq!(parse("3600").unwrap(), Some(Duration::from_secs(3600)));
        assert!(parse("1h").is_err());
        let t = now();
        let ttl = Some(Duration::from_secs(10));
        assert!(!expired(t, ttl));
        advance(Duration::from_secs(11));
        assert!(expired(t, ttl));
        assert!(!expired(t, None));
    }
}
