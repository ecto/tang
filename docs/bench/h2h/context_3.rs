//! Drafts for speculative decoding, SuffixDecoding-style: no draft model, just suffix matches
//! against text the model has already seen or written.
//!
//! Two token-level suffix automata answer "where did the last few tokens appear before, and
//! what came next most often?":
//! - one per request, over the prompt plus what's been generated so far (an agent's replies
//!   copy file contents, earlier edits and tool output from the prompt);
//! - a bounded global one over earlier completions from this server (optionally kept on disk
//!   across restarts), since agents repeat themselves across requests and sessions.
//!
//! A draft follows the most frequent continuation of the longest match. Its length adapts: each
//! drafted token gets an acceptance estimate (the continuation's frequency times a per-match-
//! length hit rate learned online), and the draft is cut where expected tokens per unit of
//! verify cost peaks, using the backend's measured cost of a k-token forward.
//!
//! Drafting never changes what's generated: the engine samples every position from the target
//! model and only keeps draft tokens that equal what it sampled.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{Read, Write};
use std::path::PathBuf;

/// Separates sequences in the global store; never drafted.
const SEP: u32 = u32::MAX;

/// Walk at most this far up suffix links when counting occurrences (bounds pathological runs
/// like a long stretch of one repeated token; counts near the root just saturate).
const COUNT_WALK: usize = 1024;

/// A fast hasher for the (state, token) transition table (splitmix64's finalizer, so both
/// halves of the key reach the low bits the table indexes by).
#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = self.0.rotate_left(5) ^ n;
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// An edge in a state's child list.
#[derive(Clone, Copy)]
struct Edge {
    tok: u32,
    to: u32,
    /// Next edge of the same state (`u32::MAX`: none).
    next: u32,
}

const NONE: u32 = u32::MAX;

/// A suffix automaton over token ids with occurrence counts (how many times each state's
/// strings occur), so it gives both the longest earlier match and continuation frequencies.
pub struct Sam {
    len: Vec<u32>,
    link: Vec<u32>,
    cnt: Vec<u32>,
    first: Vec<u32>,
    edges: Vec<Edge>,
    /// (state << 32 | token) -> edge index.
    index: FxMap<u64, u32>,
    last: u32,
    /// Tokens added.
    pub size: usize,
}

impl Default for Sam {
    fn default() -> Self {
        Self::new()
    }
}

impl Sam {
    pub fn new() -> Self {
        Self {
            len: vec![0],
            link: vec![NONE],
            cnt: vec![0],
            first: vec![NONE],
            edges: Vec::new(),
            index: FxMap::default(),
            last: 0,
            size: 0,
        }
    }

    pub fn with_capacity(tokens: usize) -> Self {
        let mut s = Self::new();
        s.len.reserve(2 * tokens);
        s.link.reserve(2 * tokens);
        s.cnt.reserve(2 * tokens);
        s.first.reserve(2 * tokens);
        s.edges.reserve(3 * tokens);
        s.index.reserve(3 * tokens);
        s
    }

    fn key(s: u32, c: u32) -> u64 {
        (s as u64) << 32 | c as u64
    }

    fn go(&self, s: u32, c: u32) -> Option<u32> {
        self.index
            .get(&Self::key(s, c))
            .map(|&e| self.edges[e as usize].to)
    }

    fn set(&mut self, s: u32, c: u32, to: u32) {
        match self.index.get(&Self::key(s, c)) {
            Some(&e) => self.edges[e as usize].to = to,
            None => {
                let e = self.edges.len() as u32;
                self.edges.push(Edge {
                    tok: c,
                    to,
                    next: self.first[s as usize],
                });
                self.first[s as usize] = e;
                self.index.insert(Self::key(s, c), e);
            }
        }
    }

    fn new_state(&mut self, len: u32, link: u32, cnt: u32) -> u32 {
        self.len.push(len);
        self.link.push(link);
