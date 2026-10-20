//! Host side of the GPU KV decoder: packs GPU-layout TKV streams
//! ([`crate::kvcodec`], `gpu = true`) into the flat tables the device kernel
//! reads (`tang-compute` `tkv_decode`, MSL source in
//! `tang-compute/src/kernels/kvdec_msl.rs`). No GPU dependency here.
//!
//! One kernel thread per lane = one (stream, block, head) arithmetic-coded
//! substream. Tables (all `u32`):
//! - `stream`: every stream's bytes, concatenated, little-endian words;
//! - `lanes`: [`LANE_WORDS`] words per lane: byte offset, byte length, first
//!   token, tokens, channel-table base, output base (elements), output token
//!   stride, scratch base (words), unit (`f32` bits), sink, sink divisor,
//!   head dim, RoPE flag, RoPE theta (`f32` bits);
//! - `chans`: 4 words per channel: mean (`i32`), rho, class, dense class
//!   index within its (stream, head);
//! - `prior`: [`crate::kvcodec::gpu_prior_table`];
//! - scratch: `scratch_words` words of adaptive-model state.

use crate::bitio::DecodeError;
use crate::kvcodec::{gpu_prior_table, KvStream, GPU_CTX_MODELS};

/// Words per lane descriptor.
pub const LANE_WORDS: usize = 14;

/// Flat decoder tables for a set of streams.
#[derive(Default)]
pub struct Batch {
    /// Concatenated stream bytes as words.
    pub stream: Vec<u32>,
    /// Lane descriptors, [`LANE_WORDS`] each.
    pub lanes: Vec<u32>,
    /// Channel table, 4 words per channel.
    pub chans: Vec<u32>,
    /// Model scratch size in words.
    pub scratch_words: usize,
    bytes: Vec<u8>,
}

/// Where a stream's output goes: element index of (head, token) row start
/// and the token stride, plus optional RoPE (K given pre-RoPE).
pub struct Placement<'f> {
    /// `(head, token) -> element offset` of that row's channel 0.
    pub row: &'f dyn Fn(usize, usize) -> usize,
    /// Apply half-split RoPE with this theta to each decoded token row.
    pub rope_theta: Option<f32>,
}

impl Batch {
    /// Empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one stream; every lane writes `row(head, t) + c` for its tokens.
    /// Lanes covering `[t0, t1)` only (whole blocks) are added.
    pub fn add(
        &mut self,
        bytes: &[u8],
        place: &Placement,
        t0: usize,
        t1: usize,
    ) -> Result<(), DecodeError> {
        let s = KvStream::parse(bytes)?;
        let tb = s
            .gpu_tables()
            .ok_or(DecodeError::Invalid("not a gpu-layout stream"))?;
        let p = &s.params;
        let base = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
        let chan0 = self.chans.len() / 4;
        // dense class index per (head, channel)
        let mut ncls = vec![0usize; p.heads];
        for hh in 0..p.heads {
            let mut seen: Vec<u32> = Vec::new();
            for c in 0..p.hd {
                let j = hh * p.hd + c;
                let cl = tb.class[j];
                let di = seen.iter().position(|&v| v == cl).unwrap_or_else(|| {
                    seen.push(cl);
                    seen.len() - 1
                });
                self.chans
                    .extend_from_slice(&[tb.mu[j] as u32, tb.rho[j], cl, di as u32]);
            }
            ncls[hh] = seen.len();
        }
        let ctx_words = 3 * GPU_CTX_MODELS;
        for l in &tb.lanes {
            let [off, len, pos0, n, hh] = *l;
            let (pos0, n, hh) = (pos0 as usize, n as usize, hh as usize);
            if pos0 + n <= t0 || pos0 >= t1 {
                continue;
            }
            let row0 = (place.row)(hh, pos0);
            let stride = if n > 1 {
                (place.row)(hh, pos0 + 1) - row0
            } else {
                p.hd
            };
            self.lanes.extend_from_slice(&[
                (base + off as usize) as u32,
                len,
                pos0 as u32,
                n as u32,
                (chan0 + hh * p.hd) as u32,
                row0 as u32,
                stride as u32,
                self.scratch_words as u32,
                p.unit().to_bits(),
                p.sink as u32,
                p.sink_div,
                p.hd as u32,
                place.rope_theta.is_some() as u32,
                place.rope_theta.unwrap_or(0.0).to_bits(),
            ]);
            self.scratch_words += ncls[hh] * ctx_words;
        }
        self.stream = self
            .bytes
            .chunks(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        Ok(())
    }

    /// Number of lanes.
    pub fn n_lanes(&self) -> usize {
        self.lanes.len() / LANE_WORDS
    }

    /// The shared prior table.
    pub fn prior() -> Vec<u32> {
        gpu_prior_table()
    }
}
