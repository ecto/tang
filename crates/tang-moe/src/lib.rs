//! tang-moe: the substrate for running MoE models whose experts don't fit in VRAM.
//!
//! - [`HostArena`]: pinned, device-mapped host memory holding expert blobs. Kernels read it
//!   directly over PCIe through its device addresses.
//! - [`CachePolicy`]: which experts live in VRAM slots, with decayed-LFU adaptation. CPU-only,
//!   tested anywhere.
//! - [`PointerTable`]: per (layer, expert) the device address of its blob, either a VRAM slot
//!   or the mapped host copy, so an expert kernel does one indirection and doesn't care where
//!   the blob lives.
//! - `ExpertCache` (feature `cuda`): the VRAM slot arena, the device-side residency and pointer
//!   tables, and swaps on a side stream.
//!
//! See `DESIGN.md` for the measurements behind the design and `BENCH.md` for how to reproduce
//! them with `tang-moe-bench`.

pub mod arena;
pub mod policy;

#[cfg(feature = "cuda")]
pub mod cache;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod kernels;

pub use arena::{ArenaOptions, Backing, HostArena};
#[cfg(feature = "cuda")]
pub use cache::{ExpertCache, SizeReport};
pub use policy::{AdaptParams, CachePolicy, Geometry, PointerTable, Stats, Swap};
