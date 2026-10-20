//! Extension point: faster implementations of ops (or whole steps) registered by another crate
//! for specific GPUs, models and quantizations, with the built-in kernels as the fallback.
//!
//! An implementation is a value of a *slot type*: a newtype per op, so each slot has its own
//! registry ([`Linear`], [`RmsNorm`], [`KvAttentionPaged`], [`AllReduce`] here; tang-llm adds a
//! whole-step slot). [`register`] adds one under a [`Match`] and a priority; the consumer calls
//! [`select`] once (at model load) with the [`Target`] it runs on and keeps what it got. With
//! nothing registered `select` returns `None` and the caller uses the built-in kernel, so the
//! hot path pays one `Option` check per call.
//!
//! Every implementation may also decline a call at run time (return `None`), e.g. a GEMV
//! kernel asked for 64 rows: the caller then falls back for that call only.
//!
//! See `crates/tang-llm/docs/extensions.md` for the design.

use crate::device::{ComputeDevice, Pages};
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// What a resolved implementation runs on: the device, the model and the weight format.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Target {
    /// [`ComputeDevice::arch`]: `sm_86`, `sm_90`, `sm_100`, `metal`, `cpu`.
    pub arch: String,
    /// The model family (`model_type` in `config.json`: `qwen3`, `gemma3_text`, ...).
    pub model: String,
    pub shape: Shape,
    /// How matrices are stored: `f32`, `bf16`, `q4`, or an extension's own format name.
    pub quant: String,
}

/// The dimensions a kernel is usually specialised for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Shape {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub vocab: usize,
}

/// Which targets an implementation is for. Empty lists and `None` match anything.
#[derive(Debug, Clone, Default)]
pub struct Match {
    pub archs: Vec<String>,
    pub models: Vec<String>,
    pub shape: Option<Shape>,
    pub quants: Vec<String>,
}

impl Match {
    pub fn any() -> Self {
        Self::default()
    }
    pub fn arch(mut self, a: &str) -> Self {
        self.archs.push(a.into());
        self
    }
    pub fn model(mut self, m: &str) -> Self {
        self.models.push(m.into());
        self
    }
    pub fn shape(mut self, s: Shape) -> Self {
        self.shape = Some(s);
        self
    }
    pub fn quant(mut self, q: &str) -> Self {
        self.quants.push(q.into());
        self
    }

    pub fn matches(&self, t: &Target) -> bool {
        let has = |l: &[String], v: &str| l.is_empty() || l.iter().any(|x| x == v);
        has(&self.archs, &t.arch)
            && has(&self.models, &t.model)
            && has(&self.quants, &t.quant)
            && self.shape.is_none_or(|s| s == t.shape)
    }
}

/// A registered implementation of slot `T`.
#[derive(Clone)]
pub struct Entry<T> {
    /// Shown in logs and parity-test failures.
    pub name: &'static str,
    pub matches: Match,
    /// The highest-priority match wins; ties go to the later registration.
    pub priority: i32,
    pub imp: T,
}

type Slots = HashMap<TypeId, Box<dyn Any + Send>>;

fn slots() -> &'static Mutex<Slots> {
    static S: OnceLock<Mutex<Slots>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// Register an implementation. Call before loading models: consumers resolve at load.
pub fn register<T: Clone + Send + 'static>(e: Entry<T>) {
    let mut s = slots().lock().unwrap();
    let v = s
        .entry(TypeId::of::<T>())
        .or_insert_with(|| Box::new(Vec::<Entry<T>>::new()));
    v.downcast_mut::<Vec<Entry<T>>>().unwrap().push(e);
}

/// The best registered implementation of slot `T` for `t`, if any. `TANG_EXT=0` turns every
/// extension off (for A/B runs against the built-in kernels).
pub fn select<T: Clone + Send + 'static>(t: &Target) -> Option<Entry<T>> {
    if disabled() {
        return None;
    }
    let s = slots().lock().unwrap();
    let v = s.get(&TypeId::of::<T>())?.downcast_ref::<Vec<Entry<T>>>()?;
    v.iter()
        .enumerate()
        .filter(|(_, e)| e.matches.matches(t))
        .max_by_key(|(i, e)| (e.priority, *i))
        .map(|(_, e)| e.clone())
}

/// Remove every registration of slot `T` (tests).
pub fn clear<T: 'static>() {
    slots().lock().unwrap().remove(&TypeId::of::<T>());
}

fn disabled() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("TANG_EXT").is_ok_and(|v| v == "0"))
}

/// `x · wᵀ` for `m` rows, as [`ComputeDevice::linear`] (any weight format the device holds).
pub type LinearFn<D> = dyn Fn(
        &D,
        &<D as ComputeDevice>::Buffer,
        &<D as ComputeDevice>::Buffer,
        usize,
        usize,
        usize,
    ) -> Option<<D as ComputeDevice>::Buffer>
    + Send
    + Sync;

/// Projections (GEMV for decode, GEMM for short batches).
pub struct Linear<D: ComputeDevice>(pub Arc<LinearFn<D>>);

/// As [`ComputeDevice::rms_norm`]: `(x, weight, rows, dim, eps)`.
pub type RmsNormFn<D> = dyn Fn(
        &D,
        &<D as ComputeDevice>::Buffer,
        &<D as ComputeDevice>::Buffer,
        usize,
        usize,
        f32,
    ) -> Option<<D as ComputeDevice>::Buffer>
    + Send
    + Sync;

pub struct RmsNorm<D: ComputeDevice>(pub Arc<RmsNormFn<D>>);

/// The arguments of [`ComputeDevice::kv_attention_paged`].
pub struct PagedAttention<'a, B> {
    pub q: &'a B,
    pub k_pool: &'a B,
    pub v_pool: &'a B,
    pub pages: &'a Pages<'a, B>,
    pub start: usize,
    pub rows: usize,
    pub heads: (usize, usize, usize),
    pub window: usize,
    pub causal: bool,
}

pub type KvAttentionPagedFn<D> = dyn Fn(
        &D,
        &PagedAttention<'_, <D as ComputeDevice>::Buffer>,
    ) -> Option<<D as ComputeDevice>::Buffer>
    + Send
    + Sync;

/// Attention over a paged KV cache.
pub struct KvAttentionPaged<D: ComputeDevice>(pub Arc<KvAttentionPagedFn<D>>);

/// Sum `bufs` (one per GPU, `len` floats each) in place, every buffer ending with the total.
/// Each buffer lives on its own device, `devs[i]`.
pub type AllReduceFn<D> =
    dyn Fn(&[&D], &mut [&mut <D as ComputeDevice>::Buffer], usize) -> Option<()> + Send + Sync;

/// Tensor-parallel all-reduce. tang has no multi-GPU path yet: the slot exists so the one it
/// grows can resolve a fused implementation the same way.
pub struct AllReduce<D: ComputeDevice>(pub Arc<AllReduceFn<D>>);

macro_rules! slot_clone {
    ($($t:ident),*) => {$(
        impl<D: ComputeDevice> Clone for $t<D> {
            fn clone(&self) -> Self {
                Self(self.0.clone())
            }
        }
    )*};
}
slot_clone!(Linear, RmsNorm, KvAttentionPaged, AllReduce);

/// The op slots resolved for one target: what a model keeps and checks on its hot path.
pub struct Ops<D: ComputeDevice> {
    pub linear: Option<Entry<Linear<D>>>,
    pub rms_norm: Option<Entry<RmsNorm<D>>>,
    pub kv_attention_paged: Option<Entry<KvAttentionPaged<D>>>,
    pub all_reduce: Option<Entry<AllReduce<D>>>,
}

impl<D: ComputeDevice> Default for Ops<D> {
    fn default() -> Self {
        Self {
            linear: None,
            rms_norm: None,
            kv_attention_paged: None,
            all_reduce: None,
        }
    }
}

impl<D: ComputeDevice + 'static> Ops<D> {
    pub fn resolve(t: &Target) -> Self {
        Self {
            linear: select(t),
            rms_norm: select(t),
            kv_attention_paged: select(t),
            all_reduce: select(t),
        }
    }

    /// The names of what resolved, for logs.
    pub fn names(&self) -> Vec<&'static str> {
        [
            self.linear.as_ref().map(|e| e.name),
            self.rms_norm.as_ref().map(|e| e.name),
            self.kv_attention_paged.as_ref().map(|e| e.name),
            self.all_reduce.as_ref().map(|e| e.name),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    pub fn linear(
        &self,
        dev: &D,
        x: &D::Buffer,
        w: &D::Buffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> D::Buffer {
        if let Some(out) = self
            .linear
            .as_ref()
            .and_then(|e| (e.imp.0)(dev, x, w, m, k, n))
        {
            return out;
        }
        dev.linear(x, w, m, k, n)
    }

    pub fn rms_norm(
        &self,
        dev: &D,
        x: &D::Buffer,
        w: &D::Buffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> D::Buffer {
        if let Some(out) = self
            .rms_norm
            .as_ref()
            .and_then(|e| (e.imp.0)(dev, x, w, rows, dim, eps))
        {
            return out;
        }
        dev.rms_norm(x, w, rows, dim, eps)
    }

    pub fn kv_attention_paged(&self, dev: &D, a: &PagedAttention<'_, D::Buffer>) -> D::Buffer {
        if let Some(out) = self
            .kv_attention_paged
            .as_ref()
            .and_then(|e| (e.imp.0)(dev, a))
        {
            return out;
        }
        dev.kv_attention_paged(
            a.q, a.k_pool, a.v_pool, a.pages, a.start, a.rows, a.heads, a.window, a.causal,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Probe(u32);

    fn target(arch: &str) -> Target {
        Target {
            arch: arch.into(),
            model: "qwen3".into(),
            quant: "bf16".into(),
            ..Default::default()
        }
    }

    #[test]
    fn highest_priority_match_wins() {
        clear::<Probe>();
        assert!(select::<Probe>(&target("sm_90")).is_none());
        let e = |name, m: Match, priority, v| Entry {
            name,
            matches: m,
            priority,
            imp: Probe(v),
        };
        register(e("any", Match::any(), 0, 1));
        register(e(
            "hopper",
            Match::any().arch("sm_90").model("qwen3"),
            10,
            2,
        ));
        register(e(
            "hopper-q4",
            Match::any().arch("sm_90").quant("q4"),
            20,
            3,
        ));
        assert_eq!(select::<Probe>(&target("sm_90")).unwrap().imp.0, 2);
        assert_eq!(select::<Probe>(&target("sm_86")).unwrap().imp.0, 1);
        let mut q4 = target("sm_90");
        q4.quant = "q4".into();
        assert_eq!(select::<Probe>(&q4).unwrap().name, "hopper-q4");
        clear::<Probe>();
    }
}
