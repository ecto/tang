//! Serve-side wiring of the GDN memstore ([`super::memstore`]; design in
//! `docs/design/gdn-memstore.md`, "Serve wiring"): which GDN layer and heads hold the store
//! ([`StoreMeta`]), the store blob a client sends and gets back (the store heads'
//! [`MemStore::snapshot_heads`] as little-endian f32), store operations as engine-internal GDN
//! windows ([`run_op`]), the per-session lifecycle ([`Sessions`]) and `tang-llm memory dump`.
//!
//! Everything here builds without CUDA; the engines ([`super::engine::Engine`],
//! [`super::coder::Coder`]) route [`run_op`] through their own device's `gdn_step` on a slot's
//! state buffer of the store layer, and `flash::serve` drives [`Sessions`].

use std::collections::HashMap;

use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use tang_compute::flash::shape::{GDN_D, GDN_HK, GDN_HV};
use tang_compute::flash::{GdnParams, GDN_STATE};
use tang_compute::{ComputeDevice, CpuDevice};

use super::memstore::{MemStore, STORE_HK, STORE_HV, VALUE_LIMIT};

/// Floats in a store blob: the 12 store v heads' 128 × 128 rows.
pub const HEAD_FLOATS: usize = GDN_D * STORE_HV.len() * GDN_D;
/// Bytes in a store blob (768 KiB).
pub const BLOB_BYTES: usize = 4 * HEAD_FLOATS;

/// Where the store lives inside a recurrent-state snapshot: `RunningState::parts()[2 · gdn_layer]`
/// (the GDN layer's `[i][hv][j]` state), v heads [`STORE_HV`] (k heads [`STORE_HK`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreMeta {
    /// Index among the model's GDN layers (the state's part `2 · gdn_layer`).
    pub gdn_layer: usize,
    /// The model layer it is (for logs and humans).
    pub layer: usize,
}

impl StoreMeta {
    /// The store's v heads in the layer state.
    pub fn heads(&self) -> &'static [usize] {
        &STORE_HV
    }

    /// The store's k heads.
    pub fn k_heads(&self) -> &'static [usize] {
        &STORE_HK
    }

    /// As JSON, for `/v1/memory` replies and logs.
    pub fn to_json(&self) -> Value {
        json!({
            "gdn_layer": self.gdn_layer,
            "layer": self.layer,
            "v_heads": STORE_HV,
            "k_heads": STORE_HK,
        })
    }

    /// The trailer a state file carries after its parts (`MEMS`, gdn layer, layer, then the
    /// v heads), so the store heads are identifiable in a file on disk.
    pub fn trailer(&self) -> Vec<u8> {
        let mut b = b"MEMS".to_vec();
        for x in [self.gdn_layer, self.layer, STORE_HV.len()] {
            b.extend_from_slice(&(x as u64).to_le_bytes());
        }
        for &h in &STORE_HV {
            b.extend_from_slice(&(h as u64).to_le_bytes());
        }
        b
    }

    /// The inverse of [`trailer`](Self::trailer) (`None`: no trailer, or another layout).
    pub fn from_trailer(b: &[u8]) -> Option<Self> {
        let b = b.strip_prefix(b"MEMS")?;
        let w = |i: usize| -> Option<usize> {
            Some(u64::from_le_bytes(b.get(8 * i..8 * i + 8)?.try_into().ok()?) as usize)
        };
        let n = w(2)?;
        let heads: Option<Vec<usize>> = (0..n).map(|i| w(3 + i)).collect();
        (heads? == STORE_HV).then_some(StoreMeta {
            gdn_layer: w(0)?,
            layer: w(1)?,
        })
    }
}

/// Can the store be laid out in a GDN layer of `hk` k heads and `hv` v heads? The memstore's
/// layout is the Flash-Next shape (16 k / 48 v heads, v head `h` on k head `h % 16`).
pub fn check_shape(hk: usize, hv: usize) -> Result<()> {
    ensure!(
        hk == GDN_HK && hv == GDN_HV,
        "the memstore layout needs GDN layers of {GDN_HK} k / {GDN_HV} v heads (Flash-Next); \
         this model's have {hk} k / {hv} v heads"
    );
    Ok(())
}

/// Zero the store heads of a layer state, leaving the base heads alone.
pub fn blank_heads(state: &mut [f32]) {
    MemStore::<CpuDevice>::restore_heads(state, &vec![0f32; HEAD_FLOATS]);
}

/// A store blob's heads (little-endian f32, [`BLOB_BYTES`] bytes).
pub fn blob_to_heads(b: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        b.len() == BLOB_BYTES,
        "a store blob is {BLOB_BYTES} bytes (12 heads × 128 × 128 f32), not {}",
        b.len()
    );
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// The blob of a store's heads.
pub fn heads_to_blob(h: &[f32]) -> Vec<u8> {
    h.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// [`MemStore::audit`] of a blob's heads: the poisoned v heads (empty: intact).
pub fn audit_heads(heads: &[f32]) -> Vec<usize> {
    let dev = CpuDevice::new();
    let mut s = vec![0f32; GDN_STATE];
    MemStore::<CpuDevice>::restore_heads(&mut s, heads);
    MemStore::<CpuDevice>::audit(&dev, &dev.upload_f32(&s))
}

/// A store operation (`POST /v1/memory/op`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreOp {
    Put { name: Vec<u8>, ids: Vec<u32> },
    Get { name: Vec<u8> },
    Delete { name: Vec<u8> },
    List,
}

impl StoreOp {
    /// From `{"op": "put"|"get"|"delete"|"list", "name": "...", "ids": [...]}`.
    pub fn parse(v: &Value) -> Result<Self> {
        let op = v["op"]
            .as_str()
            .context("op must be put, get, delete or list")?;
        let name = || -> Result<Vec<u8>> {
            Ok(v["name"]
                .as_str()
                .context("name must be a string")?
                .as_bytes()
                .to_vec())
        };
        Ok(match op {
            "put" => StoreOp::Put {
                name: name()?,
                ids: v["ids"]
                    .as_array()
                    .context("ids must be an array of token ids")?
                    .iter()
                    .map(|x| {
                        x.as_u64()
                            .filter(|&n| n < VALUE_LIMIT as u64)
                            .map(|n| n as u32)
                            .context("each id must be an integer below 2^18")
                    })
                    .collect::<Result<_>>()?,
            },
            "get" => StoreOp::Get { name: name()? },
            "delete" => StoreOp::Delete { name: name()? },
            "list" => StoreOp::List,
            o => bail!("op must be put, get, delete or list, not {o}"),
        })
    }

    /// Whether it writes the store.
    pub fn writes(&self) -> bool {
        matches!(self, StoreOp::Put { .. } | StoreOp::Delete { .. })
    }
}

/// What a [`StoreOp`] returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpOut {
    Put,
    Got(Option<Vec<u32>>),
    Deleted(bool),
    Listed(Vec<(Vec<u8>, usize)>),
}

impl OpOut {
    pub fn to_json(&self) -> Value {
        match self {
            OpOut::Put => json!({ "ok": true }),
            OpOut::Got(ids) => json!({ "found": ids.is_some(), "ids": ids }),
            OpOut::Deleted(d) => json!({ "deleted": d }),
            OpOut::Listed(files) => json!({
                "files": files
                    .iter()
                    .map(|(n, len)| json!({ "name": String::from_utf8_lossy(n), "tokens": len }))
                    .collect::<Vec<_>>()
            }),
        }
    }
}

/// Run `op` on one GDN layer's state as store windows through `dev.gdn_step` (each ≤ 8
/// tokens, `ReadOnly` for reads, `Commit` for writes): nothing but this layer's state is read
/// or written — no base layer, LM head, KV cache or sampler. Opening the store costs a
/// directory scan (16 windows) on top of the op's own windows.
pub fn run_op<Dev: ComputeDevice>(
    dev: &Dev,
    params: GdnParams<'_, Dev::Buffer>,
    eps: f32,
    state: &mut Dev::Buffer,
    op: &StoreOp,
) -> Result<OpOut> {
    let mut st = MemStore::open(dev, params, eps, state)?;
    Ok(match op {
        StoreOp::Put { name, ids } => {
            st.put(state, name, ids)?;
            OpOut::Put
        }
        StoreOp::Get { name } => OpOut::Got(st.get(state, name)?),
        StoreOp::Delete { name } => OpOut::Deleted(st.delete(state, name)?),
        StoreOp::List => OpOut::Listed(st.list(state)?),
    })
}

/// What [`Sessions`] needs from the engine: a slot's store heads in and out, and a store op on
/// its live state. `flash::serve` implements it over `ServeEngine`.
pub trait StoreHost {
    fn store_heads(&mut self, slot: usize) -> Result<Vec<f32>>;
    fn set_store_heads(&mut self, slot: usize, heads: &[f32]) -> Result<()>;
    fn store_op(&mut self, slot: usize, op: &StoreOp) -> Result<OpOut>;
}

/// Why an export failed.
#[derive(Debug)]
pub enum ExportError {
    /// No store for that key.
    Missing,
    /// [`MemStore::audit`] flagged these v heads.
    Poisoned(Vec<usize>),
}

/// Per-session stores, keyed by `prompt_cache_key`.
///
/// The session's copy (`held`) is the store between requests; it changes only by `load` and
/// store ops, each of which runs on a slot's live state: the key's slot when it has one,
/// otherwise slot 0 borrowed (its own store heads saved before and put back after, bitwise).
/// While a request of the key runs, its slot carries the store ([`attach`](Self::attach)); a
/// slot that goes to another key, or whose session ends, is [`detach`](Self::detach)ed: its
/// store heads are zeroed, and with `ephemeral` the session's copy is dropped too.
/// [`resident`](Self::resident) says which slots carry a store, so saved prefix states (RAM
/// and the disk tier) blank those heads.
#[derive(Default)]
pub struct Sessions {
    /// Zero and forget a session's store when its slot is evicted or the session ends (never
    /// kept past the slot, never written to disk).
    pub ephemeral: bool,
    held: HashMap<String, Vec<f32>>,
    slots: HashMap<usize, String>,
}

impl Sessions {
    pub fn new(ephemeral: bool) -> Self {
        Sessions {
            ephemeral,
            ..Default::default()
        }
    }

    /// Whether any session holds a store (when none does, serving is exactly as without one).
    pub fn is_empty(&self) -> bool {
        self.held.is_empty() && self.slots.is_empty()
    }

    pub fn has(&self, key: &str) -> bool {
        self.held.contains_key(key)
    }

    /// The key whose store slot `slot` carries.
    pub fn resident(&self, slot: usize) -> Option<&str> {
        self.slots.get(&slot).map(String::as_str)
    }

    /// The slot carrying `key`'s store.
    pub fn slot_of(&self, key: &str) -> Option<usize> {
        self.slots
            .iter()
            .find(|(_, k)| k.as_str() == key)
            .map(|(&s, _)| s)
    }

    /// `POST /v1/memory/load`: `blob` becomes `key`'s store (replacing any), and goes into the
    /// key's slot when it has one. Refused when the audit flags it.
    pub fn load(&mut self, host: &mut dyn StoreHost, key: &str, blob: &[u8]) -> Result<()> {
        let heads = blob_to_heads(blob)?;
        let bad = audit_heads(&heads);
        ensure!(
            bad.is_empty(),
            "the blob's store heads {bad:?} are poisoned"
        );
        if let Some(slot) = self.slot_of(key) {
            host.set_store_heads(slot, &heads)?;
        }
        self.held.insert(key.to_string(), heads);
        Ok(())
    }

    /// `POST /v1/memory/op` on `key`'s store (an empty store when none is loaded).
    pub fn op(&mut self, host: &mut dyn StoreHost, key: &str, op: &StoreOp) -> Result<OpOut> {
        let heads = self
            .held
            .get(key)
            .cloned()
            .unwrap_or_else(|| vec![0f32; HEAD_FLOATS]);
        let (slot, borrowed) = match self.slot_of(key) {
            Some(s) => (s, None),
            None => (0, Some(host.store_heads(0)?)),
        };
        host.set_store_heads(slot, &heads)?;
        let out = host.store_op(slot, op);
        let after = host.store_heads(slot);
        if let Some(prev) = &borrowed {
            host.set_store_heads(slot, prev)?;
        }
        let (out, after) = (out?, after?);
        if op.writes() || self.held.contains_key(key) {
            self.held.insert(key.to_string(), after);
        }
        Ok(out)
    }

    /// `GET /v1/memory`: `key`'s store blob, audited first.
    pub fn export(&self, key: &str) -> std::result::Result<Vec<u8>, ExportError> {
        let heads = self.held.get(key).ok_or(ExportError::Missing)?;
        let bad = audit_heads(heads);
        if !bad.is_empty() {
            return Err(ExportError::Poisoned(bad));
        }
        Ok(heads_to_blob(heads))
    }

    /// A request of `key` runs in `slot` (after the slot's state was restored or reset): a
    /// store some other key left there is detached, and `key`'s store goes in. Nothing happens
    /// when neither has a store.
    pub fn attach(
        &mut self,
        host: &mut dyn StoreHost,
        slot: usize,
        key: Option<&str>,
    ) -> Result<()> {
        if self.resident(slot).is_some_and(|k| Some(k) != key) {
            self.detach(host, slot)?;
        }
        let Some(key) = key else { return Ok(()) };
        let Some(heads) = self.held.get(key).cloned() else {
            return Ok(());
        };
        if let Some(old) = self.slot_of(key).filter(|&s| s != slot) {
            self.detach_keep(host, old)?;
        }
        host.set_store_heads(slot, &heads)?;
        self.slots.insert(slot, key.to_string());
        Ok(())
    }

    /// `slot` leaves its session (evicted, freed, or given to another key): its store heads
    /// are zeroed and, with `ephemeral`, the session's store is dropped.
    pub fn detach(&mut self, host: &mut dyn StoreHost, slot: usize) -> Result<()> {
        let Some(key) = self.slots.get(&slot).cloned() else {
            return Ok(());
        };
        self.detach_keep(host, slot)?;
        if self.ephemeral {
            self.held.remove(&key);
        }
        Ok(())
    }

    fn detach_keep(&mut self, host: &mut dyn StoreHost, slot: usize) -> Result<()> {
        if self.slots.remove(&slot).is_some() {
            host.set_store_heads(slot, &vec![0f32; HEAD_FLOATS])?;
        }
        Ok(())
    }

    /// The session `key` ends (`POST /v1/memory/end`): its slot is detached and its store
    /// dropped (ephemeral or not). Whether it had one.
    pub fn end(&mut self, host: &mut dyn StoreHost, key: &str) -> Result<bool> {
        if let Some(slot) = self.slot_of(key) {
            self.detach(host, slot)?;
        }
        Ok(self.held.remove(key).is_some())
    }

    /// Session stores, bytes held in RAM.
    pub fn summary(&self) -> Value {
        json!({
            "sessions": self.held.len(),
            "resident_slots": self.slots.len(),
            "bytes": self.held.len() * BLOB_BYTES,
            "ephemeral": self.ephemeral,
        })
    }
}

/// A store blob as text: each file's name, length and token ids, and with `decode` the text.
pub fn dump(blob: &[u8], decode: Option<&dyn Fn(&[u32]) -> Result<String>>) -> Result<String> {
    use std::fmt::Write as _;
    let heads = blob_to_heads(blob)?;
    let bad = audit_heads(&heads);
    let dev = CpuDevice::new();
    let mut s = vec![0f32; GDN_STATE];
    MemStore::<CpuDevice>::restore_heads(&mut s, &heads);
    let mut state = dev.upload_f32(&s);
    // The store's own windows pin g = 1 and β, so any finite params read it; these keep the
    // read's arithmetic plain.
    let (conv, dt, a, norm) = (
        dev.upload_f32(&[0.0]),
        dev.upload_f32(&[0.0; GDN_HV]),
        dev.upload_f32(&[-1.0; GDN_HV]),
        dev.upload_f32(&[1.0; GDN_D]),
    );
    let params = || GdnParams {
        conv: &conv,
        dt_bias: &dt,
        ssm_a: &a,
        norm: &norm,
    };
    let mut out = String::new();
    if !bad.is_empty() {
        writeln!(out, "warning: poisoned store heads {bad:?}; reads may fail")?;
    }
    let mut st = MemStore::open(&dev, params(), 1e-6, &mut state)?;
    let cap = st.capacity();
    writeln!(
        out,
        "{} files, {} tokens, {}/{} content rows",
        cap.files,
        cap.tokens,
        cap.rows_used,
        super::memstore::ROWS
    )?;
    for (name, len) in st.list(&mut state)? {
        let ids = st.get(&mut state, &name)?.unwrap_or_default();
        writeln!(
            out,
            "\n== {} ({len} tokens)",
            String::from_utf8_lossy(&name)
        )?;
        writeln!(
            out,
            "ids: {}",
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
        )?;
        if let Some(d) = decode {
            writeln!(out, "text: {}", d(&ids)?)?;
        }
    }
    Ok(out)
}

/// `tang-llm memory dump <blob> [--tokenizer tokenizer.json|model.gguf]`.
pub fn cli(args: &[String]) -> Result<()> {
    let usage =
        "usage: tang-llm memory dump <blob> [--tokenizer <tokenizer.json | first shard.gguf>]";
    ensure!(args.first().map(String::as_str) == Some("dump"), "{usage}");
    let path = args.get(1).context(usage)?;
    let mut tok_path = None;
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tokenizer" => tok_path = Some(it.next().context(usage)?.clone()),
            o => bail!("unknown option {o}; {usage}"),
        }
    }
    let blob = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let text = match tok_path {
        None => dump(&blob, None)?,
        Some(p) if p.ends_with(".gguf") => {
            let g = crate::gguf::Gguf::open(std::path::Path::new(&p))?;
            let t = super::tokenize::FlashTokenizer::from_gguf(&g)?;
            dump(&blob, Some(&|ids: &[u32]| t.decode(ids)))?
        }
        Some(p) => {
            let t = tokenizers::Tokenizer::from_file(&p)
                .map_err(|e| anyhow::anyhow!("reading {p}: {e}"))?;
            dump(
                &blob,
                Some(&|ids: &[u32]| t.decode(ids, false).map_err(|e| anyhow::anyhow!("{e}"))),
            )?
        }
    };
    print!("{text}");
    Ok(())
}

/// A state file for the disk tier (`TANGRS02`): the token ids, the running state's parts, the
/// positional parts and, with `store`, the [`StoreMeta::trailer`]. With `blank`, the store
/// heads of part `2 · store.gdn_layer` are written as zeros (the state itself is untouched).
pub fn encode_state_file(
    ids: &[u32],
    parts: &[&[f32]],
    pos_parts: &[&[u8]],
    store: Option<StoreMeta>,
    blank: bool,
) -> Vec<u8> {
    let floats: usize = parts.iter().map(|p| p.len()).sum();
    let mut bytes: Vec<u8> = Vec::with_capacity(4 * floats + 4 * ids.len() + 128);
    bytes.extend_from_slice(b"TANGRS02");
    bytes.extend_from_slice(&(ids.len() as u64).to_le_bytes());
    for &i in ids {
        bytes.extend_from_slice(&i.to_le_bytes());
    }
    let masked = store.filter(|_| blank).map(|m| 2 * m.gdn_layer);
    bytes.extend_from_slice(&(parts.len() as u64).to_le_bytes());
    for (k, p) in parts.iter().enumerate() {
        bytes.extend_from_slice(&(p.len() as u64).to_le_bytes());
        if masked == Some(k) && p.len() == GDN_STATE {
            let mut q = p.to_vec();
            blank_heads(&mut q);
            q.iter()
                .for_each(|x| bytes.extend_from_slice(&x.to_le_bytes()));
        } else {
            p.iter()
                .for_each(|x| bytes.extend_from_slice(&x.to_le_bytes()));
        }
    }
    bytes.extend_from_slice(&(pos_parts.len() as u64).to_le_bytes());
    for p in pos_parts {
        bytes.extend_from_slice(&(p.len() as u64).to_le_bytes());
        bytes.extend_from_slice(p);
    }
    if let Some(m) = store {
        bytes.extend_from_slice(&m.trailer());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use tang_compute::CpuDevice;

    const EPS: f32 = 1e-6;

    /// Slots of CPU-device layer states with base heads full of noise, standing in for an
    /// engine's slots of the store layer.
    struct Host {
        dev: CpuDevice,
        params: [<CpuDevice as ComputeDevice>::Buffer; 4],
        states: Vec<<CpuDevice as ComputeDevice>::Buffer>,
    }

    fn noise(seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..GDN_STATE)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    impl Host {
        fn new(slots: usize) -> Self {
            let dev = CpuDevice::new();
            let dt: Vec<f32> = (0..GDN_HV).map(|h| (h as f32 * 0.37).sin()).collect();
            let a: Vec<f32> = (0..GDN_HV).map(|h| -0.5 - (h % 7) as f32).collect();
            let norm: Vec<f32> = (0..GDN_D).map(|j| 0.6 + (j % 5) as f32 * 0.3).collect();
            let params = [
                dev.upload_f32(&[0.0]),
                dev.upload_f32(&dt),
                dev.upload_f32(&a),
                dev.upload_f32(&norm),
            ];
            let states = (0..slots)
                .map(|i| {
                    let mut s = noise(i as u64 + 1);
                    blank_heads(&mut s);
                    dev.upload_f32(&s)
                })
                .collect();
            Host {
                dev,
                params,
                states,
            }
        }
        fn state(&self, slot: usize) -> Vec<f32> {
            self.dev.download(&self.states[slot])
        }
    }

    impl StoreHost for Host {
        fn store_heads(&mut self, slot: usize) -> Result<Vec<f32>> {
            Ok(MemStore::<CpuDevice>::snapshot_heads(&self.state(slot)))
        }
        fn set_store_heads(&mut self, slot: usize, heads: &[f32]) -> Result<()> {
            let mut s = self.state(slot);
            MemStore::<CpuDevice>::restore_heads(&mut s, heads);
            self.states[slot] = self.dev.upload_f32(&s);
            Ok(())
        }
        fn store_op(&mut self, slot: usize, op: &StoreOp) -> Result<OpOut> {
            let p = GdnParams {
                conv: &self.params[0],
                dt_bias: &self.params[1],
                ssm_a: &self.params[2],
                norm: &self.params[3],
            };
            run_op(&self.dev, p, EPS, &mut self.states[slot], op)
        }
    }

    fn put(name: &str, ids: &[u32]) -> StoreOp {
        StoreOp::Put {
            name: name.into(),
            ids: ids.to_vec(),
        }
    }
    fn get(name: &str) -> StoreOp {
        StoreOp::Get { name: name.into() }
    }

    /// A blob holding `files`, built through the store itself.
    fn blob_with(files: &[(&str, Vec<u32>)]) -> Vec<u8> {
        let mut h = Host::new(1);
        let mut s = Sessions::new(false);
        for (n, ids) in files {
            s.op(&mut h, "build", &put(n, ids)).unwrap();
        }
        s.export("build").unwrap()
    }

    fn heads_zero(h: &Host, slot: usize) -> bool {
        MemStore::<CpuDevice>::snapshot_heads(&h.state(slot))
            .iter()
            .all(|&x| x == 0.0)
    }

    fn base_eq(a: &[f32], b: &[f32]) -> bool {
        let (mut a, mut b) = (a.to_vec(), b.to_vec());
        blank_heads(&mut a);
        blank_heads(&mut b);
        a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn op_json_parses() {
        let op = StoreOp::parse(&json!({"op": "put", "name": "a", "ids": [1, 2]})).unwrap();
        assert_eq!(op, put("a", &[1, 2]));
        assert!(StoreOp::parse(&json!({"op": "put", "name": "a", "ids": [1 << 18]})).is_err());
        assert!(StoreOp::parse(&json!({"op": "nope"})).is_err());
        assert_eq!(
            StoreOp::parse(&json!({"op": "list"})).unwrap(),
            StoreOp::List
        );
    }

    /// Load a client blob for a session, then get its files back through store windows on a
    /// slot; the borrowed slot 0's own state (base and store heads) is bitwise as it was.
    #[test]
    fn memserve_load_then_get() {
        let files = [("notes/a.md", vec![5u32, 6, 7]), ("b", (0..300).collect())];
        let blob = blob_with(&files);
        let mut h = Host::new(2);
        let before = h.state(0);
        let mut s = Sessions::new(false);
        s.load(&mut h, "k1", &blob).unwrap();
        for (n, ids) in &files {
            assert_eq!(
                s.op(&mut h, "k1", &get(n)).unwrap(),
                OpOut::Got(Some(ids.clone()))
            );
        }
        assert_eq!(
            s.op(&mut h, "k1", &get("missing")).unwrap(),
            OpOut::Got(None)
        );
        assert_eq!(
            h.state(0).iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            before.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        // another session sees nothing
        assert_eq!(s.op(&mut h, "k2", &get("b")).unwrap(), OpOut::Got(None));
        assert!(!s.has("k2"));
        // attached to its slot, the slot carries the store and ops run there
        s.attach(&mut h, 1, Some("k1")).unwrap();
        assert_eq!(s.resident(1), Some("k1"));
        assert_eq!(
            MemStore::<CpuDevice>::snapshot_heads(&h.state(1)),
            blob_to_heads(&blob).unwrap()
        );
        assert_eq!(
            s.op(&mut h, "k1", &get("notes/a.md")).unwrap(),
            OpOut::Got(Some(vec![5, 6, 7]))
        );
    }

    /// Export after read-only ops is bitwise the loaded blob; after writes it is the new store,
    /// which a fresh server loads and reads back.
    #[test]
    fn memserve_export_after_ops() {
        let blob = blob_with(&[("x", vec![1, 2, 3])]);
        let mut h = Host::new(2);
        let mut s = Sessions::new(false);
        s.load(&mut h, "k", &blob).unwrap();
        s.op(&mut h, "k", &get("x")).unwrap();
        s.op(&mut h, "k", &StoreOp::List).unwrap();
        assert!(
            s.export("k").unwrap() == blob,
            "read-only ops changed the store"
        );
        s.attach(&mut h, 1, Some("k")).unwrap();
        s.op(&mut h, "k", &put("y", &[9; 200])).unwrap();
        s.op(&mut h, "k", &StoreOp::Delete { name: "x".into() })
            .unwrap();
        let out = s.export("k").unwrap();
        assert_eq!(
            heads_to_blob(&MemStore::<CpuDevice>::snapshot_heads(&h.state(1))),
            out,
            "the slot carries what is exported"
        );
        let mut h2 = Host::new(1);
        let mut s2 = Sessions::new(false);
        s2.load(&mut h2, "k", &out).unwrap();
        assert_eq!(
            s2.op(&mut h2, "k", &get("y")).unwrap(),
            OpOut::Got(Some(vec![9; 200]))
        );
        assert_eq!(s2.op(&mut h2, "k", &get("x")).unwrap(), OpOut::Got(None));
        assert_eq!(
            s2.op(&mut h2, "k", &StoreOp::List).unwrap(),
            OpOut::Listed(vec![(b"y".to_vec(), 200)])
        );
        assert!(matches!(s.export("nobody"), Err(ExportError::Missing)));
    }

    /// `memory_ephemeral`: the slot given to another key has its store heads zeroed (base heads
    /// untouched) and the session's store is gone; a state file written for that slot carries
    /// zeroed store heads (and the store trailer), even if the heads were live.
    #[test]
    fn memserve_ephemeral_eviction_and_disk() {
        let blob = blob_with(&[("secret", vec![42; 10])]);
        let mut h = Host::new(2);
        let mut s = Sessions::new(true);
        s.load(&mut h, "k", &blob).unwrap();
        s.attach(&mut h, 1, Some("k")).unwrap();
        assert!(!heads_zero(&h, 1));

        // the disk tier, while the store is live in the slot
        let meta = StoreMeta {
            gdn_layer: 1,
            layer: 3,
        };
        let live = h.state(1);
        let hist = vec![0.25f32; 16];
        let other_gdn = noise(99);
        let parts: Vec<&[f32]> = vec![&other_gdn, &hist, &live, &hist, &[]];
        let file = encode_state_file(&[1, 2, 3], &parts, &[&[7u8; 5]], Some(meta), true);
        // parse it back: header, 3 ids, part count, then parts
        let f32s = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let mut at = 8 + 8 + 12 + 8;
        let mut got = Vec::new();
        for _ in 0..parts.len() {
            let n = u64::from_le_bytes(file[at..at + 8].try_into().unwrap()) as usize;
            got.push(f32s(&file[at + 8..at + 8 + 4 * n]));
            at += 8 + 4 * n;
        }
        assert!(MemStore::<CpuDevice>::snapshot_heads(&got[2])
            .iter()
            .all(|&x| x == 0.0));
        assert!(base_eq(&got[2], &live));
        assert_eq!(
            got[0], other_gdn,
            "other GDN layers are written as they are"
        );
        let tail = &file[file.len() - meta.trailer().len()..];
        assert_eq!(StoreMeta::from_trailer(tail), Some(meta));

        // eviction: the slot goes to another key
        let base = h.state(1);
        s.attach(&mut h, 1, Some("other")).unwrap();
        assert!(heads_zero(&h, 1));
        assert!(base_eq(&h.state(1), &base));
        assert!(!s.has("k"), "ephemeral store outlived its slot");
        assert!(s.is_empty());
        assert!(matches!(s.export("k"), Err(ExportError::Missing)));

        // not ephemeral: the slot is zeroed but the session keeps its store
        let mut s = Sessions::new(false);
        s.load(&mut h, "k", &blob).unwrap();
        s.attach(&mut h, 1, Some("k")).unwrap();
        s.attach(&mut h, 1, None).unwrap();
        assert!(heads_zero(&h, 1));
        assert_eq!(s.export("k").unwrap(), blob);
        // session end drops it either way
        assert!(s.end(&mut h, "k").unwrap());
        assert!(!s.has("k"));
    }

    /// The audit runs before an export: a store whose slot took one model-like token (g < 1,
    /// dense k) is refused with the poisoned heads named; a poisoned blob is refused at load.
    #[test]
    fn memserve_audit_flags_poison_before_export() {
        let blob = blob_with(&[("x", vec![1, 2, 3])]);
        let mut h = Host::new(2);
        let mut s = Sessions::new(false);
        s.load(&mut h, "k", &blob).unwrap();
        // poison the session's copy as a model token through its slot would
        let mut heads = blob_to_heads(&blob).unwrap();
        for x in heads.iter_mut().step_by(97) {
            *x = *x * 0.93 + 0.01;
        }
        s.held.insert("k".into(), heads.clone());
        match s.export("k") {
            Err(ExportError::Poisoned(bad)) => assert!(!bad.is_empty()),
            other => panic!("poisoned store exported: {:?}", other.map(|b| b.len())),
        }
        let mut s2 = Sessions::new(false);
        assert!(s2.load(&mut h, "k", &heads_to_blob(&heads)).is_err());
        assert!(s2.load(&mut h, "k", &blob[..100]).is_err());
    }

    #[test]
    fn memserve_dump_reads_names_and_ids() {
        let blob = blob_with(&[("notes/a.md", vec![5, 6, 7]), ("b", vec![1])]);
        let text = dump(
            &blob,
            Some(&|ids: &[u32]| Ok(format!("<{} ids>", ids.len()))),
        )
        .unwrap();
        assert!(text.contains("2 files, 4 tokens"), "{text}");
        assert!(
            text.contains("== notes/a.md (3 tokens)\nids: 5 6 7\ntext: <3 ids>"),
            "{text}"
        );
        assert!(text.contains("== b (1 tokens)\nids: 1"), "{text}");
    }

    /// The store windows through the CUDA device's `gdn_step` (what the engines route
    /// `store_op` through) leave a state bitwise equal to the CPU device's on the same inputs,
    /// and return the same results. Needs a GPU (panics without one under
    /// `TANG_REQUIRE_CUDA=1`, else skips).
    #[cfg(feature = "cuda")]
    #[test]
    fn memserve_cuda_ops_match_cpu_bitwise() {
        use tang_compute::CudaComputeDevice;
        let Ok(gpu) = CudaComputeDevice::new() else {
            assert!(
                std::env::var("TANG_REQUIRE_CUDA").is_err(),
                "TANG_REQUIRE_CUDA=1 and no CUDA device"
            );
            eprintln!("no CUDA device: skipping");
            return;
        };
        let cpu = Host::new(1);
        let init = cpu.state(0);
        let up = |v: &[f32]| gpu.upload_f32(v);
        let cp: Vec<Vec<f32>> = cpu.params.iter().map(|b| cpu.dev.download(b)).collect();
        let gb = [up(&cp[0]), up(&cp[1]), up(&cp[2]), up(&cp[3])];
        let mut gs = gpu.upload_f32(&init);
        let mut cs = cpu.dev.upload_f32(&init);
        let mut ops = Vec::new();
        for i in 0..60u32 {
            let n = format!("f{}", i % 23);
            ops.push(match i % 5 {
                0 | 1 => put(
                    &n,
                    &(0..(i * 37) % 900)
                        .map(|x| (x * 7919 + i) % (1 << 18))
                        .collect::<Vec<_>>(),
                ),
                2 => get(&n),
                3 => StoreOp::Delete { name: n.into() },
                _ => StoreOp::List,
            });
        }
        for (i, op) in ops.iter().enumerate() {
            let pg = GdnParams {
                conv: &gb[0],
                dt_bias: &gb[1],
                ssm_a: &gb[2],
                norm: &gb[3],
            };
            let pc = GdnParams {
                conv: &cpu.params[0],
                dt_bias: &cpu.params[1],
                ssm_a: &cpu.params[2],
                norm: &cpu.params[3],
            };
            let og = run_op(&gpu, pg, EPS, &mut gs, op).unwrap();
            let oc = run_op(&cpu.dev, pc, EPS, &mut cs, op).unwrap();
            assert_eq!(og, oc, "op {i} {op:?}");
            let (a, b) = (gpu.download(&gs), cpu.dev.download(&cs));
            let diff = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(diff, 0, "op {i}: {diff} state floats differ");
        }
        assert!(base_eq(&gpu.download(&gs), &init), "base heads moved");
        eprintln!("memserve: {} ops bitwise equal on CUDA and CPU", ops.len());
    }
}
