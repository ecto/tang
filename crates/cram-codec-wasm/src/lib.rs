//! Browser decoder for cram-codec GPU streams (TCG1, TCG3 static and
//! learned): the native WGSL kernels through wgpu's WebGPU backend, plus a
//! CPU fallback running the same integer reference decoder in WASM. Both
//! paths are byte-identical to native `gpuwav::decode_box_parsed` /
//! `gpuwav3::decode3_box`.
//!
//! JS surface (see `README.md`):
//! - `registerModel(bytes)` — a pretrained `.tcm` net, for learned TCG3 streams
//!   that only carry its fingerprint.
//! - `streamInfo(bytes)` — JSON header summary.
//! - `decodeCpu(bytes)` — `Float32Array`, synchronous.
//! - `TcGpu.create()` → `await gpu.decode(bytes)` — `Float32Array`;
//!   `gpu.lastTimes()` — JSON split of the last decode.
//! - `fnv1a(f32s)` — the gate hash (FNV-1a 64 over the `f32` bit patterns).

use cram_codec::gpudec::{GpuDecoder, GpuTimes};
use cram_codec::{gpuwav, gpuwav3, learned};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// Stream format, by magic (an `0xA0`/`0xA1` auto tag byte is not expected
/// here: GPU streams are written bare).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Format {
    /// TCG1 bricks.
    V1,
    /// TCG3 code-blocks (static or learned probabilities).
    V3,
}

/// Format of `bytes`.
pub fn format_of(bytes: &[u8]) -> Result<Format, String> {
    match bytes.get(..4) {
        Some(m) if m == gpuwav::MAGIC => Ok(Format::V1),
        Some(m) if m == gpuwav3::MAGIC3 => Ok(Format::V3),
        // TCG3 carrying its own decoder (verified + compiled by parse3)
        Some(m) if m == cram_codec::shipdec::MAGIC => Ok(Format::V3),
        _ => Err("not a TCG1/TCG3 stream".into()),
    }
}

/// (lanes, longest line) a GPU decoder must be built for.
pub fn decoder_key(bytes: &[u8]) -> Result<(usize, usize), String> {
    Ok(match format_of(bytes)? {
        Format::V1 => {
            let ps = gpuwav::parse(bytes).map_err(|e| format!("{e:?}"))?;
            (
                ps.lanes,
                ps.shapes.iter().flat_map(|s| s.dims).max().unwrap_or(2),
            )
        }
        Format::V3 => {
            let ps = gpuwav3::parse3(bytes).map_err(|e| format!("{e:?}"))?;
            (ps.lanes, ps.d3.iter().copied().max().unwrap_or(2))
        }
    })
}

/// CPU reference decode (single thread) to `f32` bits, field layout.
pub fn decode_cpu_bits(bytes: &[u8]) -> Result<Vec<u32>, String> {
    let e = |e| format!("{e:?}");
    match format_of(bytes)? {
        Format::V1 => {
            let ps = gpuwav::parse(bytes).map_err(e)?;
            gpuwav::decode_box_parsed(&ps, bytes, [0; 3], ps.d3, 1).map_err(e)
        }
        Format::V3 => {
            let ps = gpuwav3::parse3(bytes).map_err(e)?;
            Ok(gpuwav3::decode3_box_parsed(&ps, bytes, [0; 3], ps.d3, 1)
                .map_err(e)?
                .bits)
        }
    }
}

/// FNV-1a 64 of `f32` bit patterns (same as `gpubench encode3`'s `.hash`).
pub fn fnv1a_bits(v: &[u32]) -> u64 {
    v.iter().fold(0xcbf29ce484222325u64, |h, &x| {
        (h ^ x as u64).wrapping_mul(0x100000001b3)
    })
}

fn bits_to_f32(v: Vec<u32>) -> js_sys::Float32Array {
    let a = js_sys::Float32Array::new_with_length(v.len() as u32);
    // u32 → f32 is a bit reinterpretation; copy through a Uint32Array view
    js_sys::Uint32Array::new(&a.buffer()).copy_from(&v);
    a
}

fn f32_bits(a: &js_sys::Float32Array) -> Vec<u32> {
    js_sys::Uint32Array::new_with_byte_offset_and_length(&a.buffer(), a.byte_offset(), a.length())
        .to_vec()
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
}

/// Route Rust panics to `console.error` (runs once on module start).
#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|p| console_error(&p.to_string())));
}

/// Register a pretrained `.tcm` network; returns its fingerprint (hex).
#[wasm_bindgen(js_name = registerModel)]
pub fn register_model(bytes: &[u8]) -> Result<String, JsError> {
    let net = learned::Net::from_bytes(bytes).map_err(|e| JsError::new(&format!("{e:?}")))?;
    let fp = net.fingerprint();
    learned::register(Arc::new(net));
    Ok(format!("{fp:016x}"))
}

/// Header summary as JSON: format, dims, lanes, learned, cells, bytes.
#[wasm_bindgen(js_name = streamInfo)]
pub fn stream_info(bytes: &[u8]) -> Result<String, JsError> {
    let je = |e: String| JsError::new(&e);
    let fmt = format_of(bytes).map_err(je)?;
    let (dims, lanes, learned) = match fmt {
        Format::V1 => {
            let ps = gpuwav::parse(bytes).map_err(|e| je(format!("{e:?}")))?;
            (ps.dims, ps.lanes, false)
        }
        Format::V3 => {
            let ps = gpuwav3::parse3(bytes).map_err(|e| je(format!("{e:?}")))?;
            (ps.dims, ps.lanes, ps.net.is_some())
        }
    };
    let cells: usize = dims.iter().product();
    Ok(format!(
        "{{\"format\":\"{}\",\"dims\":{dims:?},\"lanes\":{lanes},\"learned\":{learned},\"cells\":{cells},\"bytes\":{}}}",
        if fmt == Format::V1 { "TCG1" } else { "TCG3" },
        bytes.len()
    ))
}

/// CPU (WASM, one thread) decode.
#[wasm_bindgen(js_name = decodeCpu)]
pub fn decode_cpu(bytes: &[u8]) -> Result<js_sys::Float32Array, JsError> {
    decode_cpu_bits(bytes)
        .map(bits_to_f32)
        .map_err(|e| JsError::new(&e))
}

/// FNV-1a 64 (hex) of a decoded array's bit patterns.
#[wasm_bindgen]
pub fn fnv1a(a: &js_sys::Float32Array) -> String {
    format!("{:016x}", fnv1a_bits(&f32_bits(a)))
}

/// WebGPU decoder. Pipelines are built per (lanes, line length) and cached,
/// so the first stream of each kind pays shader compilation.
#[wasm_bindgen]
pub struct TcGpu {
    decs: Vec<((usize, usize), Arc<GpuDecoder>)>,
    last: GpuTimes,
    adapter: String,
}

#[wasm_bindgen]
impl TcGpu {
    /// Open the WebGPU adapter (fails where `navigator.gpu` is missing).
    pub async fn create() -> Result<TcGpu, JsError> {
        // a throwaway decoder proves the adapter + kernels compile
        let d = GpuDecoder::new_async(8, 64)
            .await
            .map_err(|e| JsError::new(&e))?;
        let adapter = d.name.clone();
        Ok(TcGpu {
            decs: vec![((8, 64), Arc::new(d))],
            last: GpuTimes::default(),
            adapter,
        })
    }

    /// Adapter description.
    pub fn adapter(&self) -> String {
        self.adapter.clone()
    }

    async fn dec(&mut self, key: (usize, usize)) -> Result<Arc<GpuDecoder>, JsError> {
        // line buffers are sized by maxl; round up so similar fields share pipelines
        let key = (key.0, key.1.max(2).next_power_of_two());
        if let Some(d) = self.decs.iter().find(|d| d.0 == key) {
            return Ok(d.1.clone());
        }
        let d = Arc::new(
            GpuDecoder::new_async(key.0, key.1)
                .await
                .map_err(|e| JsError::new(&e))?,
        );
        self.decs.push((key, d.clone()));
        Ok(d)
    }

    /// Build (and cache) the pipelines a stream needs without decoding it.
    pub async fn warm(&mut self, bytes: Vec<u8>) -> Result<(), JsError> {
        let key = decoder_key(&bytes).map_err(|e| JsError::new(&e))?;
        self.dec(key).await.map(|_| ())
    }

    /// Decode on the GPU.
    pub async fn decode(&mut self, bytes: Vec<u8>) -> Result<js_sys::Float32Array, JsError> {
        let je = |e: String| JsError::new(&e);
        let fmt = format_of(&bytes).map_err(je)?;
        let d = self.dec(decoder_key(&bytes).map_err(je)?).await?;
        let r = match fmt {
            Format::V1 => d.decode_async(&bytes).await,
            Format::V3 => d.decode3_async(&bytes).await,
        };
        let (v, t) = r.map_err(|e| je(format!("{e:?}")))?;
        self.last = t;
        Ok(bits_to_f32(v))
    }

    /// Seconds split of the last decode, JSON: parse, upload, kernels, readback.
    #[wasm_bindgen(js_name = lastTimes)]
    pub fn last_times(&self) -> String {
        let t = self.last;
        format!(
            "{{\"parse_s\":{},\"upload_s\":{},\"kernels_s\":{},\"readback_s\":{}}}",
            t.parse_s, t.upload_s, t.kernels_s, t.readback_s
        )
    }
}
