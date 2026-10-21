//! General-purpose container: one front door for any bytes.
//!
//! [`compress`] runs a set of candidate *arms* (stored, zstd, xz, brotli and,
//! only when the [`Hint`] carries an error bound, the lossy cram float
//! portfolio) and keeps the smallest. [`decompress`] reads the method tag and
//! undoes it.
//!
//! # Stream format
//!
//! ```text
//! "CRM1"           4 bytes magic
//! method           1 byte  (see [`Arm::tag`])
//! original length  LEB128 varint (bytes)
//! crc32            4 bytes LE, CRC-32 (IEEE) of the decoded output
//! payload          the arm's stream
//! ```
//!
//! For a lossless arm the checksum is over the original input, so a corrupt
//! stream fails closed. For the lossy arm it is over the reconstruction the
//! encoder produced by decoding its own stream; decode recomputes it.
//!
//! # Guarantee
//!
//! `compress(x).len() <= min(arm(x).len()) + header` over the enabled arms,
//! where the header is `4 + 1 + varint(len) + 4` bytes. Lossless input (any
//! hint without a bound) never takes the lossy arm.

use crate::codec::ErrorBound;
use crate::{decode_any, wavcodec, Decoded};
use std::io::{Read, Write};

/// Stream magic.
pub const MAGIC: &[u8; 4] = b"CRM1";

/// What the caller knows about the input.
#[derive(Clone, Debug, PartialEq)]
pub enum Hint {
    /// Nothing; sniff the bytes.
    Auto,
    /// Opaque bytes (same policy as `Auto`).
    Bytes,
    /// Text (same lossless arms; no sniff skip).
    Text,
    /// Little-endian f32 samples of shape `dims` (row-major). With `bound`
    /// set, the lossy cram float arm may be used and the reconstruction is
    /// only guaranteed within that bound. With `bound: None` the data is
    /// treated as lossless bytes.
    F32Field {
        /// Shape; its product times 4 must equal the input length.
        dims: Vec<usize>,
        /// Error bound for the lossy arm, or `None` for lossless.
        bound: Option<ErrorBound>,
    },
}

/// A compression method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arm {
    /// Raw copy.
    Stored,
    /// zstd level 3.
    Zstd3,
    /// zstd level 19.
    Zstd19,
    /// xz / LZMA2, preset 9.
    Xz9,
    /// brotli quality 11, window 24.
    Brotli11,
    /// cram lossy f32 portfolio ([`crate::encode_auto_f32`]); needs a bound.
    CramLossy,
}

/// Every lossless arm, in tie-break order.
pub const LOSSLESS_ARMS: [Arm; 5] = [Arm::Stored, Arm::Zstd3, Arm::Zstd19, Arm::Xz9, Arm::Brotli11];

impl Arm {
    /// Method tag byte.
    pub fn tag(self) -> u8 {
        match self {
            Arm::Stored => 0,
            Arm::Zstd3 => 1,
            Arm::Zstd19 => 2,
            Arm::Xz9 => 3,
            Arm::Brotli11 => 4,
            Arm::CramLossy => 5,
        }
    }
    /// Arm for a tag byte.
    pub fn from_tag(t: u8) -> Option<Arm> {
        Some(match t {
            0 => Arm::Stored,
            1 => Arm::Zstd3,
            2 => Arm::Zstd19,
            3 => Arm::Xz9,
            4 => Arm::Brotli11,
            5 => Arm::CramLossy,
            _ => return None,
        })
    }
    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Arm::Stored => "stored",
            Arm::Zstd3 => "zstd-3",
            Arm::Zstd19 => "zstd-19",
            Arm::Xz9 => "xz-9",
            Arm::Brotli11 => "brotli-11",
            Arm::CramLossy => "cram-lossy",
        }
    }
    /// Whether the arm reproduces its input exactly.
    pub fn is_lossless(self) -> bool {
        self != Arm::CramLossy
    }
}

/// Container error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Missing or wrong magic.
    BadMagic,
    /// Unknown method tag.
    UnknownMethod(u8),
    /// Header cut short or malformed.
    Truncated,
    /// The payload did not decode.
    Payload(String),
    /// Decoded length differs from the header.
    LengthMismatch,
    /// Checksum differs: the stream is corrupt.
    Checksum,
    /// The hint does not fit the input (e.g. dims × 4 ≠ length).
    BadHint(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BadMagic => write!(f, "not a CRM1 stream"),
            Error::UnknownMethod(t) => write!(f, "unknown method tag {t}"),
            Error::Truncated => write!(f, "truncated header"),
            Error::Payload(e) => write!(f, "payload: {e}"),
            Error::LengthMismatch => write!(f, "decoded length differs from header"),
            Error::Checksum => write!(f, "checksum mismatch (corrupt stream)"),
            Error::BadHint(e) => write!(f, "bad hint: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// What [`compress_report`] did.
#[derive(Clone, Debug)]
pub struct Report {
    /// The arm whose payload was kept.
    pub pick: Arm,
    /// Every arm tried: `(arm, payload bytes)`.
    pub tried: Vec<(Arm, usize)>,
    /// Container header bytes.
    pub header: usize,
}

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn read_varint(b: &[u8]) -> Option<(u64, usize)> {
    let mut v = 0u64;
    for (i, &c) in b.iter().enumerate().take(10) {
        let part = (c & 0x7f) as u64;
        if i == 9 && part > 1 {
            return None;
        }
        v |= part << (7 * i);
        if c & 0x80 == 0 {
            return Some((v, i + 1));
        }
    }
    None
}

/// Header bytes for an input of `len` bytes.
pub fn header_len(len: usize) -> usize {
    let mut v = Vec::new();
    varint(len as u64, &mut v);
    4 + 1 + v.len() + 4
}

/// Magic-number sniff: is this an already-compressed format (gzip, zstd,
/// xz, bzip2, png, jpeg, gif, webp, zip, 7z, mp4/mov)?
pub fn looks_compressed(x: &[u8]) -> bool {
    let p = |m: &[u8]| x.starts_with(m);
    p(&[0x1f, 0x8b])
        || p(&[0x28, 0xb5, 0x2f, 0xfd])
        || p(&[0xfd, b'7', b'z', b'X', b'Z', 0])
        || p(b"BZh")
        || p(&[0x89, b'P', b'N', b'G'])
        || p(&[0xff, 0xd8, 0xff])
        || p(b"GIF8")
        || (p(b"RIFF") && x.get(8..12) == Some(b"WEBP"))
        || p(b"PK\x03\x04")
        || p(&[b'7', b'z', 0xbc, 0xaf, 0x27, 0x1c])
        || x.get(4..8) == Some(b"ftyp")
        || p(MAGIC)
}

/// The arms [`compress`] tries for `x` under `hint`, in tie-break order.
pub fn plan(x: &[u8], hint: &Hint) -> Vec<Arm> {
    let lossy = matches!(hint, Hint::F32Field { bound: Some(_), .. });
    let sniff = matches!(hint, Hint::Auto | Hint::Bytes);
    let mut arms: Vec<Arm> = if sniff && looks_compressed(x) {
        vec![Arm::Stored, Arm::Zstd3]
    } else {
        LOSSLESS_ARMS.to_vec()
    };
    if lossy {
        arms.push(Arm::CramLossy);
    }
    arms
}

fn f32s(x: &[u8]) -> Vec<f32> {
    x.as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Encode `x` with one arm, returning the bare payload. `CramLossy` needs an
/// `F32Field` hint with a bound.
pub fn encode_arm(arm: Arm, x: &[u8], hint: &Hint) -> Result<Vec<u8>, Error> {
    let io = |e: std::io::Error| Error::Payload(e.to_string());
    Ok(match arm {
        Arm::Stored => x.to_vec(),
        Arm::Zstd3 => zstd::bulk::compress(x, 3).map_err(io)?,
        Arm::Zstd19 => zstd::bulk::compress(x, 19).map_err(io)?,
        Arm::Xz9 => {
            let mut e = xz2::write::XzEncoder::new(Vec::new(), 9);
            e.write_all(x).map_err(io)?;
            e.finish().map_err(io)?
        }
        Arm::Brotli11 => {
            let mut out = Vec::new();
            {
                let mut w = brotli::CompressorWriter::new(&mut out, 1 << 16, 11, 24);
                w.write_all(x).map_err(io)?;
            }
            out
        }
        Arm::CramLossy => {
            let Hint::F32Field {
                dims,
                bound: Some(b),
            } = hint
            else {
                return Err(Error::BadHint("cram-lossy needs F32Field with a bound".into()));
            };
            if dims.iter().product::<usize>() * 4 != x.len() || dims.is_empty() {
                return Err(Error::BadHint(format!(
                    "dims {dims:?} do not match {} bytes",
                    x.len()
                )));
            }
            let cfg = wavcodec::WavConfig::new(*b);
            crate::encode_auto_f32(&f32s(x), dims, &cfg).0
        }
    })
}

/// Decode one arm's bare payload; `len` is the expected output length (the
/// decoders never produce more than `len + 1` bytes).
pub fn decode_arm(arm: Arm, payload: &[u8], len: usize) -> Result<Vec<u8>, Error> {
    let io = |e: std::io::Error| Error::Payload(e.to_string());
    let lim = len as u64 + 1;
    let out = match arm {
        Arm::Stored => payload.to_vec(),
        Arm::Zstd3 | Arm::Zstd19 => {
            let mut out = Vec::with_capacity(len);
            zstd::stream::read::Decoder::new(payload)
                .map_err(io)?
                .take(lim)
                .read_to_end(&mut out)
                .map_err(io)?;
            out
        }
        Arm::Xz9 => {
            let mut out = Vec::with_capacity(len);
            xz2::read::XzDecoder::new(payload)
                .take(lim)
                .read_to_end(&mut out)
                .map_err(io)?;
            out
        }
        Arm::Brotli11 => {
            let mut out = Vec::with_capacity(len);
            brotli::Decompressor::new(payload, 1 << 16)
                .take(lim)
                .read_to_end(&mut out)
                .map_err(io)?;
            out
        }
        Arm::CramLossy => match decode_any(payload) {
            Ok(Decoded::F32 { data, .. }) => f32_bytes(&data),
            Ok(_) => return Err(Error::Payload("cram stream is not f32".into())),
            Err(e) => return Err(Error::Payload(format!("{e:?}"))),
        },
    };
    if out.len() != len {
        return Err(Error::LengthMismatch);
    }
    Ok(out)
}

fn frame(arm: Arm, len: usize, crc: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 16);
    out.extend_from_slice(MAGIC);
    out.push(arm.tag());
    varint(len as u64, &mut out);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Compress `x`, keeping the smallest arm; see the module docs.
pub fn compress(x: &[u8], hint: &Hint) -> Result<Vec<u8>, Error> {
    compress_report(x, hint).map(|(b, _)| b)
}

/// [`compress`], also returning which arms ran and what each produced.
pub fn compress_report(x: &[u8], hint: &Hint) -> Result<(Vec<u8>, Report), Error> {
    if let Hint::F32Field { dims, .. } = hint {
        if dims.iter().product::<usize>() * 4 != x.len() {
            return Err(Error::BadHint(format!(
                "dims {dims:?} do not match {} bytes",
                x.len()
            )));
        }
    }
    let mut best: Option<(Arm, Vec<u8>)> = None;
    let mut tried = Vec::new();
    for arm in plan(x, hint) {
        let p = encode_arm(arm, x, hint)?;
        tried.push((arm, p.len()));
        if best.as_ref().is_none_or(|(_, b)| p.len() < b.len()) {
            best = Some((arm, p));
        }
    }
    let (pick, payload) = best.expect("plan is never empty");
    let crc = if pick.is_lossless() {
        crc32fast::hash(x)
    } else {
        crc32fast::hash(&decode_arm(pick, &payload, x.len())?)
    };
    let out = frame(pick, x.len(), crc, &payload);
    let header = out.len() - payload.len();
    Ok((
        out,
        Report {
            pick,
            tried,
            header,
        },
    ))
}

/// Parse a container header: `(arm, original length, crc, payload offset)`.
pub fn parse_header(s: &[u8]) -> Result<(Arm, usize, u32, usize), Error> {
    if s.len() < 4 || &s[..4] != MAGIC {
        return Err(Error::BadMagic);
    }
    let &t = s.get(4).ok_or(Error::Truncated)?;
    let arm = Arm::from_tag(t).ok_or(Error::UnknownMethod(t))?;
    let (len, n) = read_varint(&s[5..]).ok_or(Error::Truncated)?;
    let len = usize::try_from(len).map_err(|_| Error::Truncated)?;
    let at = 5 + n;
    let c = s.get(at..at + 4).ok_or(Error::Truncated)?;
    let crc = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    if arm == Arm::Stored && s.len() - (at + 4) != len {
        return Err(Error::LengthMismatch);
    }
    Ok((arm, len, crc, at + 4))
}

/// Decompress a container stream. Fails closed on any corruption.
pub fn decompress(s: &[u8]) -> Result<Vec<u8>, Error> {
    let (arm, len, crc, at) = parse_header(s)?;
    let out = decode_arm(arm, &s[at..], len)?;
    if crc32fast::hash(&out) != crc {
        return Err(Error::Checksum);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*: deterministic test bytes without a rand dependency.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            s.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    fn random(n: usize, seed: u64) -> Vec<u8> {
        let mut r = rng(seed);
        (0..n).map(|_| r() as u8).collect()
    }

    fn inputs() -> Vec<Vec<u8>> {
        let mut v = vec![
            vec![],
            vec![0],
            vec![0xff],
            vec![7; 100_000],
            b"abcabcabd".repeat(5000),
            random(1, 1),
            random(17, 2),
            random(4096, 3),
            random(200_000, 4),
            "the quick brown fox jumps over the lazy dog\n".repeat(300).into_bytes(),
        ];
        // property sweep: random lengths, random alphabet sizes
        let mut r = rng(99);
        for _ in 0..24 {
            let n = (r() % 5000) as usize;
            let k = 1 + (r() % 256) as u8 as u64;
            v.push((0..n).map(|_| (r() % k) as u8).collect());
        }
        v
    }

    #[test]
    fn round_trip_every_lossless_arm_and_container() {
        for x in inputs() {
            for arm in LOSSLESS_ARMS {
                let p = encode_arm(arm, &x, &Hint::Bytes).unwrap();
                assert_eq!(decode_arm(arm, &p, x.len()).unwrap(), x, "{arm:?}");
            }
            for h in [Hint::Auto, Hint::Bytes, Hint::Text] {
                let c = compress(&x, &h).unwrap();
                assert_eq!(decompress(&c).unwrap(), x);
            }
        }
    }

    #[test]
    fn never_worse_than_best_arm_plus_header() {
        for x in inputs() {
            let (c, rep) = compress_report(&x, &Hint::Bytes).unwrap();
            let best = plan(&x, &Hint::Bytes)
                .into_iter()
                .map(|a| encode_arm(a, &x, &Hint::Bytes).unwrap().len())
                .min()
                .unwrap();
            assert_eq!(rep.header, header_len(x.len()));
            assert!(c.len() <= best + header_len(x.len()));
            // and never more than stored + header
            assert!(c.len() <= x.len() + header_len(x.len()));
        }
    }

    #[test]
    fn deterministic() {
        let x = "log line 42 ok\n".repeat(1000).into_bytes();
        assert_eq!(
            compress(&x, &Hint::Auto).unwrap(),
            compress(&x, &Hint::Auto).unwrap()
        );
    }

    #[test]
    fn sniff_skips_hopeless_arms() {
        let mut x = vec![0x1f, 0x8b, 8, 0];
        x.extend(random(10_000, 5));
        assert_eq!(plan(&x, &Hint::Auto), vec![Arm::Stored, Arm::Zstd3]);
        let c = compress(&x, &Hint::Auto).unwrap();
        assert_eq!(decompress(&c).unwrap(), x);
        // Text skips the sniff
        assert_eq!(plan(&x, &Hint::Text).len(), 5);
    }

    #[test]
    fn lossless_hints_never_go_lossy() {
        let x: Vec<u8> = (0..4096u32)
            .flat_map(|i| ((i as f32) * 0.01).sin().to_le_bytes())
            .collect();
        for h in [
            Hint::Auto,
            Hint::Bytes,
            Hint::Text,
            Hint::F32Field {
                dims: vec![16, 16, 16],
                bound: None,
            },
        ] {
            assert!(!plan(&x, &h).contains(&Arm::CramLossy));
            let c = compress(&x, &h).unwrap();
            assert_ne!(c[4], Arm::CramLossy.tag());
            assert_eq!(decompress(&c).unwrap(), x);
        }
    }

    #[test]
    fn lossy_field_respects_bound() {
        let d = [16usize, 24, 20];
        let v: Vec<f32> = (0..d.iter().product::<usize>())
            .map(|i| ((i % 20) as f32 * 0.3).sin() + ((i / 480) as f32 * 0.2).cos())
            .collect();
        let x = f32_bytes(&v);
        let h = Hint::F32Field {
            dims: d.to_vec(),
            bound: Some(ErrorBound::MaxAbs(1e-3)),
        };
        let (c, rep) = compress_report(&x, &h).unwrap();
        assert_eq!(rep.pick, Arm::CramLossy);
        let y = f32s(&decompress(&c).unwrap());
        assert!(v.iter().zip(&y).all(|(a, b)| ((a - b) as f64).abs() <= 1e-3));
        // wrong dims fail
        let bad = Hint::F32Field {
            dims: vec![3],
            bound: Some(ErrorBound::MaxAbs(1e-3)),
        };
        assert!(matches!(compress(&x, &bad), Err(Error::BadHint(_))));
    }

    #[test]
    fn corrupt_streams_fail_closed() {
        for x in [
            "hello hello hello world\n".repeat(400).into_bytes(),
            random(3000, 7),
            vec![1, 2, 3],
        ] {
            let c = compress(&x, &Hint::Bytes).unwrap();
            // every single-byte flip is detected (or decodes to the original,
            // which a flip of a pure-padding bit could in principle allow)
            for i in 0..c.len() {
                for bit in [0x01u8, 0x80] {
                    let mut d = c.clone();
                    d[i] ^= bit;
                    match decompress(&d) {
                        Err(_) => {}
                        Ok(y) => assert_eq!(y, x, "flip at {i} decoded to wrong data"),
                    }
                }
            }
            // every truncation fails
            for n in 0..c.len() {
                assert!(decompress(&c[..n]).is_err() || x.is_empty(), "trunc {n}");
            }
            // trailing garbage on stored is rejected
            let mut d = c.clone();
            d.push(0);
            if c[4] == Arm::Stored.tag() {
                assert!(decompress(&d).is_err());
            }
        }
        assert_eq!(decompress(b"nope"), Err(Error::BadMagic));
        assert_eq!(decompress(b"CRM1\x09\x00\0\0\0\0"), Err(Error::UnknownMethod(9)));
        assert_eq!(decompress(b"CRM1\x00\xff"), Err(Error::Truncated));
    }

    #[test]
    fn varint_round_trip() {
        for v in [0u64, 1, 127, 128, 300, 1 << 35, u64::MAX] {
            let mut b = Vec::new();
            varint(v, &mut b);
            assert_eq!(read_varint(&b), Some((v, b.len())));
        }
    }
}
