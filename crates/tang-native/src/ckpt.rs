//! Checkpoints: `params.safetensors` (every parameter by name, f32),
//! `adam.safetensors` (`<name>.m`, `<name>.v`), and `meta.txt` (one
//! `key value` per line: step, Adam's t, variant, shape, schedule, data
//! seed). Safetensors is written and read directly: a little-endian u64
//! header length, a JSON header, then the raw tensors.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::Path;

pub fn save_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> io::Result<()> {
    let mut header = String::from("{");
    let mut off = 0usize;
    for (i, (name, shape, data)) in tensors.iter().enumerate() {
        let n = data.len() * 4;
        let shape: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}",
            shape.join(","),
            off + n
        ));
        off += n;
    }
    header.push('}');
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let tmp = path.with_extension("tmp");
    let mut f = io::BufWriter::new(std::fs::File::create(&tmp)?);
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    for (_, _, data) in tensors {
        for x in data {
            f.write_all(&x.to_le_bytes())?;
        }
    }
    f.flush()?;
    drop(f);
    std::fs::rename(tmp, path) // never leave a half-written checkpoint
}

/// Name → values (shapes are the caller's to check).
pub fn load_safetensors(path: &Path) -> io::Result<BTreeMap<String, Vec<f32>>> {
    let b = std::fs::read(path)?;
    let bad = |m: &str| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {m}", path.display()),
        )
    };
    let n =
        u64::from_le_bytes(b.get(..8).ok_or_else(|| bad("short"))?.try_into().unwrap()) as usize;
    let header = std::str::from_utf8(b.get(8..8 + n).ok_or_else(|| bad("header"))?)
        .map_err(|_| bad("utf8"))?;
    let data = &b[8 + n..];
    let mut out = BTreeMap::new();
    // our own writer's flat JSON: "name":{"dtype":"F32","shape":[..],"data_offsets":[a,b]}
    let mut rest = header.trim().trim_start_matches('{').trim_end_matches('}');
    while let Some(q) = rest.find('"') {
        let r = &rest[q + 1..];
        let e = r.find('"').ok_or_else(|| bad("name"))?;
        let name = r[..e].to_string();
        let k = r.find("\"data_offsets\":[").ok_or_else(|| bad("offsets"))? + 16;
        let close = r[k..].find(']').ok_or_else(|| bad("offsets"))? + k;
        let nums: Vec<usize> = r[k..close]
            .split(',')
            .map(|s| s.trim().parse().map_err(|_| bad("offset")))
            .collect::<Result<_, _>>()?;
        let (a, z) = (nums[0], nums[1]);
        let v = data
            .get(a..z)
            .ok_or_else(|| bad("range"))?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        out.insert(name, v);
        rest = &r[close + 1..];
        rest = rest.find('}').map_or("", |i| &rest[i + 1..]);
    }
    Ok(out)
}

pub fn save_meta(path: &Path, kv: &[(&str, String)]) -> io::Result<()> {
    let mut s = String::new();
    for (k, v) in kv {
        s.push_str(&format!("{k} {v}\n"));
    }
    std::fs::write(path, s)
}

pub fn load_meta(path: &Path) -> io::Result<BTreeMap<String, String>> {
    Ok(std::fs::read_to_string(path)?
        .lines()
        .filter_map(|l| l.split_once(' '))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect())
}
