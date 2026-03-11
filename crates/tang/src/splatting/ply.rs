//! PLY file loader for standard 3D Gaussian Splatting format.
//!
//! Parses binary little-endian PLY files as exported by the original
//! [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) codebase.
//!
//! # PLY Property Layout
//!
//! Each vertex (gaussian) has these properties in order:
//! - `x, y, z` — position
//! - `nx, ny, nz` — normals (unused, but present in file)
//! - `f_dc_0, f_dc_1, f_dc_2` — DC spherical harmonics (RGB)
//! - `f_rest_0 .. f_rest_{N}` — higher-order SH coefficients
//! - `opacity` — opacity logit (inverse sigmoid)
//! - `scale_0, scale_1, scale_2` — log-scale
//! - `rot_0, rot_1, rot_2, rot_3` — rotation quaternion (w, x, y, z)

use super::GaussianCloud;
use std::io::{BufRead, BufReader, Read};

/// Errors during PLY parsing.
#[derive(Debug)]
pub enum PlyError {
    Io(std::io::Error),
    /// Header is malformed or missing required fields.
    BadHeader(String),
    /// Unexpected end of data.
    UnexpectedEof,
    /// Unsupported format (we only handle binary_little_endian).
    UnsupportedFormat(String),
}

impl From<std::io::Error> for PlyError {
    fn from(e: std::io::Error) -> Self {
        PlyError::Io(e)
    }
}

impl std::fmt::Display for PlyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlyError::Io(e) => write!(f, "IO error: {}", e),
            PlyError::BadHeader(s) => write!(f, "Bad PLY header: {}", s),
            PlyError::UnexpectedEof => write!(f, "Unexpected end of PLY data"),
            PlyError::UnsupportedFormat(s) => write!(f, "Unsupported PLY format: {}", s),
        }
    }
}

impl std::error::Error for PlyError {}

/// Load a gaussian cloud from a PLY file.
pub fn load_ply(path: &std::path::Path) -> Result<GaussianCloud, PlyError> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    load_ply_reader(&mut reader)
}

/// Load from any reader (useful for testing with in-memory data).
pub fn load_ply_reader<R: Read>(reader: &mut BufReader<R>) -> Result<GaussianCloud, PlyError> {
    let header = parse_header(reader)?;
    parse_binary_data(reader, &header)
}

struct PlyHeader {
    vertex_count: usize,
    /// Ordered list of (property_name, byte_size).
    properties: Vec<(String, usize)>,
    /// Total bytes per vertex.
    vertex_bytes: usize,
}

impl PlyHeader {
    /// Find the byte offset of a property within a vertex record.
    fn offset_of(&self, name: &str) -> Option<usize> {
        let mut off = 0;
        for (pname, size) in &self.properties {
            if pname == name {
                return Some(off);
            }
            off += size;
        }
        None
    }

    /// Count how many `f_rest_*` properties exist.
    fn sh_rest_count(&self) -> usize {
        self.properties
            .iter()
            .filter(|(name, _)| name.starts_with("f_rest_"))
            .count()
    }
}

fn property_size(type_name: &str) -> Result<usize, PlyError> {
    match type_name {
        "float" | "float32" => Ok(4),
        "double" | "float64" => Ok(8),
        "uchar" | "uint8" => Ok(1),
        "short" | "int16" => Ok(2),
        "ushort" | "uint16" => Ok(2),
        "int" | "int32" => Ok(4),
        "uint" | "uint32" => Ok(4),
        other => Err(PlyError::BadHeader(format!("unknown type: {}", other))),
    }
}

fn parse_header<R: Read>(reader: &mut BufReader<R>) -> Result<PlyHeader, PlyError> {
    let mut line = String::new();

    // First line must be "ply"
    reader.read_line(&mut line)?;
    if line.trim() != "ply" {
        return Err(PlyError::BadHeader("missing ply magic".into()));
    }

    let mut format_ok = false;
    let mut vertex_count = None;
    let mut properties = Vec::new();
    let mut in_vertex_element = false;

    loop {
        line.clear();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();

        if trimmed == "end_header" {
            break;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "format" => {
                if parts.len() < 2 {
                    return Err(PlyError::BadHeader("missing format spec".into()));
                }
                if parts[1] != "binary_little_endian" {
                    return Err(PlyError::UnsupportedFormat(parts[1].to_string()));
                }
                format_ok = true;
            }
            "element" => {
                in_vertex_element = parts.len() >= 3 && parts[1] == "vertex";
                if in_vertex_element {
                    vertex_count = Some(
                        parts[2]
                            .parse::<usize>()
                            .map_err(|_| PlyError::BadHeader("bad vertex count".into()))?,
                    );
                }
            }
            "property" => {
                if in_vertex_element && parts.len() >= 3 {
                    let size = property_size(parts[1])?;
                    properties.push((parts[2].to_string(), size));
                }
            }
            _ => {} // comment, obj_info, etc.
        }
    }

    if !format_ok {
        return Err(PlyError::BadHeader("no format line found".into()));
    }
    let vertex_count =
        vertex_count.ok_or_else(|| PlyError::BadHeader("no vertex element found".into()))?;

    let vertex_bytes: usize = properties.iter().map(|(_, s)| *s).sum();

    Ok(PlyHeader {
        vertex_count,
        properties,
        vertex_bytes,
    })
}

fn parse_binary_data<R: Read>(
    reader: &mut BufReader<R>,
    header: &PlyHeader,
) -> Result<GaussianCloud, PlyError> {
    let n = header.vertex_count;

    // Resolve offsets for required properties
    let off_x = header
        .offset_of("x")
        .ok_or_else(|| PlyError::BadHeader("missing x property".into()))?;
    let off_y = header
        .offset_of("y")
        .ok_or_else(|| PlyError::BadHeader("missing y property".into()))?;
    let off_z = header
        .offset_of("z")
        .ok_or_else(|| PlyError::BadHeader("missing z property".into()))?;
    let off_opacity = header
        .offset_of("opacity")
        .ok_or_else(|| PlyError::BadHeader("missing opacity property".into()))?;
    let off_scale0 = header
        .offset_of("scale_0")
        .ok_or_else(|| PlyError::BadHeader("missing scale_0 property".into()))?;
    let off_rot0 = header
        .offset_of("rot_0")
        .ok_or_else(|| PlyError::BadHeader("missing rot_0 property".into()))?;
    let off_dc0 = header
        .offset_of("f_dc_0")
        .ok_or_else(|| PlyError::BadHeader("missing f_dc_0 property".into()))?;

    let sh_rest_count = header.sh_rest_count();
    // Determine SH degree from rest count: rest = 3 * ((deg+1)^2 - 1)
    let sh_degree = if sh_rest_count == 0 {
        0
    } else {
        let coeffs_per_channel = sh_rest_count / 3 + 1;
        let deg = (coeffs_per_channel as f64).sqrt() as u32 - 1;
        deg
    };

    let sh_per_gaussian = 3 * ((sh_degree + 1) * (sh_degree + 1)) as usize;
    let sh_rest_per_channel = if sh_degree > 0 {
        ((sh_degree + 1) * (sh_degree + 1) - 1) as usize
    } else {
        0
    };

    let off_rest0 = if sh_rest_count > 0 {
        header.offset_of("f_rest_0")
    } else {
        None
    };

    let mut positions = Vec::with_capacity(n);
    let mut scales = Vec::with_capacity(n);
    let mut rotations = Vec::with_capacity(n);
    let mut opacities = Vec::with_capacity(n);
    let mut sh_coeffs = Vec::with_capacity(n * sh_per_gaussian);

    let mut buf = vec![0u8; header.vertex_bytes];

    for _ in 0..n {
        reader.read_exact(&mut buf).map_err(|_| PlyError::UnexpectedEof)?;

        let read_f32 = |offset: usize| -> f32 {
            f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
        };

        positions.push([read_f32(off_x), read_f32(off_y), read_f32(off_z)]);

        // Scales are stored consecutively as scale_0, scale_1, scale_2
        scales.push([
            read_f32(off_scale0),
            read_f32(off_scale0 + 4),
            read_f32(off_scale0 + 8),
        ]);

        // Rotation: rot_0=w, rot_1=x, rot_2=y, rot_3=z
        rotations.push([
            read_f32(off_rot0),
            read_f32(off_rot0 + 4),
            read_f32(off_rot0 + 8),
            read_f32(off_rot0 + 12),
        ]);

        opacities.push(read_f32(off_opacity));

        // SH coefficients: DC first (interleaved RGB), then rest
        // PLY stores: f_dc_0(R), f_dc_1(G), f_dc_2(B)
        // Then f_rest: first sh_rest_per_channel for R, then G, then B
        //
        // Our layout: [R_dc, G_dc, B_dc, R_1, G_1, B_1, R_2, G_2, B_2, ...]
        // i.e., interleaved by coefficient index across channels

        // DC term
        let dc_r = read_f32(off_dc0);
        let dc_g = read_f32(off_dc0 + 4);
        let dc_b = read_f32(off_dc0 + 8);
        sh_coeffs.push(dc_r);
        sh_coeffs.push(dc_g);
        sh_coeffs.push(dc_b);

        // Higher-order SH (if any)
        if let Some(off_rest) = off_rest0 {
            for coeff_idx in 0..sh_rest_per_channel {
                // R channel
                let r = read_f32(off_rest + coeff_idx * 4);
                // G channel
                let g = read_f32(off_rest + (sh_rest_per_channel + coeff_idx) * 4);
                // B channel
                let b = read_f32(off_rest + (2 * sh_rest_per_channel + coeff_idx) * 4);
                sh_coeffs.push(r);
                sh_coeffs.push(g);
                sh_coeffs.push(b);
            }
        }
    }

    Ok(GaussianCloud {
        count: n,
        positions,
        scales,
        rotations,
        opacities,
        sh_coeffs,
        sh_degree,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_test_ply(n: usize) -> Vec<u8> {
        let header = format!(
            "ply\n\
             format binary_little_endian 1.0\n\
             element vertex {n}\n\
             property float x\n\
             property float y\n\
             property float z\n\
             property float nx\n\
             property float ny\n\
             property float nz\n\
             property float f_dc_0\n\
             property float f_dc_1\n\
             property float f_dc_2\n\
             property float opacity\n\
             property float scale_0\n\
             property float scale_1\n\
             property float scale_2\n\
             property float rot_0\n\
             property float rot_1\n\
             property float rot_2\n\
             property float rot_3\n\
             end_header\n"
        );
        let mut data = header.into_bytes();
        // 17 floats per vertex = 68 bytes
        for i in 0..n {
            let t = i as f32;
            let floats: [f32; 17] = [
                t, t + 1.0, t + 2.0, // x, y, z
                0.0, 0.0, 1.0, // nx, ny, nz
                0.5, 0.6, 0.7, // f_dc RGB
                2.0,  // opacity (logit)
                -3.0, -3.0, -3.0, // scale (log)
                1.0, 0.0, 0.0, 0.0, // rot (w,x,y,z)
            ];
            for f in floats {
                data.extend_from_slice(&f.to_le_bytes());
            }
        }
        data
    }

    #[test]
    fn test_load_degree0() {
        let data = make_test_ply(3);
        let mut reader = BufReader::new(Cursor::new(data));
        let cloud = load_ply_reader(&mut reader).unwrap();

        assert_eq!(cloud.count, 3);
        assert_eq!(cloud.sh_degree, 0);
        assert_eq!(cloud.sh_coeffs.len(), 9); // 3 gaussians * 3 DC
        assert_eq!(cloud.positions[0], [0.0, 1.0, 2.0]);
        assert_eq!(cloud.positions[2], [2.0, 3.0, 4.0]);
        assert_eq!(cloud.scales[0], [-3.0, -3.0, -3.0]);
        assert_eq!(cloud.rotations[0], [1.0, 0.0, 0.0, 0.0]);
        assert!((cloud.opacities[0] - 2.0).abs() < 1e-6);
        // DC SH
        assert!((cloud.sh_coeffs[0] - 0.5).abs() < 1e-6); // R
        assert!((cloud.sh_coeffs[1] - 0.6).abs() < 1e-6); // G
        assert!((cloud.sh_coeffs[2] - 0.7).abs() < 1e-6); // B
    }

    #[test]
    fn test_bad_magic() {
        let data = b"not a ply file\n";
        let mut reader = BufReader::new(Cursor::new(data.to_vec()));
        assert!(load_ply_reader(&mut reader).is_err());
    }
}
