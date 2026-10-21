//! Byte- and bit-level serialization primitives.
//!
//! Everything the codec emits goes through [`ByteWriter`], so the length of
//! its buffer *is* the bit account: there is no side channel.

/// Errors raised while parsing a bitstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// The stream ended before a field was complete.
    Truncated,
    /// A field held a value the format does not allow.
    Invalid(&'static str),
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::Truncated => write!(f, "bitstream truncated"),
            DecodeError::Invalid(what) => write!(f, "invalid bitstream: {what}"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Append-only byte buffer with LEB128 varints.
#[derive(Default, Debug, Clone)]
pub struct ByteWriter {
    /// The bytes written so far.
    pub buf: Vec<u8>,
}

impl ByteWriter {
    /// Empty writer.
    pub fn new() -> Self {
        Self::default()
    }
    /// One raw byte.
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    /// Unsigned LEB128.
    pub fn varint(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.buf.push(b);
                return;
            }
            self.buf.push(b | 0x80);
        }
    }
    /// Zigzag-mapped signed LEB128.
    pub fn svarint(&mut self, v: i64) {
        self.varint(zigzag(v));
    }
    /// Little-endian f64.
    pub fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    /// Length-prefixed byte block.
    pub fn block(&mut self, bytes: &[u8]) {
        self.varint(bytes.len() as u64);
        self.buf.extend_from_slice(bytes);
    }
    /// Raw bytes, no prefix.
    pub fn bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
}

/// Cursor over a byte slice, mirroring [`ByteWriter`].
#[derive(Debug, Clone)]
pub struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    /// Start reading at offset 0.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    /// Bytes consumed so far.
    pub fn position(&self) -> usize {
        self.pos
    }
    /// One raw byte.
    pub fn u8(&mut self) -> Result<u8, DecodeError> {
        let b = *self.buf.get(self.pos).ok_or(DecodeError::Truncated)?;
        self.pos += 1;
        Ok(b)
    }
    /// Unsigned LEB128.
    pub fn varint(&mut self) -> Result<u64, DecodeError> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            if shift >= 64 {
                return Err(DecodeError::Invalid("varint overflow"));
            }
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
        }
    }
    /// Zigzag-mapped signed LEB128.
    pub fn svarint(&mut self) -> Result<i64, DecodeError> {
        Ok(unzigzag(self.varint()?))
    }
    /// Little-endian f64.
    pub fn f64(&mut self) -> Result<f64, DecodeError> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }
    /// `n` raw bytes.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::Truncated)?;
        let s = self.buf.get(self.pos..end).ok_or(DecodeError::Truncated)?;
        self.pos = end;
        Ok(s)
    }
    /// Length-prefixed byte block.
    pub fn block(&mut self) -> Result<&'a [u8], DecodeError> {
        let n = self.varint()? as usize;
        self.take(n)
    }
}

/// Signed → unsigned interleaving (0, -1, 1, -2, …).
pub fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Inverse of [`zigzag`].
pub fn unzigzag(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// LSB-first bit packer for raw (uniformly distributed) extra bits.
#[derive(Default, Debug, Clone)]
pub struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    /// Empty packer.
    pub fn new() -> Self {
        Self::default()
    }
    /// Append the low `nbits` (≤ 57) of `v`.
    pub fn put(&mut self, v: u64, nbits: u32) {
        debug_assert!(nbits <= 57);
        if nbits == 0 {
            return;
        }
        self.acc |= (v & ((1u64 << nbits) - 1)) << self.n;
        self.n += nbits;
        while self.n >= 8 {
            self.buf.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    /// Flush the partial byte and return the buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.buf.push(self.acc as u8);
        }
        self.buf
    }
}

/// Reader for [`BitWriter`] output.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    acc: u64,
    n: u32,
}

impl<'a> BitReader<'a> {
    /// Start at bit 0.
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            acc: 0,
            n: 0,
        }
    }
    /// Read `nbits` (≤ 57) bits.
    pub fn get(&mut self, nbits: u32) -> Result<u64, DecodeError> {
        if nbits == 0 {
            return Ok(0);
        }
        while self.n < nbits {
            let b = *self.buf.get(self.pos).ok_or(DecodeError::Truncated)?;
            self.pos += 1;
            self.acc |= (b as u64) << self.n;
            self.n += 8;
        }
        let v = self.acc & ((1u64 << nbits) - 1);
        self.acc >>= nbits;
        self.n -= nbits;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_and_zigzag_roundtrip() {
        let mut w = ByteWriter::new();
        let vals = [0i64, -1, 1, 63, -64, 1 << 40, i64::MIN + 1, i64::MAX];
        for &v in &vals {
            w.svarint(v);
        }
        w.f64(-3.25);
        let mut r = ByteReader::new(&w.buf);
        for &v in &vals {
            assert_eq!(r.svarint().unwrap(), v);
        }
        assert_eq!(r.f64().unwrap(), -3.25);
        assert_eq!(r.u8(), Err(DecodeError::Truncated));
    }

    #[test]
    fn bits_roundtrip() {
        let mut w = BitWriter::new();
        for i in 0..200u64 {
            w.put(i * 7919, (i % 30) as u32);
        }
        let buf = w.finish();
        let mut r = BitReader::new(&buf);
        for i in 0..200u64 {
            let nb = (i % 30) as u32;
            let mask = if nb == 0 { 0 } else { (1u64 << nb) - 1 };
            assert_eq!(r.get(nb).unwrap(), (i * 7919) & mask);
        }
    }
}
