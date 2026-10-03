//! Minimal GGUF (v2/v3) reader: metadata, tensor directory, and a read-only memory map of the tensor data.
//!
//! Layout: header (magic, version, tensor count, metadata count), metadata key/values, tensor infos, padding to
//! `general.alignment`, then the tensor data section. Tensor offsets are relative to the start of that section.

use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            Value::Bool(v) => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::Str(s) if s.len() > 120 => write!(f, "{:?}... ({} bytes)", s.chars().take(100).collect::<String>(), s.len()),
            Value::Str(s) => write!(f, "{s:?}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Arr(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().take(16).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                if a.len() > 16 {
                    write!(f, ", ... ({} items)", a.len())?;
                }
                write!(f, "]")
            }
        }
    }
}

/// ggml tensor types we may meet in this model, with their block geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
    BF16,
    Other(u32),
}

impl GgmlType {
    fn from_u32(t: u32) -> Self {
        match t {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            30 => GgmlType::BF16,
            other => GgmlType::Other(other),
        }
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> Option<(u64, u64)> {
        Some(match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::BF16 => (1, 2),
            GgmlType::Q4_0 => (32, 18),
            GgmlType::Q4_1 => (32, 20),
            GgmlType::Q5_0 => (32, 22),
            GgmlType::Q5_1 => (32, 24),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q4K => (256, 144),
            GgmlType::Q5K => (256, 176),
            GgmlType::Q6K => (256, 210),
            GgmlType::Other(_) => return None,
        })
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgmlType::Q4K => write!(f, "Q4_K"),
            GgmlType::Q5K => write!(f, "Q5_K"),
            GgmlType::Q6K => write!(f, "Q6_K"),
            GgmlType::Other(t) => write!(f, "type{t}"),
            other => write!(f, "{other:?}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the contiguous dimension (row length).
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Byte offset relative to the data section.
    pub offset: u64,
    pub size: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    pub version: u32,
    pub meta: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    mmap: Mmap,
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.pos + n <= self.b.len(), "unexpected end of file at {}", self.pos);
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(elem_ty)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => bail!("unknown GGUF value type {other} at {}", self.pos),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut c = Cursor { b: &mmap, pos: 0 };

        ensure!(c.take(4)? == b"GGUF", "{} is not a GGUF file", path.display());
        let version = c.u32()?;
        ensure!(version == 2 || version == 3, "unsupported GGUF version {version}");
        let n_tensors = c.u64()?;
        let n_meta = c.u64()?;

        let mut meta = BTreeMap::new();
        for _ in 0..n_meta {
            let key = c.string()?;
            let ty = c.u32()?;
            let value = c.value(ty).with_context(|| format!("reading metadata {key}"))?;
            meta.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()?;
            let dims = (0..n_dims).map(|_| c.u64()).collect::<Result<Vec<_>>>()?;
            let ty = GgmlType::from_u32(c.u32()?);
            let offset = c.u64()?;
            let n: u64 = dims.iter().product();
            let (blk, bytes) = ty.block().with_context(|| format!("tensor {name} has unsupported type {ty}"))?;
            ensure!(dims[0] % blk == 0, "tensor {name}: row length {} not a multiple of block {blk}", dims[0]);
            tensors.push(TensorInfo { name, dims, ty, offset, size: n / blk * bytes });
        }

        let align = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
        let data_offset = (c.pos as u64).div_ceil(align) * align;
        for t in &tensors {
            ensure!(
                data_offset + t.offset + t.size <= mmap.len() as u64,
                "tensor {} extends past end of file",
                t.name
            );
        }
        Ok(Gguf { version, meta, tensors, data_offset, mmap })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        let start = (self.data_offset + t.offset) as usize;
        &self.mmap[start..start + t.size as usize]
    }

    pub fn get(&self, key: &str) -> Result<&Value> {
        self.meta.get(key).with_context(|| format!("missing GGUF metadata {key}"))
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        self.get(key)?.as_u64().with_context(|| format!("GGUF metadata {key} is not an integer"))
    }

    pub fn get_f64(&self, key: &str) -> Result<f64> {
        self.get(key)?.as_f64().with_context(|| format!("GGUF metadata {key} is not a number"))
    }

    pub fn get_str(&self, key: &str) -> Result<&str> {
        self.get(key)?.as_str().with_context(|| format!("GGUF metadata {key} is not a string"))
    }

    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }
}
