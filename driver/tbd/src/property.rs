//! XDomain property blocks: the directory of keys and values a host publishes to the hosts it is connected to
//! (vendor, device name, max HopID, and one directory per service such as `network` and `stream`).
//!
//! The layout follows Linux (drivers/thunderbolt/property.c): the block is a dword array in host order, the root
//! directory header is a magic and the length of its entries, every entry is four dwords (key, length/type,
//! value), and each directory's data (text, data) follows its entries, with one zero dword per child directory
//! after the data and the child directories after that. Keys, text and data are stored byte-swapped
//! (`cpu_to_be32`) inside the block; the control channel swaps every dword once more on the wire.

use crate::uuid::Uuid;
use anyhow::{Result, bail, ensure};

pub const ROOTDIR_MAGIC: u32 = 0x5558_4401;
const MAX_DEPTH: usize = 8;

const TYPE_DIRECTORY: u8 = 0x44;
const TYPE_DATA: u8 = 0x64;
const TYPE_TEXT: u8 = 0x74;
const TYPE_VALUE: u8 = 0x76;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Dir(Dir),
    /// Dwords as `be32_to_cpu` leaves them.
    Data(Vec<u32>),
    /// Text and its length in dwords (with the NUL and padding).
    Text(String, usize),
    Immediate(u32),
    Unknown(u8),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Property {
    pub key: String,
    pub value: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dir {
    /// Child directories have a UUID; the root has none.
    pub uuid: Option<Uuid>,
    pub props: Vec<Property>,
}

fn key_words(key: &str) -> [u32; 2] {
    let mut b = [0u8; 8];
    b[..key.len()].copy_from_slice(key.as_bytes());
    [u32::from_be_bytes(b[..4].try_into().unwrap()), u32::from_be_bytes(b[4..].try_into().unwrap())]
}

fn text_words(text: &str) -> Vec<u32> {
    let len = (text.len() + 1).div_ceil(4) * 4;
    let mut b = text.as_bytes().to_vec();
    b.resize(len, 0);
    b.chunks(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect()
}

impl Dir {
    pub fn new(uuid: Option<Uuid>) -> Self {
        Dir { uuid, props: Vec::new() }
    }

    fn push(&mut self, key: &str, value: Value) -> &mut Self {
        assert!(!key.is_empty() && key.len() <= 8, "property key {key:?}");
        self.props.push(Property { key: key.to_owned(), value });
        self
    }

    pub fn add_immediate(&mut self, key: &str, v: u32) -> &mut Self {
        self.push(key, Value::Immediate(v))
    }

    pub fn add_text(&mut self, key: &str, text: &str) -> &mut Self {
        let len = (text.len() + 1).div_ceil(4);
        self.push(key, Value::Text(text.to_owned(), len))
    }

    pub fn add_dir(&mut self, key: &str, dir: Dir) -> &mut Self {
        self.push(key, Value::Dir(dir))
    }

    pub fn immediate(&self, key: &str) -> Option<u32> {
        self.props.iter().find_map(|p| match p.value {
            Value::Immediate(v) if p.key == key => Some(v),
            _ => None,
        })
    }

    pub fn text(&self, key: &str) -> Option<&str> {
        self.props.iter().find_map(|p| match &p.value {
            Value::Text(t, _) if p.key == key => Some(t.as_str()),
            _ => None,
        })
    }

    pub fn dir(&self, key: &str) -> Option<&Dir> {
        self.props.iter().find_map(|p| match &p.value {
            Value::Dir(d) if p.key == key => Some(d),
            _ => None,
        })
    }

    pub fn dirs(&self) -> impl Iterator<Item = (&str, &Dir)> {
        self.props.iter().filter_map(|p| match &p.value {
            Value::Dir(d) => Some((p.key.as_str(), d)),
            _ => None,
        })
    }

    /// `tb_property_dir_length`: header and entries; adds the data and one padding dword per child directory to
    /// `data_len`.
    fn length(&self, recurse: bool, data_len: &mut usize) -> usize {
        let mut len = if self.uuid.is_some() { 4 } else { 2 };
        for p in &self.props {
            len += 4;
            match &p.value {
                Value::Dir(d) => {
                    if recurse {
                        len += d.length(true, data_len);
                    }
                    *data_len += 1;
                }
                Value::Data(v) => *data_len += v.len(),
                Value::Text(_, n) => *data_len += n,
                _ => {}
            }
        }
        len
    }

    /// The block, as `tb_property_format_dir` builds it.
    pub fn format(&self) -> Vec<u32> {
        let mut data_len = 0;
        let total = self.length(true, &mut data_len) + data_len;
        let mut block = vec![0u32; total];
        let end = self.format_at(&mut block, 0);
        debug_assert_eq!(end, total);
        block
    }

    fn format_at(&self, block: &mut [u32], start: usize) -> usize {
        let mut data_len = 0;
        let dir_len = self.length(false, &mut data_len);
        let mut data_offset = start + dir_len;
        let mut dir_end = start + dir_len + data_len;
        let mut entry = match self.uuid {
            Some(u) => {
                block[start..start + 4].copy_from_slice(&u.words());
                start + 4
            }
            None => {
                block[start] = ROOTDIR_MAGIC;
                block[start + 1] = (dir_len - 2) as u32;
                start + 2
            }
        };
        for p in &self.props {
            let [hi, lo] = key_words(&p.key);
            let (ty, length, value) = match &p.value {
                Value::Dir(d) => {
                    let at = dir_end;
                    dir_end = d.format_at(block, at);
                    (TYPE_DIRECTORY, d.length(false, &mut 0), at as u32)
                }
                Value::Data(v) => {
                    block[data_offset..data_offset + v.len()].copy_from_slice(&v.iter().map(|w| w.swap_bytes()).collect::<Vec<_>>());
                    data_offset += v.len();
                    (TYPE_DATA, v.len(), (data_offset - v.len()) as u32)
                }
                Value::Text(t, n) => {
                    let mut w = text_words(t);
                    w.resize(*n, 0);
                    block[data_offset..data_offset + n].copy_from_slice(&w);
                    data_offset += n;
                    (TYPE_TEXT, *n, (data_offset - n) as u32)
                }
                Value::Immediate(v) => (TYPE_VALUE, 1, *v),
                Value::Unknown(t) => (*t, 0, 0),
            };
            block[entry] = hi;
            block[entry + 1] = lo;
            block[entry + 2] = length as u32 | (ty as u32) << 24;
            block[entry + 3] = value;
            entry += 4;
        }
        dir_end
    }

    /// `tb_property_parse_dir`.
    pub fn parse(block: &[u32]) -> Result<Dir> {
        ensure!(block.len() >= 2 && block[0] == ROOTDIR_MAGIC, "no property root directory");
        let len = block[1] as usize;
        ensure!(len <= block.len(), "root directory length {len} past the block ({})", block.len());
        Self::parse_at(block, 0, len, true, 0)
    }

    fn parse_at(block: &[u32], offset: usize, len: usize, root: bool, depth: usize) -> Result<Dir> {
        ensure!(depth <= MAX_DEPTH, "property directories nested too deep");
        let (uuid, content, content_len) = if root {
            ensure!(offset + 2 + len <= block.len(), "root directory past the block");
            (None, offset + 2, len)
        } else {
            ensure!(len >= 4 && offset + len <= block.len(), "directory at {offset} of length {len}");
            (Some(Uuid::from_words(&block[offset..offset + 4])), offset + 4, len - 4)
        };
        let mut dir = Dir::new(uuid);
        for i in 0..content_len / 4 {
            let e = &block[content + i * 4..content + i * 4 + 4];
            let mut kb = [0u8; 8];
            kb[..4].copy_from_slice(&e[0].to_be_bytes());
            kb[4..].copy_from_slice(&e[1].to_be_bytes());
            let key = String::from_utf8_lossy(&kb).trim_end_matches('\0').to_owned();
            let length = (e[2] & 0xffff) as usize;
            let ty = (e[2] >> 24) as u8;
            let value = e[3] as usize;
            let in_block = |n: usize| -> Result<()> {
                ensure!(n > 0 && value.checked_add(n).is_some_and(|end| end <= block.len()), "property {key:?} past the block");
                Ok(())
            };
            let v = match ty {
                TYPE_DIRECTORY => {
                    in_block(length)?;
                    Value::Dir(Self::parse_at(block, value, length, false, depth + 1)?)
                }
                TYPE_DATA => {
                    in_block(length)?;
                    Value::Data(block[value..value + length].iter().map(|w| w.swap_bytes()).collect())
                }
                TYPE_TEXT => {
                    in_block(length)?;
                    let mut b: Vec<u8> = block[value..value + length].iter().flat_map(|w| w.to_be_bytes()).collect();
                    *b.last_mut().unwrap() = 0;
                    let end = b.iter().position(|&c| c == 0).unwrap();
                    Value::Text(String::from_utf8_lossy(&b[..end]).into_owned(), length)
                }
                TYPE_VALUE => {
                    if length != 1 {
                        bail!("immediate {key:?} of length {length}");
                    }
                    Value::Immediate(e[3])
                }
                t => Value::Unknown(t),
            };
            dir.props.push(Property { key, value: v });
        }
        Ok(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linux_like() -> Dir {
        let mut net = Dir::new(Some(Uuid::init(0xc66189ca, 0x1cce, 0x4195, [0xbd, 0xb8, 0x49, 0x59, 0x2e, 0x5f, 0x5a, 0x4f])));
        net.add_immediate("prtcid", 1).add_immediate("prtcvers", 1).add_immediate("prtcrevs", 1).add_immediate("prtcstns", 6);
        let mut root = Dir::new(None);
        root.add_immediate("vendorid", 0x1d6b).add_text("vendorid", "Linux").add_immediate("deviceid", 4);
        root.add_dir("network", net).add_text("deviceid", "host").add_immediate("maxhopid", 15);
        root
    }

    #[test]
    fn layout_matches_linux() {
        let block = linux_like().format();
        // Root: magic, 6 entries of 4 dwords.
        assert_eq!(block[0], ROOTDIR_MAGIC);
        assert_eq!(block[1], 24);
        // "vendorid" as two big-endian dwords, then immediate.
        assert_eq!(&block[2..6], &[0x76656e64, 0x6f726964, 0x7600_0001, 0x1d6b]);
        // The text entry points past the 26 dwords of entries: "Linux\0" is 2 dwords.
        assert_eq!(&block[6..10], &[0x76656e64, 0x6f726964, 0x7400_0002, 26]);
        assert_eq!(&block[26..28], &[0x4c696e75, 0x7800_0000]);
        // "deviceid" text at 28 (2 dwords), then one padding dword for the child directory at 30.
        assert_eq!(block[18 + 3], 28);
        assert_eq!(block[30], 0);
        // The directory entry: length 4 + 4*4, at dword 31.
        assert_eq!(&block[14..18], &[0x6e657477, 0x6f726b00, 0x4400_0014, 31]);
        assert_eq!(block.len(), 31 + 20);
        assert_eq!(Dir::parse(&block).unwrap(), linux_like());
    }

    #[test]
    fn rejects_bad_blocks() {
        let mut block = linux_like().format();
        assert!(Dir::parse(&block[..20]).is_err());
        block[17] = 1000;
        assert!(Dir::parse(&block).is_err());
        block[0] = 0;
        assert!(Dir::parse(&block).is_err());
    }
}
