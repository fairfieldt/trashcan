//! Read-only filesystems served by a program (fsd): the kernel asks it for directory listings and file contents
//! over a channel, and keeps both (the VFS caches listings in its tree, contents as memory objects).
//!
//! Requests, each answered by one message starting with a status (`i64`, 0 or a negative errno):
//! - `LIST` (op 1): `cookie: u64`, then the directory's path below the mount root. The reply continues with the
//!   next cookie (0 when done) and entries of `mode: u32, uid: u32, gid: u32, size: u64, mtime: u64 (ns),
//!   name length: u16, symlink target length: u16`, the name and the target.
//! - `OPEN` (op 2): the file's path. The reply continues with the size (`u64`) and carries one handle: a memory
//!   object holding the file, which the server gives up (no other handle or mapping of it remains).
use super::*;
use crate::{
    ipc::{ChannelEnd, Message},
    memory::Memory,
    object::Object,
};

pub struct Mount {
    endpoint: Arc<ChannelEnd>,
    busy: WaitCell<bool>,
}

pub struct Listed {
    pub name: String,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub mtime: u64,
    pub target: Vec<u8>,
}

struct Guard<'a>(&'a WaitCell<bool>);
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.0.update(|v| *v = false);
    }
}

impl Mount {
    pub fn new(endpoint: Arc<ChannelEnd>) -> Arc<Mount> {
        Arc::new(Mount { endpoint, busy: WaitCell::new(false) })
    }

    /// One request and its reply; one at a time per mount. Reading a large file takes as long as the disk does.
    fn rpc(&self, request: Vec<u8>) -> Result<Message, Errno> {
        self.busy
            .wait_uninterruptible(u64::MAX, |busy| (!*busy).then(|| *busy = true))
            .map_err(|_| EINTR)?;
        let _guard = Guard(&self.busy);
        self.endpoint.write(Message { bytes: request, handles: Vec::new() }).map_err(|_| 5)?;
        let reply = self.endpoint.read_transaction(dross_abi::MSG_BYTES, 1, u64::MAX).map_err(|_| 5)?;
        let status = reply.bytes.get(..8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).ok_or(5)?;
        if status < 0 { Err(-status) } else { Ok(reply) }
    }

    /// Everything in the directory at `path` (below the mount root).
    pub fn list(&self, path: &str) -> Result<Vec<Listed>, Errno> {
        let mut out = Vec::new();
        let mut cookie = 0u64;
        loop {
            let mut req = 1u32.to_le_bytes().to_vec();
            req.extend_from_slice(&cookie.to_le_bytes());
            req.extend_from_slice(path.as_bytes());
            let reply = self.rpc(req)?;
            let b = &reply.bytes;
            let u16_at = |at: usize| b.get(at..at + 2).map(|v| u16::from_le_bytes(v.try_into().unwrap()) as usize);
            let u32_at = |at: usize| b.get(at..at + 4).map(|v| u32::from_le_bytes(v.try_into().unwrap()));
            let u64_at = |at: usize| b.get(at..at + 8).map(|v| u64::from_le_bytes(v.try_into().unwrap()));
            cookie = u64_at(8).ok_or(5)?;
            let mut at = 16;
            while at < b.len() {
                let (Some(mode), Some(uid), Some(gid), Some(size), Some(mtime), Some(nl), Some(tl)) =
                    (u32_at(at), u32_at(at + 4), u32_at(at + 8), u64_at(at + 12), u64_at(at + 20), u16_at(at + 28), u16_at(at + 30))
                else {
                    return Err(5);
                };
                let name = b.get(at + 32..at + 32 + nl).ok_or(5)?;
                let target = b.get(at + 32 + nl..at + 32 + nl + tl).ok_or(5)?;
                let name = core::str::from_utf8(name).map_err(|_| 5)?;
                if !name.is_empty() && !name.contains('/') && name != "." && name != ".." {
                    out.push(Listed { name: String::from(name), mode, uid, gid, size, mtime, target: target.to_vec() });
                }
                at += 32 + nl + tl;
            }
            if cookie == 0 {
                return Ok(out);
            }
        }
    }

    /// The file at `path`, whole, read-only.
    pub fn open(&self, path: &str) -> Result<Arc<Memory>, Errno> {
        let mut req = 2u32.to_le_bytes().to_vec();
        req.extend_from_slice(path.as_bytes());
        let mut reply = self.rpc(req)?;
        let entry = reply.handles.pop().ok_or(5)?;
        let Object::Memory(m) = entry.object else {
            return Err(5);
        };
        // The server must not keep a way to change it.
        let m = Arc::try_unwrap(m).map_err(|_| 5)?;
        Ok(Arc::new(m.read_only()))
    }
}
