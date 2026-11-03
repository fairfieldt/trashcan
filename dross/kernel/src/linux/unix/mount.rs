//! Filesystem RPC for userspace fsd. The wire format and writable limits are
//! documented in user/fsd/README.md. Requests are serialized per mount.
use super::*;
use crate::{
    ipc::{ChannelEnd, Message},
    memory::Memory,
    object::{Entry, Object},
};

static NEXT_DEVICE:AtomicU64=AtomicU64::new(2);
pub struct Mount {
    pub device:u64,
    endpoint: Arc<ChannelEnd>,
    busy: WaitCell<bool>,
    mode: core::sync::atomic::AtomicU8,
}

#[derive(Clone, Copy)]
pub struct Info {pub inode:u64,pub size:u64,pub atime:u64,pub mtime:u64,pub ctime:u64,pub mode:u32,pub uid:u32,pub gid:u32,pub links:u32}
impl Info {
    fn decode(bytes:&[u8])->Result<Self,Errno>{
        let bytes=bytes.get(..56).ok_or(5)?;
        let q=|at:usize|u64::from_le_bytes(bytes[at..at+8].try_into().unwrap());
        let d=|at:usize|u32::from_le_bytes(bytes[at..at+4].try_into().unwrap());
        Ok(Self{inode:q(0),size:q(8),atime:q(16),mtime:q(24),ctime:q(32),mode:d(40),uid:d(44),gid:d(48),links:d(52)})
    }
}
/// One capability per open description. Closing its last endpoint tells fsd to
/// release the inode; kernel destructors never wait for a filesystem response.
pub struct OpenFile {endpoint:Arc<ChannelEnd>,busy:WaitCell<bool>}
impl OpenFile {
    fn rpc(&self,request:Vec<u8>)->Result<Message,Errno>{rpc(&self.endpoint,&self.busy,request,Vec::new())}
    pub fn info(&self)->Result<Info,Errno>{Info::decode(self.rpc(3u32.to_le_bytes().to_vec())?.bytes.get(8..).ok_or(5)?)}
    pub fn read(&self,offset:u64,len:usize)->Result<Vec<u8>,Errno>{
        let mut out=Vec::new();out.try_reserve_exact(len).map_err(|_|ENOMEM)?;
        while out.len()<len{let count=(len-out.len()).min(60*1024);let mut req=1u32.to_le_bytes().to_vec();req.extend_from_slice(&(offset+out.len() as u64).to_le_bytes());req.extend_from_slice(&(count as u32).to_le_bytes());
            let reply=self.rpc(req)?;let data=reply.bytes.get(8..).ok_or(5)?;if data.len()>count{return Err(5)}out.extend_from_slice(data);if data.len()<count{break}
        }Ok(out)
    }
    pub fn write(&self,offset:u64,bytes:&[u8],append:bool)->Result<(usize,u64),Errno>{
        let(mut at,mut written)=(offset,0);
        for data in bytes.chunks(48*1024){let mut req=2u32.to_le_bytes().to_vec();req.extend_from_slice(&(if append{u64::MAX}else{at}).to_le_bytes());req.extend_from_slice(data);
            let reply=match self.rpc(req){Ok(r)=>r,Err(_)if written!=0=>return Ok((written,at)),Err(e)=>return Err(e)};
            let end=reply.bytes.get(8..16).map(|b|u64::from_le_bytes(b.try_into().unwrap())).ok_or(5)?;
            let count=reply.bytes.get(16..24).map(|b|u64::from_le_bytes(b.try_into().unwrap())).ok_or(5)?;
            if count>data.len()as u64||end<count||(!append&&at.checked_add(count)!=Some(end)){return Err(5)}at=end;written+=count as usize;if count<data.len()as u64{break}
        }Ok((written,at))
    }
    pub fn truncate(&self,size:u64)->Result<(),Errno>{let mut r=4u32.to_le_bytes().to_vec();r.extend_from_slice(&size.to_le_bytes());self.rpc(r).map(|_|())}
    pub fn setattr(&self,mode:u32,uid:u32,gid:u32)->Result<(),Errno>{let mut r=5u32.to_le_bytes().to_vec();for v in[mode,uid,gid]{r.extend_from_slice(&v.to_le_bytes());}self.rpc(r).map(|_|())}
    pub fn times(&self,times:[Option<u64>;2])->Result<(),Errno>{let mut r=6u32.to_le_bytes().to_vec();let flags=times[0].is_some()as u32|(times[1].is_some()as u32)<<1;r.extend_from_slice(&flags.to_le_bytes());for v in times{r.extend_from_slice(&v.unwrap_or(0).to_le_bytes());}self.rpc(r).map(|_|())}
    pub fn sync(&self)->Result<(),Errno>{self.rpc(7u32.to_le_bytes().to_vec()).map(|_|())}
    pub fn snapshot(&self)->Result<Arc<Memory>,Errno>{memory_reply(self.rpc(8u32.to_le_bytes().to_vec())?)}
}
fn memory_reply(mut reply:Message)->Result<Arc<Memory>,Errno>{let entry=reply.handles.pop().ok_or(5)?;let Object::Memory(m)=entry.object else{return Err(5)};let m=Arc::try_unwrap(m).map_err(|_|5)?;Ok(Arc::new(m.read_only()))}
fn rpc(endpoint:&Arc<ChannelEnd>,busy:&WaitCell<bool>,request:Vec<u8>,handles:Vec<Entry>)->Result<Message,Errno>{
    busy.wait_uninterruptible(u64::MAX,|v|(!*v).then(||*v=true)).map_err(|_|EINTR)?;let _guard=Guard(busy);
    endpoint.write(Message::new(request,handles)).map_err(|_|5)?;
    let reply=endpoint.read_transaction(dross_abi::MSG_BYTES,1,u64::MAX).map_err(|_|5)?;
    let status=reply.bytes.get(..8).map(|b|i64::from_le_bytes(b.try_into().unwrap())).ok_or(5)?;if status<0{Err(-status)}else{Ok(reply)}
}

pub struct Listed {
    pub name: String,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub mtime: u64,
    pub inode: u64,
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
        Arc::new(Mount { device:NEXT_DEVICE.fetch_add(1,Ordering::Relaxed), endpoint, busy: WaitCell::new(false), mode: core::sync::atomic::AtomicU8::new(0) })
    }

    /// One request and its reply; one at a time per mount. Reading a large file takes as long as the disk does.
    fn rpc(&self,request:Vec<u8>)->Result<Message,Errno>{rpc(&self.endpoint,&self.busy,request,Vec::new())}
    pub fn open_file(&self,path:&str,flags:u32)->Result<Arc<OpenFile>,Errno>{
        let(endpoint,b)=ChannelEnd::try_pair().map_err(|_|12)?;let mut req=14u32.to_le_bytes().to_vec();req.extend_from_slice(&(flags&3).to_le_bytes());req.extend_from_slice(path.as_bytes());
        rpc(&self.endpoint,&self.busy,req,alloc::vec![Entry::new(Object::Channel(b))])?;
        Ok(Arc::new(OpenFile{endpoint,busy:WaitCell::new(false)}))
    }
    pub fn info(&self,path:&str)->Result<Info,Errno>{let mut r=15u32.to_le_bytes().to_vec();r.extend_from_slice(path.as_bytes());Info::decode(self.rpc(r)?.bytes.get(8..).ok_or(5)?)}
    pub fn link(&self,old:&str,new:&str)->Result<(),Errno>{let mut r=16u32.to_le_bytes().to_vec();r.extend_from_slice(&(old.len()as u32).to_le_bytes());r.extend_from_slice(old.as_bytes());r.extend_from_slice(new.as_bytes());self.rpc(r).map(|_|())}

    pub fn writable(&self) -> Result<bool, Errno> {
        let mode = self.mode.load(Ordering::Acquire);
        if mode != 0 { return Ok(mode == 2); }
        let reply = self.rpc(11u32.to_le_bytes().to_vec())?;
        let writable = reply.bytes.get(8) == Some(&1);
        self.mode.store(if writable { 2 } else { 1 }, Ordering::Release);
        Ok(writable)
    }
    pub fn create(&self, path: &str, mode: u32, uid: u32, gid: u32, target: &[u8]) -> Result<(), Errno> {
        let mut req = if mode & 0o170000 == 0o120000 {
            let mut req = 8u32.to_le_bytes().to_vec();
            req.extend_from_slice(&(path.len() as u32).to_le_bytes());
            req.extend_from_slice(path.as_bytes()); req.extend_from_slice(target); req
        } else {
            let mut req = 3u32.to_le_bytes().to_vec();
            req.extend_from_slice(&mode.to_le_bytes()); req.extend_from_slice(&uid.to_le_bytes()); req.extend_from_slice(&gid.to_le_bytes());
            req.extend_from_slice(path.as_bytes()); req
        };
        self.rpc(core::mem::take(&mut req)).map(|_| ())
    }
    pub fn truncate(&self, path: &str, size: u64) -> Result<(), Errno> {
        let mut req = 4u32.to_le_bytes().to_vec(); req.extend_from_slice(&size.to_le_bytes()); req.extend_from_slice(path.as_bytes()); self.rpc(req).map(|_| ())
    }
    /// Returns bytes written and the end offset, including server-selected append placement.
    pub fn write(&self, path: &str, offset: u64, bytes: &[u8], append: bool) -> Result<(usize, u64), Errno> {
        let mut at = offset;
        let mut written = 0;
        for data in bytes.chunks(48 * 1024) {
            let mut req = 5u32.to_le_bytes().to_vec(); req.extend_from_slice(&(if append { u64::MAX } else { at }).to_le_bytes());
            req.extend_from_slice(&(path.len() as u32).to_le_bytes()); req.extend_from_slice(path.as_bytes()); req.extend_from_slice(data);
            let reply = match self.rpc(req) { Ok(reply) => reply, Err(_) if written != 0 => return Ok((written, at)), Err(e) => return Err(e) };
            let end = reply.bytes.get(8..16).map(|b| u64::from_le_bytes(b.try_into().unwrap())).ok_or(5)?;
            let count = reply.bytes.get(16..24).map(|b| u64::from_le_bytes(b.try_into().unwrap())).ok_or(5)?;
            if count > data.len() as u64 || end < count || (!append && at.checked_add(count) != Some(end)) { return Err(5); }
            at = end; written += count as usize;
            if count < data.len() as u64 { break; }
        }
        Ok((written, at))
    }
    pub fn read(&self, path: &str, offset: u64, len: usize) -> Result<Vec<u8>, Errno> {
        let mut out = Vec::new(); out.try_reserve_exact(len).map_err(|_| ENOMEM)?;
        while out.len() < len {
            let count = (len - out.len()).min(60 * 1024);
            let mut req = 10u32.to_le_bytes().to_vec(); req.extend_from_slice(&(offset + out.len() as u64).to_le_bytes());
            req.extend_from_slice(&(count as u32).to_le_bytes()); req.extend_from_slice(path.as_bytes());
            let reply = self.rpc(req)?; let data = reply.bytes.get(8..).ok_or(5)?;
            if data.len() > count { return Err(5); }
            out.extend_from_slice(data); if data.len() < count { break; }
        }
        Ok(out)
    }
    pub fn remove(&self, path: &str, directory: bool) -> Result<(), Errno> {
        let mut req = 6u32.to_le_bytes().to_vec(); req.extend_from_slice(&(directory as u32).to_le_bytes()); req.extend_from_slice(path.as_bytes()); self.rpc(req).map(|_| ())
    }
    pub fn rename(&self, old: &str, new: &str, directory: bool) -> Result<(), Errno> {
        let mut req = 7u32.to_le_bytes().to_vec(); req.extend_from_slice(&(directory as u32).to_le_bytes()); req.extend_from_slice(&(old.len() as u32).to_le_bytes());
        req.extend_from_slice(old.as_bytes()); req.extend_from_slice(new.as_bytes()); self.rpc(req).map(|_| ())
    }
    pub fn setattr(&self, path: &str, mode: u32, uid: u32, gid: u32) -> Result<(), Errno> {
        let mut req = 12u32.to_le_bytes().to_vec(); req.extend_from_slice(&mode.to_le_bytes()); req.extend_from_slice(&uid.to_le_bytes()); req.extend_from_slice(&gid.to_le_bytes()); req.extend_from_slice(path.as_bytes()); self.rpc(req).map(|_| ())
    }
    pub fn times(&self,path:&str,times:[Option<u64>;2])->Result<(),Errno>{
        let flags=times[0].is_some()as u32|(times[1].is_some()as u32)<<1;let mut req=13u32.to_le_bytes().to_vec();req.extend_from_slice(&flags.to_le_bytes());
        for t in times{req.extend_from_slice(&t.unwrap_or(0).to_le_bytes());}req.extend_from_slice(path.as_bytes());self.rpc(req).map(|_|())
    }
    pub fn sync(&self) -> Result<(), Errno> { self.rpc(9u32.to_le_bytes().to_vec()).map(|_| ()) }

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
                    (u32_at(at), u32_at(at + 4), u32_at(at + 8), u64_at(at + 12), u64_at(at + 20), u16_at(at + 36), u16_at(at + 38))
                else {
                    return Err(5);
                };
                let name = b.get(at + 40..at + 40 + nl).ok_or(5)?;
                let target = b.get(at + 40 + nl..at + 40 + nl + tl).ok_or(5)?;
                let name = core::str::from_utf8(name).map_err(|_| 5)?;
                if !name.is_empty() && !name.contains('/') && name != "." && name != ".." {
                    out.push(Listed { name: String::from(name), mode, uid, gid, size, mtime, inode: u64_at(at + 28).ok_or(5)?, target: target.to_vec() });
                }
                at += 40 + nl + tl;
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
