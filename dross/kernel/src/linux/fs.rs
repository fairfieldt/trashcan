//! Files for Linux programs: the fd table, a read-only tree (the `rootfs.tar` boot module plus the ACPI tables
//! under /sys/firmware/acpi/tables), /dev/null and friends, stdout and stderr into the kernel log, pipes, memfds,
//! eventfds, fds for dross notifications, and poll over all of them.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use spin::Mutex;

use super::sock::{self, Endpoint, Listener};
use super::*;
use crate::ipc::Notification;
use crate::memory::Memory;
use crate::sched::WaitCell;

const O_ACCMODE: u32 = 3;
const O_CREAT: u32 = 0o100;
const O_NONBLOCK: u32 = 0o4000;
const O_DIRECTORY: u32 = 0o200000;
const O_CLOEXEC: u32 = 0o2000000;
const AT_FDCWD: i32 = -100;

const S_IFIFO: u32 = 0o010000;
const S_IFCHR: u32 = 0o020000;
const S_IFDIR: u32 = 0o040000;
const S_IFREG: u32 = 0o100000;
const S_IFSOCK: u32 = 0o140000;

pub const POLLIN: u16 = 1;
pub const POLLOUT: u16 = 4;
pub const POLLERR: u16 = 8;
pub const POLLHUP: u16 = 16;
const POLLNVAL: u16 = 32;

/// The read-only tree: path (no leading slash) to contents.
static TREE: Mutex<BTreeMap<String, &'static [u8]>> = Mutex::new(BTreeMap::new());

/// Indexes a ustar archive (regular files and directories) into the tree.
pub fn init_rootfs(data: &'static [u8]) {
    #[cfg(feature = "unix")]
    super::unix::vfs::archive(data);
    let mut t = TREE.lock();
    let mut at = 0;
    let mut n = 0;
    while at + 512 <= data.len() {
        let h = &data[at..at + 512];
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let field = |s: &[u8]| -> String { String::from_utf8_lossy(&s[..s.iter().position(|&b| b == 0).unwrap_or(s.len())]).into_owned() };
        let mut name = field(&h[0..100]);
        if &h[257..262] == b"ustar" {
            let prefix = field(&h[345..500]);
            if !prefix.is_empty() {
                name = alloc::format!("{prefix}/{name}");
            }
        }
        let size = usize::from_str_radix(field(&h[124..136]).trim(), 8).unwrap_or(0);
        let ty = h[156];
        let body = at + 512;
        if body + size > data.len() {
            break;
        }
        let path = String::from(name.trim_start_matches("./").trim_end_matches('/'));
        if (ty == b'0' || ty == 0) && !path.is_empty() {
            t.insert(path, &data[body..body + size]);
            n += 1;
        }
        at = body + size.next_multiple_of(512);
    }
    crate::println!("linux: rootfs: {} files", n);
}

/// Adds a firmware table as /sys/firmware/acpi/tables/SIG.
pub fn publish_table(sig: &str, data: &'static [u8]) {
    #[cfg(feature = "unix")]
    super::unix::vfs::import(&alloc::format!("/sys/firmware/acpi/tables/{sig}"), 0o100444, data);
    TREE.lock().insert(alloc::format!("sys/firmware/acpi/tables/{sig}"), data);
}

/// A file in the tree (an interpreter for exec).
#[cfg(not(feature = "unix"))]
pub(super) fn tree_file(path: &str) -> Option<&'static [u8]> {
    TREE.lock().get(&normalize(path)).copied()
}

fn is_dir(path: &str) -> bool {
    path.is_empty() || TREE.lock().range(String::from(path)..).next().is_some_and(|(k, _)| k.starts_with(path) && k.as_bytes().get(path.len()) == Some(&b'/'))
}

/// Normalizes a path against `/` (the only working directory).
fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for c in path.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            c => parts.push(c),
        }
    }
    parts.join("/")
}

pub struct Log {
    name: String,
    line: Mutex<Vec<u8>>,
}

impl Log {
    fn write(&self, data: &[u8]) {
        let mut l = self.line.lock();
        for &b in data {
            if b == b'\n' {
                crate::println!("{}: {}", self.name, String::from_utf8_lossy(&l));
                l.clear();
            } else if l.len() < 4096 {
                l.push(b);
            }
        }
    }
}

impl Drop for Log {
    fn drop(&mut self) {
        let l = self.line.lock();
        if !l.is_empty() {
            crate::println!("{}: {}", self.name, String::from_utf8_lossy(&l));
        }
    }
}

pub enum Kind {
    Log(Arc<Log>),
    #[cfg(feature = "unix")]
    Node(Arc<super::unix::vfs::Node>),
    #[cfg(feature = "unix")]
    Pty(Arc<super::unix::pty::End>),
    #[cfg(feature = "unix")]
    Net(Arc<super::unix::net::Socket>),
    #[cfg(feature = "unix")]
    Channel(Arc<super::unix::channel::Stream>),
    Null,
    Zero,
    Random,
    Data(&'static [u8]),
    Dir(String),
    Memfd(Mutex<Option<Arc<Memory>>>),
    Stream(Arc<Endpoint>),
    Socket(Mutex<sock::State>),
    Listener(Arc<Listener>),
    EventFd(WaitCell<u64>),
    Notify(Arc<Notification>),
    /// Reading /dev/kmsg: the kernel log from the oldest record kept, one record per read.
    Kmsg(Mutex<u64>),
}

#[cfg(feature = "unix")]
struct PositionGuard<'a>(&'a WaitCell<bool>);
#[cfg(feature = "unix")]
impl Drop for PositionGuard<'_> {
    fn drop(&mut self) { self.0.update(|busy| *busy = false); }
}

pub struct File {
    pub kind: Kind,
    pub(super) pos: Mutex<u64>,
    pub flags: AtomicU32,
    /// Rights retained by an fd created from a native memory capability.
    memory_rights: Option<u32>,
    size: AtomicU64,
    /// An open mounted file retains its backing independently of cache eviction.
    #[cfg(feature = "unix")]
    remote_content: Mutex<Option<Arc<Memory>>>,
    #[cfg(feature = "unix")]
    opened: Mutex<Option<Arc<super::unix::mount::OpenFile>>>,
    #[cfg(feature = "unix")]
    position_busy: WaitCell<bool>,
}

impl File {
    pub fn new(kind: Kind, flags: u32) -> Arc<File> {
        Self::with_memory_rights(kind, flags, None)
    }

    fn with_memory_rights(kind: Kind, flags: u32, memory_rights: Option<u32>) -> Arc<File> {
        let size = match &kind {
            #[cfg(feature = "unix")]
            Kind::Node(node) => node.len(),
            Kind::Data(d) => d.len() as u64,
            _ => 0,
        };
        Arc::new(File { kind, pos: Mutex::new(0), flags: AtomicU32::new(flags), memory_rights, size: AtomicU64::new(size),
            #[cfg(feature = "unix")]
            remote_content: Mutex::new(None),
            #[cfg(feature = "unix")]
            opened: Mutex::new(None),
            #[cfg(feature = "unix")]
            position_busy: WaitCell::new(false),
        })
    }

    #[cfg(feature = "unix")]
    pub fn mounted(&self)->Option<Arc<super::unix::mount::OpenFile>>{self.opened.lock().clone()}
    #[cfg(feature = "unix")]
    pub fn open_mounted(&self)->Result<(),Errno>{
        if let Kind::Node(node)=&self.kind{let opened=node.open_file(self.flags.load(Ordering::Relaxed))?;*self.opened.lock()=opened;}self.refresh()
    }
    #[cfg(feature = "unix")]
    pub fn refresh(&self)->Result<(),Errno>{if let Kind::Node(node)=&self.kind{if let Some(fd)=self.mounted(){node.apply_info(fd.info()?);}else{node.refresh()?;}}Ok(())}
    #[cfg(feature = "unix")]
    pub fn sync(&self)->Result<(),Errno>{if let Some(fd)=self.mounted(){return fd.sync()}match &self.kind{Kind::Node(node)=>node.sync(),Kind::Dir(path)=>super::unix::vfs::lookup(&alloc::format!("/{path}"),true)?.1.sync(),_=>Ok(())}}

    #[cfg(feature = "unix")]
    fn position_guard(&self) -> Result<PositionGuard<'_>, Errno> {
        self.position_busy.wait(u64::MAX, |busy| if *busy { None } else { *busy = true; Some(()) }).map_err(wait_errno)?;
        Ok(PositionGuard(&self.position_busy))
    }

    #[cfg(feature = "unix")]
    fn remote_content(&self) -> Result<Option<Arc<Memory>>, Errno> {
        let Kind::Node(node) = &self.kind else { return Ok(None) };
        if node.writable_remote()? { return Ok(None); }
        if let Some(m) = self.remote_content.lock().clone() { return Ok(Some(m)); }
        let Some(m) = node.content()? else { return Ok(None) };
        Ok(Some(self.remote_content.lock().get_or_insert(m).clone()))
    }

    /// Native capability ceilings survive conversion to an fd and shared mmap.
    pub fn memory_permissions(&self) -> Option<u64> {
        self.memory_rights.map(|r| {
            let mut flags = 0;
            if r & dross_abi::rights::READ != 0 { flags |= vm::READ; }
            if r & dross_abi::rights::WRITE != 0 { flags |= vm::WRITE; }
            if r & dross_abi::rights::EXECUTE != 0 { flags |= vm::EXECUTE; }
            flags
        })
    }

    pub fn nonblocking(&self) -> bool {
        self.flags.load(Ordering::Relaxed) & O_NONBLOCK != 0
    }

    /// The memory behind a shared mapping of this file at `off` (memfds, mounted files), or None for a regular
    /// file, which is mapped by copying it.
    pub fn backing(&self, off: u64, len: u64) -> Result<Option<Arc<Memory>>, Errno> {
        match &self.kind {
            #[cfg(feature = "unix")]
            // A mounted file's contents (read-only) map directly; a RAM file maps as a copy.
            Kind::Node(node) => {let backing=if let Some(fd)=self.mounted(){Some(fd.snapshot()?)}else{node.content()?};Ok(backing.filter(|m|off+len<=m.size()))},
            Kind::Data(_) => Ok(None),
            Kind::Memfd(m) => {
                let m = m.lock().clone().ok_or(EINVAL)?;
                if off + len > m.size() {
                    return Err(EINVAL);
                }
                Ok(Some(m))
            }
            _ => Err(EACCES),
        }
    }

    /// A copy of [off, off+len) of the file (zeros past its end), for a private mapping.
    pub fn copy(&self, off: u64, len: u64) -> Result<Memory, Errno> {
        if self.memory_rights.is_some_and(|r| r & dross_abi::rights::READ == 0) { return Err(EACCES); }
        let m = Memory::anonymous(len, false, false).map_err(|_| ENOMEM)?;
        let put = |d: &[u8]| {
            let from = d.len().min(off as usize);
            m.write(0, &d[from..d.len().min(from + len as usize)]).map_err(|_| EINVAL)
        };
        match &self.kind {
            #[cfg(feature = "unix")]
            Kind::Node(node) => match if let Some(fd)=self.mounted(){Some(fd.snapshot()?)}else{node.content()?} {
                Some(c) => m.copy_from(0, &c, off, len.min(c.size().saturating_sub(off))).map_err(|_| EINVAL)?,
                None => put(&node.data.lock())?,
            },
            Kind::Data(d) => put(d)?,
            Kind::Memfd(mf) => {
                let mf = mf.lock().clone().ok_or(EINVAL)?;
                m.copy_from(0, &mf, off, len.min(mf.size().saturating_sub(off))).map_err(|_| EINVAL)?
            }
            _ => return Err(EACCES),
        }
        Ok(m)
    }

    fn mode(&self) -> u32 {
        match &self.kind {
            #[cfg(feature = "unix")]
            Kind::Node(node) => node.attr.lock().mode,
            #[cfg(feature = "unix")]
            Kind::Pty(end) => end.pty.node.attr.lock().mode,
            #[cfg(feature = "unix")]
            Kind::Net(_) => S_IFSOCK | 0o600,
            Kind::Data(_) | Kind::Memfd(_) => S_IFREG | 0o644,
            #[cfg(feature = "unix")]
            Kind::Dir(path) => super::unix::vfs::lookup(&alloc::format!("/{path}"), true).map(|(_, n)| n.attr.lock().mode).unwrap_or(S_IFDIR | 0o755),
            #[cfg(not(feature = "unix"))]
            Kind::Dir(_) => S_IFDIR | 0o755,
            Kind::Stream(e) if e.is_pipe() => S_IFIFO | 0o600,
            Kind::Stream(_) | Kind::Socket(_) | Kind::Listener(_) => S_IFSOCK | 0o777,
            _ => S_IFCHR | 0o666,
        }
    }

    /// Readiness for poll; registers `me` with whatever would change it if nothing is ready.
    pub fn poll(&self, me: &Arc<Thread>, events: u16) -> u16 {
        match &self.kind {
            #[cfg(feature = "unix")]
            Kind::Pty(end) => end.poll(me, events),
            #[cfg(feature = "unix")]
            Kind::Net(socket) => socket.poll(events),
            #[cfg(feature = "unix")]
            Kind::Channel(stream) => stream.poll(events),
            Kind::Stream(e) => e.poll(me, events),
            Kind::Listener(l) => l.poll(me),
            Kind::EventFd(c) => c.poll(me, |v| if *v > 0 { POLLIN } else { 0 } | POLLOUT) & (events | POLLERR | POLLHUP),
            Kind::Notify(n) => n.poll(me),
            Kind::Socket(_) => POLLOUT & events,
            _ => (POLLIN | POLLOUT) & events,
        }
    }

    pub fn unpoll(&self, me: &Arc<Thread>) {
        match &self.kind {
            #[cfg(feature = "unix")]
            Kind::Pty(end) => end.unpoll(me),
            Kind::Stream(e) => e.unpoll(me),
            Kind::Listener(l) => l.unpoll(me),
            Kind::EventFd(c) => c.unregister(me),
            Kind::Notify(n) => n.unpoll(me),
            _ => {}
        }
    }

    pub fn read(&self, _p: &Process, buf_len: usize, at: Option<u64>) -> Result<Vec<u8>, Errno> {
        match &self.kind {
            #[cfg(feature = "unix")]
            Kind::Pty(end) => end.read(buf_len, self.nonblocking()),
            #[cfg(feature = "unix")]
            Kind::Net(socket) => socket.read(buf_len, self.nonblocking(), false),
            #[cfg(feature = "unix")]
            Kind::Channel(stream) => stream.read(buf_len, self.nonblocking()),
            #[cfg(feature = "unix")]
            Kind::Node(node) => {
                if self.flags.load(Ordering::Relaxed) & 3 == 1 { return Err(EBADF); }
                let _position = self.position_guard()?;
                let offset = at.unwrap_or(*self.pos.lock());
                let out = if let Some(fd)=self.mounted(){fd.read(offset,buf_len.min(1<<20))?}else if let Some(m) = self.remote_content()? {
                    let n = (buf_len.min(1 << 20) as u64).min(node.len().saturating_sub(offset)) as usize;
                    let mut out = Vec::new();
                    out.try_reserve_exact(n).map_err(|_| ENOMEM)?;
                    out.resize(n, 0);
                    m.read(offset.min(node.len()), &mut out).map_err(|_| 5)?;
                    out
                } else { node.read_at(offset, buf_len.min(1 << 20))? };
                if at.is_none() { *self.pos.lock() = offset + out.len() as u64; } Ok(out)
            }
            Kind::Null => Ok(Vec::new()),
            Kind::Zero => Ok(alloc::vec![0; buf_len]),
            Kind::Random => {
                let mut v = alloc::vec![0; buf_len.min(1 << 20)];
                random_bytes(&mut v);
                Ok(v)
            }
            Kind::Log(_) => Err(EBADF),
            Kind::Data(d) => {
                let mut pos = self.pos.lock();
                let from = at.unwrap_or(*pos) as usize;
                let n = buf_len.min(d.len().saturating_sub(from));
                let v = d[from.min(d.len())..from.min(d.len()) + n].to_vec();
                if at.is_none() {
                    *pos += n as u64;
                }
                Ok(v)
            }
            Kind::Memfd(m) => {
                if self.memory_rights.is_some_and(|r| r & dross_abi::rights::READ == 0) { return Err(EBADF); }
                let m = m.lock().clone();
                let size = self.size.load(Ordering::Relaxed);
                let mut pos = self.pos.lock();
                let from = at.unwrap_or(*pos);
                let n = (buf_len as u64).min(size.saturating_sub(from));
                let mut v = alloc::vec![0u8; n as usize];
                if let Some(m) = m {
                    m.read(from, &mut v).map_err(|_| EINVAL)?;
                }
                if at.is_none() {
                    *pos += n;
                }
                Ok(v)
            }
            Kind::Dir(_) => Err(EISDIR),
            Kind::Stream(e) => e.recv(buf_len, 0, self.nonblocking(), false).map(|(d, _, _)| d),
            Kind::EventFd(c) => {
                if buf_len < 8 {
                    return Err(EINVAL);
                }
                let nb = self.nonblocking();
                let v = c.wait(if nb { 0 } else { u64::MAX }, |v| if *v > 0 { Some(core::mem::take(v)) } else { None });
                match v {
                    Ok(v) => Ok(v.to_le_bytes().to_vec()),
                    Err(WaitError::TimedOut) => Err(EAGAIN),
                    Err(e) => Err(wait_errno(e)),
                }
            }
            Kind::Notify(n) => {
                if buf_len < 8 {
                    return Err(EINVAL);
                }
                match n.wait(if self.nonblocking() { 0 } else { u64::MAX }) {
                    Ok(_) => Ok(1u64.to_le_bytes().to_vec()),
                    Err(WaitError::TimedOut) => Err(EAGAIN),
                    Err(e) => Err(wait_errno(e)),
                }
            }
            Kind::Socket(_) | Kind::Listener(_) => Err(ENOTCONN),
            Kind::Kmsg(cursor) => loop {
                // The log has no waiters to wake (it is written from anywhere, with any lock held): poll it.
                if let Some(r) = crate::log::read_record(&mut cursor.lock()) {
                    if r.len() > buf_len {
                        return Err(EINVAL);
                    }
                    return Ok(r);
                }
                if self.nonblocking() {
                    return Err(EAGAIN);
                }
                sched::sleep_ns(50_000_000).map_err(|_| EINTR)?;
            },
        }
    }

    pub fn write(&self, data: &[u8], at: Option<u64>) -> Result<usize, Errno> {
        match &self.kind {
            #[cfg(feature = "unix")]
            Kind::Pty(end) => end.write(data, self.nonblocking()),
            #[cfg(feature = "unix")]
            Kind::Net(socket) => socket.write(data, self.nonblocking()),
            #[cfg(feature = "unix")]
            Kind::Channel(stream) => stream.write(data, self.nonblocking()),
            #[cfg(feature = "unix")]
            Kind::Node(node) => {
                let flags = self.flags.load(Ordering::Relaxed); if flags & 3 == 0 { return Err(EBADF); }
                let _position = self.position_guard()?; let from = at.unwrap_or(*self.pos.lock());
                let (written,end)=if let Some(fd)=self.mounted(){let(n,end)=fd.write(from,data,flags&0x400!=0)?;(n,usize::try_from(end).map_err(|_|EINVAL)?)}else{super::unix::vfs::write(node,usize::try_from(from).map_err(|_|EINVAL)?,data,flags&0x400!=0)?};
                if at.is_none() { *self.pos.lock() = end as u64; } Ok(written)
            }
            Kind::Null | Kind::Zero | Kind::Random => Ok(data.len()),
            Kind::Log(l) => {
                l.write(data);
                Ok(data.len())
            }
            Kind::Memfd(m) => {
                if self.memory_rights.is_some_and(|r| r & dross_abi::rights::WRITE == 0) { return Err(EBADF); }
                let m = m.lock().clone().ok_or(EINVAL)?;
                if !m.writable() { return Err(EACCES); }
                let mut pos = self.pos.lock();
                let from = at.unwrap_or(*pos);
                if from + data.len() as u64 > m.size() {
                    return Err(EINVAL);
                }
                m.write(from, data).map_err(|_| EINVAL)?;
                if at.is_none() {
                    *pos += data.len() as u64;
                }
                self.size.fetch_max(from + data.len() as u64, Ordering::Relaxed);
                Ok(data.len())
            }
            Kind::Stream(e) => e.send(data, Vec::new(), self.nonblocking()),
            Kind::EventFd(c) => {
                if data.len() < 8 {
                    return Err(EINVAL);
                }
                let v = u64::from_le_bytes(data[..8].try_into().unwrap());
                c.update(|x| *x = x.saturating_add(v));
                Ok(8)
            }
            Kind::Notify(n) => {
                n.signal(1);
                Ok(data.len().min(8))
            }
            Kind::Data(_) | Kind::Dir(_) | Kind::Kmsg(_) => Err(EBADF),
            Kind::Socket(_) | Kind::Listener(_) => Err(ENOTCONN),
        }
    }
}

#[derive(Clone)]
struct Fd {
    file: Arc<File>,
    cloexec: bool,
}

pub struct Files {
    fds: Mutex<Vec<Option<Fd>>>,
}

impl Files {
    pub fn new(name: &str) -> Files {
        let log = Arc::new(Log { name: String::from(name), line: Mutex::new(Vec::new()) });
        let fds = alloc::vec![
            Some(Fd { file: File::new(Kind::Null, 0), cloexec: false }),
            Some(Fd { file: File::new(Kind::Log(log.clone()), 1), cloexec: false }),
            Some(Fd { file: File::new(Kind::Log(log), 1), cloexec: false }),
        ];
        Files { fds: Mutex::new(fds) }
    }

    #[cfg(feature = "unix")]
    pub fn inherit(&self, source: &Files) { *self.fds.lock() = source.fds.lock().clone(); }

    #[cfg(feature = "unix")]
    pub fn exec_close(&self) {
        let mut fds = self.fds.lock();
        let mut closed = Vec::new();
        for entry in fds.iter_mut() { if entry.as_ref().is_some_and(|e| e.cloexec) { closed.push(entry.take()); } }
        drop(fds); drop(closed);
        sock::collect_cycles();
    }

    pub fn get(&self, fd: i32) -> Result<Arc<File>, Errno> {
        if fd < 0 {
            return Err(EBADF);
        }
        self.fds.lock().get(fd as usize).and_then(|f| f.as_ref()).map(|f| f.file.clone()).ok_or(EBADF)
    }

    pub fn insert(&self, file: Arc<File>, cloexec: bool) -> Result<u64, Errno> {
        self.insert_from(file, cloexec, 0)
    }

    /// Reserve slots, copy their numbers out, then publish together. The short
    /// user copy never waits; failures leave no partially installed descriptors.
    pub fn insert_group(&self, files: &[Arc<File>], cloexec: bool,
        publish: impl FnOnce(&[usize]) -> Result<(), Errno>) -> Result<(), Errno> {
        if files.len() > sock::MAX_RIGHTS { return Err(EINVAL); }
        if files.is_empty() { return publish(&[]); }
        let mut slots = [0usize; sock::MAX_RIGHTS];
        let mut count = 0;
        let mut table = self.fds.lock();
        for index in 0..4096 {
            if table.get(index).is_none_or(Option::is_none) {
                slots[count] = index;
                count += 1;
                if count == files.len() { break; }
            }
        }
        if count != files.len() { return Err(EMFILE); }
        let needed = slots[count - 1] + 1;
        let extra = needed.saturating_sub(table.len());
        table.try_reserve(extra).map_err(|_| ENOMEM)?;
        while table.len() < needed { table.push(None); }
        publish(&slots[..count])?;
        for (file, &slot) in files.iter().zip(&slots[..count]) {
            table[slot] = Some(Fd { file: file.clone(), cloexec });
        }
        Ok(())
    }
    pub fn insert_pair(&self, first: Arc<File>, second: Arc<File>, cloexec: bool,
        publish: impl FnOnce(u64, u64) -> Result<(), Errno>) -> Result<(), Errno> {
        self.insert_group(&[first, second], cloexec, |slots| publish(slots[0] as u64, slots[1] as u64))
    }

    fn insert_from(&self, file: Arc<File>, cloexec: bool, min: usize) -> Result<u64, Errno> {
        if min >= 4096 { return Err(EMFILE); }
        let mut t = self.fds.lock();
        let i = match t.iter().enumerate().skip(min).position(|(_, f)| f.is_none()) {
            Some(i) => i + min,
            None => {
                if t.len() >= 4096 {
                    return Err(EMFILE);
                }
                let extra = min.max(t.len()) + 1 - t.len();
                t.try_reserve(extra).map_err(|_| ENOMEM)?;
                while t.len() < min {
                    t.push(None);
                }
                t.push(None);
                t.len() - 1
            }
        };
        t[i] = Some(Fd { file, cloexec });
        Ok(i as u64)
    }

    fn set(&self, fd: usize, file: Arc<File>, cloexec: bool) -> Option<Arc<File>> {
        let mut t = self.fds.lock();
        while t.len() <= fd {
            t.push(None);
        }
        t[fd].replace(Fd { file, cloexec }).map(|f| f.file)
    }

    /// Puts `file` in place of the file at `fd` (a socket turning into a listener or a connection).
    pub fn fds_replace(&self, fd: i32, file: Arc<File>) -> Result<(), Errno> {
        let mut t = self.fds.lock();
        let e = t.get_mut(fd as usize).and_then(|f| f.as_mut()).ok_or(EBADF)?;
        let old = core::mem::replace(&mut e.file, file);
        drop(t);
        drop(old);
        sock::collect_cycles();
        Ok(())
    }

    pub fn close(&self, fd: i32) -> Result<Arc<File>, Errno> {
        if fd < 0 {
            return Err(EBADF);
        }
        self.fds.lock().get_mut(fd as usize).and_then(|f| f.take()).map(|f| f.file).ok_or(EBADF)
    }

    pub fn release_all(&self) {
        let old = core::mem::take(&mut *self.fds.lock());
        drop(old);
        sock::collect_cycles();
    }


}

pub(super) fn stat(p: &Process, f: &File, ptr: u64) -> R {
    #[cfg(feature = "unix")]
    f.refresh()?;
    let mut st = [0u8; 144];
    let size = match &f.kind {
        #[cfg(feature = "unix")]
        Kind::Node(node) => node.len(),
        Kind::Data(d) => d.len() as u64,
        Kind::Memfd(_) => f.size.load(Ordering::Relaxed),
        _ => 0,
    };
    // Each file its own inode, the same through every descriptor of it: musl's dynamic linker takes a library whose
    // (st_dev, st_ino) it has seen for one it already loaded, and libgpud_drm finds its device's memfd that way.
    let ino = if let Kind::Data(d) = &f.kind { d.as_ptr() as u64 } else { f as *const File as u64 };
    #[cfg(feature = "unix")]
    let ino = match &f.kind { Kind::Node(node) => node.inode(), Kind::Pty(end) => end.pty.node.ino, _ => ino };
    #[cfg(feature = "unix")]
    if let Some(node) = match &f.kind { Kind::Node(n) => Some(n), Kind::Pty(end) => Some(&end.pty.node), _ => None } {
        st[0..8].copy_from_slice(&node.device().to_le_bytes());
        let attr = node.attr.lock(); st[28..32].copy_from_slice(&attr.uid.to_le_bytes()); st[32..36].copy_from_slice(&attr.gid.to_le_bytes());
    }
    st[8..16].copy_from_slice(&ino.to_le_bytes());
    let links=1u64;
    #[cfg(feature = "unix")]
    let links=match &f.kind{Kind::Node(node)=>node.links(),_=>links};
    st[16..24].copy_from_slice(&links.to_le_bytes());
    st[24..28].copy_from_slice(&f.mode().to_le_bytes());
    st[48..56].copy_from_slice(&size.to_le_bytes());
    st[56..64].copy_from_slice(&4096u64.to_le_bytes());
    st[64..72].copy_from_slice(&size.div_ceil(512).to_le_bytes());
    #[cfg(feature = "unix")]
    if let Kind::Node(node)=&f.kind{let attr=node.attr.lock();for(at,time)in[(72,attr.atime),(88,attr.mtime),(104,attr.ctime)]{st[at..at+8].copy_from_slice(&(time/1_000_000_000).to_le_bytes());st[at+8..at+16].copy_from_slice(&(time%1_000_000_000).to_le_bytes());}}
    write_user(p, ptr, &st).map(|_| 0)
}

/// A statx reply with the basic fields.
pub(super) fn statx(p: &Process, f: &File, ptr: u64) -> R {
    #[cfg(feature = "unix")]
    f.refresh()?;
    let mut st = [0u8; 256];
    let size = match &f.kind {
        #[cfg(feature = "unix")]
        Kind::Node(node) => node.len(),
        Kind::Data(d) => d.len() as u64,
        Kind::Memfd(_) => f.size.load(Ordering::Relaxed),
        _ => 0,
    };
    st[0..4].copy_from_slice(&0x7ffu32.to_le_bytes());
    st[4..8].copy_from_slice(&4096u32.to_le_bytes());
    st[16..20].copy_from_slice(&1u32.to_le_bytes());
    st[28..30].copy_from_slice(&(f.mode() as u16).to_le_bytes());
    st[40..48].copy_from_slice(&size.to_le_bytes());
    st[48..56].copy_from_slice(&size.div_ceil(512).to_le_bytes());
    #[cfg(feature = "unix")]
    if let Kind::Node(node)=&f.kind{let attr=node.attr.lock();st[16..20].copy_from_slice(&(node.links()as u32).to_le_bytes());st[20..24].copy_from_slice(&attr.uid.to_le_bytes());st[24..28].copy_from_slice(&attr.gid.to_le_bytes());st[32..40].copy_from_slice(&node.inode().to_le_bytes());st[140..144].copy_from_slice(&(node.device()as u32).to_le_bytes());for(at,time)in[(64,attr.atime),(112,attr.mtime),(96,attr.ctime)]{st[at..at+8].copy_from_slice(&(time/1_000_000_000).to_le_bytes());st[at+8..at+12].copy_from_slice(&((time%1_000_000_000)as u32).to_le_bytes());}}
    write_user(p, ptr, &st).map(|_| 0)
}

/// Opens a path in the tree or a device.
pub(super) fn open_path(path: &str, flags: u32) -> Result<Arc<File>, Errno> {
    let n = normalize(path);
    let kind = match n.as_str() {
        "dev/null" => Kind::Null,
        "dev/zero" => Kind::Zero,
        "dev/urandom" | "dev/random" => Kind::Random,
        "dev/kmsg" if flags & O_ACCMODE == 0 => Kind::Kmsg(Mutex::new(0)),
        "dev/kmsg" | "dev/console" | "dev/tty" => Kind::Log(Arc::new(Log { name: String::from("console"), line: Mutex::new(Vec::new()) })),
        _ => {
            let data = TREE.lock().get(&n).copied();
            match data {
                Some(d) if flags & O_DIRECTORY == 0 => {
                    if flags & O_ACCMODE != 0 {
                        return Err(EACCES);
                    }
                    Kind::Data(d)
                }
                Some(_) => return Err(ENOTDIR),
                None if is_dir(&n) => Kind::Dir(n),
                None if flags & O_CREAT != 0 => return Err(EACCES),
                None => return Err(ENOENT),
            }
        }
    };
    Ok(File::new(kind, flags & !O_CLOEXEC))
}

fn path_at(p: &Process, dirfd: i32, ptr: u64) -> Result<String, Errno> {
    let path = read_cstr(p, ptr, 4096)?;
    if path.starts_with('/') || dirfd == AT_FDCWD {
        return Ok(path);
    }
    match &linux(p).files.get(dirfd)?.kind {
        Kind::Dir(d) => Ok(alloc::format!("/{d}/{path}")),
        _ => Err(ENOTDIR),
    }
}

fn getdents(p: &Process, f: &File, ptr: u64, len: u64) -> R {
    let Kind::Dir(d) = &f.kind else { return Err(ENOTDIR) };
    let prefix = if d.is_empty() { String::new() } else { alloc::format!("{d}/") };
    let mut names: Vec<(String, bool)> = Vec::new();
    for k in TREE.lock().keys() {
        if let Some(rest) = k.strip_prefix(prefix.as_str()) {
            let (name, dir) = match rest.split_once('/') {
                Some((n, _)) => (n, true),
                None => (rest, false),
            };
            if names.last().is_none_or(|l| l.0 != name) {
                names.push((String::from(name), dir));
            }
        }
    }
    let mut pos = f.pos.lock();
    let mut out = Vec::new();
    while (*pos as usize) < names.len() {
        let (name, dir) = &names[*pos as usize];
        let reclen = (19 + name.len() + 1).next_multiple_of(8);
        if out.len() + reclen > len as usize {
            break;
        }
        let mut e = alloc::vec![0u8; reclen];
        e[0..8].copy_from_slice(&(*pos + 1000).to_le_bytes());
        e[8..16].copy_from_slice(&(*pos + 1).to_le_bytes());
        e[16..18].copy_from_slice(&(reclen as u16).to_le_bytes());
        e[18] = if *dir { 4 } else { 8 };
        e[19..19 + name.len()].copy_from_slice(name.as_bytes());
        out.extend_from_slice(&e);
        *pos += 1;
    }
    if out.is_empty() && (*pos as usize) < names.len() {
        return Err(EINVAL);
    }
    write_user(p, ptr, &out)?;
    Ok(out.len() as u64)
}

/// poll/ppoll over pollfds at `ptr`; `deadline` in TSC.
fn poll(p: &Process, ptr: u64, n: u64, deadline: u64) -> R {
    if n > 1024 {
        return Err(EINVAL);
    }
    let raw = read_user(p, ptr, n * 8)?;
    let fds: Vec<(i32, u16)> = raw.chunks_exact(8).map(|c| (i32::from_le_bytes(c[..4].try_into().unwrap()), u16::from_le_bytes(c[4..6].try_into().unwrap()))).collect();
    let files: Vec<Option<Arc<File>>> = fds.iter().map(|&(fd, _)| if fd < 0 { None } else { linux(p).files.get(fd).ok() }).collect();
    let me = sched::current();
    loop {
        sched::prepare_block(&me, deadline);
        let mut rev = alloc::vec![0u16; fds.len()];
        let mut ready = 0;
        for (i, &(fd, ev)) in fds.iter().enumerate() {
            rev[i] = match (&files[i], fd) {
                (_, fd) if fd < 0 => 0,
                (None, _) => POLLNVAL,
                (Some(f), _) => f.poll(&me, ev),
            };
            if rev[i] != 0 {
                ready += 1;
            }
        }
        if ready > 0 || cpu::rdtsc() >= deadline || me.killed() {
            sched::cancel_block(&me);
        } else {
            sched::schedule();
        }
        for f in files.iter().flatten() {
            f.unpoll(&me);
        }
        if ready > 0 || cpu::rdtsc() >= deadline {
            let mut out = raw.clone();
            for (i, r) in rev.iter().enumerate() {
                out[i * 8 + 6..i * 8 + 8].copy_from_slice(&r.to_le_bytes());
            }
            write_user(p, ptr, &out)?;
            return Ok(ready);
        }
        if me.killed() {
            return Err(EINTR);
        }
    }
}

pub fn syscall(p: &Arc<Process>, n: u64, a: [u64; 6]) -> R {
    let _collect = sock::CollectOnReturn;
    let files = &linux(p).files;
    let fd = a[0] as i32;
    match n {
        // read, pread64
        0 | 17 => {
            let f = files.get(fd)?;
            let d = f.read(p, a[2].min(16 << 20) as usize, (n == 17).then_some(a[3]))?;
            write_user(p, a[1], &d)?;
            Ok(d.len() as u64)
        }
        // write, pwrite64
        1 | 18 => {
            let f = files.get(fd)?;
            let d = read_user(p, a[1], a[2].min(16 << 20))?;
            f.write(&d, (n == 18).then_some(a[3])).map(|n| n as u64)
        }
        19 => {
            let f = files.get(fd)?;
            let iov = read_iov(p, a[1], a[2])?;
            let total: u64 = iov.iter().map(|v| v.1).sum();
            let d = f.read(p, total.min(16 << 20) as usize, None)?;
            scatter(p, &iov, &d).map(|n| n as u64)
        }
        20 => {
            let f = files.get(fd)?;
            let iov = read_iov(p, a[1], a[2])?;
            let d = gather(p, &iov)?;
            f.write(&d, None).map(|n| n as u64)
        }
        // open, openat
        2 | 257 => {
            let (path, flags) = if n == 2 { (read_cstr(p, a[0], 4096)?, a[1] as u32) } else { (path_at(p, a[0] as i32, a[1])?, a[2] as u32) };
            let f = open_path(&path, flags)?;
            files.insert(f, flags & O_CLOEXEC != 0)
        }
        3 => files.close(fd).map(|f| {
            drop(f);
            0
        }),
        // stat, lstat, fstat, newfstatat
        4 | 6 => {
            let f = open_path(&read_cstr(p, a[0], 4096)?, 0)?;
            stat(p, &f, a[1])
        }
        5 => stat(p, &*files.get(fd)?, a[1]),
        262 => {
            let path = read_cstr(p, a[1], 4096)?;
            if path.is_empty() && a[3] & 0x1000 != 0 {
                return stat(p, &*files.get(a[0] as i32)?, a[2]);
            }
            let f = open_path(&path_at(p, a[0] as i32, a[1])?, 0)?;
            stat(p, &f, a[2])
        }
        332 => {
            let path = read_cstr(p, a[1], 4096)?;
            let f = if path.is_empty() && a[2] & 0x1000 != 0 { files.get(a[0] as i32)? } else { open_path(&path_at(p, a[0] as i32, a[1])?, 0)? };
            statx(p, &f, a[4])
        }
        // access, faccessat, faccessat2
        21 | 269 | 439 => {
            let path = if n == 21 { read_cstr(p, a[0], 4096)? } else { path_at(p, a[0] as i32, a[1])? };
            open_path(&path, 0).map(|_| 0)
        }
        8 => {
            let f = files.get(fd)?;
            #[cfg(feature = "unix")]
            let _position = f.position_guard()?;
            #[cfg(feature = "unix")]
            f.refresh()?;
            let size = match &f.kind {
                #[cfg(feature = "unix")]
                Kind::Node(node) => node.len() as i64,
                Kind::Data(d) => d.len() as i64,
                Kind::Memfd(_) => f.size.load(Ordering::Relaxed) as i64,
                Kind::Dir(_) => 0,
                _ => return Err(ESPIPE),
            };
            let mut pos = f.pos.lock();
            let new = match a[2] {
                0 => a[1] as i64,
                1 => *pos as i64 + a[1] as i64,
                2 => size + a[1] as i64,
                _ => return Err(EINVAL),
            };
            if new < 0 {
                return Err(EINVAL);
            }
            *pos = new as u64;
            Ok(new as u64)
        }
        16 => {
            let f = files.get(fd)?;
            #[cfg(feature = "unix")]
            if let Kind::Pty(end) = &f.kind { return end.ioctl(p, a[1] as u32 as u64, a[2]); }
            match a[1] {
                // FIONBIO
                0x5421 => {
                    let on = read_u32(p, a[2])? != 0;
                    if on { f.flags.fetch_or(O_NONBLOCK, Ordering::Relaxed) } else { f.flags.fetch_and(!O_NONBLOCK, Ordering::Relaxed) };
                    Ok(0)
                }
                // FIOCLEX, FIONCLEX
                0x5451 | 0x5450 => {
                    let mut table=files.fds.lock();let entry=table.get_mut(fd as usize).and_then(Option::as_mut).ok_or(EBADF)?;
                    entry.cloexec=a[1]==0x5451;Ok(0)
                },
                _ => Err(ENOTTY),
            }
        }
        // pipe, pipe2
        22 | 293 => {
            let (r, w) = Endpoint::pipe()?;
            let flags = if n == 293 { a[1] as u32 } else { 0 };
            files.insert_pair(File::new(Kind::Stream(r), flags & O_NONBLOCK),
                File::new(Kind::Stream(w), flags & O_NONBLOCK), flags & O_CLOEXEC != 0,
                |first, second| {
                    let mut bytes = [0u8; 8];
                    bytes[..4].copy_from_slice(&(first as u32).to_le_bytes());
                    bytes[4..].copy_from_slice(&(second as u32).to_le_bytes());
                    write_user(p, a[0], &bytes)
                }).map(|()| 0)
        }
        32 => files.insert(files.get(fd)?, false),
        33 | 292 => {
            if a[1] >= 4096 { return Err(EBADF); }
            let f = files.get(fd)?;
            if n == 33 && a[0] == a[1] { files.get(fd)?; return Ok(a[1]); }
            if n == 292 && a[0] == a[1] {
                return Err(EINVAL);
            }
            let old = files.set(a[1] as usize, f, n == 292 && a[2] as u32 & O_CLOEXEC != 0);
            drop(old);
            Ok(a[1])
        }
        72 => {
            let f = files.get(fd)?;
            match a[1] {
                0 | 1030 => files.insert_from(f, a[1] == 1030, a[2] as usize),
                1 => Ok(files.fds.lock()[fd as usize].as_ref().map_or(0, |f| f.cloexec as u64)),
                2 => {
                    if let Some(Some(e)) = files.fds.lock().get_mut(fd as usize) {
                        e.cloexec = a[2] & 1 != 0;
                    }
                    Ok(0)
                }
                3 => Ok(f.flags.load(Ordering::Relaxed) as u64),
                4 => {
                    let keep = f.flags.load(Ordering::Relaxed) & !O_NONBLOCK;
                    f.flags.store(keep | a[2] as u32 & O_NONBLOCK, Ordering::Relaxed);
                    Ok(0)
                }
                _ => Err(EINVAL),
            }
        }
        77 => {
            let f = files.get(fd)?;
            let Kind::Memfd(m) = &f.kind else { return Err(EINVAL) };
            // Native memory objects have fixed backing; truncation must not replace it.
            if f.memory_rights.is_some() { return Err(EACCES); }
            let size = a[1];
            if size > i64::MAX as u64 { return Err(EINVAL); }
            let rounded = size.checked_add(4095).ok_or(EINVAL)? & !4095;
            let mut g = m.lock();
            let cur = g.as_ref().map_or(0, |m| m.size());
            if rounded > cur {
                let new = Arc::new(Memory::anonymous(rounded.max(4096), false, false).map_err(|_| ENOMEM)?);
                if let Some(old) = g.as_ref() {
                    new.copy_from(0, old, 0, old.size()).map_err(|_| EINVAL)?;
                }
                *g = Some(new);
            }
            f.size.store(size, Ordering::Relaxed);
            Ok(0)
        }
        // fallocate: memfds are allocated in full already; punching holes keeps the memory.
        285 => files.get(fd).map(|_| 0),
        217 | 78 => {
            let f = files.get(fd)?;
            getdents(p, &f, a[1], a[2])
        }
        79 => {
            if a[1] < 2 {
                return Err(34);
            }
            write_user(p, a[0], b"/\0").map(|_| a[0])
        }
        80 => {
            let path = read_cstr(p, a[0], 4096)?;
            if normalize(&path).is_empty() { Ok(0) } else { Err(EACCES) }
        }
        // mkdir, mkdirat, unlink, unlinkat, chmod, fchmod, chown, fchown, fchownat, fchmodat, umask: the tree is
        // read-only, but socket paths need no directories, so creating them succeeds.
        83 | 258 | 90 | 91 | 92 | 93 | 260 | 268 | 87 | 263 => Ok(0),
        95 => Ok(0o022),
        89 | 267 => Err(EINVAL),
        // eventfd, eventfd2
        284 | 290 => {
            let flags = if n == 290 { a[1] as u32 } else { 0 };
            files.insert(File::new(Kind::EventFd(WaitCell::new(a[0])), flags & O_NONBLOCK), flags & O_CLOEXEC != 0)
        }
        319 => {
            let flags = a[1] as u32;
            files.insert(File::new(Kind::Memfd(Mutex::new(None)), 2), flags & 1 != 0)
        }
        7 => {
            let ms = a[2] as i64;
            let deadline = if ms < 0 { u64::MAX } else { time::tsc_at_ns(time::now_ns().saturating_add((ms as u64).saturating_mul(1_000_000))) };
            poll(p, a[0], a[1], deadline)
        }
        271 => {
            let deadline = if a[2] == 0 { u64::MAX } else { time::tsc_at_ns(time::now_ns().saturating_add(read_timespec(p, a[2])?)) };
            poll(p, a[0], a[1], deadline)
        }
        _ => Err(ENOSYS),
    }
}

/// An fd for a dross memory object (memory or device memory such as a BAR): mappable, like a memfd.
pub fn memory_fd(p: &Process, m: Arc<Memory>, rights: u32) -> R {
    let size = m.size();
    let flags = if rights & dross_abi::rights::WRITE == 0 { 0 } else if rights & dross_abi::rights::READ == 0 { 1 } else { 2 };
    let f = File::with_memory_rights(Kind::Memfd(Mutex::new(Some(m))), flags, Some(rights));
    f.size.store(size, Ordering::Relaxed);
    linux(p).files.insert(f, true)
}

/// An fd for a dross notification: readable while bits are pending; reading takes them (8 bytes: 1).
pub fn notification_fd(p: &Process, n: Arc<Notification>) -> R {
    linux(p).files.insert(File::new(Kind::Notify(n), 0), true)
}
