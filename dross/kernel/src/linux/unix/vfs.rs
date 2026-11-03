//! Bounded RAM filesystem and imported boot files. Open descriptions hold node Arcs,
//! so unlink/rename never invalidate an open fd. Symlinks resolve component by component.
use super::*;
use alloc::collections::BTreeMap;

pub const REG: u32 = 0o100000;
pub const DIR: u32 = 0o040000;
pub const LINK: u32 = 0o120000;
use core::sync::atomic::AtomicUsize;
static BYTES: AtomicUsize = AtomicUsize::new(0);
const MAX_FILE: usize = 8 << 20;
const MAX_TOTAL: usize = 24 << 20;
const MAX_NODES: usize = 4096;
const EBUSY: Errno = 16;
const EROFS: Errno = 30;
#[derive(Clone, Copy)]
pub struct Attr {
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub atime: u64,
    pub mtime: u64,
    pub ctime: u64,
}
pub struct Node {
    pub attr: Mutex<Attr>,
    /// A file's contents, a symlink's target.
    pub data: Mutex<Vec<u8>>,
    pub ino: u64,
    links: AtomicU64,
    /// A node served by a userspace filesystem (`mount.rs`).
    pub remote: Option<Remote>,
}
pub struct Remote {
    mount: Arc<super::mount::Mount>,
    /// Below the mount's root.
    path: Mutex<String>,
    size: AtomicU64,
    inode: AtomicU64,
    links: AtomicU64,
    /// A directory whose entries are in the tree.
    listed: core::sync::atomic::AtomicBool,
    /// Active users retain backing; the node alone must not pin the file in RAM.
    content: Mutex<Weak<crate::memory::Memory>>,
}
// Cache-owned references are separate from open descriptions and mappings.
// An oversized file remains usable but is never retained just for caching.
const CONTENT_CACHE_BYTES: u64 = 32 << 20;
const CONTENT_CACHE_ENTRIES: usize = 64;
struct CachedContent {
    ino: u64,
    memory: Arc<crate::memory::Memory>,
}
struct ContentCache {
    entries: [Option<CachedContent>; CONTENT_CACHE_ENTRIES],
    next: usize,
    bytes: u64,
    limit: u64,
}
impl ContentCache {
    const fn new(limit: u64) -> Self {
        Self { entries: [const { None }; CONTENT_CACHE_ENTRIES], next: 0, bytes: 0, limit }
    }
    fn retain(&mut self, ino: u64, memory: Arc<crate::memory::Memory>) {
        let size = memory.size();
        if size > self.limit || self.entries.iter().flatten().any(|e| e.ino == ino) { return; }
        // FIFO bounds both bytes and metadata without allocating cache metadata.
        loop {
            if let Some(old) = self.entries[self.next].take() { self.bytes -= old.memory.size(); }
            if self.bytes + size <= self.limit { break; }
            self.next = (self.next + 1) % CONTENT_CACHE_ENTRIES;
        }
        self.entries[self.next] = Some(CachedContent { ino, memory });
        self.bytes += size;
        self.next = (self.next + 1) % CONTENT_CACHE_ENTRIES;
    }
}
static CONTENT_CACHE: Mutex<ContentCache> = Mutex::new(ContentCache::new(CONTENT_CACHE_BYTES));

/// Drops cache ownership; open descriptions and mappings remain valid.
pub fn reclaim_content_cache() -> u64 {
    let (entries, bytes) = {
        let mut cache = CONTENT_CACHE.lock();
        let entries = core::mem::replace(&mut cache.entries, [const { None }; CONTENT_CACHE_ENTRIES]);
        let bytes = cache.bytes;
        cache.bytes = 0;
        cache.next = 0;
        (entries, bytes)
    };
    // Freeing many pages must not hold the cache registry lock.
    drop(entries);
    bytes
}

#[cfg(feature = "audit-test")]
pub fn audit_content_cache() {
    use crate::memory::Memory;
    let mut cache = ContentCache::new(8192);
    let a = Arc::new(Memory::anonymous(4096, false, false).unwrap());
    let weak = Arc::downgrade(&a);
    cache.retain(1, a);
    cache.retain(2, Arc::new(Memory::anonymous(4096, false, false).unwrap()));
    let live = Arc::new(Memory::anonymous(4096, false, false).unwrap());
    cache.retain(3, live.clone());
    assert!(weak.upgrade().is_none());
    assert_eq!(cache.bytes, 8192);
    cache.retain(4, Arc::new(Memory::anonymous(12288, false, false).unwrap()));
    assert_eq!(cache.bytes, 8192);
    drop(cache);
    assert_eq!(live.size(), 4096);
    crate::println!("vfs: bounded content cache tests passed");
}
impl Node {
    pub fn open_view(&self)->Arc<Self>{
        let remote=self.remote.as_ref().map(|r|Remote{mount:r.mount.clone(),path:Mutex::new(r.path.lock().clone()),size:AtomicU64::new(r.size.load(Ordering::Acquire)),inode:AtomicU64::new(r.inode.load(Ordering::Acquire)),links:AtomicU64::new(r.links.load(Ordering::Acquire)),listed:false.into(),content:Mutex::new(Weak::new())});
        // Writable regular files keep no bytes in Node.data; this detached view
        // prevents an old open target's metadata from changing its replacement.
        Arc::new(Self{attr:Mutex::new(*self.attr.lock()),data:Mutex::new(Vec::new()),ino:self.ino,links:AtomicU64::new(self.links()),remote})
    }
    pub fn mount_root(&self)->bool{self.remote.as_ref().is_some_and(|r|*r.path.lock()=="/")}
    pub fn device(&self)->u64{self.remote.as_ref().map_or(1,|r|r.mount.device)}
    pub fn inode(&self)->u64{self.remote.as_ref().map_or(self.ino,|r|r.inode.load(Ordering::Acquire))}
    pub fn links(&self)->u64{self.remote.as_ref().map_or_else(||self.links.load(Ordering::Acquire),|r|r.links.load(Ordering::Acquire))}
    pub fn apply_info(&self,info:super::mount::Info){
        if let Some(r)=&self.remote{r.inode.store(info.inode,Ordering::Release);r.size.store(info.size,Ordering::Release);r.links.store(info.links as u64,Ordering::Release);}
        *self.attr.lock()=Attr{mode:info.mode,uid:info.uid,gid:info.gid,atime:info.atime,mtime:info.mtime,ctime:info.ctime};
    }
    pub fn refresh(&self)->Result<(),Errno>{if let Some(r)=&self.remote{let path=r.path.lock().clone();let info=r.mount.info(&path)?;self.apply_info(info);}Ok(())}
    pub fn open_file(&self,flags:u32)->Result<Option<Arc<super::mount::OpenFile>>,Errno>{
        let Some(r)=&self.remote else{return Ok(None)};
        if !r.mount.writable()?{return Ok(None)}
        let path=r.path.lock().clone();r.mount.open_file(&path,flags).map(Some)
    }
    /// A file's size.
    pub fn len(&self) -> u64 {
        match &self.remote {
            Some(r) if self.attr.lock().mode & 0o170000 == REG => r.size.load(Ordering::Acquire),
            _ => self.data.lock().len() as u64,
        }
    }
    pub fn writable_remote(&self) -> Result<bool, Errno> {
        self.remote.as_ref().map_or(Ok(false), |r| r.mount.writable())
    }
    pub fn persist_attr(&self, mode: u32, uid: u32, gid: u32) -> Result<(), Errno> {
        if let Some(r) = &self.remote { let path = r.path.lock().clone(); r.mount.setattr(&path, mode, uid, gid)?; }
        Ok(())
    }
    pub fn persist_times(&self, times: [Option<u64>; 2]) -> Result<(), Errno> {
        if let Some(r) = &self.remote { let path = r.path.lock().clone(); r.mount.times(&path, times)?; }
        Ok(())
    }
    pub fn sync(&self) -> Result<(), Errno> {
        self.remote.as_ref().map_or(Ok(()), |r| r.mount.sync())
    }
    /// A mounted file's contents, shared with active readers and the bounded cache.
    pub fn content(&self) -> Result<Option<Arc<crate::memory::Memory>>, Errno> {
        let Some(r) = self.remote.as_ref().filter(|_| self.attr.lock().mode & 0o170000 == REG) else {
            return Ok(None);
        };
        if r.mount.writable()? { let path = r.path.lock().clone(); return r.mount.open(&path).map(Some); }
        if let Some(m) = r.content.lock().upgrade() {
            return Ok(Some(m));
        }
        // RPC can block: do not hold a spinlock across it. Concurrent opens may
        // load twice, but all subsequent users share the first published backing.
        let path = r.path.lock().clone();
        let loaded = r.mount.open(&path)?;
        let m = {
            let mut content = r.content.lock();
            if let Some(m) = content.upgrade() { m } else {
                *content = Arc::downgrade(&loaded);
                loaded
            }
        };
        CONTENT_CACHE.lock().retain(self.ino, m.clone());
        Ok(Some(m))
    }
    /// `len` bytes of a file from `offset`.
    pub fn read_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, Errno> {
        if let Some(r) = self.remote.as_ref().filter(|_| self.attr.lock().mode & 0o170000 == REG) {
            if r.mount.writable()? { let path = r.path.lock().clone(); return r.mount.read(&path, offset, len); }
        }
        if let Some(m) = self.content()? {
            let from = offset.min(self.len());
            let n = len.min((self.len() - from) as usize);
            let mut out = Vec::new();
            out.try_reserve_exact(n).map_err(|_| ENOMEM)?;
            out.resize(n, 0);
            m.read(from, &mut out).map_err(|_| 5)?;
            return Ok(out);
        }
        let data = self.data.lock();
        let from = (offset as usize).min(data.len());
        Ok(data[from..from + len.min(data.len() - from)].to_vec())
    }
}
struct Tree {
    nodes: BTreeMap<String, Arc<Node>>,
    next: u64,
}
static TREE: Mutex<Tree> = Mutex::new(Tree {
    nodes: BTreeMap::new(),
    next: 1,
});

static NAMESPACE_BUSY:WaitCell<bool>=WaitCell::new(false);
pub struct NamespaceGuard;
impl Drop for NamespaceGuard{fn drop(&mut self){NAMESPACE_BUSY.update(|busy|*busy=false);}}
pub fn namespace_guard()->Result<NamespaceGuard,Errno>{NAMESPACE_BUSY.wait_uninterruptible(u64::MAX,|busy|(!*busy).then(||*busy=true)).map_err(|_|EINTR)?;Ok(NamespaceGuard)}

pub fn normalize(path: &str) -> String {
    let mut parts = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            part => parts.push(part),
        }
    }
    alloc::format!("/{}", parts.join("/"))
}
fn parent(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some(("", _)) => "/",
        Some((p, _)) => p,
        _ => "/",
    }
}
fn insert(
    tree: &mut Tree,
    path: String,
    mode: u32,
    uid: u32,
    gid: u32,
    data: Vec<u8>,
) -> Result<Arc<Node>, Errno> {
    if tree.nodes.len() >= MAX_NODES
        || BYTES.load(Ordering::Relaxed) + data.len() > MAX_TOTAL
        || data.len() > MAX_FILE
    {
        return Err(28);
    }
    let now = clock_ns(0);
    let node = new_node(tree, Attr { mode, uid, gid, atime: now, mtime: now, ctime: now }, data, None);
    tree.nodes.insert(path, node.clone());
    Ok(node)
}
fn new_node(tree: &mut Tree, attr: Attr, data: Vec<u8>, remote: Option<Remote>) -> Arc<Node> {
    BYTES.fetch_add(data.len(), Ordering::Relaxed);
    let ino = tree.next;
    tree.next += 1;
    Arc::new(Node { attr: Mutex::new(attr), data: Mutex::new(data), ino, links: AtomicU64::new(if attr.mode & 0o170000 == DIR { 2 } else { 1 }), remote })
}
/// Puts the filesystem `mount` serves at `path` (an empty directory, made if missing).
pub fn mount(path: &str, mount: Arc<super::mount::Mount>) -> Result<(), Errno> {
    let _namespace=namespace_guard()?;
    let path = normalize(path);
    if path == "/" {
        return Err(EBUSY);
    }
    import(&path, DIR | 0o755, &[]);
    let mut tree = TREE.lock();
    if tree.nodes.keys().any(|k| k.starts_with(&alloc::format!("{path}/"))) {
        return Err(EBUSY);
    }
    let now = clock_ns(0);
    let remote = Remote { mount, path: Mutex::new(String::from("/")), size: AtomicU64::new(0), inode: AtomicU64::new(2), links: AtomicU64::new(2), listed: false.into(), content: Mutex::new(Weak::new()) };
    let node = new_node(&mut tree, Attr { mode: DIR | 0o755, uid: 0, gid: 0, atime: now, mtime: now, ctime: now }, Vec::new(), Some(remote));
    tree.nodes.insert(path, node);
    Ok(())
}
/// Brings a mounted directory's entries into the tree, once.
fn list(dir: &Arc<Node>, at: &str) -> Result<(), Errno> {
    let Some(r) = dir.remote.as_ref().filter(|r| !r.listed.load(Ordering::Acquire)) else {
        return Ok(());
    };
    let path = r.path.lock().clone();
    let entries = r.mount.list(&path)?;
    let mut tree = TREE.lock();
    for e in entries {
        let path = if at == "/" { alloc::format!("/{}", e.name) } else { alloc::format!("{at}/{}", e.name) };
        if tree.nodes.contains_key(&path) {
            continue;
        }
        let base = r.path.lock();
        let remote_path = if *base == "/" { alloc::format!("/{}", e.name) } else { alloc::format!("{}/{}", *base, e.name) };
        let remote = Remote { mount: r.mount.clone(), path: Mutex::new(remote_path), size: AtomicU64::new(e.size), inode: AtomicU64::new(e.inode), links: AtomicU64::new(1), listed: false.into(), content: Mutex::new(Weak::new()) };
        let attr = Attr { mode: e.mode, uid: e.uid, gid: e.gid, atime: e.mtime, mtime: e.mtime, ctime: e.mtime };
        let node = new_node(&mut tree, attr, e.target, Some(remote));
        tree.nodes.insert(path, node);
    }
    r.listed.store(true, Ordering::Release);
    Ok(())
}
/// Lists a mounted directory before its entries are read.
pub fn ensure_listed(p: &Process, path: &str) -> Result<(), Errno> {
    let (at, node) = lookup_as(p, path, true)?;
    list(&node, &at)
}
pub fn init() {
    let mut tree = TREE.lock();
    for path in [
        "/",
        "/bin",
        "/sbin",
        "/usr",
        "/usr/bin",
        "/usr/sbin",
        "/etc",
        "/etc/dropbear",
        "/dev",
        "/dev/pts",
        "/tmp",
        "/root",
        "/root/.ssh",
        "/proc",
        "/run",
    ] {
        if !tree.nodes.contains_key(path) {
            let _ = insert(
                &mut tree,
                String::from(path),
                DIR | if path == "/tmp" { 0o1777 } else { 0o755 },
                0,
                0,
                Vec::new(),
            );
        }
    }
}
pub fn import(path: &str, mode: u32, bytes: &[u8]) {
    let path = normalize(path);
    let mut tree = TREE.lock();
    let mut parents = Vec::new();
    let mut p = parent(&path);
    while p != "/" {
        parents.push(String::from(p));
        p = parent(p);
    }
    for p in parents.into_iter().rev() {
        if !tree.nodes.contains_key(&p) {
            let _ = insert(&mut tree, p, DIR | 0o755, 0, 0, Vec::new());
        }
    }
    tree.nodes.remove(&path);
    let _ = insert(&mut tree, path, mode, 0, 0, bytes.to_vec());
}
pub fn archive(bytes: &'static [u8]) {
    init();
    let mut at = 0;
    while at + 512 <= bytes.len() {
        let header = &bytes[at..at + 512];
        if header.iter().all(|&b| b == 0) {
            break;
        }
        let field = |raw: &[u8]| {
            String::from_utf8_lossy(&raw[..raw.iter().position(|&b| b == 0).unwrap_or(raw.len())])
                .into_owned()
        };
        let size = usize::from_str_radix(field(&header[124..136]).trim(), 8).unwrap_or(0);
        let mode =
            u32::from_str_radix(field(&header[100..108]).trim(), 8).unwrap_or(0o644) & 0o7777;
        let name = field(&header[..100]);
        let prefix = field(&header[345..500]);
        let path = if prefix.is_empty() {
            name
        } else {
            alloc::format!("{prefix}/{name}")
        };
        let start = at + 512;
        if size > bytes.len() - start {
            break;
        }
        match header[156] {
            0 | b'0' => import(&path, REG | mode, &bytes[start..start + size]),
            b'5' => import(&path, DIR | mode, &[]),
            b'2' => import(&path, LINK | 0o777, field(&header[157..257]).as_bytes()),
            b'1' => {
                let target = normalize(&field(&header[157..257]));
                let mut tree = TREE.lock();
                if let Some(node) = tree.nodes.get(&target).cloned() {
                    tree.nodes.insert(normalize(&path), node);
                }
            }
            _ => {}
        }
        at = start + size.next_multiple_of(512);
    }
}
pub fn lookup(path: &str, follow: bool) -> Result<(String, Arc<Node>), Errno> {
    lookup_listing(None, path, follow)
}
pub fn lookup_as(p: &Process, path: &str, follow: bool) -> Result<(String, Arc<Node>), Errno> {
    let found=lookup_listing(Some(p),path,follow)?;found.1.refresh()?;Ok(found)
}
/// Why a lookup stopped: an error, or a mounted directory whose entries aren't in the tree yet.
enum Miss {
    Error(Errno),
    List(Arc<Node>, String),
}
impl From<Errno> for Miss {
    fn from(e: Errno) -> Miss {
        Miss::Error(e)
    }
}
fn lookup_listing(process: Option<&Process>, path: &str, follow: bool) -> Result<(String, Arc<Node>), Errno> {
    // Each listing adds a directory's entries, so a path needs at most one per component.
    for _ in 0..256 {
        match resolve(process, path, follow) {
            Ok(found) => return Ok(found),
            Err(Miss::Error(e)) => return Err(e),
            Err(Miss::List(dir, at)) => list(&dir, &at)?,
        }
    }
    Err(40)
}
/// A path's missing entry: ENOENT, or the listing its directory still needs.
fn missing(tree: &Tree, path: &str) -> Miss {
    let dir = parent(path);
    match tree.nodes.get(dir).filter(|d| d.remote.as_ref().is_some_and(|r| !r.listed.load(Ordering::Acquire))) {
        Some(d) => Miss::List(d.clone(), String::from(dir)),
        None => Miss::Error(ENOENT),
    }
}
fn resolve(
    process: Option<&Process>,
    path: &str,
    follow: bool,
) -> Result<(String, Arc<Node>), Miss> {
    let mut path = String::from(path);
    let tree = TREE.lock();
    for _ in 0..40 {
        if let Some(p) = process
            && let Some(root) = tree.nodes.get("/")
        {
            check(p, root, 1)?;
        }
        let components: Vec<&str> = path
            .split('/')
            .filter(|p| !p.is_empty() && *p != ".")
            .collect();
        let mut built = String::new();
        let mut restart = None;
        for (i, part) in components.iter().enumerate() {
            if *part == ".." {
                built = String::from(parent(if built.is_empty() { "/" } else { &built }));
                if built == "/" {
                    built.clear();
                }
                continue;
            }
            built.push('/');
            built.push_str(part);
            let Some(node) = tree.nodes.get(&built) else {
                return Err(missing(&tree, &built));
            };
            let mode = node.attr.lock().mode;
            if mode & 0o170000 == LINK && (follow || i + 1 != components.len()) {
                let data = node.data.lock();
                let target = core::str::from_utf8(&data).map_err(|_| Miss::Error(EINVAL))?;
                let rest = components[i + 1..].join("/");
                restart = Some(if target.starts_with('/') {
                    alloc::format!("{target}/{rest}")
                } else {
                    alloc::format!("{}/{target}/{rest}", parent(&built))
                });
                break;
            }
            if i + 1 != components.len() {
                if mode & 0o170000 != DIR {
                    return Err(Miss::Error(ENOTDIR));
                }
                if let Some(p) = process {
                    check(p, node, 1)?;
                }
            }
        }
        if let Some(next) = restart {
            path = next;
            continue;
        }
        if built.is_empty() {
            built.push('/');
        }
        return match tree.nodes.get(&built) {
            Some(n) => Ok((built, n.clone())),
            None => Err(missing(&tree, &built)),
        };
    }
    Err(Miss::Error(40))
}
pub fn check(p: &Process, node: &Node, need: u32) -> Result<(), Errno> {
    let credentials = state(p).credentials.lock();
    let a = node.attr.lock();
    if credentials.uid[1] == 0 {
        if need & 1 != 0 && a.mode & 0o170000 != DIR && a.mode & 0o111 == 0 {
            return Err(EACCES);
        }
        return Ok(());
    }
    let shift = if credentials.uid[1] == a.uid {
        6
    } else if credentials.gid[1] == a.gid || credentials.groups.contains(&a.gid) {
        3
    } else {
        0
    };
    if (a.mode >> shift) & need == need {
        Ok(())
    } else {
        Err(EACCES)
    }
}
fn writable_parent(p: &Process, path: &str) -> Result<(), Errno> {
    let (_, node) = lookup_as(p, parent(path), true)?;
    if node.attr.lock().mode & 0o170000 != DIR {
        return Err(ENOTDIR);
    }
    if let Some(r) = &node.remote { if !r.mount.writable()? { return Err(EROFS); } }
    check(p, &node, 3)
}
pub fn create(p: &Process, path: &str, mode: u32, data: Vec<u8>) -> Result<Arc<Node>, Errno> {
    let path = destination(p, path)?;
    writable_parent(p, &path)?;
    let (uid, gid) = {
        let credentials = state(p).credentials.lock();
        (credentials.uid[1], credentials.gid[1])
    };
    let (_, parent_node) = lookup_as(p, parent(&path), true)?;
    if let Some(r) = &parent_node.remote {
        let name = path.rsplit('/').next().ok_or(EINVAL)?;
        let remote_path = alloc::format!("{}/{}", r.path.lock().trim_end_matches('/'), name);
        r.mount.create(&remote_path, mode, uid, gid, &data)?;
        let now = clock_ns(0);
        let remote = Remote { mount: r.mount.clone(), path: Mutex::new(remote_path), size: AtomicU64::new(data.len() as u64), inode: AtomicU64::new(0), links: AtomicU64::new(1), listed: false.into(), content: Mutex::new(Weak::new()) };
        let mut tree = TREE.lock();
        let node = new_node(&mut tree, Attr { mode, uid, gid, atime: now, mtime: now, ctime: now }, data, Some(remote));
        tree.nodes.insert(path, node.clone());
        return Ok(node);
    }
    let mut tree = TREE.lock();
    if tree.nodes.contains_key(&path) { return Err(EEXIST); }
    insert(&mut tree, path, mode, uid, gid, data)
}
pub fn resize(node: &Node, size: usize) -> Result<(), Errno> {
    if let Some(r) = &node.remote {
        if !r.mount.writable()? { return Err(EROFS); }
        let path = r.path.lock().clone();
        r.mount.truncate(&path, size as u64)?;
        r.size.store(size as u64, Ordering::Release);
        return Ok(());
    }
    if size > MAX_FILE {
        return Err(28);
    }
    let _tree = TREE.lock();
    let mut data = node.data.lock();
    let total = BYTES
        .load(Ordering::Relaxed)
        .saturating_sub(data.len())
        .checked_add(size)
        .ok_or(28)?;
    if total > MAX_TOTAL {
        return Err(28);
    }
    let old = data.len();
    let extra = size.saturating_sub(old);
    data.try_reserve_exact(extra).map_err(|_| ENOMEM)?;
    data.resize(size, 0);
    data.shrink_to_fit();
    if size >= old {
        BYTES.fetch_add(size - old, Ordering::Relaxed);
    } else {
        BYTES.fetch_sub(old - size, Ordering::Relaxed);
    }
    Ok(())
}
pub fn write(node: &Node, offset: usize, bytes: &[u8], append: bool) -> Result<(usize, usize), Errno> {
    if let Some(r) = &node.remote {
        if !r.mount.writable()? { return Err(EROFS); }
        let path = r.path.lock().clone();
        let (written, end) = r.mount.write(&path, offset as u64, bytes, append)?;
        if written != 0 { r.size.fetch_max(end, Ordering::AcqRel); }
        return Ok((written, usize::try_from(end).map_err(|_| EINVAL)?));
    }
    let _tree = TREE.lock();
    let mut data = node.data.lock();
    let offset = if append { data.len() } else { offset };
    let end = offset.checked_add(bytes.len()).ok_or(28)?;
    if end > MAX_FILE {
        return Err(28);
    }
    if end > data.len() {
        let total = BYTES.load(Ordering::Relaxed) + end - data.len();
        if total > MAX_TOTAL {
            return Err(28);
        }
        let extra = end - data.len();
        data.try_reserve_exact(extra).map_err(|_| ENOMEM)?;
        data.resize(end, 0);
        BYTES.fetch_add(extra, Ordering::Relaxed);
    }
    data[offset..end].copy_from_slice(bytes);
    Ok((bytes.len(), end))
}
pub fn entries(path: &str) -> Vec<(String, Arc<Node>)> {
    let prefix = if path == "/" {
        String::from("/")
    } else {
        alloc::format!("{path}/")
    };
    TREE.lock()
        .nodes
        .iter()
        .filter_map(|(name, node)| {
            name.strip_prefix(&prefix)
                .filter(|name| !name.is_empty() && !name.contains('/'))
                .map(|name| (String::from(name), node.clone()))
        })
        .collect()
}
pub fn remove(p: &Process, path: &str, directory: bool) -> R {
    let (path, node) = lookup_as(p, path, false)?;
    if path == "/" || node.mount_root() { return Err(EBUSY); }
    writable_parent(p, &path)?;
    if directory {
        if node.attr.lock().mode & 0o170000 != DIR {
            return Err(ENOTDIR);
        }
        if !entries(&path).is_empty() {
            return Err(39);
        }
    } else if node.attr.lock().mode & 0o170000 == DIR {
        return Err(EISDIR);
    }
    // Sticky directories allow only the owner of the directory or file (or root).
    let (_, parent) = lookup(parent(&path), true)?;
    let uid = state(p).credentials.lock().uid[1];
    if parent.attr.lock().mode & 0o1000 != 0
        && uid != 0
        && uid != node.attr.lock().uid
        && uid != parent.attr.lock().uid
    {
        return Err(1);
    }
    if let Some(r) = &node.remote {
        let path = r.path.lock().clone();
        r.mount.remove(&path, directory)?;
    }
    TREE.lock().nodes.remove(&path);
    if node.remote.is_none(){node.links.fetch_sub(1,Ordering::AcqRel);}
    Ok(0)
}
pub fn rename(p: &Process, old: &str, new: &str) -> R {
    let (old, node) = lookup_as(p, old, false)?;
    let new = destination(p, new)?;
    if node.mount_root(){return Err(EBUSY)}
    if old == "/" || new.starts_with(&alloc::format!("{old}/")) {
        return Err(EINVAL);
    }
    writable_parent(p, &old)?;
    writable_parent(p, &new)?;
    if old == new { return Ok(0); }
    if let Ok((_,existing))=lookup(&new,false){
        if existing.mount_root(){return Err(EBUSY)}
        if existing.device()==node.device() && existing.inode()==node.inode(){return Ok(0)}
        let isdir=node.attr.lock().mode & 0o170000 == DIR;
        if isdir != (existing.attr.lock().mode & 0o170000 == DIR){return Err(if isdir{ENOTDIR}else{EISDIR})}
        if isdir {ensure_listed(p,&new)?;if !entries(&new).is_empty(){return Err(39)}}
    }
    let (_, new_parent) = lookup_as(p, parent(&new), true)?;
    match (&node.remote, &new_parent.remote) {
        (Some(src), Some(dst)) if Arc::ptr_eq(&src.mount, &dst.mount) => {
            if lookup(&new,false).is_ok_and(|(_,target)|target.inode()==node.inode()){return Ok(0)}
            let next = alloc::format!("{}/{}", dst.path.lock().trim_end_matches('/'), new.rsplit('/').next().ok_or(EINVAL)?);
            let path = src.path.lock().clone();
            let directory = node.attr.lock().mode & 0o170000 == DIR;
            src.mount.rename(&path, &next, directory)?;
        }
        (None, None) => {}
        _ => return Err(18),
    }
    let mut tree = TREE.lock();
    if let Some(replaced)=tree.nodes.remove(&new){if replaced.remote.is_none(){replaced.links.fetch_sub(1,Ordering::AcqRel);}}
    let prefix = alloc::format!("{old}/");
    let names: Vec<_> = tree
        .nodes
        .keys()
        .filter(|k| **k == old || k.starts_with(&prefix))
        .cloned()
        .collect();
    for name in names {
        let n = tree.nodes.remove(&name).unwrap();
        if let (Some(r), Some(dst)) = (&n.remote, &new_parent.remote) {
            *r.path.lock() = alloc::format!("{}/{}{}", dst.path.lock().trim_end_matches('/'), new.rsplit('/').next().unwrap(), &name[old.len()..]);
        }
        tree.nodes.insert(alloc::format!("{new}{}", &name[old.len()..]), n);
    }
    Ok(0)
}
pub fn link(p:&Process,old:&str,new:&str,follow:bool)->R{
    let(_,source)=lookup_as(p,old,follow)?;if source.attr.lock().mode & 0o170000 == DIR{return Err(1)}
    let new=destination(p,new)?;writable_parent(p,&new)?;if lookup(&new,false).is_ok(){return Err(EEXIST)}
    let(_,parent_node)=lookup_as(p,parent(&new),true)?;
    match(&source.remote,&parent_node.remote){
        (Some(src),Some(dst))if Arc::ptr_eq(&src.mount,&dst.mount)=>{
            let old=src.path.lock().clone();let next=alloc::format!("{}/{}",dst.path.lock().trim_end_matches('/'),new.rsplit('/').next().ok_or(EINVAL)?);src.mount.link(&old,&next)?;
            let remote=Remote{mount:src.mount.clone(),path:Mutex::new(next),size:AtomicU64::new(source.len()),inode:AtomicU64::new(source.inode()),links:AtomicU64::new(source.links()+1),listed:false.into(),content:Mutex::new(Weak::new())};
            let attr=*source.attr.lock();let mut tree=TREE.lock();let node=new_node(&mut tree,attr,source.data.lock().clone(),Some(remote));tree.nodes.insert(new,node);drop(tree);source.refresh()?;
        }
        (None,None)=>{source.links.fetch_add(1,Ordering::AcqRel);TREE.lock().nodes.insert(new,source);}
        _=>return Err(18),
    }Ok(0)
}
pub fn read_executable(p: &Process, path: &str) -> Result<Vec<u8>, Errno> {
    let _namespace=namespace_guard()?;
    let (_, node) = lookup_as(p, path, true)?;
    if node.attr.lock().mode & 0o170000 != REG {
        return Err(EACCES);
    }
    check(p, &node, 1)?;
    node.read_at(0, node.len() as usize)
}

pub fn device_node(path: &str, uid: u32) -> Result<Arc<Node>, Errno> {
    insert(
        &mut TREE.lock(),
        String::from(path),
        0o020600,
        uid,
        0,
        Vec::new(),
    )
}
pub fn remove_device(path: &str) {
    TREE.lock().nodes.remove(path);
}

impl Drop for Node {
    fn drop(&mut self) {
        BYTES.fetch_sub(self.data.get_mut().len(), Ordering::Relaxed);
    }
}

fn destination(p: &Process, path: &str) -> Result<String, Errno> {
    let trimmed = path.trim_end_matches('/');
    let name = trimmed.rsplit('/').next().ok_or(EINVAL)?;
    if matches!(name, "" | "." | "..") {
        return Err(EEXIST);
    }
    let (parent, node) = lookup_as(p, parent(trimmed), true)?;
    if node.attr.lock().mode & 0o170000 != DIR {
        return Err(ENOTDIR);
    }
    Ok(if parent == "/" {
        alloc::format!("/{name}")
    } else {
        alloc::format!("{parent}/{name}")
    })
}

pub fn sync_mounts() -> Result<(), Errno> {
    let mounts = {
        let tree = TREE.lock();
        let mut mounts: Vec<Arc<super::mount::Mount>> = Vec::new();
        for node in tree.nodes.values() {
            if let Some(r) = &node.remote {
                if !mounts.iter().any(|m| Arc::ptr_eq(m, &r.mount)) { mounts.push(r.mount.clone()); }
            }
        }
        mounts
    };
    for mount in mounts { mount.sync()?; }
    Ok(())
}
