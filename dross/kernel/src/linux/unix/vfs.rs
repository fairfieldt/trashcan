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
    /// A node of a mounted filesystem (`mount.rs`): read-only.
    pub remote: Option<Remote>,
}
pub struct Remote {
    mount: Arc<super::mount::Mount>,
    /// Below the mount's root.
    path: String,
    size: u64,
    /// A directory whose entries are in the tree.
    listed: core::sync::atomic::AtomicBool,
    /// A file's contents, once opened.
    content: Mutex<Option<Arc<crate::memory::Memory>>>,
}
impl Node {
    /// A file's size.
    pub fn len(&self) -> u64 {
        match &self.remote {
            Some(r) if self.attr.lock().mode & 0o170000 == REG => r.size,
            _ => self.data.lock().len() as u64,
        }
    }
    /// A mounted file's contents (read from the filesystem the first time).
    pub fn content(&self) -> Result<Option<Arc<crate::memory::Memory>>, Errno> {
        let Some(r) = self.remote.as_ref().filter(|_| self.attr.lock().mode & 0o170000 == REG) else {
            return Ok(None);
        };
        if let Some(m) = r.content.lock().clone() {
            return Ok(Some(m));
        }
        // Read outside the lock; a concurrent open may read it too, and one copy is kept.
        let m = r.mount.open(&r.path)?;
        Ok(Some(r.content.lock().get_or_insert(m).clone()))
    }
    /// `len` bytes of a file from `offset`.
    pub fn read_at(&self, offset: u64, len: usize) -> Result<Vec<u8>, Errno> {
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
    Arc::new(Node { attr: Mutex::new(attr), data: Mutex::new(data), ino, remote })
}
/// Puts the filesystem `mount` serves at `path` (an empty directory, made if missing).
pub fn mount(path: &str, mount: Arc<super::mount::Mount>) -> Result<(), Errno> {
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
    let remote = Remote { mount, path: String::from("/"), size: 0, listed: false.into(), content: Mutex::new(None) };
    let node = new_node(&mut tree, Attr { mode: DIR | 0o755, uid: 0, gid: 0, atime: now, mtime: now, ctime: now }, Vec::new(), Some(remote));
    tree.nodes.insert(path, node);
    Ok(())
}
/// Brings a mounted directory's entries into the tree, once.
fn list(dir: &Arc<Node>, at: &str) -> Result<(), Errno> {
    let Some(r) = dir.remote.as_ref().filter(|r| !r.listed.load(Ordering::Acquire)) else {
        return Ok(());
    };
    let entries = r.mount.list(&r.path)?;
    let mut tree = TREE.lock();
    for e in entries {
        let path = if at == "/" { alloc::format!("/{}", e.name) } else { alloc::format!("{at}/{}", e.name) };
        if tree.nodes.contains_key(&path) {
            continue;
        }
        let remote_path = if r.path == "/" { alloc::format!("/{}", e.name) } else { alloc::format!("{}/{}", r.path, e.name) };
        let remote = Remote { mount: r.mount.clone(), path: remote_path, size: e.size, listed: false.into(), content: Mutex::new(None) };
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
    lookup_listing(Some(p), path, follow)
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
    if node.remote.is_some() || lookup(path, false).is_ok_and(|(_, n)| n.remote.is_some()) {
        return Err(EROFS);
    }
    check(p, &node, 3)
}
pub fn create(p: &Process, path: &str, mode: u32, data: Vec<u8>) -> Result<Arc<Node>, Errno> {
    let path = destination(p, path)?;
    writable_parent(p, &path)?;
    let (uid, gid) = {
        let credentials = state(p).credentials.lock();
        (credentials.uid[1], credentials.gid[1])
    };
    let mut tree = TREE.lock();
    if tree.nodes.contains_key(&path) {
        return Err(EEXIST);
    }
    insert(&mut tree, path, mode, uid, gid, data)
}
pub fn resize(node: &Node, size: usize) -> Result<(), Errno> {
    if node.remote.is_some() {
        return Err(EROFS);
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
pub fn write(node: &Node, offset: usize, bytes: &[u8], append: bool) -> Result<usize, Errno> {
    if node.remote.is_some() {
        return Err(EROFS);
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
    Ok(end)
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
    if path == "/" {
        return Err(16);
    }
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
    TREE.lock().nodes.remove(&path);
    Ok(0)
}
pub fn rename(p: &Process, old: &str, new: &str) -> R {
    let (old, node) = lookup_as(p, old, false)?;
    let new = destination(p, new)?;
    if old == "/" || new.starts_with(&alloc::format!("{old}/")) {
        return Err(EINVAL);
    }
    writable_parent(p, &old)?;
    writable_parent(p, &new)?;
    let mut tree = TREE.lock();
    if let Some(existing) = tree.nodes.get(&new) {
        let isdir = node.attr.lock().mode & 0o170000 == DIR;
        if isdir != (existing.attr.lock().mode & 0o170000 == DIR) {
            return Err(if isdir { ENOTDIR } else { EISDIR });
        }
        if tree
            .nodes
            .keys()
            .any(|k| k.starts_with(&alloc::format!("{new}/")))
        {
            return Err(39);
        }
    }
    let prefix = alloc::format!("{old}/");
    let names: Vec<_> = tree
        .nodes
        .keys()
        .filter(|k| **k == old || k.starts_with(&prefix))
        .cloned()
        .collect();
    for name in names {
        let n = tree.nodes.remove(&name).unwrap();
        tree.nodes
            .insert(alloc::format!("{new}{}", &name[old.len()..]), n);
    }
    Ok(0)
}
pub fn read_executable(p: &Process, path: &str) -> Result<Vec<u8>, Errno> {
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
