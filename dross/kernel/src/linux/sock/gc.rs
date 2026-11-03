//! Trace only SCM_RIGHTS and pending-accept ownership. Weak peer/name links do
//! not keep a socket alive. Queue mutation and weak-name upgrades share GRAPH;
//! no queue wait or resource destruction runs while that lock is held.
use super::*;

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
static REQUESTED: AtomicBool = AtomicBool::new(false);
static COLLECTING: AtomicBool = AtomicBool::new(false);
static PRUNE: AtomicBool = AtomicBool::new(false);
pub(super) fn retired() {
    PRUNE.store(true, Ordering::Release);
    REQUESTED.store(true, Ordering::Release);
}
pub(super) fn needs_collection() -> bool {
    has_rights() || PRUNE.load(Ordering::Acquire)
}
static RIGHTS: AtomicUsize = AtomicUsize::new(0);
pub(super) fn has_rights() -> bool {
    RIGHTS.load(Ordering::Acquire) != 0
}
pub(super) fn added_rights(count: usize) {
    RIGHTS.fetch_add(count, Ordering::AcqRel);
}
pub(super) fn removed_rights(count: usize) {
    RIGHTS.fetch_sub(count, Ordering::AcqRel);
}
static GRAPH: Mutex<()> = Mutex::new(());
struct Registry {
    endpoints: Vec<Weak<Endpoint>>,
    listeners: Vec<Weak<Listener>>,
}
static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    endpoints: Vec::new(),
    listeners: Vec::new(),
});
pub(super) fn mutation() -> spin::MutexGuard<'static, ()> {
    GRAPH.lock()
}
pub(super) fn endpoints(a: &Arc<Endpoint>, b: &Arc<Endpoint>) -> Result<(), Errno> {
    let _graph = mutation();
    let mut registry = REGISTRY.lock();
    registry.endpoints.retain(|end| end.strong_count() != 0);
    registry.endpoints.try_reserve(2).map_err(|_| ENOMEM)?;
    registry.endpoints.push(Arc::downgrade(a));
    registry.endpoints.push(Arc::downgrade(b));
    Ok(())
}
pub(super) fn listener(listener: &Arc<Listener>) -> Result<(), Errno> {
    let _graph = mutation();
    let mut registry = REGISTRY.lock();
    registry
        .listeners
        .retain(|listener| listener.strong_count() != 0);
    registry.listeners.try_reserve(1).map_err(|_| ENOMEM)?;
    registry.listeners.push(Arc::downgrade(listener));
    Ok(())
}
struct EndNode {
    end: Arc<Endpoint>,
    internal: usize,
    live: bool,
}
struct ListenNode {
    listener: Arc<Listener>,
    internal: usize,
    live: bool,
}
struct FileNode {
    file: Arc<File>,
    queued: usize,
    live: bool,
}
#[derive(Clone, Copy)]
enum Work {
    End(usize),
    Listener(usize),
    File(usize),
}
enum Garbage {
    Packets(VecDeque<Packet>),
    Pending(VecDeque<Arc<Endpoint>>),
}
fn end_index(ends: &[EndNode], end: &Arc<Endpoint>) -> Option<usize> {
    ends.binary_search_by_key(&(Arc::as_ptr(end) as usize), |n| {
        Arc::as_ptr(&n.end) as usize
    })
    .ok()
}
fn listener_index(listeners: &[ListenNode], listener: &Arc<Listener>) -> Option<usize> {
    listeners
        .binary_search_by_key(&(Arc::as_ptr(listener) as usize), |n| {
            Arc::as_ptr(&n.listener) as usize
        })
        .ok()
}
fn file_index(files: &[FileNode], file: &Arc<File>) -> usize {
    files
        .binary_search_by_key(&(Arc::as_ptr(file) as usize), |n| {
            Arc::as_ptr(&n.file) as usize
        })
        .unwrap()
}

/// Deferred work is retried at syscall/scheduler safe points. Reentrant pressure
/// only marks the request; it never recursively traces this graph.
pub(super) fn has_deferred() -> bool {
    REQUESTED.load(Ordering::Acquire) && !COLLECTING.load(Ordering::Acquire)
}
pub(super) fn reap_deferred() -> usize {
    if has_deferred() { collect() } else { 0 }
}
pub(super) fn collect() -> usize {
    REQUESTED.store(true, Ordering::Release);
    if COLLECTING
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return 0;
    }
    let mut released = 0;
    for _ in 0..2 {
        REQUESTED.store(false, Ordering::Release);
        match collect_once() {
            Attempt::Done(count) => released += count,
            Attempt::Busy => {
                REQUESTED.store(true, Ordering::Release);
                break;
            }
            // Waiting idle CPUs must not spin trying to allocate the same
            // unavailable scratch space. A new operation/pressure retries.
            Attempt::NoMemory => {
                REQUESTED.store(false, Ordering::Release);
                break;
            }
        }
        if !REQUESTED.load(Ordering::Acquire) {
            break;
        }
    }
    COLLECTING.store(false, Ordering::Release);
    released
}
enum Attempt {
    Done(usize),
    Busy,
    NoMemory,
}
fn collect_once() -> Attempt {
    // These owners precede GRAPH so every failure path unlocks before dropping
    // temporary last references or resources with recursive destructors.
    let mut ends: Vec<EndNode> = Vec::new();
    let mut listeners: Vec<ListenNode> = Vec::new();
    let mut files: Vec<FileNode> = Vec::new();
    let mut garbage: Vec<Garbage> = Vec::new();
    let Some(graph) = GRAPH.try_lock() else {
        return Attempt::Busy;
    };
    {
        let mut registry = REGISTRY.lock();
        PRUNE.store(false, Ordering::Release);
        registry.endpoints.retain(|end| end.strong_count() != 0);
        registry
            .listeners
            .retain(|listener| listener.strong_count() != 0);
        if ends.try_reserve_exact(registry.endpoints.len()).is_err()
            || listeners
                .try_reserve_exact(registry.listeners.len())
                .is_err()
        {
            return Attempt::NoMemory;
        }
        ends.extend(
            registry
                .endpoints
                .iter()
                .filter_map(Weak::upgrade)
                .map(|end| EndNode {
                    end,
                    internal: 0,
                    live: false,
                }),
        );
        listeners.extend(
            registry
                .listeners
                .iter()
                .filter_map(Weak::upgrade)
                .map(|listener| ListenNode {
                    listener,
                    internal: 0,
                    live: false,
                }),
        );
    }
    ends.sort_unstable_by_key(|n| Arc::as_ptr(&n.end) as usize);
    listeners.sort_unstable_by_key(|n| Arc::as_ptr(&n.listener) as usize);
    let mut count = 0usize;
    for end in &ends {
        let n = end.end.rx.with(|rx| {
            rx.packets
                .iter()
                .map(|packet| packet.fds.len())
                .sum::<usize>()
        });
        let Some(total) = count.checked_add(n) else {
            return Attempt::NoMemory;
        };
        count = total;
    }
    if files.try_reserve_exact(count).is_err() {
        return Attempt::NoMemory;
    }
    for end in &ends {
        end.end.rx.with(|rx| {
            for packet in &rx.packets {
                for file in &packet.fds {
                    files.push(FileNode {
                        file: file.clone(),
                        queued: 1,
                        live: false,
                    });
                }
            }
        });
    }
    files.sort_unstable_by_key(|n| Arc::as_ptr(&n.file) as usize);
    files.dedup_by(|later, earlier| {
        if Arc::ptr_eq(&later.file, &earlier.file) {
            earlier.queued += later.queued;
            true
        } else {
            false
        }
    });
    // A File owns exactly one reference to its endpoint/listener, independent
    // of how many descriptor tables or packets refer to that File.
    for file in &files {
        match &file.file.kind {
            Kind::Stream(end) => {
                if let Some(i) = end_index(&ends, end) {
                    ends[i].internal += 1;
                }
            }
            Kind::Listener(listener) => {
                if let Some(i) = listener_index(&listeners, listener) {
                    listeners[i].internal += 1;
                }
            }
            _ => {}
        }
    }
    for listener in &listeners {
        listener.listener.queue.with(|pending| {
            for end in pending {
                if let Some(i) = end_index(&ends, end) {
                    ends[i].internal += 1;
                }
            }
        });
    }
    let mut work = Vec::new();
    let Some(nodes) = ends
        .len()
        .checked_add(listeners.len())
        .and_then(|n| n.checked_add(files.len()))
    else {
        return Attempt::NoMemory;
    };
    if work.try_reserve_exact(nodes).is_err()
        || garbage
            .try_reserve_exact(ends.len() + listeners.len())
            .is_err()
    {
        return Attempt::NoMemory;
    }
    for (i, end) in ends.iter_mut().enumerate() {
        if Arc::strong_count(&end.end) > end.internal + 1 {
            end.live = true;
            work.push(Work::End(i));
        }
    }
    for (i, listener) in listeners.iter_mut().enumerate() {
        if Arc::strong_count(&listener.listener) > listener.internal + 1 {
            listener.live = true;
            work.push(Work::Listener(i));
        }
    }
    for (i, file) in files.iter_mut().enumerate() {
        if Arc::strong_count(&file.file) > file.queued + 1 {
            file.live = true;
            work.push(Work::File(i));
        }
    }
    while let Some(node) = work.pop() {
        match node {
            Work::End(i) => ends[i].end.rx.with(|rx| {
                for packet in &rx.packets {
                    for file in &packet.fds {
                        let i = file_index(&files, file);
                        if !files[i].live {
                            files[i].live = true;
                            work.push(Work::File(i));
                        }
                    }
                }
            }),
            Work::Listener(i) => listeners[i].listener.queue.with(|pending| {
                for end in pending {
                    if let Some(i) = end_index(&ends, end) {
                        if !ends[i].live {
                            ends[i].live = true;
                            work.push(Work::End(i));
                        }
                    }
                }
            }),
            Work::File(i) => match &files[i].file.kind {
                Kind::Stream(end) => {
                    if let Some(i) = end_index(&ends, end) {
                        if !ends[i].live {
                            ends[i].live = true;
                            work.push(Work::End(i));
                        }
                    }
                }
                Kind::Listener(listener) => {
                    if let Some(i) = listener_index(&listeners, listener) {
                        if !listeners[i].live {
                            listeners[i].live = true;
                            work.push(Work::Listener(i));
                        }
                    }
                }
                _ => {}
            },
        }
    }
    let mut released = 0;
    for end in &ends {
        if !end.live {
            let packets = end.end.rx.with(|rx| {
                rx.bytes = 0;
                core::mem::take(&mut rx.packets)
            });
            let rights = packets.iter().map(|packet| packet.fds.len()).sum::<usize>();
            removed_rights(rights);
            released += rights;
            garbage.push(Garbage::Packets(packets));
        }
    }
    for listener in &listeners {
        if !listener.live {
            let pending = listener.listener.queue.with(core::mem::take);
            released += pending.len();
            garbage.push(Garbage::Pending(pending));
        }
    }
    drop(graph);
    // File/endpoint destructors can wake peers or re-enter close/collection.
    // Keep all of them out of GRAPH and out of queue locks.
    for item in garbage {
        match item {
            Garbage::Packets(packets) => drop(packets),
            Garbage::Pending(pending) => drop(pending),
        }
    }
    drop(files);
    drop(listeners);
    drop(ends);
    {
        let mut registry = REGISTRY.lock();
        registry.endpoints.retain(|end| end.strong_count() != 0);
        registry
            .listeners
            .retain(|listener| listener.strong_count() != 0);
        if registry.endpoints.is_empty() {
            registry.endpoints = Vec::new();
        }
        if registry.listeners.is_empty() {
            registry.listeners = Vec::new();
        }
    }
    Attempt::Done(released)
}

#[cfg(feature = "audit-test")]
pub(super) fn audit() {
    // Group admission is all-or-nothing when only one slot remains, and a
    // failed user copy must leave both chosen numbers unoccupied.
    {
        let table = super::super::fs::Files::new("socket-group-audit");
        let file = File::new(Kind::Null, 0);
        for _ in 0..4092 {
            table.insert(file.clone(), false).unwrap();
        }
        assert_eq!(
            table.insert_group(&[file.clone(), file.clone()], false, |_| Ok(())),
            Err(EMFILE)
        );
        assert_eq!(table.insert(file.clone(), false).unwrap(), 4095);
        drop(table.close(4095).unwrap());
        drop(table.close(4094).unwrap());
        assert_eq!(
            table.insert_group(&[file.clone(), file], false, |slots| {
                assert_eq!(slots, &[4094, 4095]);
                Err(EFAULT)
            }),
            Err(EFAULT)
        );
        assert!(table.get(4094).is_err());
        assert!(table.get(4095).is_err());
        table.release_all();
    }

    // A live sender's weak peer link does not make an unreachable receive
    // queue (including a descriptor to itself and a memory payload) live.
    {
        let (a, b) = Endpoint::pair(SOCK_SEQPACKET).unwrap();
        let file = File::new(Kind::Stream(b.clone()), 2);
        let weak_end = Arc::downgrade(&b);
        let weak_file = Arc::downgrade(&file);
        let memory = Arc::new(crate::memory::Memory::anonymous(4096, false, false).unwrap());
        let weak_memory = Arc::downgrade(&memory);
        let payload = File::new(Kind::Memfd(Mutex::new(Some(memory))), 2);
        a.send(b"x", alloc::vec![file.clone(), payload], true)
            .unwrap();
        drop(file);
        drop(b);
        assert!(collect() >= 2);
        assert_eq!(weak_file.strong_count(), 0);
        assert_eq!(weak_end.strong_count(), 0);
        assert_eq!(weak_memory.strong_count(), 0);
        assert_eq!(a.send(b"x", Vec::new(), true), Err(EPIPE));
    }
    // Two receive queues own one another. Retaining either open description
    // keeps both live and permits receiving both capabilities normally.
    {
        let (a, b) = Endpoint::pair(SOCK_SEQPACKET).unwrap();
        let fa = File::new(Kind::Stream(a.clone()), 2);
        let fb = File::new(Kind::Stream(b.clone()), 2);
        let wa = Arc::downgrade(&a);
        let wb = Arc::downgrade(&b);
        a.send(b"a", alloc::vec![fa.clone()], true).unwrap();
        b.send(b"b", alloc::vec![fb.clone()], true).unwrap();
        drop(a);
        drop(b);
        drop(fb);
        assert_eq!(collect(), 0);
        let Kind::Stream(a) = &fa.kind else {
            unreachable!()
        };
        let (_, received, _) = a.recv(1, 0, true, false).unwrap();
        assert_eq!(received.len(), 1);
        let Kind::Stream(b) = &received[0].kind else {
            unreachable!()
        };
        let (_, returned, _) = b.recv(1, 0, true, false).unwrap();
        assert!(Arc::ptr_eq(&returned[0], &fa));
        drop(returned);
        drop(received);
        drop(fa);
        collect();
        assert_eq!(wa.strong_count(), 0);
        assert_eq!(wb.strong_count(), 0);
    }
    {
        let (a, b) = Endpoint::pair(SOCK_SEQPACKET).unwrap();
        let fa = File::new(Kind::Stream(a.clone()), 2);
        let fb = File::new(Kind::Stream(b.clone()), 2);
        let wa = Arc::downgrade(&a);
        let wb = Arc::downgrade(&b);
        a.send(b"a", alloc::vec![fa.clone()], true).unwrap();
        b.send(b"b", alloc::vec![fb.clone()], true).unwrap();
        drop(a);
        drop(b);
        drop(fa);
        drop(fb);
        assert!(collect() >= 2);
        assert_eq!(wa.strong_count(), 0);
        assert_eq!(wb.strong_count(), 0);
    }
    // A pending accept is an owning edge, not an unconditional external root.
    {
        let (a, b) = Endpoint::pair(SOCK_SEQPACKET).unwrap();
        let listener = Arc::new(Listener {
            _charge: crate::account::current().reserve(512).unwrap(),
            ty: SOCK_SEQPACKET,
            path: "audit-gc".into(),
            queue: WaitCell::new(VecDeque::new()),
        });
        self::listener(&listener).unwrap();
        let file = File::new(Kind::Listener(listener.clone()), 2);
        let wl = Arc::downgrade(&listener);
        let wb = Arc::downgrade(&b);
        {
            let _graph = mutation();
            listener.queue.with(|pending| pending.push_back(b.clone()));
        }
        a.send(b"l", alloc::vec![file.clone()], true).unwrap();
        drop(file);
        drop(listener);
        drop(b);
        assert!(collect() >= 2);
        assert_eq!(wl.strong_count(), 0);
        assert_eq!(wb.strong_count(), 0);
    }
    #[cfg(feature = "unix")]
    {
        // SCM ownership can be the sole external root of a native channel
        // graph. Releasing the File requests a later native pass, after the
        // wrapper's endpoint field has actually dropped.
        use crate::object::{Entry, Object};
        let (native_sender, native_receiver) = crate::ipc::ChannelEnd::try_pair().unwrap();
        let memory = Arc::new(crate::memory::Memory::anonymous(4096, false, false).unwrap());
        let weak_memory = Arc::downgrade(&memory);
        assert!(
            native_sender
                .write(crate::ipc::Message::new(
                    alloc::vec![1],
                    alloc::vec![
                        Entry::new(Object::Channel(native_receiver.clone())),
                        Entry::new(Object::Memory(memory)),
                    ]
                ))
                .is_ok()
        );
        let channel = File::new(
            Kind::Channel(Arc::new(crate::linux::unix::channel::Stream::new(
                native_receiver,
            ))),
            2,
        );
        let (sender, receiver) = Endpoint::pair(SOCK_SEQPACKET).unwrap();
        let socket = File::new(Kind::Stream(receiver.clone()), 2);
        sender
            .send(b"c", alloc::vec![socket.clone(), channel], true)
            .unwrap();
        drop(socket);
        drop(receiver);
        crate::ipc::collect_cycles();
        assert!(weak_memory.strong_count() > 0);
        assert!(collect() >= 2);
        crate::ipc::reap_deferred();
        assert_eq!(weak_memory.strong_count(), 0);
        drop(native_sender);
    }
    collect();
    assert!(!has_rights());
    crate::println!("audit: Linux queued-fd cycles reclaimed; live roots preserved");
}
