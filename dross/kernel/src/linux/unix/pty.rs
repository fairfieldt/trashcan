//! A shared kernel PTY mechanism for local and SSH terminals.
use super::super::fs::{File, Kind, POLLHUP, POLLIN, POLLOUT};
use super::*;
use alloc::collections::{BTreeMap, VecDeque};
const LIMIT: usize = 65536;
static NEXT: AtomicU32 = AtomicU32::new(0);
static PTYS: Mutex<BTreeMap<u32, Weak<Pty>>> = Mutex::new(BTreeMap::new());
struct Buffers {
    input: VecDeque<u8>,
    output: VecDeque<u8>,
    line: Vec<u8>,
    eof: bool,
    master: bool,
    slaves: usize,
    locked: bool,
    attrs: [u8; 36],
    size: [u8; 8],
    fg: u32,
    session: u32,
}
pub struct Pty {
    id: u32,
    pub node: Arc<vfs::Node>,
    buffers: WaitCell<Buffers>,
}
pub struct End {
    pub pty: Arc<Pty>,
    pub master: bool,
}
fn flag(s: &Buffers, at: usize) -> u32 {
    u32::from_le_bytes(s.attrs[at..at + 4].try_into().unwrap())
}
impl Pty {
    fn new(id: u32, uid: u32) -> Result<Self, Errno> {
        let mut attrs = [0; 36];
        attrs[..4].copy_from_slice(&0x100u32.to_le_bytes());
        attrs[4..8].copy_from_slice(&5u32.to_le_bytes());
        attrs[8..12].copy_from_slice(&0xbfu32.to_le_bytes());
        attrs[12..16].copy_from_slice(&0x823bu32.to_le_bytes());
        attrs[17..24].copy_from_slice(&[3, 28, 127, 21, 4, 0, 1]);
        attrs[27] = 26;
        Ok(Self {
            id,
            node: vfs::device_node(&alloc::format!("/dev/pts/{id}"), uid)?,
            buffers: WaitCell::new(Buffers {
                input: VecDeque::new(),
                output: VecDeque::new(),
                line: Vec::new(),
                eof: false,
                master: true,
                slaves: 0,
                locked: true,
                attrs,
                size: [24, 0, 80, 0, 0, 0, 0, 0],
                fg: 0,
                session: 0,
            }),
        })
    }
}
impl End {
    pub fn name(&self) -> String {
        if self.master {
            String::from("/dev/ptmx")
        } else {
            alloc::format!("/dev/pts/{}", self.pty.id)
        }
    }
    pub fn read(&self, len: usize, nb: bool) -> Result<Vec<u8>, Errno> {
        if len == 0 {
            return Ok(Vec::new());
        }
        if !self.master {
            let p = sched::current_process();
            let group = pgid(&p);
            let fg = self.pty.buffers.with(|s| s.fg);
            let controlling = state(&p)
                .ctty
                .lock()
                .as_ref()
                .is_some_and(|tty| Arc::ptr_eq(tty, &self.pty));
            if controlling && fg != 0 && fg != group {
                signal_group(group, 21);
                return Err(EINTR);
            }
        }
        let deadline = self.pty.buffers.with(|s| {
            if nb {
                0
            } else if !self.master && flag(s, 12) & 2 == 0 && s.attrs[22] != 0 {
                time::tsc_at_ns(time::now_ns() + s.attrs[22] as u64 * 100_000_000)
            } else {
                u64::MAX
            }
        });
        let result = self.pty.buffers.wait(deadline, |s| {
            let canonical = flag(s, 12) & 2 != 0;
            let immediate = !self.master && !canonical && s.attrs[23] == 0 && s.attrs[22] == 0;
            if if self.master {
                s.output.is_empty() && s.slaves != 0
            } else {
                s.input.is_empty() && s.master && !s.eof && !immediate
            } {
                return None;
            }
            if !self.master && s.eof && s.input.is_empty() {
                s.eof = false;
                return Some(Vec::new());
            }
            let queue = if self.master {
                &mut s.output
            } else {
                &mut s.input
            };
            let mut out = Vec::new();
            while out.len() < len {
                let Some(b) = queue.pop_front() else {
                    break;
                };
                out.push(b);
                if !self.master && canonical && b == b'\n' {
                    break;
                }
            }
            Some(out)
        });
        self.pty.buffers.update(|_| ());
        match result {
            Ok(out) => Ok(out),
            Err(WaitError::TimedOut) if !nb => Ok(Vec::new()),
            Err(WaitError::TimedOut) => Err(EAGAIN),
            Err(_) => Err(EINTR),
        }
    }
    pub fn write(&self, bytes: &[u8], nb: bool) -> Result<usize, Errno> {
        let mut done = 0;
        for chunk in bytes.chunks(2048) {
            let result = self.pty.buffers.wait(if nb { 0 } else { u64::MAX }, |s| {
                if !s.master {
                    return Some(Err(5));
                }
                if if self.master {
                    s.input.len() + s.line.len() + chunk.len() > LIMIT
                        || s.output.len() + chunk.len() * 4 > LIMIT
                } else {
                    s.output.len() + chunk.len() * 2 > LIMIT
                } {
                    return None;
                }
                let mut signals = Vec::new();
                if self.master {
                    let iflag = flag(s, 0);
                    let local = flag(s, 12);
                    for &raw in chunk {
                        let b = if raw == 13 && iflag & 0x100 != 0 {
                            10
                        } else {
                            raw
                        };
                        if raw == 13 && iflag & 0x80 != 0 {
                            continue;
                        }
                        if local & 1 != 0 {
                            let sig = if b != 0 && b == s.attrs[17] {
                                2
                            } else if b != 0 && b == s.attrs[18] {
                                3
                            } else if b != 0 && b == s.attrs[27] {
                                20
                            } else {
                                0
                            };
                            if sig != 0 {
                                s.line.clear();
                                s.input.clear();
                                if s.fg != 0 {
                                    signals.push((s.fg, sig));
                                }
                                if local & 8 != 0 {
                                    s.output.extend([b'^', b ^ 0x40, b'\r', b'\n']);
                                }
                                continue;
                            }
                        }
                        if local & 2 != 0 {
                            if b != 0 && b == s.attrs[19] {
                                if s.line.pop().is_some() && local & 8 != 0 {
                                    s.output.extend(b"\x08 \x08");
                                }
                                continue;
                            }
                            if b != 0 && b == s.attrs[20] {
                                s.line.clear();
                                if local & 8 != 0 {
                                    s.output.extend(b"\r\n");
                                }
                                continue;
                            }
                            if b != 0 && b == s.attrs[21] {
                                if s.line.is_empty() {
                                    s.eof = true;
                                } else {
                                    s.input.extend(s.line.drain(..));
                                }
                                continue;
                            }
                            s.line.push(b);
                            if b == 10 {
                                s.input.extend(s.line.drain(..));
                            }
                        } else {
                            s.input.push_back(b);
                        }
                        if local & 8 != 0 {
                            if b == 10 {
                                s.output.push_back(13);
                            }
                            s.output.push_back(b);
                        }
                    }
                } else {
                    let output = flag(s, 4);
                    for &b in chunk {
                        if output & 5 == 5 && b == 10 {
                            s.output.push_back(13);
                        }
                        s.output.push_back(b);
                    }
                }
                Some(Ok(signals))
            });
            match result {
                Ok(Ok(signals)) => {
                    self.pty.buffers.update(|_| ());
                    for (group, sig) in signals {
                        signal_group(group, sig);
                    }
                    done += chunk.len();
                }
                Ok(Err(e)) => return if done > 0 { Ok(done) } else { Err(e) },
                Err(e) => {
                    return if done > 0 {
                        Ok(done)
                    } else {
                        Err(if e == WaitError::TimedOut {
                            EAGAIN
                        } else {
                            EINTR
                        })
                    };
                }
            }
        }
        Ok(done)
    }
    pub fn poll(&self, thread: &Arc<Thread>, events: u16) -> u16 {
        self.pty.buffers.poll(thread, |s| {
            let (read, write, hup) = if self.master {
                (
                    !s.output.is_empty(),
                    s.input.len() + s.line.len() < LIMIT,
                    s.slaves == 0,
                )
            } else {
                (
                    !s.input.is_empty() || s.eof,
                    s.output.len() < LIMIT,
                    !s.master,
                )
            };
            (if read { POLLIN } else { 0 }
                | if write { POLLOUT } else { 0 }
                | if hup { POLLHUP } else { 0 })
                & (events | POLLHUP)
        })
    }
    pub fn unpoll(&self, thread: &Arc<Thread>) {
        self.pty.buffers.unregister(thread);
    }
    pub fn ioctl(&self, p: &Arc<Process>, request: u64, arg: u64) -> R {
        match request {
            0x5401 => write_user(p, arg, &self.pty.buffers.with(|s| s.attrs)).map(|_| 0),
            0x5402..=0x5404 => {
                let attrs = read_user(p, arg, 36)?;
                self.pty.buffers.update(|s| {
                    s.attrs.copy_from_slice(&attrs);
                    if request == 0x5404 {
                        s.input.clear();
                        s.line.clear();
                    }
                });
                Ok(0)
            }
            0x8004_5430 => write_u32(p, arg, self.pty.id).map(|_| 0),
            0x4004_5431 => {
                if !self.master {
                    return Err(ENOTTY);
                }
                let locked = read_u32(p, arg)? != 0;
                self.pty.buffers.update(|s| s.locked = locked);
                Ok(0)
            }
            0x5413 => write_user(p, arg, &self.pty.buffers.with(|s| s.size)).map(|_| 0),
            0x5414 => {
                let size = read_user(p, arg, 8)?;
                let fg = self.pty.buffers.update(|s| {
                    s.size.copy_from_slice(&size);
                    s.fg
                });
                if fg != 0 {
                    signal_group(fg, 28);
                }
                Ok(0)
            }
            0x540e => {
                let session = sid(p);
                if session != p.pid() {
                    return Err(1);
                }
                self.pty.buffers.update(|s| {
                    if s.session != 0 && s.session != session {
                        return Err(1);
                    }
                    s.session = session;
                    s.fg = pgid(p);
                    Ok(0)
                })?;
                *state(p).ctty.lock() = Some(self.pty.clone());
                Ok(0)
            }
            0x540f => write_u32(p, arg, self.pty.buffers.with(|s| s.fg)).map(|_| 0),
            0x5410 => {
                let group = read_u32(p, arg)?;
                if group == 0 {
                    return Err(EINVAL);
                }
                let controlling = state(p)
                    .ctty
                    .lock()
                    .as_ref()
                    .is_some_and(|tty| Arc::ptr_eq(tty, &self.pty));
                let session = self.pty.buffers.with(|s| s.session);
                if !controlling || session != sid(p) {
                    return Err(ENOTTY);
                }
                if !super::processes::group_in_session(group, session) {
                    return Err(1);
                }
                self.pty.buffers.update(|s| s.fg = group);
                Ok(0)
            }
            0x5422 => {
                *state(p).ctty.lock() = None;
                Ok(0)
            }
            0x541b => write_u32(
                p,
                arg,
                self.pty.buffers.with(|s| {
                    if self.master {
                        s.output.len()
                    } else {
                        s.input.len()
                    }
                }) as u32,
            )
            .map(|_| 0),
            0x540b => {
                self.pty.buffers.update(|s| {
                    if arg == 0 || arg == 2 {
                        s.input.clear();
                        s.line.clear();
                    }
                    if arg == 1 || arg == 2 {
                        s.output.clear();
                    }
                });
                Ok(0)
            }
            _ => Err(ENOTTY),
        }
    }
}
impl Drop for End {
    fn drop(&mut self) {
        let fg = self.pty.buffers.update(|s| {
            if self.master {
                s.master = false;
                s.fg
            } else {
                s.slaves = s.slaves.saturating_sub(1);
                0
            }
        });
        if fg != 0 {
            signal_group(fg, 1);
        }
        if self.pty.buffers.with(|s| !s.master && s.slaves == 0) {
            vfs::remove_device(&alloc::format!("/dev/pts/{}", self.pty.id));
        }
    }
}
pub fn open(p: &Process, path: &str, flags: u32) -> Option<Result<Arc<File>, Errno>> {
    if path == "/dev/ptmx" {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let mut all = PTYS.lock();
        all.retain(|_, p| p.strong_count() != 0);
        if all.len() == 128 {
            return Some(Err(28));
        }
        let pty = match Pty::new(id, state(p).credentials.lock().uid[1]) {
            Ok(pty) => Arc::new(pty),
            Err(e) => return Some(Err(e)),
        };
        all.insert(id, Arc::downgrade(&pty));
        return Some(Ok(File::new(
            Kind::Pty(Arc::new(End { pty, master: true })),
            flags & !0x80000,
        )));
    }
    let pty = if path == "/dev/tty" {
        match state(p).ctty.lock().clone() {
            Some(p) => p,
            None => return Some(Err(6)),
        }
    } else {
        let id = path
            .strip_prefix("/dev/pts/")
            .and_then(|s| s.parse::<u32>().ok())?;
        match PTYS.lock().get(&id).and_then(Weak::upgrade) {
            Some(p) => p,
            None => return Some(Err(ENOENT)),
        }
    };
    let need = match flags & 3 {
        0 => 4,
        1 => 2,
        2 => 6,
        _ => return Some(Err(EINVAL)),
    };
    if let Err(e) = vfs::check(p, &pty.node, need) {
        return Some(Err(e));
    }
    if let Err(e) = pty.buffers.update(|s| {
        if s.locked || !s.master {
            Err(5)
        } else {
            s.slaves += 1;
            Ok(())
        }
    }) {
        return Some(Err(e));
    }
    if flags & 0x100 == 0
        && state(p).ctty.lock().is_none()
        && state(p).sid.load(Ordering::Relaxed) == p.pid()
    {
        let acquired = pty.buffers.update(|s| {
            if s.session == 0 {
                s.session = p.pid();
                s.fg = pgid(p);
                true
            } else {
                false
            }
        });
        if acquired {
            *state(p).ctty.lock() = Some(pty.clone());
        }
    }
    Some(Ok(File::new(
        Kind::Pty(Arc::new(End { pty, master: false })),
        flags & !0x80000,
    )))
}
