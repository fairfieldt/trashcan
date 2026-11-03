use super::*;
use crate::memory::Memory;
use alloc::collections::BTreeMap;

static PROCESSES: Mutex<BTreeMap<u32, Weak<Process>>> = Mutex::new(BTreeMap::new());
fn track(p: &Arc<Process>) {
    PROCESSES.lock().insert(p.pid(), Arc::downgrade(p));
}
pub fn signal_group(group: u32, sig: u32) {
    let procs: Vec<_> = PROCESSES
        .lock()
        .values()
        .filter_map(Weak::upgrade)
        .collect();
    for p in procs {
        if pgid(&p) == group {
            super::signals::send(&p, sig);
        }
    }
}
pub(super) fn group_in_session(group: u32, session: u32) -> bool {
    PROCESSES
        .lock()
        .values()
        .filter_map(Weak::upgrade)
        .any(|p| pgid(&p) == group && sid(&p) == session)
}
fn inherit(parent: &Arc<Process>, child: &Arc<Process>) {
    let src = state(parent);
    let dst = state(child);
    linux(child).files.inherit(&linux(parent).files);
    *linux(child).sigmask.lock() = *linux(parent).sigmask.lock();
    *linux(child).sigactions.lock() = *linux(parent).sigactions.lock();
    *dst.cwd.lock() = src.cwd.lock().clone();
    *dst.credentials.lock() = src.credentials.lock().clone();
    dst.mask
        .store(src.mask.load(Ordering::Relaxed), Ordering::Relaxed);
    *dst.parent.lock() = Arc::downgrade(parent);
    dst.pgid.store(pgid(parent), Ordering::Relaxed);
    dst.sid.store(sid(parent), Ordering::Relaxed);
    *dst.ctty.lock() = src.ctty.lock().clone();
    *dst.saved_signals.lock() = src.saved_signals.lock().clone();
}
fn fork(p: &Arc<Process>, f: &Frame, stack: u64, shared: bool) -> R {
    // Stop-the-world fork of a multithreaded address space is not implemented yet.
    if p.live_threads() != 1 {
        return Err(EAGAIN);
    }
    {
        let mut processes = PROCESSES.lock();
        processes.retain(|_, p| p.strong_count() != 0);
        if processes.len() >= 256 {
            return Err(EAGAIN);
        }
    }
    let child = Process::new_linux(&p.name).map_err(|_| ENOMEM)?;
    if shared {
        p.aspace.vfork_into(&child.aspace)
    } else {
        p.aspace.fork_into(&child.aspace, child.account.clone())
    }
    .map_err(|_| ENOMEM)?;
    let gate = if shared {
        Some(Arc::new(WaitCell::new(false)))
    } else {
        None
    };
    *state(&child).vfork.lock() = gate.clone();
    inherit(p, &child);
    let mut regs = [0u64; 22];
    // SAFETY: Frame is repr(C), exactly 22 u64s. The child resumes after syscall.
    unsafe {
        core::ptr::copy_nonoverlapping(f as *const Frame as *const u64, regs.as_mut_ptr(), 22);
    }
    if stack != 0 {
        regs[20] = stack;
    }
    regs[14] = 0;
    regs[17] = f.rcx;
    regs[19] = f.r11;
    // Parent and child have independent address spaces, fd tables and TLS storage.
    // Schedule on this CPU so registry/parent linkage is installed before user code.
    let thread = sched::spawn_with(
        child.clone(),
        cpu::index(),
        regs,
        sched::current().fs_base.load(Ordering::Relaxed),
    ).map_err(|_| ENOMEM)?;
    sched::copy_fpu_to(&thread);
    state(p).children.update(|v| v.push(child.clone()));
    track(p);
    track(&child);
    if let Some(gate) = gate {
        while !gate.with(|ready| *ready) {
            if gate
                .wait(u64::MAX, |ready| if *ready { Some(()) } else { None })
                .is_err()
            {
                if sched::current().killed() {
                    return Err(EINTR);
                }
                // Defer parent signal delivery until the child detaches/exits.
                sched::schedule();
            }
        }
    }
    Ok(child.pid() as u64)
}
fn string_array(p: &Process, ptr: u64) -> Result<Vec<u8>, Errno> {
    let mut out = Vec::new();
    if ptr == 0 {
        return Ok(out);
    }
    for i in 0..4096 {
        let at = read_u64(p, ptr.checked_add(i * 8).ok_or(EFAULT)?)?;
        if at == 0 {
            return Ok(out);
        }
        let s = read_cstr(p, at, 128 * 1024 - out.len())?;
        out.extend_from_slice(s.as_bytes());
        out.push(0);
        if out.len() >= 128 * 1024 {
            return Err(7);
        }
    }
    Err(7)
}
fn exec(p: &Arc<Process>, frame: &mut Frame, a: [u64; 6]) -> R {
    if p.live_threads() != 1 {
        return Err(EAGAIN);
    }
    let path = read_cstr(p, a[0], 4096)?;
    let mut argv = string_array(p, a[1])?;
    let envp = string_array(p, a[2])?;
    let mut path = super::files::absolute(p, &path);
    // AT_EXECFN: the path as given, also for a script.
    let execfn = path.clone();
    let mut contents = vfs::read_executable(p, &path)?;
    // Interpret at most four shebang layers. Preserve argv[1..], replacing argv[0].
    for _ in 0..4 {
        if !contents.starts_with(b"#!") {
            break;
        }
        let line = contents[..contents.len().min(256)]
            .split(|&b| b == b'\n')
            .next()
            .ok_or(8)?;
        let line = core::str::from_utf8(&line[2..]).map_err(|_| 8i64)?.trim();
        let mut words = line.splitn(2, |c: char| c.is_whitespace());
        let interpreter = words.next().filter(|s| s.starts_with('/')).ok_or(8)?;
        let mut replacement = Vec::new();
        replacement.extend_from_slice(interpreter.as_bytes());
        replacement.push(0);
        if let Some(arg) = words.next().map(str::trim).filter(|s| !s.is_empty()) {
            replacement.extend_from_slice(arg.as_bytes());
            replacement.push(0);
        }
        replacement.extend_from_slice(path.as_bytes());
        replacement.push(0);
        if let Some(at) = argv.iter().position(|&b| b == 0) {
            replacement.extend_from_slice(&argv[at + 1..]);
        }
        argv = replacement;
        path = String::from(interpreter);
        contents = vfs::read_executable(p, &path)?;
    }
    if contents.starts_with(b"#!") {
        return Err(40);
    }
    let image = Memory::anonymous(
        (contents.len() as u64).next_multiple_of(4096).max(4096),
        false,
        false,
    )
    .map_err(|_| ENOMEM)?;
    image.write(0, &contents).map_err(|_| EFAULT)?;
    let (loaded, entry, sp) = super::super::exec::load(&image, execfn.as_bytes(), &argv, &envp, {
        let c = state(p).credentials.lock();
        [
            c.uid[0] as u64,
            c.uid[1] as u64,
            c.gid[0] as u64,
            c.gid[1] as u64,
        ]
    }, Some(p.account.clone()))
    // A missing interpreter is ENOENT, as on Linux.
    .map_err(|e| if e == dross_abi::Error::NotFound { ENOENT } else { 8 })?;
    // Nothing changes until the full image has been validated and loaded.
    p.aspace.replace_from(&loaded.aspace);
    if let Some(gate) = state(p).vfork.lock().take() {
        gate.update(|ready| *ready = true);
    }
    linux(p).files.exec_close();
    for action in linux(p).sigactions.lock().iter_mut() {
        if action[0] != 1 {
            *action = [0; 4];
        }
    }
    state(p).saved_signals.lock().clear();
    sched::set_fs_base(0);
    sched::current().clear_tid.store(0, Ordering::Relaxed);
    // SAFETY: overwrite the user frame with initialized registers, preserving selectors.
    let cs = frame.cs;
    let ss = frame.ss;
    unsafe {
        core::ptr::write_bytes(frame as *mut Frame, 0, 1);
    }
    frame.cs = cs;
    frame.ss = ss;
    frame.rflags = 0x202;
    frame.rip = entry;
    frame.rsp = sp;
    Ok(0)
}
fn wait(p: &Arc<Process>, a: [u64; 6]) -> R {
    let wanted = a[0] as i32;
    let options = a[2];
    if options & !0xb != 0 {
        return Err(EINVAL);
    }
    loop {
        let result = state(p).children.with(|children| {
            let mut any = false;
            for child in children.iter() {
                let group = state(child).pgid.load(Ordering::Relaxed);
                if wanted > 0 && child.pid() != wanted as u32
                    || wanted == 0 && group != pgid(p)
                    || wanted < -1 && group != wanted.unsigned_abs()
                {
                    continue;
                }
                any = true;
                if let Some(code) = child.exit.with(|exit| *exit) {
                    return Ok(Some((child.pid(), code, true)));
                }
                let event = state(child).wait_event.load(Ordering::Acquire);
                if event != 0
                    && (event == 0xffff && options & 8 != 0 || event != 0xffff && options & 2 != 0)
                {
                    return Ok(Some((child.pid(), event as i64, false)));
                }
            }
            if any { Ok(None) } else { Err(10i64) }
        })?;
        if let Some((pid, code, exited)) = result {
            // Validate output before consuming the zombie.
            let signal = state(p).children.with(|v| {
                v.iter()
                    .find(|c| c.pid() == pid)
                    .map_or(0, |c| state(c).term_signal.load(Ordering::Relaxed))
            });
            let status = if !exited {
                code as u32
            } else if signal != 0 {
                signal
            } else {
                (code as u32 & 255) << 8
            };
            if a[1] != 0 {
                write_u32(p, a[1], status)?;
            }
            if a[3] != 0 {
                write_user(p, a[3], &[0; 144])?;
            }
            if exited {
                state(p)
                    .children
                    .update(|v| v.retain(|child| child.pid() != pid));
                PROCESSES.lock().remove(&pid);
            } else {
                state(p).children.with(|v| {
                    if let Some(child) = v.iter().find(|c| c.pid() == pid) {
                        state(child).wait_event.store(0, Ordering::Release);
                    }
                });
            }
            return Ok(pid as u64);
        }
        if options & 1 != 0 {
            return Ok(0);
        }
        if super::interrupted(p) {
            return Err(EINTR);
        }
        // Bounded sleep handles both child exits and signals; no lost wakeups.
        sched::sleep_ns(1_000_000).map_err(|_| EINTR)?;
    }
}
pub fn dispatch(p: &Arc<Process>, f: &mut Frame, n: u64, a: [u64; 6]) -> Option<R> {
    if n == 56 && a[0] & 0x10000 != 0 {
        return None;
    }
    if !matches!(n, 56..=59 | 61 | 62 | 109..=112 | 121 | 124 | 200 | 234) {
        return None;
    }
    Some(inner(p, f, n, a))
}
fn inner(p: &Arc<Process>, f: &mut Frame, n: u64, a: [u64; 6]) -> R {
    match n {
        57 => fork(p, f, 0, false),
        58 => fork(p, f, 0, true),
        56 if a[0] & 0x10000 == 0 => {
            // Only the vfork/posix_spawn process-clone shape is supported.
            if a[0] & !0x4111 != 0 {
                Err(EINVAL)
            } else {
                fork(p, f, a[1], a[0] & 0x100 != 0)
            }
        }
        59 => exec(p, f, a),
        61 => wait(p, a),
        110 => Ok(state(p).parent.lock().upgrade().map_or(1, |p| p.pid()) as u64),
        109 => {
            track(p);
            if a[0] > i32::MAX as u64 || a[1] > i32::MAX as u64 {
                return Err(EINVAL);
            }
            let target = if a[0] == 0 {
                p.clone()
            } else {
                PROCESSES
                    .lock()
                    .get(&(a[0] as u32))
                    .and_then(Weak::upgrade)
                    .ok_or(3)?
            };
            if target.pid() != p.pid()
                && state(&target)
                    .parent
                    .lock()
                    .upgrade()
                    .is_none_or(|parent| parent.pid() != p.pid())
            {
                return Err(3);
            }
            let session = sid(p);
            let group = if a[1] == 0 { target.pid() } else { a[1] as u32 };
            if sid(&target) != session
                || state(&target).sid.load(Ordering::Relaxed) == target.pid()
                || group != target.pid() && !group_in_session(group, session)
            {
                return Err(1);
            }
            state(&target).pgid.store(group, Ordering::Relaxed);
            Ok(0)
        }
        111 => Ok(pgid(p) as u64),
        121 | 124 => {
            track(p);
            let target = if a[0] == 0 {
                p.clone()
            } else {
                PROCESSES
                    .lock()
                    .get(&(a[0] as u32))
                    .and_then(Weak::upgrade)
                    .ok_or(3)?
            };
            Ok(if n == 121 {
                pgid(&target)
            } else {
                sid(&target)
            } as u64)
        }
        112 => {
            // Native-spawned Linux programs start without a Unix session. A Unix
            // process group leader cannot create a second session afterwards.
            if state(p).sid.load(Ordering::Relaxed) != 0 && pgid(p) == p.pid() {
                return Err(1);
            }
            state(p).sid.store(p.pid(), Ordering::Relaxed);
            state(p).pgid.store(p.pid(), Ordering::Relaxed);
            *state(p).ctty.lock() = None;
            track(p);
            Ok(p.pid() as u64)
        }
        62 | 200 | 234 => {
            let sig = if n == 234 { a[2] } else { a[1] } as u32;
            if sig > 64 {
                Err(EINVAL)
            } else {
                track(p);
                let target = a[0] as i32;
                let targets: Vec<_> = PROCESSES
                    .lock()
                    .values()
                    .filter_map(Weak::upgrade)
                    .filter(|t| {
                        if target > 0 {
                            t.pid() == target as u32
                        } else {
                            pgid(t)
                                == if target == 0 {
                                    pgid(p)
                                } else {
                                    target.unsigned_abs()
                                }
                        }
                    })
                    .collect();
                if targets.is_empty() {
                    Err(3)
                } else {
                    let uid = state(p).credentials.lock().uid[1];
                    let mut allowed = false;
                    for t in targets {
                        if uid == 0 || uid == state(&t).credentials.lock().uid[0] {
                            allowed = true;
                            if sig != 0 {
                                super::signals::send(&t, sig);
                            }
                        }
                    }
                    if allowed { Ok(0) } else { Err(1) }
                }
            }
        }
        _ => Err(ENOSYS),
    }
}
