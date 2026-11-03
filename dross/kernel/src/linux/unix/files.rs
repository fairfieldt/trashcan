use super::super::fs::{self, File, Kind};
use super::*;

pub fn absolute(p: &Process, path: &str) -> String {
    if path.starts_with('/') {
        String::from(path)
    } else {
        alloc::format!("{}/{path}", state(p).cwd.lock())
    }
}
fn at(p: &Process, fd: i32, ptr: u64) -> Result<String, Errno> {
    let path = read_cstr(p, ptr, 4096)?;
    if path.starts_with('/') {
        return Ok(path);
    }
    if fd == -100 {
        return Ok(absolute(p, &path));
    }
    let f = linux(p).files.get(fd)?;
    match &f.kind {
        Kind::Dir(dir) => Ok(alloc::format!("/{dir}/{path}")),
        _ => Err(ENOTDIR),
    }
}
pub fn open(p: &Process, path: &str, flags: u32, mode: u32) -> Result<Arc<File>, Errno> {
    let path = absolute(p, path);
    if let Some(result) = pty::open(p, &path, flags) {
        return result;
    }
    // A descriptor's file again (libgpud_drm hands RADV its buffer memfd as /proc/self/fd/N); it shares the
    // offset, unlike Linux, which opens the file anew.
    if let Some(fd) = path.strip_prefix("/proc/self/fd/").and_then(|s| s.parse::<i32>().ok()) {
        return linux(p).files.get(fd).map_err(|_| ENOENT);
    }
    if matches!(
        path.as_str(),
        "/dev/null" | "/dev/zero" | "/dev/random" | "/dev/urandom" | "/dev/kmsg" | "/dev/console"
    ) {
        return fs::open_path(&path, flags);
    }
    let found = vfs::lookup_as(p, &path, flags & 0x20000 == 0);
    let (resolved, node) = match found {
        Ok((path, node)) => {
            if flags & 0xc0 == 0xc0 {
                return Err(EEXIST);
            }
            if node.attr.lock().mode & 0o170000 == vfs::LINK {
                return Err(40);
            }
            (path, node)
        }
        Err(ENOENT) if flags & 0x40 != 0 => {
            let node = vfs::create(
                p,
                &path,
                vfs::REG | (mode & 0o777 & !state(p).mask.load(Ordering::Relaxed)),
                Vec::new(),
            )?;
            (vfs::normalize(&path), node)
        }
        Err(e) => return Err(e),
    };
    let directory = node.attr.lock().mode & 0o170000 == vfs::DIR;
    if flags & 0x10000 != 0 && !directory {
        return Err(ENOTDIR);
    }
    if directory {
        if flags & 3 != 0 {
            return Err(EISDIR);
        }
        vfs::check(p, &node, 5)?;
        return Ok(File::new(
            Kind::Dir(String::from(resolved.trim_start_matches('/'))),
            flags & !0x80000,
        ));
    }
    let need = match flags & 3 {
        0 => 4,
        1 => 2,
        2 => 6,
        _ => return Err(EINVAL),
    };
    vfs::check(p, &node, need)?;
    if flags & 0x200 != 0 {
        if need & 2 == 0 {
            return Err(EACCES);
        }
        vfs::resize(&node, 0)?;
    }
    Ok(File::new(Kind::Node(node), flags & !0x80000))
}
fn node_of(p: &Process, fd: i32) -> Result<Arc<vfs::Node>, Errno> {
    match &linux(p).files.get(fd)?.kind {
        Kind::Node(node) => Ok(node.clone()),
        Kind::Dir(path) => {
            vfs::lookup_as(p, &alloc::format!("/{path}"), true).map(|(_, node)| node)
        }
        _ => Err(EINVAL),
    }
}
fn stat(p: &Process, path: &str, follow: bool, ptr: u64, extended: bool) -> R {
    // A descriptor's file (RADV stats the render node libgpud_drm names as /proc/self/fd/N).
    if let Some(fd) = path.strip_prefix("/proc/self/fd/").and_then(|s| s.parse::<i32>().ok()) {
        let f = linux(p).files.get(fd).map_err(|_| ENOENT)?;
        return if extended { fs::statx(p, &f, ptr) } else { fs::stat(p, &f, ptr) };
    }
    let (_, node) = vfs::lookup_as(p, path, follow)?;
    let attr = *node.attr.lock();
    let len = node.len();
    let mut bytes = alloc::vec![0; if extended { 256 } else { 144 }];
    if extended {
        bytes[0..4].copy_from_slice(&0x7ffu32.to_le_bytes());
        bytes[4..8].copy_from_slice(&4096u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&1u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&attr.uid.to_le_bytes());
        bytes[24..28].copy_from_slice(&attr.gid.to_le_bytes());
        bytes[28..30].copy_from_slice(&(attr.mode as u16).to_le_bytes());
        bytes[32..40].copy_from_slice(&node.ino.to_le_bytes());
        bytes[40..48].copy_from_slice(&len.to_le_bytes());
    } else {
        bytes[8..16].copy_from_slice(&node.ino.to_le_bytes());
        bytes[16..24].copy_from_slice(&1u64.to_le_bytes());
        bytes[24..28].copy_from_slice(&attr.mode.to_le_bytes());
        bytes[28..32].copy_from_slice(&attr.uid.to_le_bytes());
        bytes[32..36].copy_from_slice(&attr.gid.to_le_bytes());
        bytes[48..56].copy_from_slice(&len.to_le_bytes());
        bytes[56..64].copy_from_slice(&4096u64.to_le_bytes());
    }
    for (at, time) in if extended {
        [(64, attr.atime), (112, attr.mtime), (96, attr.ctime)]
    } else {
        [(72, attr.atime), (88, attr.mtime), (104, attr.ctime)]
    } {
        bytes[at..at + 8].copy_from_slice(&(time / 1000000000).to_le_bytes());
        bytes[at + 8..at + 12].copy_from_slice(&((time % 1000000000) as u32).to_le_bytes());
    }
    write_user(p, ptr, &bytes).map(|_| 0)
}
fn chmod(p: &Process, node: &vfs::Node, mode: u32) -> R {
    let uid = state(p).credentials.lock().uid[1];
    let mut attr = node.attr.lock();
    if uid != 0 && uid != attr.uid {
        return Err(1);
    }
    attr.mode = attr.mode & 0o170000 | (mode & 0o7777);
    Ok(0)
}
fn chown(p: &Process, node: &vfs::Node, uid: u32, gid: u32) -> R {
    if state(p).credentials.lock().uid[1] != 0 {
        return Err(1);
    }
    let mut attr = node.attr.lock();
    if uid != u32::MAX {
        attr.uid = uid;
    }
    if gid != u32::MAX {
        attr.gid = gid;
    }
    attr.mode &= !0o6000;
    Ok(0)
}
fn dirents(p: &Process, fd: i32, ptr: u64, size: u64) -> R {
    let file = linux(p).files.get(fd)?;
    let Kind::Dir(path) = &file.kind else {
        return Err(ENOTDIR);
    };
    let mut entries = alloc::vec![
        (
            String::from("."),
            vfs::lookup_as(p, &alloc::format!("/{path}"), true)?.1
        ),
        (String::from(".."), vfs::lookup_as(p, "/", true)?.1)
    ];
    vfs::ensure_listed(p, &alloc::format!("/{path}"))?;
    entries.extend(vfs::entries(&alloc::format!("/{path}")));
    let mut offset = file.pos.lock();
    let mut bytes = Vec::new();
    while let Some((name, node)) = entries.get(*offset as usize) {
        let len = (19 + name.len() + 1).next_multiple_of(8);
        if bytes.len() + len > size as usize {
            break;
        }
        let mut entry = alloc::vec![0; len];
        entry[..8].copy_from_slice(&node.ino.to_le_bytes());
        entry[8..16].copy_from_slice(&(*offset + 1).to_le_bytes());
        entry[16..18].copy_from_slice(&(len as u16).to_le_bytes());
        entry[18] = (node.attr.lock().mode >> 12) as u8;
        entry[19..19 + name.len()].copy_from_slice(name.as_bytes());
        bytes.extend(entry);
        *offset += 1;
    }
    if bytes.is_empty() && *offset < entries.len() as u64 {
        return Err(EINVAL);
    }
    write_user(p, ptr, &bytes)?;
    Ok(bytes.len() as u64)
}
pub fn dispatch(p: &Arc<Process>, n: u64, a: [u64; 6]) -> Option<R> {
    if !matches!(n, 2 | 4 | 6 | 21 | 76..=95 | 257 | 258 | 260..=269 | 235 | 280 | 316 | 332 | 439 | 217 | 102 | 104..=108 | 113..=120 | 122 | 123)
    {
        return None;
    }
    Some(inner(p, n, a))
}
fn inner(p: &Arc<Process>, n: u64, a: [u64; 6]) -> R {
    let files = &linux(p).files;
    match n {
        2 | 85 | 257 => {
            let (path, flags, mode) = if n == 257 {
                (at(p, a[0] as i32, a[1])?, a[2] as u32, a[3] as u32)
            } else {
                (
                    absolute(p, &read_cstr(p, a[0], 4096)?),
                    if n == 85 { 0x241 } else { a[1] as u32 },
                    a[if n == 85 { 1 } else { 2 }] as u32,
                )
            };
            files.insert(open(p, &path, flags, mode)?, flags & 0x80000 != 0)
        }
        4 | 6 | 262 | 332 => {
            if (n == 262 || n == 332)
                && read_cstr(p, a[1], 4096)?.is_empty()
                && a[if n == 332 { 2 } else { 3 }] & 0x1000 != 0
            {
                return fs::syscall(p, n, a);
            }
            let (path, flags, out) = if n == 262 || n == 332 {
                (
                    at(p, a[0] as i32, a[1])?,
                    a[if n == 332 { 2 } else { 3 }],
                    a[if n == 332 { 4 } else { 2 }],
                )
            } else {
                (
                    absolute(p, &read_cstr(p, a[0], 4096)?),
                    if n == 6 { 0x100 } else { 0 },
                    a[1],
                )
            };
            match stat(p, &path, flags & 0x100 == 0, out, n == 332) {
                Err(ENOENT) if path.starts_with("/dev/") => fs::syscall(p, n, a),
                result => result,
            }
        }
        21 | 269 | 439 => {
            let path = if n == 21 {
                absolute(p, &read_cstr(p, a[0], 4096)?)
            } else {
                at(p, a[0] as i32, a[1])?
            };
            let mode = a[if n == 21 { 1 } else { 2 }] as u32;
            if mode & !7 != 0 {
                return Err(EINVAL);
            }
            match vfs::lookup_as(p, &path, true) {
                Ok((_, node)) => vfs::check(p, &node, mode).map(|_| 0),
                Err(ENOENT) if path.starts_with("/dev/") => open(p, &path, 0, 0).map(|_| 0),
                Err(e) => Err(e),
            }
        }
        79 => {
            let cwd = state(p).cwd.lock().clone();
            if a[1] < cwd.len() as u64 + 1 {
                return Err(34);
            }
            let mut bytes = cwd.into_bytes();
            bytes.push(0);
            write_user(p, a[0], &bytes)?;
            Ok(bytes.len() as u64)
        }
        80 | 81 => {
            let path = if n == 80 {
                absolute(p, &read_cstr(p, a[0], 4096)?)
            } else {
                let file = files.get(a[0] as i32)?;
                let Kind::Dir(dir) = &file.kind else {
                    return Err(ENOTDIR);
                };
                alloc::format!("/{dir}")
            };
            let (path, node) = vfs::lookup_as(p, &path, true)?;
            if node.attr.lock().mode & 0o170000 != vfs::DIR {
                return Err(ENOTDIR);
            }
            vfs::check(p, &node, 1)?;
            *state(p).cwd.lock() = path;
            Ok(0)
        }
        83 | 258 => {
            let path = if n == 83 {
                absolute(p, &read_cstr(p, a[0], 4096)?)
            } else {
                at(p, a[0] as i32, a[1])?
            };
            let mode = a[if n == 83 { 1 } else { 2 }] as u32;
            vfs::create(
                p,
                &path,
                vfs::DIR | (mode & 0o7777 & !state(p).mask.load(Ordering::Relaxed)),
                Vec::new(),
            )
            .map(|_| 0)
        }
        84 | 87 | 263 => {
            let path = if n == 263 {
                at(p, a[0] as i32, a[1])?
            } else {
                absolute(p, &read_cstr(p, a[0], 4096)?)
            };
            vfs::remove(p, &path, n == 84 || n == 263 && a[2] & 0x200 != 0)
        }
        82 | 264 | 316 => {
            if n == 316 && a[4] != 0 {
                return Err(EINVAL);
            }
            let (old, new) = if n == 82 {
                (
                    absolute(p, &read_cstr(p, a[0], 4096)?),
                    absolute(p, &read_cstr(p, a[1], 4096)?),
                )
            } else {
                (at(p, a[0] as i32, a[1])?, at(p, a[2] as i32, a[3])?)
            };
            vfs::rename(p, &old, &new)
        }
        88 | 266 => {
            let target = read_cstr(p, a[0], 4096)?;
            let path = if n == 88 {
                absolute(p, &read_cstr(p, a[1], 4096)?)
            } else {
                at(p, a[1] as i32, a[2])?
            };
            vfs::create(p, &path, vfs::LINK | 0o777, target.into_bytes()).map(|_| 0)
        }
        89 | 267 => {
            let path = if n == 89 {
                absolute(p, &read_cstr(p, a[0], 4096)?)
            } else {
                at(p, a[0] as i32, a[1])?
            };
            if let Some(fd) = path
                .strip_prefix("/proc/self/fd/")
                .and_then(|s| s.parse::<i32>().ok())
            {
                let file = files.get(fd)?;
                let Kind::Pty(end) = &file.kind else {
                    return Err(ENOENT);
                };
                let name = end.name();
                let count = name.len().min(a[if n == 89 { 2 } else { 3 }] as usize);
                write_user(p, a[if n == 89 { 1 } else { 2 }], &name.as_bytes()[..count])?;
                return Ok(count as u64);
            }
            let (_, node) = vfs::lookup_as(p, &path, false)?;
            if node.attr.lock().mode & 0o170000 != vfs::LINK {
                return Err(EINVAL);
            }
            let bytes = node.data.lock();
            let len = bytes.len().min(a[if n == 89 { 2 } else { 3 }] as usize);
            write_user(p, a[if n == 89 { 1 } else { 2 }], &bytes[..len])?;
            Ok(len as u64)
        }
        90 | 91 | 268 => {
            let node = if n == 91 {
                node_of(p, a[0] as i32)?
            } else {
                let path = if n == 268 {
                    at(p, a[0] as i32, a[1])?
                } else {
                    absolute(p, &read_cstr(p, a[0], 4096)?)
                };
                vfs::lookup_as(p, &path, true)?.1
            };
            chmod(p, &node, a[if n == 268 { 2 } else { 1 }] as u32)
        }
        92..=94 | 260 => {
            let node = if n == 93 {
                node_of(p, a[0] as i32)?
            } else {
                let path = if n == 260 {
                    at(p, a[0] as i32, a[1])?
                } else {
                    absolute(p, &read_cstr(p, a[0], 4096)?)
                };
                vfs::lookup_as(p, &path, n != 94)?.1
            };
            chown(
                p,
                &node,
                a[if n == 260 { 2 } else { 1 }] as u32,
                a[if n == 260 { 3 } else { 2 }] as u32,
            )
        }
        95 => Ok(state(p).mask.swap(a[0] as u32 & 0o777, Ordering::Relaxed) as u64),
        76 | 77 => {
            let node = if n == 77 {
                match node_of(p, a[0] as i32) {
                    Ok(n) => n,
                    Err(_) => return fs::syscall(p, n, a),
                }
            } else {
                vfs::lookup_as(p, &absolute(p, &read_cstr(p, a[0], 4096)?), true)?.1
            };
            vfs::check(p, &node, 2)?;
            vfs::resize(&node, usize::try_from(a[1]).map_err(|_| EINVAL)?).map(|_| 0)
        }
        235 | 280 => {
            let node = if n == 280 && a[1] == 0 {
                node_of(p, a[0] as i32)?
            } else {
                let path = if n == 235 {
                    absolute(p, &read_cstr(p, a[0], 4096)?)
                } else {
                    at(p, a[0] as i32, a[1])?
                };
                vfs::lookup_as(p, &path, n != 280 || a[3] & 0x100 == 0)?.1
            };
            let ptr = a[if n == 235 { 1 } else { 2 }];
            let now = clock_ns(0);
            let values = if ptr == 0 {
                [Some(now), Some(now)]
            } else {
                let raw = read_user(p, ptr, 32)?;
                let mut values = [None; 2];
                for i in 0..2 {
                    let sec = i64::from_le_bytes(raw[i * 16..i * 16 + 8].try_into().unwrap());
                    let frac = i64::from_le_bytes(raw[i * 16 + 8..i * 16 + 16].try_into().unwrap());
                    if n == 280 && frac == 1073741822 {
                        continue;
                    }
                    if n == 280 && frac == 1073741823 {
                        values[i] = Some(now);
                        continue;
                    }
                    let unit = if n == 235 { 1000 } else { 1 };
                    if sec < 0 || frac < 0 || frac >= 1000000000 / unit {
                        return Err(EINVAL);
                    }
                    values[i] = Some(
                        (sec as u64)
                            .saturating_mul(1000000000)
                            .saturating_add(frac as u64 * unit as u64),
                    );
                }
                values
            };
            vfs::check(p, &node, 2)?;
            let mut attr = node.attr.lock();
            if let Some(v) = values[0] {
                attr.atime = v;
            }
            if let Some(v) = values[1] {
                attr.mtime = v;
            }
            attr.ctime = now;
            Ok(0)
        }
        78 | 217 => dirents(p, a[0] as i32, a[1], a[2]),
        102 | 104 | 107 | 108 => {
            let c = state(p).credentials.lock();
            Ok(match n {
                102 => c.uid[0],
                107 => c.uid[1],
                104 => c.gid[0],
                _ => c.gid[1],
            } as u64)
        }
        105 | 106 | 113..=120 | 122 | 123 => credentials(p, n, a),
        _ => Err(ENOSYS),
    }
}
fn credentials(p: &Process, n: u64, a: [u64; 6]) -> R {
    let mut c = state(p).credentials.lock();
    let root = c.uid[1] == 0;
    if n == 115 {
        if a[0] == 0 {
            return Ok(c.groups.len() as u64);
        }
        if a[0] < c.groups.len() as u64 {
            return Err(EINVAL);
        }
        write_user(
            p,
            a[1],
            &c.groups
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        return Ok(c.groups.len() as u64);
    }
    if n == 116 {
        if !root {
            return Err(1);
        }
        if a[0] > 32 {
            return Err(EINVAL);
        }
        c.groups = read_user(p, a[1], a[0] * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        return Ok(0);
    }
    let ids = if matches!(n, 105 | 113 | 117 | 118 | 122) {
        &mut c.uid
    } else {
        &mut c.gid
    };
    if n == 118 || n == 120 {
        for i in 0..3 {
            write_u32(p, a[i], ids[i])?;
        }
        return Ok(0);
    }
    if n == 122 || n == 123 {
        return Err(ENOSYS);
    }
    let mut new = *ids;
    if n == 105 || n == 106 {
        if root {
            new.fill(a[0] as u32);
        } else if ids.contains(&(a[0] as u32)) {
            new[1] = a[0] as u32;
        } else {
            return Err(1);
        }
    } else {
        let count = if n == 113 || n == 114 { 2 } else { 3 };
        for i in 0..count {
            if a[i] as u32 != u32::MAX {
                if !root && !ids.contains(&(a[i] as u32)) {
                    return Err(1);
                }
                new[i] = a[i] as u32;
            }
        }
        if count == 2 && (a[0] as u32 != u32::MAX || a[1] as u32 != u32::MAX && new[1] != ids[0]) {
            new[2] = new[1];
        }
    }
    *ids = new;
    Ok(0)
}
