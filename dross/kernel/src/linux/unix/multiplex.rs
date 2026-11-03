//! Poll/select compatibility across ordinary fds and userspace network capabilities.
//! The current transport has no readiness subscription, so waits use a bounded timer.
use super::*;
pub fn dispatch(p: &Arc<Process>, n: u64, a: [u64; 6]) -> Option<R> {
    if !matches!(n, 7 | 23 | 270 | 271) {
        return None;
    }
    Some(inner(p, n, a))
}
fn inner(p: &Arc<Process>, n: u64, a: [u64; 6]) -> R {
    let select = n == 23 || n == 270;
    let nfds = if select { a[0] } else { a[1] };
    if nfds > 1024 {
        return Err(EINVAL);
    }
    if n == 270 && a[5] != 0 {
        let mask = read_user(p, a[5], 16)?;
        if u64::from_le_bytes(mask[..8].try_into().unwrap()) != 0 {
            return Err(ENOSYS);
        }
    }
    let size = nfds.div_ceil(64) as usize * 8;
    let mut wanted = Vec::new();
    let mut sets = Vec::new();
    if select {
        for ptr in &a[1..4] {
            sets.push(if *ptr == 0 {
                alloc::vec![0; size]
            } else {
                read_user(p, *ptr, size as u64)?
            });
        }
        for fd in 0..nfds {
            let bit = 1 << (fd % 8);
            let index = fd as usize / 8;
            let events = if sets[0][index] & bit != 0 { 1 } else { 0 }
                | if sets[1][index] & bit != 0 { 4 } else { 0 }
                | if sets[2][index] & bit != 0 { 2 } else { 0 };
            if events != 0 {
                linux(p).files.get(fd as i32)?;
                wanted.push((fd as i32, events));
            }
        }
    } else {
        for bytes in read_user(p, a[0], nfds * 8)?.as_chunks::<8>().0 {
            wanted.push((
                i32::from_le_bytes(bytes[..4].try_into().unwrap()),
                u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            ));
        }
    }
    let duration = if n == 7 {
        if a[2] as i64 >= 0 {
            a[2].saturating_mul(1_000_000)
        } else {
            u64::MAX
        }
    } else if select && a[4] != 0 {
        let bytes = read_user(p, a[4], 16)?;
        let sec = i64::from_le_bytes(bytes[..8].try_into().unwrap());
        let frac = i64::from_le_bytes(bytes[8..].try_into().unwrap());
        let unit = if n == 23 { 1000 } else { 1 };
        if sec < 0 || frac < 0 || frac >= 1_000_000_000 / unit {
            return Err(EINVAL);
        }
        (sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(frac as u64 * unit as u64)
    } else if n == 271 && a[2] != 0 {
        read_timespec(p, a[2])?
    } else {
        u64::MAX
    };
    let deadline = time::now_ns().saturating_add(duration);
    let me = sched::current();
    loop {
        let mut ready = Vec::with_capacity(wanted.len());
        let mut count = 0;
        for &(fd, events) in &wanted {
            let flags = if fd < 0 {
                0
            } else {
                match linux(p).files.get(fd) {
                    Ok(file) => {
                        let flags = file.poll(&me, events);
                        file.unpoll(&me);
                        flags
                    }
                    Err(_) => 32,
                }
            };
            if flags != 0 {
                count += 1;
            }
            ready.push(flags);
        }
        if count != 0 || time::now_ns() >= deadline {
            if select {
                let mut output = [
                    alloc::vec![0u8; size],
                    alloc::vec![0u8; size],
                    alloc::vec![0u8; size],
                ];
                for (i, &(fd, events)) in wanted.iter().enumerate() {
                    let flags = ready[i];
                    let index = fd as usize / 8;
                    let bit = 1 << (fd as usize % 8);
                    if events & 1 != 0 && flags & (1 | 8 | 16) != 0 {
                        output[0][index] |= bit;
                    }
                    if events & 4 != 0 && flags & (4 | 8) != 0 {
                        output[1][index] |= bit;
                    }
                    if events & 2 != 0 && flags & 2 != 0 {
                        output[2][index] |= bit;
                    }
                }
                for i in 0..3 {
                    if a[i + 1] != 0 {
                        write_user(p, a[i + 1], &output[i])?;
                    }
                }
            } else {
                for (i, flags) in ready.iter().enumerate() {
                    write_user(p, a[0] + i as u64 * 8 + 6, &flags.to_le_bytes())?;
                }
            }
            return Ok(count);
        }
        if super::interrupted(p) {
            return Err(EINTR);
        }
        sched::sleep_ns((deadline.saturating_sub(time::now_ns())).min(1_000_000))
            .map_err(|_| EINTR)?;
    }
}
