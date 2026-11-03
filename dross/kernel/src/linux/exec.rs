//! Starting a static Linux executable: its segments (static or static-PIE, which relocates itself), an 8 MiB
//! stack with argv, envp and the auxiliary vector, and the first thread.

use alloc::sync::Arc;
use alloc::vec::Vec;
use dross_abi::{vm, Error, USER_TOP};

use crate::memory::Memory;
use crate::object::Entry;
use crate::process::Process;
use crate::sched;

const STACK_SIZE: u64 = 8 << 20;
const STACK_TOP: u64 = 0x0000_7fff_ffff_0000;
/// Where position-independent executables go.
const PIE_BASE: u64 = 0x0000_5555_0000_0000;

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes(b[o..o + 2].try_into().unwrap())
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

/// `argv` and `envp` are NUL-separated strings. `bootstrap` becomes native handle 1.
pub fn spawn(image: &Memory, argv: &[u8], envp: &[u8], bootstrap: Option<Entry>) -> Result<Arc<Process>, Error> {
    let (p, entry, sp) = load(image, argv, envp, [0; 4])?;
    if let Some(e) = bootstrap { let h = p.handles.lock().insert(e); debug_assert_eq!(h, 1); }
    sched::spawn_user(p.clone(), sched::pick_cpu(), entry, sp, 0);
    Ok(p)
}

pub(super) fn load(image: &Memory, argv: &[u8], envp: &[u8], ids: [u64; 4]) -> Result<(Arc<Process>, u64, u64), Error> {
    let packed = |bytes: &[u8]| -> Vec<Vec<u8>> {
        if bytes.is_empty() { Vec::new() } else { bytes.strip_suffix(&[0]).unwrap_or(bytes).split(|&b| b == 0).map(|s| s.to_vec()).collect() }
    };
    let args = packed(argv); let envs = packed(envp);
    let name = args.first().and_then(|a| core::str::from_utf8(a).ok()).map(|a| a.rsplit('/').next().unwrap_or(a)).unwrap_or("linux");

    let mut hdr = [0u8; 64];
    image.read(0, &mut hdr).map_err(|_| Error::BadElf)?;
    let kind = u16_at(&hdr, 16);
    if &hdr[..4] != b"\x7fELF" || hdr[4] != 2 || hdr[5] != 1 || !(kind == 2 || kind == 3) || u16_at(&hdr, 18) != 0x3e {
        return Err(Error::BadElf);
    }
    let base = if kind == 3 { PIE_BASE } else { 0 };
    let (phoff, phentsize, phnum) = (u64_at(&hdr, 32), u16_at(&hdr, 54) as u64, u16_at(&hdr, 56) as u64);
    if phentsize < 56 || phnum > 64 {
        return Err(Error::BadElf);
    }
    let p = Process::new_linux(name);
    let mut phdr_addr = 0;
    let mut first_load = None;
    for i in 0..phnum {
        let mut ph = [0u8; 56];
        image.read(phoff + i * phentsize, &mut ph).map_err(|_| Error::BadElf)?;
        let ty = u32_at(&ph, 0);
        let (flags, offset, vaddr, filesz, memsz) = (u32_at(&ph, 4), u64_at(&ph, 8), u64_at(&ph, 16), u64_at(&ph, 32), u64_at(&ph, 40));
        if ty == 3 { return Err(Error::BadElf); }
        if ty == 6 {
            phdr_addr = base + vaddr;
        }
        if ty != 1 || memsz == 0 {
            continue;
        }
        if first_load.is_none() {
            first_load = Some((vaddr, offset));
        }
        let va = base.checked_add(vaddr).ok_or(Error::BadElf)?;
        let start = va & !0xfff;
        let end = va.checked_add(memsz).and_then(|v| v.checked_add(4095)).ok_or(Error::BadElf)? & !4095;
        if filesz > memsz || offset.checked_add(filesz).is_none_or(|end| end > image.size()) || end > USER_TOP || start == 0 {
            return Err(Error::BadElf);
        }
        let seg = Memory::anonymous(end - start, false, false)?;
        let mut buf = alloc::vec![0u8; filesz as usize];
        image.read(offset, &mut buf).map_err(|_| Error::BadElf)?;
        seg.write(va - start, &buf)?;
        let mut f = vm::READ;
        if flags & 2 != 0 {
            f |= vm::WRITE;
        }
        if flags & 1 != 0 {
            f |= vm::EXECUTE;
        }
        p.aspace.map(Arc::new(seg), 0, end - start, start, f)?;
    }
    if phdr_addr == 0 {
        let (v, off) = first_load.ok_or(Error::BadElf)?;
        phdr_addr = base + v - off + phoff;
    }
    let entry = base.checked_add(u64_at(&hdr, 24)).filter(|entry| *entry < USER_TOP && *entry >= 4096).ok_or(Error::BadElf)?;

    // The stack, built in a kernel buffer and copied into its top page(s).
    let stack = Arc::new(Memory::anonymous(STACK_SIZE, false, false)?);
    p.aspace.map(stack.clone(), 0, STACK_SIZE, STACK_TOP - STACK_SIZE, vm::READ | vm::WRITE)?;
    let mut strings: Vec<u8> = Vec::new();
    fn place(strings: &mut Vec<u8>, s: &[u8]) -> u64 {
        let at = strings.len() as u64;
        strings.extend_from_slice(s);
        strings.push(0);
        at
    }
    let arg_off: Vec<u64> = args.iter().map(|a| place(&mut strings, a)).collect();
    let env_off: Vec<u64> = envs.iter().map(|e| place(&mut strings, e)).collect();
    let rand_off = strings.len() as u64;
    let mut random = [0u8; 16];
    super::random_bytes(&mut random);
    strings.extend_from_slice(&random);
    let platform_off = place(&mut strings, b"x86_64");
    let strings_at = (STACK_TOP - strings.len() as u64 - 64) & !15;
    let (_, _, _, edx) = crate::cpu::cpuid(1, 0);
    let aux: [(u64, u64); 13] = [
        (3, phdr_addr),
        (4, phentsize),
        (5, phnum),
        (6, 4096),
        (7, 0),
        (9, entry),
        (11, ids[0]),
        (12, ids[1]),
        (13, ids[2]),
        (14, ids[3]),
        (16, edx as u64),
        (25, strings_at + rand_off),
        (15, strings_at + platform_off),
    ];
    let mut words: Vec<u64> = Vec::new();
    words.push(args.len() as u64);
    words.extend(arg_off.iter().map(|o| strings_at + o));
    words.push(0);
    words.extend(env_off.iter().map(|o| strings_at + o));
    words.push(0);
    for (k, v) in aux {
        words.push(k);
        words.push(v);
    }
    words.push(0);
    words.push(0);
    let sp = (strings_at - words.len() as u64 * 8) & !15;
    let total = STACK_TOP - sp;
    let mut img = alloc::vec![0u8; total as usize];
    for (i, w) in words.iter().enumerate() {
        img[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
    }
    let so = (strings_at - sp) as usize;
    img[so..so + strings.len()].copy_from_slice(&strings);
    stack.write(STACK_SIZE - total, &img)?;

    Ok((p, entry, sp))
}
