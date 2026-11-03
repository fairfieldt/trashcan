//! The client protocol, mirroring driver/shim/include/gpud_proto.h (same layouts, checked below).

pub const VERSION: u32 = 1;
pub const MAX_MSG: usize = 65536;

pub const HELLO: u32 = 1;
pub const INFO: u32 = 2;
pub const BO_CREATE: u32 = 3;
pub const BO_FREE: u32 = 4;
pub const BO_QUERY: u32 = 5;
pub const VA: u32 = 6;
pub const SUBMIT: u32 = 7;
pub const WAIT: u32 = 8;
pub const RING: u32 = 9;
pub const BO_IMPORT: u32 = 10;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BoImportReq {
    pub offset: u64,
    pub size: u64,
}

pub const NO_MMAP: u64 = u64::MAX;

pub fn fence(ring: i32, seq: u64) -> u64 {
    ((ring as u64) << 56) | (seq & 0x00ff_ffff_ffff_ffff)
}

pub fn fence_ring(f: u64) -> i32 {
    (f >> 56) as i32
}

pub fn fence_seq(f: u64) -> u64 {
    f & 0x00ff_ffff_ffff_ffff
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Req {
    pub op: u32,
    pub len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Rep {
    pub status: i32,
    pub len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HelloRep {
    pub version: u32,
    pub vmid: u32,
    pub bo_file_size: u64,
    pub pci_domain: u16,
    pub pci_bus: u8,
    pub pci_dev: u8,
    pub pci_func: u8,
    pub revision: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub subvendor_id: u16,
    pub subdevice_id: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct InfoReq {
    pub query: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BoCreateReq {
    pub size: u64,
    pub alignment: u64,
    pub domains: u64,
    pub flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BoHandle {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BoRep {
    pub handle: u32,
    pub domain: u32,
    pub size: u64,
    pub mmap_offset: u64,
    pub alignment: u64,
    pub domains: u64,
    pub flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VaReq {
    pub handle: u32,
    pub operation: u32,
    pub flags: u32,
    pub pad: u32,
    pub va: u64,
    pub offset: u64,
    pub size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IbDesc {
    pub va: u64,
    pub size_dw: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SubmitReq {
    pub ip_type: u32,
    pub ring: u32,
    pub num_ibs: u32,
    pub num_deps: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FenceRep {
    pub fence: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WaitReq {
    pub fence: u64,
    pub timeout_ns: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RingReq {
    pub ip_type: u32,
    pub ring: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RingRep {
    pub ring: i32,
    pub pad: u32,
    pub last_fence: u64,
}

const _: () = {
    assert!(size_of::<Req>() == 8 && size_of::<Rep>() == 8);
    assert!(size_of::<HelloRep>() == 32);
    assert!(size_of::<InfoReq>() == 32);
    assert!(size_of::<BoCreateReq>() == 32 && size_of::<BoHandle>() == 8 && size_of::<BoRep>() == 48);
    assert!(size_of::<VaReq>() == 40 && size_of::<IbDesc>() == 16 && size_of::<SubmitReq>() == 16);
    assert!(size_of::<FenceRep>() == 8 && size_of::<WaitReq>() == 16);
    assert!(size_of::<RingReq>() == 8 && size_of::<RingRep>() == 16);
};

/// A plain-old-data value read from (or written into) a message.
pub fn read<T: Copy>(buf: &[u8], off: usize) -> Option<T> {
    if off + size_of::<T>() > buf.len() {
        return None;
    }
    // SAFETY: in bounds; T is a repr(C) Copy type with no invalid bit patterns (integers only).
    Some(unsafe { std::ptr::read_unaligned(buf.as_ptr().add(off).cast::<T>()) })
}

pub fn bytes<T: Copy>(v: &T) -> &[u8] {
    // SAFETY: T is plain old data.
    unsafe { std::slice::from_raw_parts((v as *const T).cast::<u8>(), size_of::<T>()) }
}
