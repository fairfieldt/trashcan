//! USB mass storage through bulk-only transport. A single block client receives
//! a read/write DMA window. INFO=0, READ=1, WRITE=2, FLUSH=3; read/write requests
//! carry a u64 LBA and u32 count. Replies start with a negative errno or zero.
//! FLUSH uses SYNCHRONIZE CACHE with IMMED clear, and must succeed before fsync
//! may report durability. See user/fsd/README.md for the full block contract.

use super::*;
use dross_abi::rights;

/// The DMA buffer reads land in, shared with the client.
pub const WINDOW: usize = 1 << 20;
const CBW_SIGNATURE: u32 = 0x4342_5355;
const CSW_SIGNATURE: u32 = 0x5342_5355;
const EIO: i64 = -5;
const EINVAL: i64 = -22;

pub struct Storage {
    slot: u8,
    out: Ring,
    out_dci: u8,
    inp: Ring,
    in_dci: u8,
    window: DmaBuffer,
    /// Command and status wrappers.
    small: DmaBuffer,
    tag: u32,
    pub block_size: u32,
    pub blocks: u64,
}

/// A bulk-only mass storage interface's endpoints, from the configuration descriptor: (OUT, IN), each as
/// (endpoint number, max packet size, max burst).
pub type Bulk = ((u8, u16, u8), (u8, u16, u8));

impl Xhci {
    /// Bulk endpoint contexts for the input context of a Configure Endpoint command; the rings to keep.
    pub(crate) fn storage_endpoints(&mut self, input: &DmaBuffer, bulk: Bulk) -> ((Ring, u8), (Ring, u8)) {
        let mut ring = |(ep, mps, burst): (u8, u16, u8), dir_in: bool| {
            let dci = ep * 2 + dir_in as u8;
            let ring = Ring::new(&mut self.space);
            let c = dci as usize + 1;
            // Error count 3, type bulk OUT (2) or IN (6), max burst, max packet size; average TRB length.
            let ty: u32 = if dir_in { 6 } else { 2 };
            self.ctx_w(input, c, 1, 3 << 1 | ty << 3 | (burst as u32) << 8 | (mps as u32) << 16);
            self.ctx_w(input, c, 2, ring.buf.iova as u32 | 1);
            self.ctx_w(input, c, 3, (ring.buf.iova >> 32) as u32);
            self.ctx_w(input, c, 4, 3072);
            (ring, dci)
        };
        let out = ring(bulk.0, false);
        let inp = ring(bulk.1, true);
        (out, inp)
    }

    /// Brings the disk up: waits until it is ready and reads its capacity.
    pub(crate) fn storage_start(&mut self, slot: u8, (out, out_dci): (Ring, u8), (inp, in_dci): (Ring, u8)) -> Option<Storage> {
        let window = self.space.alloc(WINDOW).ok()?;
        let small = self.space.alloc(4096).ok()?;
        let mut s = Storage { slot, out, out_dci, inp, in_dci, window, small, tag: 0, block_size: 0, blocks: 0 };
        // A disk can take a while to spin up or report ready.
        let end = sys::after_ms(5000);
        while self.scsi(&mut s, &[0x00, 0, 0, 0, 0, 0], 0).is_none() {
            if sys::clock() > end {
                println!("disk: not ready");
                return None;
            }
            let _ = self.scsi(&mut s, &[0x03, 0, 0, 0, 18, 0], 18);
            let _ = sys::notify_wait(sys::notify_create().ok()?, sys::after_ms(100));
        }
        self.scsi(&mut s, &[0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8)?;
        let w = s.window.slice();
        let (last, size) = (u32::from_be_bytes(w[0..4].try_into().unwrap()), u32::from_be_bytes(w[4..8].try_into().unwrap()));
        if last == u32::MAX {
            // READ CAPACITY (16) for disks of 2 TiB and more.
            self.scsi(&mut s, &[0x9e, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0], 32)?;
            let w = s.window.slice();
            s.blocks = u64::from_be_bytes(w[0..8].try_into().unwrap()) + 1;
            s.block_size = u32::from_be_bytes(w[8..12].try_into().unwrap());
        } else {
            s.blocks = last as u64 + 1;
            s.block_size = size;
        }
        if s.block_size == 0 || s.block_size as usize > WINDOW {
            println!("disk: block size {}", s.block_size);
            return None;
        }
        println!("disk: {} blocks of {} bytes ({} MiB)", s.blocks, s.block_size, s.blocks * s.block_size as u64 >> 20);
        Some(s)
    }

    /// One SCSI command with `len` bytes coming back into the window; None if it failed.
    fn scsi(&mut self, s: &mut Storage, cdb: &[u8], len: usize) -> Option<()> {
        self.scsi_transfer(s, cdb, len, true)
    }

    fn scsi_transfer(&mut self, s: &mut Storage, cdb: &[u8], len: usize, input: bool) -> Option<()> {
        s.tag = s.tag.wrapping_add(1);
        let mut cbw = [0u8; 31];
        cbw[0..4].copy_from_slice(&CBW_SIGNATURE.to_le_bytes());
        cbw[4..8].copy_from_slice(&s.tag.to_le_bytes());
        cbw[8..12].copy_from_slice(&(len as u32).to_le_bytes());
        cbw[12] = if len > 0 && input { 0x80 } else { 0 };
        cbw[14] = cdb.len() as u8;
        cbw[15..15 + cdb.len()].copy_from_slice(cdb);
        s.small.slice()[..31].copy_from_slice(&cbw);
        let small = s.small.iova;
        self.bulk(s.slot, &mut s.out, s.out_dci, small, 31)?;
        if len > 0 {
            let window = s.window.iova;
            if input { self.bulk(s.slot, &mut s.inp, s.in_dci, window, len)?; }
            else { self.bulk(s.slot, &mut s.out, s.out_dci, window, len)?; }
        }
        self.bulk(s.slot, &mut s.inp, s.in_dci, small + 512, 13)?;
        let csw = &s.small.slice()[512..525];
        let ok = u32::from_le_bytes(csw[0..4].try_into().unwrap()) == CSW_SIGNATURE
            && u32::from_le_bytes(csw[4..8].try_into().unwrap()) == s.tag
            && u32::from_le_bytes(csw[8..12].try_into().unwrap()) == 0
            && csw[12] == 0;
        ok.then_some(())
    }

    /// One transfer on a bulk endpoint, split into TRBs that don't cross 64 KiB; None on an error or timeout.
    fn bulk(&mut self, slot: u8, ring: &mut Ring, dci: u8, iova: u64, len: usize) -> Option<()> {
        let mut addrs = Vec::new();
        let (mut at, end) = (iova, iova + len as u64);
        while at < end {
            let n = ((at | 0xffff) + 1).min(end) - at;
            let last = at + n == end;
            // Interrupt on short packet; chain all but the last, which interrupts on completion.
            let flags = TRB_NORMAL << 10 | 1 << 2 | if last { 1 << 5 } else { 1 << 4 };
            addrs.push(ring.push([at as u32, (at >> 32) as u32, n as u32, flags]));
            at += n;
        }
        self.doorbell(slot, dci);
        let ev = self.try_wait_event(sys::after_ms(5000), |e| {
            trb_type(e) == EV_TRANSFER && addrs.contains(&(e[0] as u64 | (e[1] as u64) << 32))
        })?;
        matches!(code(&ev), 1 | 13).then_some(())
    }

    /// Serves the block protocol: one request from the client, if there is one.
    pub(crate) fn storage_serve(&mut self, ch: Handle, shared: &mut bool, view: Option<(u64,u64)>) -> Result<(), dross_rt::Error> {
        let mut req = [0u8; 16];
        let (n, _) = match sys::channel_read(ch, &mut req, &mut [], 0) {
            Ok(r) => r,
            Err(dross_rt::Error::TimedOut | dross_rt::Error::ShouldWait) => return Ok(()),
            Err(e) => return Err(e),
        };
        let Some(mut s) = self.storage.take() else {
            return sys::channel_write(ch, &(-19i64).to_le_bytes(), &[]);
        };
        let (base,blocks)=view.unwrap_or((0,s.blocks));
        if blocks==0 || base.checked_add(blocks).is_none_or(|end|end>s.blocks){
            self.storage=Some(s);return sys::channel_write(ch,&EINVAL.to_le_bytes(),&[]);
        }
        let r = match (n, req[0]) {
            (1, 0) => {
                let mut reply = alloc::vec::Vec::from(0i64.to_le_bytes());
                reply.extend_from_slice(&s.block_size.to_le_bytes());
                reply.extend_from_slice(&blocks.to_le_bytes());
                if *shared {sys::channel_write(ch,&reply,&[])}else{
                    match sys::dup(s.window.mem,rights::READ|rights::WRITE|rights::MAP|rights::TRANSFER){
                        Ok(window)=>{let result=sys::channel_write(ch,&reply,&[window]);if result.is_ok(){*shared=true}else{let _=sys::close(window);}result}
                        Err(_)=>sys::channel_write(ch,&(-12i64).to_le_bytes(),&[]),
                    }
                }
            }
            (13, op @ (1 | 2)) => {
                let lba = u64::from_le_bytes(req[1..9].try_into().unwrap());
                let count = u32::from_le_bytes(req[9..13].try_into().unwrap());
                let len = count as usize * s.block_size as usize;
                let status = if count == 0 || len > WINDOW || lba.checked_add(count as u64).is_none_or(|e| e > blocks) {
                    EINVAL
                } else {
                    // The client sees only the delegated range. Both the bounds
                    // check above and checked view geometry precede translation.
                    let lba=base+lba;
                    let cdb = if lba + count as u64 <= u32::MAX as u64 && count <= u16::MAX as u32 {
                        let (l, c) = ((lba as u32).to_be_bytes(), (count as u16).to_be_bytes());
                        alloc::vec![if op == 1 { 0x28 } else { 0x2a }, 0, l[0], l[1], l[2], l[3], 0, c[0], c[1], 0]
                    } else {
                        let (l, c) = (lba.to_be_bytes(), count.to_be_bytes());
                        let mut v = alloc::vec![if op == 1 { 0x88 } else { 0x8a }, 0];
                        v.extend_from_slice(&l);
                        v.extend_from_slice(&c);
                        v.extend_from_slice(&[0, 0]);
                        v
                    };
                    match self.scsi_transfer(&mut s, &cdb, len, op == 1) {
                        Some(()) => 0,
                        None => EIO,
                    }
                };
                sys::channel_write(ch, &status.to_le_bytes(), &[])
            }
            (1, 3) => {
                // SYNCHRONIZE CACHE (10), IMMED clear: success means the device
                // reports all preceding writes on stable storage, not just queued.
                let status = if self.scsi(&mut s, &[0x35, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0).is_some() { 0i64 } else { EIO };
                sys::channel_write(ch, &status.to_le_bytes(), &[])
            }
            _ => sys::channel_write(ch, &EINVAL.to_le_bytes(), &[]),
        };
        self.storage = Some(s);
        r
    }
}
