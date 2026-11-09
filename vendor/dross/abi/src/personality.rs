//! Private per-call channels are the authority to complete or copy memory for that call.
//! All words are little endian. No application-supplied handle enters a request.
pub const VERSION: u64 = 1;
pub const REQUEST_BYTES: usize = 80;
/// The stopped caller's registers and TLS base, which follow a forwarded call's request.
pub const CONTEXT_BYTES: usize = crate::task::CONTEXT_BYTES;
pub const CALL_BYTES: usize = REQUEST_BYTES + CONTEXT_BYTES;
pub const HEADER_BYTES: usize = 32;
pub const MAX_COPY: usize = 60 * 1024;
pub const COMPLETE: u64 = 0;
pub const LEGACY: u64 = 1;
pub const READ: u64 = 2;
pub const WRITE: u64 = 3;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub number: u64,
    pub pid: u64,
    pub tid: u64,
    pub args: [u64; 6],
}
impl Request {
    pub fn encode(self) -> [u8; REQUEST_BYTES] {
        let mut out = [0; REQUEST_BYTES];
        for (slot, word) in out.chunks_exact_mut(8).zip(
            [VERSION, self.number, self.pid, self.tid]
                .into_iter()
                .chain(self.args),
        ) {
            slot.copy_from_slice(&word.to_le_bytes());
        }
        out
    }
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != REQUEST_BYTES {
            return None;
        }
        let mut words = [0; 10];
        for (word, slot) in words.iter_mut().zip(bytes.chunks_exact(8)) {
            *word = u64::from_le_bytes(slot.try_into().ok()?);
        }
        if words[0] != VERSION {
            return None;
        }
        Some(Self {
            number: words[1],
            pid: words[2],
            tid: words[3],
            args: words[4..].try_into().ok()?,
        })
    }
}
/// The context words of a forwarded call, after its request.
pub fn decode_context(bytes: &[u8]) -> Option<[u64; crate::task::CONTEXT_WORDS]> {
    if bytes.len() != CALL_BYTES {
        return None;
    }
    let mut words = [0; crate::task::CONTEXT_WORDS];
    for (word, slot) in words.iter_mut().zip(bytes[REQUEST_BYTES..].chunks_exact(8)) {
        *word = u64::from_le_bytes(slot.try_into().ok()?);
    }
    Some(words)
}
pub fn header(op: u64, value: u64, address: u64, length: u64) -> [u8; HEADER_BYTES] {
    let mut out = [0; HEADER_BYTES];
    for (slot, word) in out.chunks_exact_mut(8).zip([op, value, address, length]) {
        slot.copy_from_slice(&word.to_le_bytes());
    }
    out
}
pub fn decode_header(bytes: &[u8]) -> Option<[u64; 4]> {
    let mut words = [0; 4];
    for (word, slot) in words
        .iter_mut()
        .zip(bytes.get(..HEADER_BYTES)?.chunks_exact(8))
    {
        *word = u64::from_le_bytes(slot.try_into().ok()?);
    }
    Some(words)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_roundtrip_and_framing() {
        let r = Request {
            number: 63,
            pid: 7,
            tid: 9,
            args: [u64::MAX, 2, 3, 4, 5, 6],
        };
        let b = r.encode();
        assert_eq!(Request::decode(&b), Some(r));
        for n in 0..REQUEST_BYTES {
            assert_eq!(Request::decode(&b[..n]), None);
        }
        let mut oversized = [0; REQUEST_BYTES + 1];
        oversized[..REQUEST_BYTES].copy_from_slice(&b);
        assert_eq!(Request::decode(&oversized), None);
        let mut b = b;
        b[0] = 2;
        assert_eq!(Request::decode(&b), None);
        assert_eq!(
            decode_header(&header(WRITE, 0, 4096, 3)),
            Some([WRITE, 0, 4096, 3])
        );
    }
    #[test]
    fn context_follows_the_request() {
        let mut call = [0; CALL_BYTES];
        for (index, slot) in call[REQUEST_BYTES..].chunks_exact_mut(8).enumerate() {
            slot.copy_from_slice(&(index as u64).to_le_bytes());
        }
        let words = decode_context(&call).unwrap();
        assert_eq!((words[0], words[crate::task::REGISTER_WORDS]), (0, crate::task::REGISTER_WORDS as u64));
        assert_eq!(decode_context(&call[..CALL_BYTES - 1]), None);
        assert_eq!(decode_context(&[0; REQUEST_BYTES]), None);
    }
}

/// Supervisor launch request. Args[0..2] are packed argv/envp lengths.
pub const SPAWN: u64 = u64::MAX;

/// Capture the stopped caller's execution context, an opaque FPU snapshot and its capabilities.
pub const CAPTURE: u64 = 4;
/// Resume with task::Context payload and optional FPU snapshot handle. Terminal operation.
pub const RESUME: u64 = 5;
/// Replace the stopped caller's address space from a prepared process, then resume.
/// Handles: prepared process, optional FPU snapshot. Payload: task::Context.
/// Header value 1 requests an 8-byte zero acknowledgment after the image commit.
pub const REPLACE: u64 = 6;
/// Generic user-return event after THREAD_INTERRUPT. Same per-call transport as syscalls.
pub const UPCALL: u64 = u64::MAX - 1;

pub const ACCEPT: u64 = 7;

pub const EXIT: u64 = 8;

pub const FAULT: u64 = u64::MAX - 2;

pub const NET_REGISTER: u64 = u64::MAX - 3;

pub const MOUNT: u64 = u64::MAX - 4;

pub const IMAGE: u64 = u64::MAX - 5;

pub const HANDLE_FD: u64 = u64::MAX - 6;

pub const PUBLISH: u64 = u64::MAX - 7;

/// The machine is about to reset: every disk finishes. The reply comes when they have, or within 20 s.
pub const UNMOUNT: u64 = u64::MAX - 8;

/// PERSONALITY_NEXT's result: the message came from the kernel's forwarding path,
/// so its pid, tid and context are the kernel's own. Nothing else can set this.
pub const NEXT_FORWARDED: u64 = 1 << 63;
