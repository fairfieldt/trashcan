extern crate alloc;
use std::sync::atomic::{AtomicUsize, Ordering};
static LIVE: AtomicUsize = AtomicUsize::new(0);
mod owned {
    pub struct OwnedHandle;
    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            super::LIVE.fetch_sub(1, super::Ordering::SeqCst);
        }
    }
    pub struct Message {
        bytes: Vec<u8>,
        handles: Vec<Option<OwnedHandle>>,
    }
    pub struct PushHandleError {
        pub handle: OwnedHandle,
    }
    impl Message {
        pub fn try_from_bytes(b: &[u8]) -> Result<Self, ()> {
            Ok(Self {
                bytes: b.to_vec(),
                handles: Vec::new(),
            })
        }
        pub fn bytes(&self) -> &[u8] {
            &self.bytes
        }
        pub fn handle_count(&self) -> usize {
            self.handles.len()
        }
        pub fn remaining_handles(&self) -> usize {
            self.handles.iter().flatten().count()
        }
        pub fn take_handle(&mut self, i: usize) -> Option<OwnedHandle> {
            self.handles.get_mut(i)?.take()
        }
        pub fn try_push_handle(&mut self, h: OwnedHandle) -> Result<(), PushHandleError> {
            self.handles.push(Some(h));
            Ok(())
        }
    }
}
#[path = "."]
mod protocol {
    #[path = "../../user/rt/src/protocol/bootstrap.rs"]
    pub mod bootstrap;
    #[path = "../../user/rt/src/protocol/frame.rs"]
    pub mod frame;
    #[path = "../../user/rt/src/protocol/schema.rs"]
    pub mod schema;
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ProtocolError {
        Frame(frame::FrameError),
        Malformed,
        Unsupported,
        Allocation,
        Limit,
        Missing,
        WrongProtocol,
    }
    impl From<frame::FrameError> for ProtocolError {
        fn from(e: frame::FrameError) -> Self {
            Self::Frame(e)
        }
    }
    pub(crate) fn valid_name(s: &str) -> bool {
        !s.is_empty() && s.len() <= schema::MAX_NAME_BYTES && !s.chars().any(char::is_control)
    }
}
fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    use protocol::{
        frame::{Frame, Header},
        schema::*,
    };
    if a[1] == "encode" {
        let body = unhex(&a[9]);
        let mut h = Header::new(
            a[2].parse().unwrap(),
            a[3].parse().unwrap(),
            a[4].parse().unwrap(),
            a[5].parse().unwrap(),
            a[6].parse().unwrap(),
            body.len(),
            a[8].parse().unwrap(),
        )
        .unwrap();
        h.flags = a[7].parse().unwrap();
        let mut out = vec![0; 32];
        h.encode(&mut out).unwrap();
        out.extend_from_slice(&body);
        println!("{}", hex(&out));
        return;
    }
    let bytes = unhex(&a[2]);
    let count: usize = a[3].parse().unwrap();
    if a[1] == "bootstrap" {
        let mut m = owned::Message::try_from_bytes(&bytes).unwrap();
        for _ in 0..count {
            LIVE.fetch_add(1, Ordering::SeqCst);
            m.try_push_handle(owned::OwnedHandle).ok().unwrap();
        }
        let result = protocol::bootstrap::BootstrapV2::decode(m);
        let ok = result.is_ok();
        drop(result);
        assert_eq!(LIVE.load(Ordering::SeqCst), 0, "bootstrap leaked handles");
        println!("{}", if ok { 0 } else { -1 });
        return;
    }
    match Frame::parse(&bytes, count) {
        Ok(f) => {
            let h = f.header;
            println!(
                "{} {} {} {} {} {} {} {} {}",
                if h.supported(PROTOCOL_ECHO, MAJOR_ECHO) {
                    0
                } else {
                    1
                },
                h.protocol,
                h.major,
                h.kind,
                h.ordinal,
                h.request_id,
                h.handle_count,
                h.flags,
                hex(f.body)
            );
        }
        Err(_) => println!("-1"),
    }
}
