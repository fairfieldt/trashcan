//! gpud's connections (driver/shim/include/gpud_proto.h). A gpud channel serves one client; whoever holds one can
//! have gpud make another, for a program it starts, which gets it as `gpu` in its bootstrap message.
use crate::bootstrap::{Bootstrap, Builder};
use crate::deadline::{Deadline, Duration};
use crate::sys::{self, Handle};
use crate::Error;
use alloc::vec::Vec;

/// gpud's CONNECT operation.
const CONNECT: u32 = 11;

/// A new connection to the gpud that `channel` reaches. gpud answers once it has brought the GPU up; a reply not
/// within ten seconds fails. Not on a channel that this process's own libgpud_drm.so uses.
pub fn connect(channel: Handle) -> Result<Handle, Error> {
    let mut request = [0u8; 8];
    request[..4].copy_from_slice(&CONNECT.to_le_bytes());
    sys::channel_write(channel, &request, &[])?;
    let mut reply = [0u8; 8];
    let mut handles = [Handle(0); 1];
    let deadline = Deadline::from_now(Duration::from_secs(10));
    let (n, nh) = sys::channel_read(channel, &mut reply, &mut handles, deadline.as_monotonic_ns())?;
    let status = i32::from_le_bytes(reply[..4].try_into().unwrap());
    match (n, nh, status) {
        (8, 1, 0) => Ok(handles[0]),
        (_, 1, _) => {
            let _ = sys::close(handles[0]);
            Err(Error::InvalidArgs)
        }
        _ => Err(Error::InvalidArgs),
    }
}

/// `child` with a connection of its own, as `gpu`, to each gpud that `parent` names `gpu`, in the same order.
pub fn connections(parent: &Bootstrap, mut child: Builder) -> Result<Builder, Error> {
    let fresh = connection_handles(parent, connect, |handle| { let _ = sys::close(handle); })?;
    for handle in fresh {
        child = child.handle("gpu", handle);
    }
    Ok(child)
}

fn connection_handles(
    parent: &Bootstrap,
    mut connect: impl FnMut(Handle) -> Result<Handle, Error>,
    mut close: impl FnMut(Handle),
) -> Result<Vec<Handle>, Error> {
    let mut fresh = Vec::new();
    for (name, handle) in &parent.handles {
        if name == "gpu" {
            match connect(*handle) {
                Ok(handle) => fresh.push(handle),
                Err(error) => {
                    for handle in fresh { close(handle); }
                    return Err(error);
                }
            }
        }
    }
    Ok(fresh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::String, vec};
    use core::cell::RefCell;

    fn parent() -> Bootstrap {
        Bootstrap {
            handles: vec![(String::from("gpu"), Handle(7)), (String::from("window"), Handle(9)),
                          (String::from("gpu"), Handle(8))],
            values: Vec::new(),
        }
    }

    #[test]
    fn partial_failure_closes_only_new_connections() {
        let closed = RefCell::new(Vec::new());
        let result = connection_handles(&parent(), |handle| {
            if handle == Handle(7) { Ok(Handle(100)) } else { Err(Error::PeerClosed) }
        }, |handle| closed.borrow_mut().push(handle));
        assert_eq!(result, Err(Error::PeerClosed));
        assert_eq!(*closed.borrow(), vec![Handle(100)]);
    }

    #[test]
    fn success_returns_distinct_connections_in_parent_order() {
        let called = RefCell::new(Vec::new());
        let result = connection_handles(&parent(), |handle| {
            called.borrow_mut().push(handle);
            Ok(Handle(handle.0 + 100))
        }, |_| panic!("successful connections must remain owned by the child"));
        assert_eq!(result, Ok(vec![Handle(107), Handle(108)]));
        assert_eq!(*called.borrow(), vec![Handle(7), Handle(8)]);
    }
}
