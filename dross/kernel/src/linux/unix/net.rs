//! Capability RPC to the userspace IP service. No network driver or TCP stack in ring 0.
use super::*;
use crate::{
    ipc::{ChannelEnd, Message},
    object::{Entry, Object},
};
use core::sync::atomic::AtomicBool;
static SERVICE: Mutex<Option<Arc<ChannelEnd>>> = Mutex::new(None);
pub fn register(channel: Arc<ChannelEnd>) -> Result<(), dross_abi::Error> {
    let mut service = SERVICE.lock();
    if service
        .as_ref()
        .is_some_and(|channel| channel.readiness() & 16 == 0)
    {
        return Err(dross_abi::Error::Busy);
    }
    *service = Some(channel);
    Ok(())
}
pub struct Socket {
    endpoint: Arc<ChannelEnd>,
    busy: WaitCell<bool>,
    dead: AtomicBool,
}
struct Guard<'a>(&'a WaitCell<bool>);
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        self.0.update(|v| *v = false);
    }
}
impl Socket {
    fn new() -> Result<Arc<Self>, Errno> {
        let service = SERVICE.lock().clone().ok_or(EAFNOSUPPORT)?;
        let (a, b) = ChannelEnd::pair();
        let endpoint = Arc::new(a);
        service
            .write(Message {
                bytes: b"ONET\x01\0\0\0".to_vec(),
                handles: alloc::vec![Entry::new(Object::Channel(Arc::new(b)))],
            })
            .map_err(|_| EAGAIN)?;
        let socket = Arc::new(Self {
            endpoint,
            busy: WaitCell::new(false),
            dead: AtomicBool::new(false),
        });
        socket.rpc(0, &[])?;
        Ok(socket)
    }
    fn rpc(&self, op: u32, bytes: &[u8]) -> Result<Message, Errno> {
        if self.dead.load(Ordering::Acquire) {
            return Err(5);
        }
        let deadline = time::tsc_at_ns(time::now_ns() + 2_000_000_000);
        self.busy
            .wait_uninterruptible(deadline, |busy| {
                if !*busy {
                    *busy = true;
                    Some(())
                } else {
                    None
                }
            })
            .map_err(|_| EINTR)?;
        let _guard = Guard(&self.busy);
        let mut request = op.to_le_bytes().to_vec();
        request.extend_from_slice(bytes);
        self.endpoint
            .write(Message {
                bytes: request,
                handles: Vec::new(),
            })
            .map_err(|_| EPIPE)?;
        let reply = match self
            .endpoint
            .read_transaction(dross_abi::MSG_BYTES, 1, deadline)
        {
            Ok(reply) => reply,
            Err(_) => {
                self.dead.store(true, Ordering::Release);
                return Err(ETIMEDOUT);
            }
        };
        if reply.bytes.len() < 8 {
            self.dead.store(true, Ordering::Release);
            return Err(5);
        }
        let status = i64::from_le_bytes(reply.bytes[..8].try_into().unwrap());
        if status < 0 { Err(-status) } else { Ok(reply) }
    }
    fn retry(&self, op: u32, bytes: &[u8], nonblocking: bool) -> Result<Message, Errno> {
        loop {
            match self.rpc(op, bytes) {
                Err(EAGAIN) if !nonblocking => {
                    if super::interrupted(&sched::current_process()) {
                        return Err(EINTR);
                    }
                    sched::sleep_ns(1_000_000).map_err(|_| EINTR)?;
                }
                result => return result,
            }
        }
    }
    pub fn read(&self, len: usize, nonblocking: bool, peek: bool) -> Result<Vec<u8>, Errno> {
        let mut request = (len.min(60 * 1024) as u32).to_le_bytes().to_vec();
        request.extend_from_slice(&u32::from(peek).to_le_bytes());
        Ok(self.retry(5, &request, nonblocking)?.bytes[8..].to_vec())
    }
    pub fn write(&self, bytes: &[u8], nonblocking: bool) -> Result<usize, Errno> {
        let reply = self.retry(6, &bytes[..bytes.len().min(60 * 1024)], nonblocking)?;
        Ok(u64::from_le_bytes(reply.bytes[..8].try_into().unwrap()) as usize)
    }
    pub fn poll(&self, events: u16) -> u16 {
        self.rpc(7, &[])
            .map(|r| u64::from_le_bytes(r.bytes[..8].try_into().unwrap()) as u16 & (events | 0x18))
            .unwrap_or(0x18)
    }
}
fn get(p: &Process, fd: u64) -> Result<Arc<Socket>, Errno> {
    match &linux(p).files.get(fd as i32)?.kind {
        super::super::fs::Kind::Net(socket) => Ok(socket.clone()),
        _ => Err(ENOTSOCK),
    }
}
fn address(p: &Process, ptr: u64, len: u64) -> Result<Vec<u8>, Errno> {
    if len < 16 {
        return Err(EINVAL);
    }
    let bytes = read_user(p, ptr, 16)?;
    if bytes[..2] != [2, 0] {
        return Err(EAFNOSUPPORT);
    }
    Ok(bytes)
}
fn address_out(p: &Process, ptr: u64, len: u64, bytes: &[u8]) -> R {
    if ptr != 0 {
        let capacity = read_u32(p, len)? as usize;
        write_user(p, ptr, &bytes[..capacity.min(bytes.len())])?;
        write_u32(p, len, bytes.len() as u32)?;
    }
    Ok(0)
}
pub fn dispatch(p: &Arc<Process>, n: u64, a: [u64; 6]) -> Option<R> {
    if n == 41 && a[0] == 2 {
        return Some((|| {
            if a[1] & 15 != 1 || !matches!(a[2], 0 | 6) {
                return Err(93);
            }
            linux(p).files.insert(
                super::super::fs::File::new(
                    super::super::fs::Kind::Net(Socket::new()?),
                    a[1] as u32 & 0x800,
                ),
                a[1] & 0x80000 != 0,
            )
        })());
    }
    if !matches!(n, 42..=52 | 54 | 55 | 288) || get(p, a[0]).is_err() {
        return None;
    }
    Some(inner(p, n, a))
}
fn inner(p: &Arc<Process>, n: u64, a: [u64; 6]) -> R {
    let socket = get(p, a[0])?;
    let file = linux(p).files.get(a[0] as i32)?;
    let nb = file.nonblocking();
    match n {
        49 => {
            let address = address(p, a[1], a[2])?;
            let port = u16::from_be_bytes([address[2], address[3]]);
            if port != 0 && port < 1024 && state(p).credentials.lock().uid[1] != 0 {
                return Err(EACCES);
            }
            socket.rpc(1, &address)?;
            Ok(0)
        }
        50 => {
            socket.rpc(2, &(a[1].min(16) as u32).to_le_bytes())?;
            Ok(0)
        }
        42 => {
            let request = address(p, a[1], a[2])?;
            match socket.retry(3, &request, nb) {
                Ok(_) => Ok(0),
                Err(EAGAIN) if nb => Err(115),
                Err(e) => Err(e),
            }
        }
        43 | 288 => {
            if n == 288 && a[3] & !(0x80000 | 0x800) != 0 {
                return Err(EINVAL);
            }
            let reply = socket.retry(4, &[], nb)?;
            address_out(p, a[1], a[2], &reply.bytes[8..])?;
            let endpoint = match reply.handles.into_iter().next().map(|e| e.object) {
                Some(Object::Channel(ch)) => ch,
                _ => return Err(5),
            };
            let accepted = Arc::new(Socket {
                endpoint,
                busy: WaitCell::new(false),
                dead: AtomicBool::new(false),
            });
            linux(p).files.insert(
                super::super::fs::File::new(
                    super::super::fs::Kind::Net(accepted),
                    if n == 288 { a[3] as u32 & 0x800 } else { 0 },
                ),
                n == 288 && a[3] & 0x80000 != 0,
            )
        }
        44 => {
            if a[4] != 0 {
                return Err(106);
            }
            let data = read_user(p, a[1], a[2].min(60 * 1024))?;
            socket
                .write(&data, nb || a[3] & 0x40 != 0)
                .map(|n| n as u64)
        }
        45 => {
            let data = socket.read(a[2] as usize, nb || a[3] & 0x40 != 0, a[3] & 2 != 0)?;
            write_user(p, a[1], &data)?;
            Ok(data.len() as u64)
        }
        46 => {
            let hdr = read_user(p, a[1], 56)?;
            let u = |at| u64::from_le_bytes(hdr[at..at + 8].try_into().unwrap());
            let data = gather(p, &read_iov(p, u(16), u(24))?)?;
            socket
                .write(&data, nb || a[2] & 0x40 != 0)
                .map(|n| n as u64)
        }
        47 => {
            let hdr = read_user(p, a[1], 56)?;
            let u = |at| u64::from_le_bytes(hdr[at..at + 8].try_into().unwrap());
            let iov = read_iov(p, u(16), u(24))?;
            let capacity = iov.iter().try_fold(0u64, |sum, &(_, len)| sum.checked_add(len)).ok_or(EINVAL)?;
            if capacity > i64::MAX as u64 { return Err(EINVAL); }
            let capacity = capacity.min(60 * 1024);
            let data = socket.read(capacity as usize, nb || a[2] & 0x40 != 0, a[2] & 2 != 0)?;
            scatter(p, &iov, &data).map(|n| n as u64)
        }
        48 => {
            if a[1] > 2 {
                return Err(EINVAL);
            }
            socket.rpc(8, &(a[1] as u32).to_le_bytes())?;
            Ok(0)
        }
        51 | 52 => {
            let reply = socket.rpc(if n == 51 { 9 } else { 10 }, &[])?;
            address_out(p, a[1], a[2], &reply.bytes[8..])
        }
        54 => {
            if a[1] == 1 && matches!(a[2], 2 | 9) || a[1] == 6 && a[2] == 1 {
                read_user(p, a[3], 4)?;
                Ok(0)
            } else {
                Err(92)
            }
        }
        55 => {
            let value: u32 = if a[1] == 1 && a[2] == 4 {
                let reply = socket.rpc(11, &[])?;
                u64::from_le_bytes(reply.bytes[..8].try_into().unwrap()) as u32
            } else if a[1] == 1 && a[2] == 3 {
                1
            } else {
                return Err(92);
            };
            if read_u32(p, a[4])? < 4 {
                return Err(EINVAL);
            }
            write_u32(p, a[3], value)?;
            write_u32(p, a[4], 4)?;
            Ok(0)
        }
        _ => Err(ENOSYS),
    }
}
