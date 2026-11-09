//! Whole datagrams over a family-isolated UDP capability. Empty packets are
//! data, never EOF. A send result is local acceptance, not remote delivery.
use super::{
    ProtocolError,
    frame::{Frame, Header, get32},
    network::{self, Address, Endpoint, Info, Network, NetworkError, Observation, Rpc},
    schema::*,
};
use crate::{
    deadline::Deadline,
    owned::{Message, OwnedHandle},
};
pub mod rights {
    pub const SEND: u32 = 1;
    pub const RECEIVE: u32 = 2;
    pub const OBSERVE: u32 = 4;
    pub const INFO: u32 = 8;
    pub const ALL: u32 = 15;
}
pub const MAX_INLINE: usize = MAX_NETWORK_BYTES;
pub const RECEIVE_HEADER_BYTES: usize = 72;
pub const TRUNCATED: u32 = 1;
#[derive(Debug, PartialEq, Eq)]
pub enum Operation<'a> {
    Open {
        local: Endpoint,
        rights: u32,
    },
    SendTo {
        destination: Endpoint,
        bytes: &'a [u8],
    },
    Receive {
        maximum: u32,
    },
    Info,
    Observe {
        mask: u32,
        key: u64,
    },
    Rearm {
        subscription: u64,
        revision: u64,
    },
    Cancel {
        subscription: u64,
    },
}
#[derive(Debug)]
pub struct WireRequest<'a> {
    pub header: Header,
    pub deadline: Deadline,
    pub operation: Operation<'a>,
}
pub fn decode_request(bytes: &[u8], handles: usize) -> Result<WireRequest<'_>, ProtocolError> {
    let frame = Frame::parse(bytes, handles)?;
    let h = frame.header;
    if h.kind != KIND_REQUEST || !h.supported(h.protocol, 1) {
        return Err(ProtocolError::Unsupported);
    }
    let b = frame.body;
    if b.len() < 8 {
        return Err(ProtocolError::Malformed);
    }
    let deadline = Deadline::at_monotonic_ns(super::frame::get64(b, 0));
    let operation = match (h.protocol, h.ordinal) {
        (PROTOCOL_NETWORK, NETWORK_OPEN_DATAGRAM) => {
            if b.len() != 40 || handles != 0 {
                return Err(ProtocolError::Malformed);
            }
            if get32(b, 36) != 0 {
                return Err(ProtocolError::Unsupported);
            }
            let rights = get32(b, 32);
            if rights & !rights::ALL != 0 {
                return Err(ProtocolError::Malformed);
            }
            Operation::Open {
                local: Endpoint::decode(&b[8..32])?,
                rights,
            }
        }
        (PROTOCOL_DATAGRAM_SOCKET, DATAGRAM_SOCKET_SEND_TO) => {
            if b.len() < 40 || handles != 0 || get32(b, 36) != 0 {
                return Err(ProtocolError::Malformed);
            }
            let count = get32(b, 32) as usize;
            if count > MAX_INLINE || b.len() != 40 + count {
                return Err(ProtocolError::Malformed);
            }
            let destination = Endpoint::decode(&b[8..32])?;
            if destination.port == 0 {
                return Err(ProtocolError::Malformed);
            }
            Operation::SendTo {
                destination,
                bytes: &b[40..],
            }
        }
        (PROTOCOL_DATAGRAM_SOCKET, DATAGRAM_SOCKET_RECEIVE) => {
            if b.len() != 16 || handles != 0 || get32(b, 12) != 0 {
                return Err(ProtocolError::Malformed);
            }
            let maximum = get32(b, 8);
            if maximum as usize > MAX_INLINE {
                return Err(ProtocolError::Limit);
            }
            Operation::Receive { maximum }
        }
        (PROTOCOL_DATAGRAM_SOCKET, 4..=7) => {
            let translated = network::decode_request_as(bytes, handles, PROTOCOL_DATAGRAM_SOCKET)?;
            match translated.operation {
                network::Operation::Info => Operation::Info,
                network::Operation::Observe { mask, key } => Operation::Observe { mask, key },
                network::Operation::Rearm {
                    subscription,
                    revision,
                } => Operation::Rearm {
                    subscription,
                    revision,
                },
                network::Operation::Cancel { subscription } => Operation::Cancel { subscription },
                _ => return Err(ProtocolError::Unsupported),
            }
        }
        _ => return Err(ProtocolError::Unsupported),
    };
    Ok(WireRequest {
        header: h,
        deadline,
        operation,
    })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Metadata {
    pub sender: Endpoint,
    pub local: Endpoint,
    pub interface: u32,
    pub original_len: u32,
    pub copied_len: u32,
    pub truncated: bool,
}
#[derive(Debug, PartialEq, Eq)]
pub enum ReceiveReply<'a> {
    Packet { metadata: Metadata, bytes: &'a [u8] },
    Error(i32),
}
pub fn decode_receive_reply(
    body: &[u8],
    handles: usize,
    maximum: usize,
) -> Result<ReceiveReply<'_>, ProtocolError> {
    if body.len() < 4 || handles != 0 || maximum > MAX_INLINE {
        return Err(ProtocolError::Malformed);
    }
    let status = get32(body, 0) as i32;
    if status > 0 {
        return Err(ProtocolError::Malformed);
    }
    if status != 0 {
        return if body.len() == 4 {
            Ok(ReceiveReply::Error(status))
        } else {
            Err(ProtocolError::Malformed)
        };
    }
    if body.len() < 72 || get32(body, 4) & !TRUNCATED != 0 || get32(body, 68) != 0 {
        return Err(ProtocolError::Malformed);
    }
    let original_len = get32(body, 8);
    let copied_len = get32(body, 12);
    let truncated = get32(body, 4) & TRUNCATED != 0;
    if original_len > 65535
        || copied_len as usize != maximum.min(original_len as usize)
        || body.len() != 72 + copied_len as usize
        || truncated != (copied_len < original_len)
    {
        return Err(ProtocolError::Malformed);
    }
    let sender = Endpoint::decode(&body[16..40])?;
    let local = Endpoint::decode(&body[40..64])?;
    let interface = get32(body, 64);
    if local.port == 0
        || interface == 0
        || matches!(
            (sender.address, local.address),
            (Address::V4(_), Address::V6 { .. }) | (Address::V6 { .. }, Address::V4(_))
        )
    {
        return Err(ProtocolError::Malformed);
    }
    Ok(ReceiveReply::Packet {
        metadata: Metadata {
            sender,
            local,
            interface,
            original_len,
            copied_len,
            truncated,
        },
        bytes: &body[72..],
    })
}
pub fn encode_metadata(metadata: Metadata) -> Result<[u8; 72], ProtocolError> {
    if metadata.original_len > 65535
        || metadata.copied_len > metadata.original_len
        || metadata.copied_len as usize > MAX_INLINE
        || metadata.truncated != (metadata.copied_len < metadata.original_len)
        || metadata.interface == 0
        || metadata.local.port == 0
        || matches!(
            (metadata.sender.address, metadata.local.address),
            (Address::V4(_), Address::V6 { .. }) | (Address::V6 { .. }, Address::V4(_))
        )
    {
        return Err(ProtocolError::Malformed);
    }
    let mut out = [0; 72];
    out[4..8].copy_from_slice(&u32::from(metadata.truncated).to_le_bytes());
    out[8..12].copy_from_slice(&metadata.original_len.to_le_bytes());
    out[12..16].copy_from_slice(&metadata.copied_len.to_le_bytes());
    out[16..40].copy_from_slice(&metadata.sender.encode());
    out[40..64].copy_from_slice(&metadata.local.encode());
    out[64..68].copy_from_slice(&metadata.interface.to_le_bytes());
    Ok(out)
}
pub fn decode_send_reply(
    body: &[u8],
    handles: usize,
    offered: usize,
) -> Result<network::WriteReply, ProtocolError> {
    let reply = network::decode_write_reply(body, handles, offered)?;
    if matches!(reply,network::WriteReply::Accepted(n) if n!=offered) {
        return Err(ProtocolError::Malformed);
    }
    Ok(reply)
}
#[derive(Debug)]
pub struct Datagram {
    message: Message,
    metadata: Metadata,
}
impl Datagram {
    pub fn bytes(&self) -> &[u8] {
        &self.message.bytes()[104..]
    }
    pub fn metadata(&self) -> Metadata {
        self.metadata
    }
    pub fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }
    pub fn len(&self) -> usize {
        self.bytes().len()
    }
}
#[derive(Debug)]
pub struct DatagramSocket {
    rpc: Rpc,
}
impl Network {
    pub fn open_datagram(
        &mut self,
        local: Endpoint,
        requested: u32,
        deadline: Deadline,
    ) -> Result<DatagramSocket, NetworkError> {
        if requested & !rights::ALL != 0 {
            return Err(ProtocolError::Malformed.into());
        }
        let mut body = [0; 32];
        body[..24].copy_from_slice(&local.encode());
        body[24..28].copy_from_slice(&requested.to_le_bytes());
        let mut reply = self
            .rpc
            .call(NETWORK_OPEN_DATAGRAM, &[&body], None, deadline)?;
        match network::decode_grant_reply(&reply.bytes()[32..], reply.handle_count(), requested) {
            Ok(network::GrantReply::Granted(_)) => DatagramSocket::new(
                reply
                    .take_handle(0)
                    .ok_or(NetworkError::InvalidReply(ProtocolError::Malformed))?,
            ),
            _ => self.rpc.invalid(ProtocolError::Malformed),
        }
    }
}
impl DatagramSocket {
    pub fn new(endpoint: OwnedHandle) -> Result<Self, NetworkError> {
        Ok(Self {
            rpc: Rpc::new(endpoint, PROTOCOL_DATAGRAM_SOCKET)?,
        })
    }
    pub fn is_retired(&self) -> bool {
        self.rpc.retired()
    }
    pub fn info(&mut self, deadline: Deadline) -> Result<Info, NetworkError> {
        let info = self.rpc.info(deadline)?;
        if info.peer.is_some() {
            return self.rpc.invalid(ProtocolError::Malformed);
        }
        Ok(info)
    }
    pub fn send_to(
        &mut self,
        destination: Endpoint,
        bytes: &[u8],
        deadline: Deadline,
    ) -> Result<usize, NetworkError> {
        if bytes.len() > MAX_INLINE {
            return Err(NetworkError::MessageTooLarge);
        }
        if destination.port == 0 {
            return Err(ProtocolError::Malformed.into());
        }
        let mut body = [0; 32];
        body[..24].copy_from_slice(&destination.encode());
        body[24..28].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        let reply = self
            .rpc
            .call(DATAGRAM_SOCKET_SEND_TO, &[&body, bytes], None, deadline)?;
        match decode_send_reply(&reply.bytes()[32..], reply.handle_count(), bytes.len()) {
            Ok(network::WriteReply::Accepted(n)) => Ok(n),
            Ok(network::WriteReply::Error(s)) => Err(network::status_error(s)),
            Err(e) => self.rpc.invalid(e),
        }
    }
    pub fn receive(
        &mut self,
        maximum: usize,
        deadline: Deadline,
    ) -> Result<Datagram, NetworkError> {
        if maximum > MAX_INLINE {
            return Err(ProtocolError::Limit.into());
        }
        let mut body = [0; 8];
        body[..4].copy_from_slice(&(maximum as u32).to_le_bytes());
        let reply = self
            .rpc
            .call(DATAGRAM_SOCKET_RECEIVE, &[&body], None, deadline)?;
        match decode_receive_reply(&reply.bytes()[32..], reply.handle_count(), maximum) {
            Ok(ReceiveReply::Packet { metadata, .. }) => Ok(Datagram {
                message: reply,
                metadata,
            }),
            Ok(ReceiveReply::Error(s)) => Err(network::status_error(s)),
            Err(e) => self.rpc.invalid(e),
        }
    }
    pub fn observe(
        &mut self,
        mask: u32,
        key: u64,
        deadline: Deadline,
    ) -> Result<Observation, NetworkError> {
        self.rpc.observe(mask, key, deadline)
    }
    pub fn rearm(
        &mut self,
        watch: &mut Observation,
        deadline: Deadline,
    ) -> Result<(), NetworkError> {
        self.rpc.rearm(watch, deadline)
    }
    pub fn cancel_observation(
        &mut self,
        watch: &mut Observation,
        deadline: Deadline,
    ) -> Result<(), NetworkError> {
        self.rpc.cancel(watch, deadline)
    }
    pub fn receive_wait(
        &mut self,
        maximum: usize,
        deadline: Deadline,
    ) -> Result<Datagram, NetworkError> {
        let mut watch: Option<Observation> = None;
        loop {
            match self.receive(maximum, deadline) {
                Err(NetworkError::WouldBlock) => {
                    if watch.is_none() {
                        watch = Some(self.observe(
                            network::signals::READABLE | network::signals::ERROR,
                            0,
                            deadline,
                        )?);
                    }
                    let observer = watch.as_mut().unwrap();
                    if observer.dequeued {
                        self.rearm(observer, deadline)?;
                    }
                    let event = observer.receive(deadline)?;
                    if event.error != 0 && event.signals & network::signals::READABLE == 0 {
                        return Err(network::status_error(event.error));
                    }
                }
                result => return result,
            }
        }
    }
}
