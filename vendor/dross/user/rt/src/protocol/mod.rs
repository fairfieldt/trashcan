//! Draft public service protocols. Existing service protocols are unchanged.
pub mod bootstrap;
pub mod directory;
pub mod filesystem;
pub mod frame;
pub mod schema;
pub mod network;
pub mod gpu;
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

pub mod datagram;
pub mod resolver;
