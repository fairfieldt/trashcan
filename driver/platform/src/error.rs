use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// A system call failed.
    Os {
        op: &'static str,
        errno: i32,
    },
    /// The platform or device can't do what was asked.
    Unsupported(String),
    /// An argument is out of range: an offset, a size, an alignment, a malformed table.
    Invalid(String),
    /// Nothing matches: a device, a region, a firmware image.
    NotFound(String),
    /// The device address space has no room left.
    Exhausted(String),
    Timeout,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) fn last_os(op: &'static str) -> Self {
        Error::Os { op, errno: std::io::Error::last_os_error().raw_os_error().unwrap_or(0) }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Os { op, errno } => write!(f, "{op}: {}", std::io::Error::from_raw_os_error(*errno)),
            Error::Unsupported(s) => write!(f, "unsupported: {s}"),
            Error::Invalid(s) => write!(f, "invalid: {s}"),
            Error::NotFound(s) => write!(f, "not found: {s}"),
            Error::Exhausted(s) => write!(f, "out of device address space: {s}"),
            Error::Timeout => write!(f, "timed out"),
        }
    }
}

impl std::error::Error for Error {}
