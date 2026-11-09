//! Directory and file capabilities from a filesystem service. A directory reaches only what is below it: paths are
//! relative, with no `..`, and none passes through or ends at a symbolic link. A narrowed directory or file can be
//! handed to another program. Each object has one request/reply reader.
use super::network::{NetworkError, Rpc};
use super::schema::{PROTOCOL_FILE, PROTOCOL_FILES};
use crate::deadline::Deadline;
use crate::owned::{Message, OwnedHandle};
use alloc::{string::String, vec::Vec};

/// Results on this interface. They aren't system call errors or Linux error numbers.
pub mod code {
    pub const OK: i32 = 0;
    pub const NOT_FOUND: i32 = 1;
    pub const EXISTS: i32 = 2;
    pub const DENIED: i32 = 3;
    pub const NOT_DIRECTORY: i32 = 4;
    pub const IS_DIRECTORY: i32 = 5;
    pub const NOT_EMPTY: i32 = 6;
    pub const NO_SPACE: i32 = 7;
    pub const INVALID: i32 = 8;
    pub const LIMIT: i32 = 9;
    pub const FAILED: i32 = 10;
}
pub mod open_flags {
    pub const READ: u32 = 1;
    pub const WRITE: u32 = 2;
    pub const CREATE: u32 = 4;
    pub const TRUNCATE: u32 = 8;
    pub const EXCLUSIVE: u32 = 16;
    pub const ALL: u32 = 31;
}
pub mod directory_ops {
    pub const OPEN_FILE: u32 = 1;
    pub const OPEN_DIRECTORY: u32 = 2;
    pub const LIST: u32 = 3;
    pub const INFO: u32 = 4;
    pub const CREATE_DIRECTORY: u32 = 5;
    pub const REMOVE: u32 = 6;
    pub const RENAME: u32 = 7;
    pub const SYNC: u32 = 8;
}
pub mod file_ops {
    pub const READ: u32 = 1;
    pub const WRITE: u32 = 2;
    pub const INFO: u32 = 3;
    pub const TRUNCATE: u32 = 4;
    pub const SYNC: u32 = 5;
    pub const SET_PERMISSIONS: u32 = 6;
}

/// The most bytes one read or write carries.
pub const MAX_TRANSFER: usize = 60 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileError {
    NotFound,
    Exists,
    Denied,
    NotDirectory,
    IsDirectory,
    NotEmpty,
    NoSpace,
    Invalid,
    Limit,
    Failed,
    /// More than the caller's limit, or too little memory for it.
    TooLarge,
    /// The service couldn't be asked, or its answer made no sense.
    Service(NetworkError),
}
impl From<NetworkError> for FileError {
    fn from(error: NetworkError) -> Self {
        Self::Service(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Info {
    pub inode: u64,
    pub size: u64,
    pub modified: u64,
    pub mode: u32,
}
impl Info {
    pub fn is_directory(&self) -> bool {
        self.mode & 0o170000 == 0o040000
    }
    fn decode(bytes: &[u8]) -> Result<Self, FileError> {
        let bytes = bytes.get(..56).ok_or(FileError::Failed)?;
        let long = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        Ok(Self { inode: long(0), size: long(8), modified: long(24), mode: u32::from_le_bytes(bytes[40..44].try_into().unwrap()) })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub mode: u32,
    pub size: u64,
}

/// The result after a reply's status, and what it attached.
fn result(mut message: Message) -> Result<(Vec<u8>, Option<OwnedHandle>), FileError> {
    let body = message.bytes().get(36..).ok_or(FileError::Failed)?;
    let wire = i32::from_le_bytes(body.get(..4).ok_or(FileError::Failed)?.try_into().unwrap());
    let payload = body[4..].to_vec();
    match wire {
        code::OK => Ok((payload, message.take_handle(0))),
        code::NOT_FOUND => Err(FileError::NotFound),
        code::EXISTS => Err(FileError::Exists),
        code::DENIED => Err(FileError::Denied),
        code::NOT_DIRECTORY => Err(FileError::NotDirectory),
        code::IS_DIRECTORY => Err(FileError::IsDirectory),
        code::NOT_EMPTY => Err(FileError::NotEmpty),
        code::NO_SPACE => Err(FileError::NoSpace),
        code::INVALID => Err(FileError::Invalid),
        code::LIMIT => Err(FileError::Limit),
        _ => Err(FileError::Failed),
    }
}

#[derive(Debug)]
pub struct Directory {
    rpc: Rpc,
}

impl Directory {
    /// Consumes a fresh capability, which no other client has used.
    pub fn new(endpoint: OwnedHandle) -> Result<Self, FileError> {
        Ok(Self { rpc: Rpc::new(endpoint, PROTOCOL_FILES)? })
    }

    fn call(&mut self, ordinal: u32, parts: &[&[u8]], deadline: Deadline) -> Result<(Vec<u8>, Option<OwnedHandle>), FileError> {
        result(self.rpc.call(ordinal, parts, None, deadline)?)
    }

    pub fn open(&mut self, path: &str, flags: u32, deadline: Deadline) -> Result<File, FileError> {
        let (_, handle) = self.call(directory_ops::OPEN_FILE, &[&flags.to_le_bytes(), path.as_bytes()], deadline)?;
        Ok(File { rpc: Rpc::new(handle.ok_or(FileError::Failed)?, PROTOCOL_FILE)? })
    }

    /// A capability for a directory below this one, read-only unless `writable` and this one is writable.
    pub fn open_directory(&mut self, path: &str, writable: bool, deadline: Deadline) -> Result<Directory, FileError> {
        Directory::new(self.open_directory_endpoint(path, writable, deadline)?)
    }

    /// A fresh, unused directory connection for delegation to another process.
    pub fn open_directory_endpoint(&mut self, path: &str, writable: bool, deadline: Deadline) -> Result<OwnedHandle, FileError> {
        let flag = u32::from(writable).to_le_bytes();
        let (_, handle) = self.call(directory_ops::OPEN_DIRECTORY, &[&flag, path.as_bytes()], deadline)?;
        handle.ok_or(FileError::Failed)
    }

    /// What is in this directory, from `cursor` (0 to start). The second value is the next cursor, or 0 at the end.
    pub fn list(&mut self, cursor: u64, deadline: Deadline) -> Result<(Vec<Entry>, u64), FileError> {
        let (bytes, _) = self.call(directory_ops::LIST, &[&cursor.to_le_bytes()], deadline)?;
        let next = u64::from_le_bytes(bytes.get(..8).ok_or(FileError::Failed)?.try_into().unwrap());
        let mut at = 8;
        let mut entries = Vec::new();
        while at < bytes.len() {
            let row = bytes.get(at..at + 40).ok_or(FileError::Failed)?;
            let mode = u32::from_le_bytes(row[0..4].try_into().unwrap());
            let size = u64::from_le_bytes(row[12..20].try_into().unwrap());
            let name_length = usize::from(u16::from_le_bytes(row[36..38].try_into().unwrap()));
            let target_length = usize::from(u16::from_le_bytes(row[38..40].try_into().unwrap()));
            let name = bytes.get(at + 40..at + 40 + name_length).ok_or(FileError::Failed)?;
            entries.push(Entry { name: String::from_utf8_lossy(name).into_owned(), mode, size });
            at += 40 + name_length + target_length;
        }
        Ok((entries, next))
    }

    pub fn info(&mut self, path: &str, deadline: Deadline) -> Result<Info, FileError> {
        Info::decode(&self.call(directory_ops::INFO, &[path.as_bytes()], deadline)?.0)
    }

    pub fn create_directory(&mut self, path: &str, deadline: Deadline) -> Result<(), FileError> {
        self.call(directory_ops::CREATE_DIRECTORY, &[path.as_bytes()], deadline).map(|_| ())
    }

    /// Removes a file or an empty directory.
    pub fn remove(&mut self, path: &str, deadline: Deadline) -> Result<(), FileError> {
        self.call(directory_ops::REMOVE, &[path.as_bytes()], deadline).map(|_| ())
    }

    /// Moves within this directory, replacing what `to` names.
    pub fn rename(&mut self, from: &str, to: &str, deadline: Deadline) -> Result<(), FileError> {
        let length = (from.len() as u32).to_le_bytes();
        self.call(directory_ops::RENAME, &[&length, from.as_bytes(), to.as_bytes()], deadline).map(|_| ())
    }

    /// Everything written so far is on the disk when this returns.
    pub fn sync(&mut self, deadline: Deadline) -> Result<(), FileError> {
        self.call(directory_ops::SYNC, &[], deadline).map(|_| ())
    }
}

/// An open file. Closing the last handle closes it on the service.
#[derive(Debug)]
pub struct File {
    rpc: Rpc,
}

impl File {
    fn call(&mut self, ordinal: u32, parts: &[&[u8]], deadline: Deadline) -> Result<Vec<u8>, FileError> {
        Ok(result(self.rpc.call(ordinal, parts, None, deadline)?)?.0)
    }

    /// Up to `length` bytes, at most `MAX_TRANSFER`. Fewer only at the end of the file.
    pub fn read(&mut self, offset: u64, length: usize, deadline: Deadline) -> Result<Vec<u8>, FileError> {
        let count = length.min(MAX_TRANSFER) as u32;
        self.call(file_ops::READ, &[&offset.to_le_bytes(), &count.to_le_bytes()], deadline)
    }

    /// The whole file, which must be no more than `maximum` bytes.
    pub fn read_to_end(&mut self, maximum: usize, deadline: Deadline) -> Result<Vec<u8>, FileError> {
        let mut all = Vec::new();
        loop {
            let piece = self.read(all.len() as u64, MAX_TRANSFER, deadline)?;
            if piece.is_empty() {
                return Ok(all);
            }
            if all.len() + piece.len() > maximum {
                return Err(FileError::TooLarge);
            }
            all.try_reserve(piece.len()).map_err(|_| FileError::TooLarge)?;
            all.extend_from_slice(&piece);
        }
    }

    /// How much of `bytes` was written, at most `MAX_TRANSFER`.
    pub fn write(&mut self, offset: u64, bytes: &[u8], deadline: Deadline) -> Result<usize, FileError> {
        let piece = &bytes[..bytes.len().min(MAX_TRANSFER)];
        let reply = self.call(file_ops::WRITE, &[&offset.to_le_bytes(), piece], deadline)?;
        let written = u64::from_le_bytes(reply.get(..8).ok_or(FileError::Failed)?.try_into().unwrap());
        Ok(written as usize)
    }

    pub fn write_all(&mut self, mut offset: u64, mut bytes: &[u8], deadline: Deadline) -> Result<(), FileError> {
        while !bytes.is_empty() {
            let written = self.write(offset, bytes, deadline)?;
            if written == 0 || written > bytes.len() {
                return Err(FileError::Failed);
            }
            offset += written as u64;
            bytes = &bytes[written..];
        }
        Ok(())
    }

    pub fn info(&mut self, deadline: Deadline) -> Result<Info, FileError> {
        Info::decode(&self.call(file_ops::INFO, &[], deadline)?)
    }

    pub fn truncate(&mut self, size: u64, deadline: Deadline) -> Result<(), FileError> {
        self.call(file_ops::TRUNCATE, &[&size.to_le_bytes()], deadline).map(|_| ())
    }

    /// Everything written so far is on the disk when this returns.
    pub fn sync(&mut self, deadline: Deadline) -> Result<(), FileError> {
        self.call(file_ops::SYNC, &[], deadline).map(|_| ())
    }

    /// Sets the read, write and execute bits for owner, group and others, as `Info::mode`'s low bits. For a new file that takes
    /// the place of another, so that replacing a script leaves it a script. The file must be open for writing.
    pub fn set_permissions(&mut self, bits: u32, deadline: Deadline) -> Result<(), FileError> {
        self.call(file_ops::SET_PERMISSIONS, &[&bits.to_le_bytes()], deadline).map(|_| ())
    }
}
