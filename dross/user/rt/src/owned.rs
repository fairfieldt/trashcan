//! Owned capabilities over the existing native ABI.
//!
//! Owned handles close on drop and never implement `Clone`. Duplication is an
//! explicit, fallible operation. Received attachments remain untyped until an
//! actual kernel operation verifies their type. Protocol names alone are not
//! kernel type checks. The free channel operations accept an untyped borrowed
//! transport and let the kernel check its type and rights on every call.
//!
//! The legacy `sys` API remains available. Do not close or transfer a raw number
//! obtained from a borrow, or independently unmap a range owned by `Mapping`.
//! Raw interoperability must preserve these ownership rules, just as raw adoption
//! must. Untyped handles and messages deliberately do not implement Send/Sync:
//! accepting an unknown object does not establish a thread-sharing contract.

use crate::deadline::Deadline;
use crate::{Handle, sys};
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use dross_abi::{Error, MSG_BYTES, MSG_HANDLES, vm};

type Result<T> = core::result::Result<T, Error>;

/// A received capability whose object type has not been established.
pub struct Untyped(PhantomData<Rc<()>>);
pub enum ChannelObject {}
pub enum MemoryObject {}
pub enum ProcessObject {}
pub enum TimerObject {}
pub enum EventQueueObject {}

pub type Channel = OwnedHandle<ChannelObject>;
pub type Memory = OwnedHandle<MemoryObject>;
pub type Process = OwnedHandle<ProcessObject>;
pub type Timer = OwnedHandle<TimerObject>;
pub type EventQueue = OwnedHandle<EventQueueObject>;

/// Exactly one responsibility for closing a process-local handle.
/// Known kernel object markers permit Send/Sync because the kernel serializes
/// their operations; shared memory access still requires separate synchronization.
pub struct OwnedHandle<T = Untyped> {
    raw: Handle,
    marker: PhantomData<T>,
}
impl<T> OwnedHandle<T> {
    /// Adopt one valid handle of the declared object type.
    ///
    /// # Safety
    /// `raw` must be valid, uniquely owned by the caller, and of type `T`.
    /// No other owner may close or transfer its numeric slot. A received handle's
    /// protocol annotation is not sufficient proof of its kernel object type.
    pub unsafe fn from_raw(raw: Handle) -> Self {
        Self {
            raw,
            marker: PhantomData,
        }
    }
    pub fn as_handle(&self) -> BorrowedHandle<'_, T> {
        BorrowedHandle {
            raw: self.raw,
            owner: PhantomData,
        }
    }
    pub fn into_raw(self) -> Handle {
        ManuallyDrop::new(self).raw
    }
    pub fn into_untyped(self) -> OwnedHandle<Untyped> {
        // SAFETY: erasure keeps ownership and makes no claim about object type.
        unsafe { OwnedHandle::from_raw(self.into_raw()) }
    }
    pub fn try_duplicate(&self, rights: u32) -> Result<Self> {
        self.as_handle().try_duplicate(rights)
    }
}
impl<T> Drop for OwnedHandle<T> {
    fn drop(&mut self) {
        let _ = sys::close(self.raw);
    }
}
impl<T> fmt::Debug for OwnedHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("OwnedHandle").field(&self.raw).finish()
    }
}

/// A handle borrow. Its raw number is not a second owner.
pub struct BorrowedHandle<'a, T = Untyped> {
    raw: Handle,
    owner: PhantomData<&'a OwnedHandle<T>>,
}
impl<T> Copy for BorrowedHandle<'_, T> {}
impl<T> Clone for BorrowedHandle<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> BorrowedHandle<'a, T> {
    /// Borrow a valid raw handle without closing it on drop.
    ///
    /// # Safety
    /// The actual owner must keep this handle valid and of type `T` for `'a`.
    /// It must not close or transfer the slot during that lifetime.
    pub unsafe fn from_raw(raw: Handle) -> Self {
        Self {
            raw,
            owner: PhantomData,
        }
    }
    pub fn as_raw(self) -> Handle {
        self.raw
    }
    pub fn as_untyped(self) -> BorrowedHandle<'a, Untyped> {
        BorrowedHandle {
            raw: self.raw,
            owner: PhantomData,
        }
    }
    pub fn try_duplicate(self, rights: u32) -> Result<OwnedHandle<T>> {
        let raw = sys::dup(self.raw, rights)?;
        // SAFETY: successful duplicate returns a fresh owner of the same object.
        Ok(unsafe { OwnedHandle::from_raw(raw) })
    }
}
impl<T> fmt::Debug for BorrowedHandle<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BorrowedHandle").field(&self.raw).finish()
    }
}

/// Owns bytes and every still-present attachment, including during decode errors.
/// Taking an attachment leaves a hole so a second use of that position fails.
pub struct Message {
    bytes: Vec<u8>,
    handles: Vec<Option<OwnedHandle<Untyped>>>,
}
impl Message {
    /// No allocation occurs until bytes or handles are added.
    pub const fn new() -> Self {
        Self {
            bytes: Vec::new(),
            handles: Vec::new(),
        }
    }
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut message = Self::new();
        message.try_extend_bytes(bytes)?;
        Ok(message)
    }
    pub fn try_extend_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let len = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|&len| len <= MSG_BYTES)
            .ok_or(Error::InvalidArgs)?;
        self.bytes
            .try_reserve_exact(len - self.bytes.len())
            .map_err(|_| Error::NoMemory)?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
    /// Number of attachment positions, including positions already taken.
    pub fn handles_len(&self) -> usize {
        self.handles.len()
    }
    pub fn handle_count(&self) -> usize {
        self.handles_len()
    }
    pub fn remaining_handles(&self) -> usize {
        self.handles.iter().flatten().count()
    }
    pub fn take_handle(&mut self, position: usize) -> Option<OwnedHandle<Untyped>> {
        self.handles.get_mut(position)?.take()
    }
    pub fn try_push_handle(
        &mut self,
        handle: OwnedHandle<Untyped>,
    ) -> core::result::Result<(), PushHandleError> {
        if self.handles.len() >= MSG_HANDLES {
            return Err(PushHandleError {
                error: Error::InvalidArgs,
                handle,
            });
        }
        if self.handles.try_reserve_exact(1).is_err() {
            return Err(PushHandleError {
                error: Error::NoMemory,
                handle,
            });
        }
        self.handles.push(Some(handle));
        Ok(())
    }
}
impl Default for Message {
    fn default() -> Self {
        Self::new()
    }
}
impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Message")
            .field("bytes", &self.bytes.len())
            .field("attachment_positions", &self.handles.len())
            .field("remaining_handles", &self.remaining_handles())
            .finish()
    }
}
#[derive(Debug)]
pub struct PushHandleError {
    pub error: Error,
    pub handle: OwnedHandle<Untyped>,
}
#[derive(Debug)]
pub struct SendError {
    pub error: Error,
    pub message: Message,
}

/// Send on a borrowed transport. Failure returns the exact still-owned message.
/// Missing/taken attachment positions are rejected rather than silently renumbered.
pub fn try_send(
    channel: BorrowedHandle<'_, Untyped>,
    mut message: Message,
) -> core::result::Result<(), SendError> {
    if message.bytes.len() > MSG_BYTES
        || message.handles.len() > MSG_HANDLES
        || message.remaining_handles() != message.handles.len()
    {
        return Err(SendError {
            error: Error::InvalidArgs,
            message,
        });
    }
    let mut raw = [Handle(0); MSG_HANDLES];
    for (raw, owned) in raw.iter_mut().zip(&message.handles) {
        *raw = owned.as_ref().unwrap().raw;
    }
    if let Err(error) =
        sys::channel_write(channel.raw, &message.bytes, &raw[..message.handles.len()])
    {
        return Err(SendError { error, message });
    }
    for owned in &mut message.handles {
        let _ = owned.take().unwrap().into_raw();
    }
    Ok(())
}

/// Reserve all output storage before dequeueing. Every returned capability is
/// immediately owned, and unused attachments close when the message is dropped.
pub fn receive(channel: BorrowedHandle<'_, Untyped>, deadline: Deadline) -> Result<Message> {
    let mut message = Message::new();
    message
        .bytes
        .try_reserve_exact(MSG_BYTES)
        .map_err(|_| Error::NoMemory)?;
    message.bytes.resize(MSG_BYTES, 0);
    message
        .handles
        .try_reserve_exact(MSG_HANDLES)
        .map_err(|_| Error::NoMemory)?;
    let mut raw = [Handle(0); MSG_HANDLES];
    let (bytes, count) = sys::channel_read(
        channel.raw,
        &mut message.bytes,
        &mut raw,
        deadline.as_monotonic_ns(),
    )?;
    // Only a successful syscall publishes output slots. On error, tentative
    // numbers written before publication must not be treated as owned handles.
    struct Received([Handle; MSG_HANDLES]);
    impl Drop for Received {
        fn drop(&mut self) {
            for &handle in &self.0 {
                if handle.0 != 0 {
                    let _ = sys::close(handle);
                }
            }
        }
    }
    let mut received = Received(raw);
    if bytes > MSG_BYTES || count > MSG_HANDLES || received.0[..count].iter().any(|h| h.0 == 0) {
        return Err(Error::InvalidArgs);
    }
    for handle in &mut received.0[..count] {
        // SAFETY: a successful receive publishes fresh, uniquely owned slots.
        let owned = unsafe { OwnedHandle::from_raw(*handle) };
        *handle = Handle(0);
        message.handles.push(Some(owned));
    }
    message.bytes.truncate(bytes);
    Ok(message)
}
impl OwnedHandle<ChannelObject> {
    pub fn pair() -> Result<(Self, Self)> {
        let (a, b) = sys::channel_create()?;
        // SAFETY: CHANNEL_CREATE returns fresh channel handles.
        Ok(unsafe { (Self::from_raw(a), Self::from_raw(b)) })
    }
    pub fn try_send(&self, message: Message) -> core::result::Result<(), SendError> {
        try_send(self.as_handle().as_untyped(), message)
    }
    pub fn receive(&self, deadline: Deadline) -> Result<Message> {
        receive(self.as_handle().as_untyped(), deadline)
    }
}
impl BorrowedHandle<'_, ChannelObject> {
    pub fn try_send(self, message: Message) -> core::result::Result<(), SendError> {
        try_send(self.as_untyped(), message)
    }
    pub fn receive(self, deadline: Deadline) -> Result<Message> {
        receive(self.as_untyped(), deadline)
    }
}

/// A failed checked conversion returns ownership of the original capability.
#[derive(Debug)]
pub struct TypeError {
    pub error: Error,
    pub handle: OwnedHandle<Untyped>,
}
impl OwnedHandle<Untyped> {
    /// MEM_SIZE verifies actual memory type without requiring additional rights.
    pub fn try_into_memory(self) -> core::result::Result<Memory, TypeError> {
        if let Err(error) = sys::mem_size(self.raw) {
            return Err(TypeError {
                error,
                handle: self,
            });
        }
        // SAFETY: MEM_SIZE verified the object; only ownership changes wrappers.
        Ok(unsafe { Memory::from_raw(self.into_raw()) })
    }
}
impl OwnedHandle<MemoryObject> {
    pub fn create(bytes: u64, flags: u64) -> Result<Self> {
        let raw = sys::mem_create(bytes, flags)?;
        // SAFETY: MEM_CREATE returns a fresh memory capability.
        Ok(unsafe { Self::from_raw(raw) })
    }
    pub fn size(&self) -> Result<u64> {
        self.as_handle().size()
    }
    pub fn map(&self, offset: u64, bytes: u64, permissions: u64) -> Result<Mapping> {
        self.as_handle().map(offset, bytes, permissions)
    }
}
impl BorrowedHandle<'_, MemoryObject> {
    pub fn size(self) -> Result<u64> {
        sys::mem_size(self.raw)
    }
    /// The kernel selects a fresh address. Mapping owns its reference to backing
    /// and may outlive this memory handle. No Rust references to shared bytes are
    /// created: another mapping, process or device may still modify them.
    pub fn map(self, offset: u64, bytes: u64, permissions: u64) -> Result<Mapping> {
        let address = sys::vm_map(self.raw, offset, bytes, 0, permissions)?;
        Ok(Mapping {
            address,
            bytes,
            permissions,
            active: true,
            not_send: PhantomData,
        })
    }
}

/// Owns a CPU mapping, not exclusive access to its bytes. Drop tries to unmap;
/// use `unmap` when an error must be observed. Independent raw unmap/protect/remap
/// calls must not invalidate this owner's mapping. No Send/Sync is promised.
pub struct Mapping {
    address: u64,
    bytes: u64,
    permissions: u64,
    active: bool,
    not_send: PhantomData<Rc<()>>,
}
impl Mapping {
    pub fn address(&self) -> u64 {
        self.address
    }
    pub fn len(&self) -> u64 {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }
    pub fn permissions(&self) -> u64 {
        self.permissions
    }
    /// Dereferencing the result requires unsafe code. The caller must establish
    /// valid bounds, continued mapping lifetime and synchronization with every
    /// other CPU mapping or device accessing these bytes. READ does not prove
    /// that another participant cannot write the same backing.
    pub fn as_ptr(&self) -> *const u8 {
        self.address as *const u8
    }
    /// A raw writable pointer, only for mappings granted WRITE. As with as_ptr,
    /// dereferencing requires external lifetime, bounds and aliasing guarantees.
    pub fn as_mut_ptr(&self) -> Option<*mut u8> {
        (self.permissions & vm::WRITE != 0).then_some(self.address as *mut u8)
    }
    /// Failure returns this mapping so the owner can retry or inspect it.
    pub fn unmap(mut self) -> core::result::Result<(), UnmapError> {
        if let Err(error) = sys::vm_unmap(self.address, self.bytes) {
            return Err(UnmapError {
                error,
                mapping: self,
            });
        }
        self.active = false;
        Ok(())
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        if self.active {
            let _ = sys::vm_unmap(self.address, self.bytes);
        }
    }
}
impl fmt::Debug for Mapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mapping")
            .field("address", &self.address)
            .field("bytes", &self.bytes)
            .field("permissions", &self.permissions)
            .finish()
    }
}
#[derive(Debug)]
pub struct UnmapError {
    pub error: Error,
    pub mapping: Mapping,
}

impl OwnedHandle<ProcessObject> {
    pub fn wait_until(&self, deadline: Deadline) -> Result<i64> {
        self.as_handle().wait_until(deadline)
    }
}
impl BorrowedHandle<'_, ProcessObject> {
    pub fn wait_until(self, deadline: Deadline) -> Result<i64> {
        sys::process_wait(self.raw, deadline.as_monotonic_ns())
    }
}
impl OwnedHandle<TimerObject> {
    pub fn create() -> Result<Self> {
        let raw = sys::timer_create()?;
        Ok(unsafe { Self::from_raw(raw) })
    }
    pub fn arm(&self, deadline: Deadline) -> Result<()> {
        self.as_handle().arm(deadline)
    }
    pub fn cancel(&self) -> Result<()> {
        self.as_handle().cancel()
    }
}
impl BorrowedHandle<'_, TimerObject> {
    pub fn arm(self, deadline: Deadline) -> Result<()> {
        sys::timer_arm(self.raw, deadline.as_monotonic_ns())
    }
    pub fn cancel(self) -> Result<()> {
        sys::timer_cancel(self.raw)
    }
}
impl OwnedHandle<EventQueueObject> {
    pub fn create(capacity: usize) -> Result<Self> {
        let raw = sys::event_queue_create(capacity)?;
        Ok(unsafe { Self::from_raw(raw) })
    }
    pub fn subscribe<T>(
        &self,
        source: BorrowedHandle<'_, T>,
        signals: u64,
        key: u64,
    ) -> Result<Subscription<'_>> {
        self.as_handle().subscribe(source, signals, key)
    }
    pub fn wait_until(&self, deadline: Deadline) -> Result<dross_abi::event::Event> {
        self.as_handle().wait_until(deadline)
    }
}
impl<'a> BorrowedHandle<'a, EventQueueObject> {
    pub fn subscribe<T>(
        self,
        source: BorrowedHandle<'_, T>,
        signals: u64,
        key: u64,
    ) -> Result<Subscription<'a>> {
        let id = sys::object_subscribe(self.raw, source.raw, signals, key)?;
        Ok(Subscription {
            queue: self,
            id,
            active: true,
        })
    }
    pub fn wait_until(self, deadline: Deadline) -> Result<dross_abi::event::Event> {
        sys::event_queue_wait(self.raw, deadline.as_monotonic_ns())
    }
}
/// A queue registration. It borrows the queue, not the source; closing a source
/// still cancels observation according to the kernel's logical-handle contract.
pub struct Subscription<'a> {
    queue: BorrowedHandle<'a, EventQueueObject>,
    id: u64,
    active: bool,
}
impl Subscription<'_> {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn rearm(&self) -> Result<()> {
        sys::subscription_rearm(self.queue.raw, self.id)
    }
    pub fn cancel(mut self) -> Result<()> {
        sys::subscription_cancel(self.queue.raw, self.id)?;
        self.active = false;
        Ok(())
    }
}
impl Drop for Subscription<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = sys::subscription_cancel(self.queue.raw, self.id);
        }
    }
}
