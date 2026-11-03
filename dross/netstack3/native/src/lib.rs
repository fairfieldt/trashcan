//! Off-path Oscar userspace bindings for Netstack3. No Fuchsia testutils.
//!
//! This is a finite TCP proof worker, not a public networking service. It does
//! not register as netd or use a hardware NIC. Queues and TCP payloads are
//! bounded; infallible core allocations can still exhaust the process heap.
#![no_std]
#![recursion_limit = "256"]
extern crate alloc;
pub mod buffers;
pub mod runner;
mod sync;
pub mod time;

use crate::sync::Mutex;
use crate::time::{AtomicInstant, Instant};
use alloc::collections::VecDeque;
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::fmt::{self, Debug};
use core::num::{NonZeroU64, NonZeroUsize};
use core::ops::Deref;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use net_types::UnicastAddr;
use net_types::ip::{Ip, IpVersion};
use netstack3_base::sync::DynDebugReferences;
use netstack3_base::{
    BindingsSyncTypes, BufferSizeSettings, DeferredResourceRemovalContext, EventContext,
    InstantBindingsTypes, InstantContext, LinkDevice, MarkDomain, Marks, MatcherBindingsTypes,
    ReferenceNotifiers, RemoveResourceResult, RemoveResourceResultWithContext, RngContext,
    SettingsContext, SocketDiagnosticsSeed, TimerBindingsTypes, TimerContext,
    TxMetadataBindingsTypes,
};
use netstack3_core::{CoreTxMetadata, TimerId};
use netstack3_device::ethernet::EthernetDeviceId;
use netstack3_device::loopback::LoopbackDeviceId;
use netstack3_device::pure_ip::PureIpDeviceId;
use netstack3_device::queue::{
    ReceiveQueueBindingsContext, TransmitQueueBindingsContext, TxBufferAllocator,
};
use netstack3_device::socket::{
    DeviceSocketBindingsContext, DeviceSocketTypes, Frame, ReceiveFrameError, SocketId,
};
use netstack3_device::{
    DeviceBufferBindingsTypes, DeviceClassMatcher, DeviceId, DeviceIdAndNameMatcher,
    DeviceLayerEventDispatcher, DeviceLayerStateTypes, DeviceSendFrameError, WeakDeviceId,
};
use netstack3_filter::{
    FilterIpExt, FilterIpPacket, SocketEgressFilterResult, SocketIngressFilterResult,
    SocketOpsFilter, SocketOpsFilterBindingContext,
};
use netstack3_icmp_echo::{
    IcmpEchoBindingsContext, IcmpEchoBindingsTypes, IcmpSocketId, ReceiveIcmpEchoError,
};
use netstack3_ip::nud::{LinkResolutionContext, LinkResolutionNotifier};
use netstack3_ip::raw::{
    RawIpSocketId, RawIpSocketsBindingsContext, RawIpSocketsBindingsTypes, ReceivePacketError,
};
use netstack3_ip::{IpRoutingBindingsTypes, MarksBindingsContext};
use netstack3_tcp::{
    BufferSizes, TcpBindingsTypes, TcpSettings, TcpSocketDestructionContext, TcpSocketDiagnostics,
};
use netstack3_udp::{
    ReceiveUdpError, UdpBindingsTypes, UdpPacketMeta, UdpReceiveBindingsContext, UdpSocketId,
};
use packet::{Buf, BufferMut};
use zerocopy::SplitByteSlice;

pub const MAX_FRAMES: usize = 64;
pub const MAX_FRAME_BYTES: usize = 2048;
pub const MAX_TIMERS: usize = 256;
const MAX_DEFERRED: usize = 64;

struct TimerEntry {
    id: u64,
    at: Instant,
    dispatch: TimerId<Bindings>,
}
struct TimerPool {
    next: u64,
    entries: Vec<TimerEntry>,
    high_water: usize,
}
#[derive(Clone)]
pub struct Completion(Arc<AtomicBool>);
pub struct CompletionNotifier(Arc<AtomicBool>);
impl<T> netstack3_sync::rc::Notifier<T> for CompletionNotifier {
    fn notify(&mut self, value: T) {
        drop(value);
        self.0.store(true, Ordering::Release);
    }
}

/// Drop cancels the outstanding schedule. The dispatch ID never refers to an
/// unrelated replacement timer because IDs monotonically increase.
pub struct Timer {
    id: u64,
    dispatch: TimerId<Bindings>,
    pool: Weak<Mutex<TimerPool>>,
}
impl Debug for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("NativeTimer").field(&self.id).finish()
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            pool.lock().entries.retain(|entry| entry.id != self.id);
        }
    }
}

#[derive(Debug)]
pub struct FrameBytes(pub Vec<u8>);
struct State {
    now: AtomicU64,
    timers: Arc<Mutex<TimerPool>>,
    frames: Mutex<VecDeque<FrameBytes>>,
    frame_high_water: AtomicU64,
    deferred: Mutex<Vec<Completion>>,
    events: AtomicU64,
    wakeups: AtomicU64,
    failed: AtomicBool,
    rx_pending: AtomicBool,
}

pub struct Bindings {
    state: Arc<State>,
    settings: TcpSettings,
}
impl Bindings {
    pub fn new() -> Self {
        Self::with_tcp_bytes(buffers::TCP_BYTES)
    }
    pub fn with_tcp_bytes(bytes: usize) -> Self {
        let size = NonZeroUsize::new(bytes.min(buffers::MAX_TCP_BYTES).max(1)).unwrap();
        let bound = BufferSizeSettings::new(size, size, size).unwrap();
        Self {
            state: Arc::new(State {
                now: AtomicU64::new(0),
                timers: Arc::new(Mutex::new(TimerPool {
                    next: 1,
                    entries: Vec::with_capacity(MAX_TIMERS),
                    high_water: 0,
                })),
                frames: Mutex::new(VecDeque::with_capacity(MAX_FRAMES)),
                frame_high_water: AtomicU64::new(0),
                deferred: Mutex::new(Vec::with_capacity(MAX_DEFERRED)),
                events: AtomicU64::new(0),
                wakeups: AtomicU64::new(0),
                failed: AtomicBool::new(false),
                rx_pending: AtomicBool::new(false),
            }),
            settings: TcpSettings {
                receive_buffer: bound,
                send_buffer: bound,
            },
        }
    }
    pub fn set_now(&self, now: Instant) {
        self.state.now.fetch_max(now.0, Ordering::Relaxed);
    }
    pub fn next_deadline(&self) -> Option<Instant> {
        self.state
            .timers
            .lock()
            .entries
            .iter()
            .map(|entry| entry.at)
            .min()
    }
    pub fn take_due_timer(&mut self) -> Option<(TimerId<Self>, u64)> {
        let mut pool = self.state.timers.lock();
        let (index, _) = pool
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.at <= self.now())
            .min_by_key(|(_, entry)| (entry.at, entry.id))?;
        let entry = pool.entries.swap_remove(index);
        Some((entry.dispatch, entry.id))
    }
    pub fn pop_frame(&mut self) -> Option<FrameBytes> {
        self.state.frames.lock().pop_front()
    }
    pub fn failed(&self) -> bool {
        self.state.failed.load(Ordering::Acquire)
    }
    pub fn take_rx_pending(&self) -> bool {
        self.state.rx_pending.swap(false, Ordering::AcqRel)
    }
    pub fn frame_count(&self) -> usize {
        self.state.frames.lock().len()
    }
    pub fn timer_count(&self) -> usize {
        self.state.timers.lock().entries.len()
    }
    pub fn high_water(&self) -> (u64, usize) {
        (
            self.state.frame_high_water.load(Ordering::Relaxed),
            self.state.timers.lock().high_water,
        )
    }
    fn track_deferred(&self, receiver: Completion) {
        let mut pending = self.state.deferred.lock();
        pending.retain(|completion| !completion.0.load(Ordering::Acquire));
        if pending.len() < MAX_DEFERRED {
            pending.push(receiver);
        } else {
            self.state.failed.store(true, Ordering::Release);
        }
    }
    pub fn deferred_pending(&self) -> usize {
        self.state
            .deferred
            .lock()
            .iter()
            .filter(|completion| !completion.0.load(Ordering::Acquire))
            .count()
    }
}
impl Default for Bindings {
    fn default() -> Self {
        Self::new()
    }
}
impl BindingsSyncTypes for Bindings {
    type Mutex = sync::RawMutex;
    type RwLock = sync::RawRwLock;
    type Once = sync::Once;
}
impl InstantBindingsTypes for Bindings {
    type Instant = Instant;
    type AtomicInstant = AtomicInstant;
}
impl InstantContext for Bindings {
    fn now(&self) -> Instant {
        Instant(self.state.now.load(Ordering::Relaxed))
    }
}
impl TimerBindingsTypes for Bindings {
    type Timer = Timer;
    type DispatchId = TimerId<Self>;
    type UniqueTimerId = u64;
}
impl TimerContext for Bindings {
    fn new_timer(&mut self, dispatch: Self::DispatchId) -> Timer {
        let id = {
            let mut pool = self.state.timers.lock();
            let id = pool.next;
            pool.next = id.checked_add(1).expect("native timer ID overflow");
            id
        };
        Timer {
            id,
            dispatch,
            pool: Arc::downgrade(&self.state.timers),
        }
    }
    fn schedule_timer_instant(&mut self, at: Instant, timer: &mut Timer) -> Option<Instant> {
        let mut pool = self.state.timers.lock();
        let old = pool
            .entries
            .iter()
            .position(|entry| entry.id == timer.id)
            .map(|index| pool.entries.swap_remove(index).at);
        if pool.entries.len() >= MAX_TIMERS {
            self.state.failed.store(true, Ordering::Release);
            return old;
        }
        pool.entries.push(TimerEntry {
            id: timer.id,
            at,
            dispatch: timer.dispatch.clone(),
        });
        pool.high_water = pool.high_water.max(pool.entries.len());
        old
    }
    fn cancel_timer(&mut self, timer: &mut Timer) -> Option<Instant> {
        let mut pool = self.state.timers.lock();
        pool.entries
            .iter()
            .position(|entry| entry.id == timer.id)
            .map(|index| pool.entries.swap_remove(index).at)
    }
    fn scheduled_instant(&self, timer: &mut Timer) -> Option<Instant> {
        self.state
            .timers
            .lock()
            .entries
            .iter()
            .find(|entry| entry.id == timer.id)
            .map(|entry| entry.at)
    }
    fn unique_timer_id(&self, timer: &Timer) -> u64 {
        timer.id
    }
}

/// Uses Oscar's secure-entropy syscall. Failure aborts this proof worker instead
/// of substituting a predictable clock or fixed seed.
pub struct SysRng;
impl rand::TryRng for SysRng {
    type Error = !;
    fn try_next_u32(&mut self) -> Result<u32, !> {
        let mut bytes = [0; 4];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
    fn try_next_u64(&mut self) -> Result<u64, !> {
        let mut bytes = [0; 8];
        self.try_fill_bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }
    fn try_fill_bytes(&mut self, bytes: &mut [u8]) -> Result<(), !> {
        dross_rt::sys::random(bytes).expect("native proof requires secure entropy");
        Ok(())
    }
}
impl rand::TryCryptoRng for SysRng {}
impl RngContext for Bindings {
    type Rng<'a> = SysRng;
    fn rng(&mut self) -> SysRng {
        SysRng
    }
}
impl TxMetadataBindingsTypes for Bindings {
    type TxMetadata = CoreTxMetadata<Self>;
}
impl SettingsContext<TcpSettings> for Bindings {
    fn settings(&self) -> impl Deref<Target = TcpSettings> + '_ {
        &self.settings
    }
}
impl<T> EventContext<T> for Bindings {
    fn on_event(&mut self, _event: T) {
        self.state.events.fetch_add(1, Ordering::Relaxed);
    }
}
impl ReferenceNotifiers for Bindings {
    type ReferenceReceiver<T: 'static> = Completion;
    type ReferenceNotifier<T: Send + 'static> = CompletionNotifier;
    fn new_reference_notifier<T: Send + 'static>(
        _debug: DynDebugReferences,
    ) -> (CompletionNotifier, Completion) {
        let signal = Arc::new(AtomicBool::new(false));
        (CompletionNotifier(signal.clone()), Completion(signal))
    }
}
impl DeferredResourceRemovalContext for Bindings {
    fn defer_removal<T: Send + 'static>(&mut self, receiver: Completion) {
        self.track_deferred(receiver);
    }
}
impl TcpSocketDestructionContext for Bindings {
    fn defer_tcp_socket_destruction<I, S>(&self, result: RemoveResourceResultWithContext<S, Self>)
    where
        I: Ip,
        S: SocketDiagnosticsSeed<Output = TcpSocketDiagnostics<I, Instant>> + Send + 'static,
    {
        if let RemoveResourceResult::Deferred(receiver) = result {
            self.track_deferred(receiver);
        }
    }
}
impl TcpBindingsTypes for Bindings {
    type ReceiveBuffer = buffers::Rx;
    type SendBuffer = buffers::Ring;
    type ReturnedBuffers = buffers::ClientBuffers;
    type ListenerNotifierOrProvidedBuffers = buffers::SocketExtra;
    fn new_passive_open_buffers(
        sizes: BufferSizes,
    ) -> (Self::ReceiveBuffer, Self::SendBuffer, Self::ReturnedBuffers) {
        buffers::new_buffers(sizes)
    }
}

#[derive(Debug)]
pub struct Identifier(pub NonZeroU64);
impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl DeviceIdAndNameMatcher for Identifier {
    fn id_matches(&self, id: &NonZeroU64) -> bool {
        self.0 == *id
    }
    fn name_matches(&self, _name: &str) -> bool {
        false
    }
}
#[derive(Debug)]
pub struct DeviceState;
impl DeviceClassMatcher<()> for DeviceState {
    fn device_class_matches(&self, _class: &()) -> bool {
        true
    }
}
impl DeviceLayerStateTypes for Bindings {
    type LoopbackDeviceState = DeviceState;
    type EthernetDeviceState = DeviceState;
    type BlackholeDeviceState = DeviceState;
    type PureIpDeviceState = DeviceState;
    type DeviceIdentifier = Identifier;
}
impl MatcherBindingsTypes for Bindings {
    type DeviceClass = ();
    type BindingsPacketMatcher = !;
}
#[derive(Default)]
pub struct TxAllocator;
impl TxBufferAllocator<Buf<Vec<u8>>> for TxAllocator {
    type Error = ();
    fn alloc(&mut self, length: usize, queue_length: usize) -> Result<Buf<Vec<u8>>, ()> {
        if length > MAX_FRAME_BYTES || queue_length >= MAX_FRAMES {
            return Err(());
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|_| ())?;
        bytes.resize(length, 0);
        Ok(Buf::new(bytes, ..))
    }
}
impl DeviceBufferBindingsTypes for Bindings {
    type TxBuffer = Buf<Vec<u8>>;
    type TxAllocator = TxAllocator;
}
impl ReceiveQueueBindingsContext<LoopbackDeviceId<Self>> for Bindings {
    fn wake_rx_task(&mut self, _device: &LoopbackDeviceId<Self>) {
        self.state.rx_pending.store(true, Ordering::Release);
        self.state.wakeups.fetch_add(1, Ordering::Relaxed);
    }
}
impl<D> TransmitQueueBindingsContext<D> for Bindings {
    fn wake_tx_task(&mut self, _device: &D) {
        self.state.wakeups.fetch_add(1, Ordering::Relaxed);
    }
}
impl DeviceLayerEventDispatcher for Bindings {
    type DequeueContext = ();
    fn send_ethernet_frame(
        &mut self,
        _device: &EthernetDeviceId<Self>,
        frame: Buf<Vec<u8>>,
        _dequeue: Option<&mut ()>,
        _offload: Option<netstack3_base::ChecksumOffloadResult>,
    ) -> Result<(), DeviceSendFrameError> {
        let mut queue = self.state.frames.lock();
        if queue.len() >= MAX_FRAMES || frame.as_ref().len() > MAX_FRAME_BYTES {
            return Err(DeviceSendFrameError::NoBuffers);
        }
        queue.push_back(FrameBytes(frame.into_inner()));
        self.state
            .frame_high_water
            .fetch_max(queue.len() as u64, Ordering::Relaxed);
        Ok(())
    }
    fn send_ip_packet(
        &mut self,
        _device: &PureIpDeviceId<Self>,
        _packet: Buf<Vec<u8>>,
        _version: IpVersion,
        _dequeue: Option<&mut ()>,
        _offload: Option<netstack3_base::ChecksumOffloadResult>,
    ) -> Result<(), DeviceSendFrameError> {
        Err(DeviceSendFrameError::NoBuffers)
    }
}
#[derive(Debug)]
pub struct NeighborNotifier;
impl<D: LinkDevice> LinkResolutionContext<D> for Bindings {
    type Notifier = NeighborNotifier;
}
impl<D: LinkDevice> LinkResolutionNotifier<D> for NeighborNotifier {
    type Observer = ();
    fn new() -> (Self, ()) {
        (Self, ())
    }
    fn notify(
        self,
        _result: Result<UnicastAddr<D::Address>, netstack3_base::AddressResolutionFailed>,
    ) {
    }
}
impl IpRoutingBindingsTypes for Bindings {
    type RoutingTableId = ();
}
impl MarksBindingsContext for Bindings {
    fn marks_to_keep_on_egress() -> &'static [MarkDomain] {
        &[]
    }
    fn marks_to_set_on_ingress() -> &'static [MarkDomain] {
        &[]
    }
}
struct PassFilter;
impl SocketOpsFilter<DeviceId<Bindings>> for PassFilter {
    fn on_egress<I: FilterIpExt, P: FilterIpPacket<I>>(
        &self,
        _packet: &P,
        _device: &DeviceId<Bindings>,
        _socket: netstack3_base::socket::SocketInfo,
        _marks: &Marks,
        _gso: Option<netstack3_base::GsoInfo>,
    ) -> SocketEgressFilterResult {
        SocketEgressFilterResult::Pass { congestion: false }
    }
    fn on_ingress(
        &self,
        _version: IpVersion,
        _packet: packet::FragmentedByteSlice<'_, &[u8]>,
        _header: usize,
        _device: &DeviceId<Bindings>,
        _socket: netstack3_base::socket::SocketInfo,
        _marks: &Marks,
        _kind: netstack3_base::PacketType,
        _gso: Option<netstack3_base::GsoInfo>,
    ) -> SocketIngressFilterResult {
        SocketIngressFilterResult::Accept
    }
}
impl SocketOpsFilterBindingContext<DeviceId<Self>> for Bindings {
    fn socket_ops_filter(&self) -> impl SocketOpsFilter<DeviceId<Self>> {
        PassFilter
    }
}
impl UdpBindingsTypes for Bindings {
    type ExternalData<I: Ip> = ();
    type SendToken = ();
}
impl<I: netstack3_core::IpExt> UdpReceiveBindingsContext<I, DeviceId<Self>> for Bindings {
    fn receive_udp(
        &mut self,
        _id: &UdpSocketId<I, WeakDeviceId<Self>, Self>,
        _device: &DeviceId<Self>,
        _meta: UdpPacketMeta<I>,
        _body: &[u8],
    ) -> Result<(), ReceiveUdpError> {
        Err(ReceiveUdpError::QueueFull)
    }
    fn on_socket_error(
        &mut self,
        _id: &UdpSocketId<I, WeakDeviceId<Self>, Self>,
        _error: netstack3_datagram::PendingDatagramSocketError,
    ) {
    }
}
impl IcmpEchoBindingsTypes for Bindings {
    type ExternalData<I: Ip> = ();
    type SendToken = ();
}
impl<I: netstack3_core::IpExt> IcmpEchoBindingsContext<I, DeviceId<Self>> for Bindings {
    fn receive_icmp_echo_reply<B: BufferMut>(
        &mut self,
        _id: &IcmpSocketId<I, WeakDeviceId<Self>, Self>,
        _device: &DeviceId<Self>,
        _src: I::Addr,
        _dst: I::Addr,
        _id_number: u16,
        _data: B,
    ) -> Result<(), ReceiveIcmpEchoError> {
        Err(ReceiveIcmpEchoError::QueueFull)
    }
}
impl DeviceSocketTypes for Bindings {
    type SocketState<D: Send + Sync + Debug> = ();
}
impl DeviceSocketBindingsContext<DeviceId<Self>> for Bindings {
    fn receive_frame(
        &self,
        _id: &SocketId<Self>,
        _device: &DeviceId<Self>,
        _frame: Frame<&[u8]>,
        _raw: &[u8],
    ) -> Result<(), ReceiveFrameError> {
        Err(ReceiveFrameError::QueueFull)
    }
}
impl RawIpSocketsBindingsTypes for Bindings {
    type RawIpSocketState<I: Ip> = ();
}
impl<I: netstack3_core::IpExt> RawIpSocketsBindingsContext<I, DeviceId<Self>> for Bindings {
    fn receive_packet<B: SplitByteSlice>(
        &self,
        _socket: &RawIpSocketId<I, WeakDeviceId<Self>, Self>,
        _packet: &I::Packet<B>,
        _device: &DeviceId<Self>,
    ) -> Result<(), ReceivePacketError> {
        Err(ReceivePacketError::QueueFull)
    }
}
