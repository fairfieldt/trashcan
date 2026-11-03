//! Off-path Netstack3 host proof. This uses upstream fake bindings, not Oscar syscalls.
//!
//! The packet frontdoor and application writes are bounded. Upstream fake
//! bindings still allocate internally; callback-boundary assertions are NOT a
//! hard memory quota or evidence of graceful OOM handling.
mod allocations;

use std::collections::VecDeque;
use std::num::{NonZeroU16, NonZeroUsize};
use std::time::Duration;

use net_types::ZonedAddr;
use net_types::ip::Ipv4;
use netstack3_base::TcpSocketState;
use netstack3_base::testutil::{FakeInstant, InstantAndData, TEST_ADDRS_V4, WithFakeTimerContext};
use netstack3_core::device::{EthernetDeviceId, WeakDeviceId};
use netstack3_core::socket::ShutdownType;
use netstack3_core::testutil::{CtxPairExt, FakeBindingsCtx, FakeCtx, FakeCtxBuilder};
use netstack3_tcp::testutil::{
    ClientBuffers, ProvidedBuffers, RingBuffer, TestSendBuffer, WriteBackClientBuffers,
};
use netstack3_tcp::{AcceptError, Buffer, TcpSocketId};
use packet::Buf;

const BUFFER_BYTES: usize = 16 * 1024;
const MAX_FRAME_BYTES: usize = 2048;
const MAX_PACKETS: usize = 64;
const MAX_TIMERS_PER_ENDPOINT: usize = 256;
const MAX_STEPS: usize = 16_384;
const PORT: NonZeroU16 = NonZeroU16::new(22222).unwrap();
type Socket = TcpSocketId<Ipv4, WeakDeviceId<FakeBindingsCtx>, FakeBindingsCtx>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Host {
    Client = 0,
    Server = 1,
}
impl Host {
    fn index(self) -> usize {
        self as usize
    }
    fn peer(self) -> Self {
        match self {
            Self::Client => Self::Server,
            Self::Server => Self::Client,
        }
    }
    const ALL: [Host; 2] = [Host::Client, Host::Server];
}

struct Packet {
    destination: Host,
    length: usize,
    bytes: [u8; MAX_FRAME_BYTES],
}
impl Packet {
    fn new(destination: Host, bytes: &[u8]) -> Option<Self> {
        if bytes.len() > MAX_FRAME_BYTES {
            return None;
        }
        let mut packet = Self {
            destination,
            length: bytes.len(),
            bytes: [0; MAX_FRAME_BYTES],
        };
        packet.bytes[..bytes.len()].copy_from_slice(bytes);
        Some(packet)
    }
}

/// Models a bounded native NIC-channel frontdoor; rejection is explicit.
struct PacketQueue {
    packets: VecDeque<Packet>,
    high_water: usize,
    rejected: usize,
}
impl PacketQueue {
    fn new() -> Self {
        Self {
            packets: VecDeque::with_capacity(MAX_PACKETS),
            high_water: 0,
            rejected: 0,
        }
    }
    fn offer(&mut self, destination: Host, bytes: &[u8]) -> bool {
        if self.packets.len() == MAX_PACKETS {
            self.rejected += 1;
            return false;
        }
        let Some(packet) = Packet::new(destination, bytes) else {
            self.rejected += 1;
            return false;
        };
        self.packets.push_back(packet);
        self.high_water = self.high_water.max(self.packets.len());
        true
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LossPlan {
    syn: bool,
    data: bool,
}
#[derive(Debug, Default, PartialEq, Eq)]
struct Report {
    emitted_frames: usize,
    client_syn_frames: usize,
    delivered_frames: usize,
    dropped_syn: usize,
    dropped_data: usize,
    timers_fired: usize,
    timer_high_water: usize,
    packet_high_water: usize,
    client_retransmits: u64,
    elapsed_nanos: u128,
}

/// Deterministic event ordering: emitted frames in endpoint order, FIFO packets,
/// then earliest deadline with client-first tie breaking. No wall-clock sleeps.
struct Lab {
    endpoints: [FakeCtx; 2],
    devices: [Option<EthernetDeviceId<FakeBindingsCtx>>; 2],
    packets: PacketQueue,
    now: FakeInstant,
    loss: LossPlan,
    report: Report,
    steps: usize,
}
impl Lab {
    fn new(loss: LossPlan) -> Self {
        let (client, mut client_devices) = FakeCtxBuilder::with_addrs(TEST_ADDRS_V4).build();
        let (server, mut server_devices) = FakeCtxBuilder::with_addrs(TEST_ADDRS_V4.swap()).build();
        client.bindings_ctx.seed_rng(0x4f53434152);
        server.bindings_ctx.seed_rng(0x44524f5353);
        Self {
            endpoints: [client, server],
            devices: [
                Some(client_devices.remove(0)),
                Some(server_devices.remove(0)),
            ],
            packets: PacketQueue::new(),
            now: FakeInstant::default(),
            loss,
            report: Report::default(),
            steps: 0,
        }
    }
    fn ctx(&mut self, host: Host) -> &mut FakeCtx {
        &mut self.endpoints[host.index()]
    }
    fn state(&mut self, host: Host, socket: &Socket) -> TcpSocketState {
        self.ctx(host)
            .core_api()
            .tcp::<Ipv4>()
            .get_tcp_info(socket)
            .state
    }
    fn audit_callbacks(&mut self) {
        for host in Host::ALL {
            let count = self.ctx(host).bindings_ctx.timer_ctx().timers.len();
            self.report.timer_high_water = self.report.timer_high_water.max(count);
            assert!(
                count <= MAX_TIMERS_PER_ENDPOINT,
                "fake timer budget exceeded"
            );
            // These are upstream diagnostic recordings, not application queues.
            self.ctx(host).bindings_ctx.take_events();
            self.ctx(host)
                .bindings_ctx
                .state_mut()
                .packet_metadata
                .clear();
        }
    }
    fn collect_frames(&mut self) {
        for source in Host::ALL {
            for (_, frame) in self.ctx(source).bindings_ctx.take_ethernet_frames() {
                self.report.emitted_frames += 1;
                let tcp = tcp_segment(&frame);
                if source == Host::Client && tcp.is_some_and(|(syn, _)| syn) {
                    self.report.client_syn_frames += 1;
                }
                let drop_syn =
                    source == Host::Client && self.loss.syn && tcp.is_some_and(|(syn, _)| syn);
                let drop_data =
                    source == Host::Client && self.loss.data && tcp.is_some_and(|(_, data)| data);
                if drop_syn {
                    self.loss.syn = false;
                    self.report.dropped_syn += 1;
                } else if drop_data {
                    self.loss.data = false;
                    self.report.dropped_data += 1;
                } else {
                    assert!(
                        self.packets.offer(source.peer(), &frame),
                        "bounded packet queue rejected a workload frame"
                    );
                }
            }
        }
        self.audit_callbacks();
    }
    fn step(&mut self, deadline: FakeInstant) -> bool {
        self.steps += 1;
        assert!(self.steps <= MAX_STEPS, "simulation step budget exhausted");
        self.collect_frames();
        if let Some(packet) = self.packets.packets.pop_front() {
            let device = self.devices[packet.destination.index()]
                .as_ref()
                .unwrap()
                .clone();
            self.ctx(packet.destination)
                .test_api()
                .receive_ethernet_frame(&device, Buf::new(packet.bytes, 0..packet.length));
            self.report.delivered_frames += 1;
            return true;
        }
        let next = Host::ALL
            .into_iter()
            .filter_map(|host| {
                self.ctx(host)
                    .with_fake_timer_ctx(|timers| timers.timers.peek().map(|entry| (entry.0, host)))
            })
            .min_by_key(|(instant, host)| (*instant, host.index()));
        let Some((instant, host)) = next else {
            return false;
        };
        if instant > deadline {
            return false;
        }
        assert!(instant >= self.now, "timer moved backwards");
        self.now = instant;
        for endpoint in &mut self.endpoints {
            endpoint.with_fake_timer_ctx_mut(|timers| timers.instant.time = instant);
        }
        let InstantAndData(_, timer) = self
            .ctx(host)
            .with_fake_timer_ctx_mut(|timers| timers.timers.pop().unwrap());
        let (dispatch, unique) = timer.into_dispatch_and_timer_id();
        self.ctx(host).core_api().handle_timer(dispatch, unique);
        self.report.timers_fired += 1;
        true
    }
    fn until(&mut self, seconds: u64, mut done: impl FnMut(&mut Self) -> bool) {
        let deadline = self.now + Duration::from_secs(seconds);
        for _ in 0..MAX_STEPS {
            if done(self) {
                self.audit_callbacks();
                return;
            }
            assert!(
                self.step(deadline),
                "condition unmet before simulation deadline; report={:?}",
                self.report
            );
        }
        panic!("condition exceeded simulation step budget");
    }
    fn finish(mut self) -> Report {
        self.collect_frames();
        assert!(
            self.packets.packets.is_empty(),
            "unconsumed frames after close"
        );
        assert_eq!(self.packets.rejected, 0);
        self.report.packet_high_water = self.packets.high_water;
        self.report.elapsed_nanos = self.now.offset.as_nanos();
        for host in Host::ALL {
            let device = self.devices[host.index()].take().unwrap();
            self.ctx(host)
                .test_api()
                .clear_routes_and_remove_device(device);
            assert!(
                self.ctx(host).bindings_ctx.timer_ctx().timers.is_empty(),
                "timer retained after device removal"
            );
        }
        self.audit_callbacks();
        self.report
    }
}

/// The proof sends only plain, unfragmented IPv4 Ethernet frames. Classification
/// is solely for deterministic fault injection, not a protocol implementation.
fn tcp_segment(frame: &[u8]) -> Option<(bool, bool)> {
    if frame.len() < 14 + 20 || frame[12..14] != [8, 0] || frame[23] != 6 {
        return None;
    }
    let ip_header = usize::from(frame[14] & 15) * 4;
    let tcp = 14 + ip_header;
    if ip_header < 20 || frame.len() < tcp + 20 {
        return None;
    }
    let tcp_header = usize::from(frame[tcp + 12] >> 4) * 4;
    let total = usize::from(u16::from_be_bytes([frame[16], frame[17]]));
    Some((frame[tcp + 13] & 2 != 0, total > ip_header + tcp_header))
}

#[derive(Debug, PartialEq, Eq)]
enum Read {
    Data(Vec<u8>),
    WouldBlock,
    Eof,
}
#[derive(Debug, PartialEq, Eq)]
enum WriteError {
    WouldBlock,
    Closed,
}

/// Owned test-side stream object. It demonstrates a bounded native-facing
/// adapter over core APIs; it is not yet the production Oscar capability SDK.
struct Stream {
    host: Host,
    socket: Socket,
    buffers: ClientBuffers,
    send_closed: bool,
}
impl Stream {
    fn new(lab: &mut Lab, host: Host, socket: Socket, buffers: ClientBuffers) -> Self {
        // Fake bindings supply a default-size private send ring. Replace it
        // before application traffic with the configured bounded ring.
        lab.ctx(host)
            .core_api()
            .tcp::<Ipv4>()
            .with_send_buffer(&socket, |send| {
                assert_eq!(send.limits().len, 0);
                *send = TestSendBuffer::new(buffers.send.clone(), RingBuffer::new(BUFFER_BYTES));
            })
            .expect("connected socket has send buffer");
        assert_eq!(buffers.receive.lock().limits().capacity, BUFFER_BYTES);
        Self {
            host,
            socket,
            buffers,
            send_closed: false,
        }
    }
    fn write(&self, lab: &mut Lab, bytes: &[u8]) -> Result<usize, WriteError> {
        if self.send_closed {
            return Err(WriteError::Closed);
        }
        let pending = lab
            .ctx(self.host)
            .core_api()
            .tcp::<Ipv4>()
            .with_send_buffer(&self.socket, |send| send.limits().len)
            .ok_or(WriteError::Closed)?;
        if bytes.len() > BUFFER_BYTES.saturating_sub(pending) {
            return Err(WriteError::WouldBlock);
        }
        self.buffers.send.lock().extend_from_slice(bytes);
        lab.ctx(self.host)
            .core_api()
            .tcp::<Ipv4>()
            .do_send(&self.socket);
        Ok(bytes.len())
    }
    fn read(&self, lab: &mut Lab) -> Read {
        let mut bytes = Vec::new();
        self.buffers.receive.lock().read_with(|regions| {
            for region in regions {
                bytes.extend_from_slice(region);
            }
            bytes.len()
        });
        assert!(bytes.len() <= BUFFER_BYTES);
        if !bytes.is_empty() {
            lab.ctx(self.host)
                .core_api()
                .tcp::<Ipv4>()
                .on_receive_buffer_read(&self.socket);
            return Read::Data(bytes);
        }
        assert!(
            lab.ctx(self.host)
                .core_api()
                .tcp::<Ipv4>()
                .get_socket_error(&self.socket)
                .is_none(),
            "socket error must not masquerade as EOF"
        );
        match lab.state(self.host, &self.socket) {
            TcpSocketState::CloseWait
            | TcpSocketState::LastAck
            | TcpSocketState::Closing
            | TcpSocketState::TimeWait
            | TcpSocketState::Close => Read::Eof,
            _ => Read::WouldBlock,
        }
    }
    fn shutdown_send(&mut self, lab: &mut Lab) {
        assert_eq!(
            lab.ctx(self.host)
                .core_api()
                .tcp::<Ipv4>()
                .shutdown(&self.socket, ShutdownType::Send),
            Ok(true)
        );
        self.send_closed = true;
    }
    fn close(self, lab: &mut Lab) {
        lab.ctx(self.host)
            .core_api()
            .tcp::<Ipv4>()
            .close(self.socket);
    }
}

fn configure(lab: &mut Lab, host: Host, socket: &Socket) {
    let core = lab.ctx(host).core_api();
    let mut tcp = core.tcp::<Ipv4>();
    tcp.set_send_buffer_size(socket, BUFFER_BYTES);
    tcp.set_receive_buffer_size(socket, BUFFER_BYTES);
}
fn receive_exact(lab: &mut Lab, stream: &Stream, expected: &[u8]) {
    let mut received = Vec::new();
    lab.until(30, |lab| {
        match stream.read(lab) {
            Read::Data(part) => received.extend_from_slice(&part),
            Read::WouldBlock => {}
            Read::Eof => panic!("EOF before exact payload"),
        }
        assert!(
            received.len() <= expected.len(),
            "duplicate or unexpected application bytes"
        );
        received.len() == expected.len()
    });
    assert_eq!(received, expected);
}

fn expect_eof(lab: &mut Lab, stream: &Stream) -> bool {
    match stream.read(lab) {
        Read::Eof => true,
        Read::WouldBlock => false,
        Read::Data(bytes) => panic!("unexpected bytes while waiting for EOF: {bytes:?}"),
    }
}

fn exercise(loss: LossPlan) -> Report {
    let mut lab = Lab::new(loss);
    let listener = lab
        .ctx(Host::Server)
        .core_api()
        .tcp::<Ipv4>()
        .create(ProvidedBuffers::NoBuffers);
    configure(&mut lab, Host::Server, &listener);
    {
        let core = lab.ctx(Host::Server).core_api();
        let mut tcp = core.tcp::<Ipv4>();
        tcp.bind(&listener, None, Some(PORT)).unwrap();
        tcp.listen(&listener, NonZeroUsize::new(1).unwrap())
            .unwrap();
        assert!(matches!(
            tcp.accept(&listener),
            Err(AcceptError::WouldBlock)
        ));
    }
    let written_back = WriteBackClientBuffers::default();
    let client = lab
        .ctx(Host::Client)
        .core_api()
        .tcp::<Ipv4>()
        .create(ProvidedBuffers::Buffers(written_back.clone()));
    configure(&mut lab, Host::Client, &client);
    lab.ctx(Host::Client)
        .core_api()
        .tcp::<Ipv4>()
        .connect(
            &client,
            Some(ZonedAddr::Unzoned(TEST_ADDRS_V4.remote_ip)),
            PORT,
        )
        .unwrap();
    let mut accepted = None;
    lab.until(30, |lab| {
        match lab
            .ctx(Host::Server)
            .core_api()
            .tcp::<Ipv4>()
            .accept(&listener)
        {
            Ok(value) => {
                accepted = Some(value);
                true
            }
            Err(AcceptError::WouldBlock) => false,
            Err(error) => panic!("accept failed: {error:?}"),
        }
    });
    assert_eq!(
        lab.state(Host::Client, &client),
        TcpSocketState::Established
    );
    let handshake_timers = lab.report.timers_fired;
    let (server, remote, server_buffers) = accepted.unwrap();
    assert_eq!(remote.ip, ZonedAddr::Unzoned(TEST_ADDRS_V4.local_ip));
    let client_buffers = written_back
        .0
        .lock()
        .take()
        .expect("active-open buffers returned");
    let mut client = Stream::new(&mut lab, Host::Client, client, client_buffers);
    let mut server = Stream::new(&mut lab, Host::Server, server, server_buffers);
    assert_eq!(client.read(&mut lab), Read::WouldBlock);
    assert_eq!(server.read(&mut lab), Read::WouldBlock);
    assert_eq!(
        client.write(&mut lab, &vec![0; BUFFER_BYTES + 1]),
        Err(WriteError::WouldBlock)
    );
    assert!(client.buffers.send.lock().is_empty());

    // A single data segment in the data-loss run ensures recovery needs a
    // timeout instead of being satisfied only by fast retransmit duplicate ACKs.
    let request: Vec<u8> = (0..if loss.data { 512 } else { 9217 })
        .map(|n| (n * 37 + 11) as u8)
        .collect();
    let response: Vec<u8> = (0..4099).map(|n| (n * 19 + 3) as u8).collect();
    let retransmits_before = lab
        .ctx(Host::Client)
        .core_api()
        .tcp::<Ipv4>()
        .get_tcp_info(&client.socket)
        .retransmits;
    let timers_before = lab.report.timers_fired;
    assert_eq!(client.write(&mut lab, &request), Ok(request.len()));
    assert_eq!(server.write(&mut lab, &response), Ok(response.len()));
    receive_exact(&mut lab, &server, &request);
    receive_exact(&mut lab, &client, &response);
    let info = lab
        .ctx(Host::Client)
        .core_api()
        .tcp::<Ipv4>()
        .get_tcp_info(&client.socket);
    if loss.data {
        assert_eq!(lab.report.dropped_data, 1);
        assert!(
            info.retransmits > retransmits_before,
            "data loss did not cause retransmission"
        );
        assert!(
            lab.report.timers_fired > timers_before,
            "data recovery did not fire a timer"
        );
    }
    if loss.syn {
        assert_eq!(lab.report.dropped_syn, 1);
        assert!(lab.report.client_syn_frames >= 2, "lost SYN was not emitted again");
        assert!(handshake_timers > 0, "lost SYN did not require a timer");
    }
    lab.report.client_retransmits = info.retransmits;

    client.shutdown_send(&mut lab);
    assert_eq!(
        client.write(&mut lab, b"forbidden"),
        Err(WriteError::Closed)
    );
    lab.until(30, |lab| expect_eof(lab, &server));
    assert_eq!(
        lab.state(Host::Server, &server.socket),
        TcpSocketState::CloseWait
    );
    let final_reply = b"reply after peer half-close\0with exact bytes";
    assert_eq!(server.write(&mut lab, final_reply), Ok(final_reply.len()));
    receive_exact(&mut lab, &client, final_reply);
    server.shutdown_send(&mut lab);
    lab.until(30, |lab| expect_eof(lab, &client));
    assert_eq!(client.read(&mut lab), Read::Eof);
    assert_eq!(server.read(&mut lab), Read::Eof);

    client.close(&mut lab);
    server.close(&mut lab);
    lab.ctx(Host::Server)
        .core_api()
        .tcp::<Ipv4>()
        .close(listener);
    lab.until(240, |lab| {
        Host::ALL.into_iter().all(|host| {
            let mut diagnostics = Vec::new();
            let all: &[netstack3_core::socket::IpSocketMatcher<()>] = &[];
            lab.ctx(host).core_api().tcp::<Ipv4>()
                .bound_sockets_diagnostics(all, &mut diagnostics, false);
            diagnostics.is_empty()
        })
    });
    // Deliver an already-emitted final ACK before removing the NICs.
    lab.collect_frames();
    while !lab.packets.packets.is_empty() {
        assert!(lab.step(lab.now));
        lab.collect_frames();
    }
    lab.finish()
}

fn main() {
    for (label, loss) in [
        ("baseline", LossPlan::default()),
        (
            "lost SYN",
            LossPlan {
                syn: true,
                data: false,
            },
        ),
        (
            "lost data",
            LossPlan {
                syn: false,
                data: true,
            },
        ),
    ] {
        let (report, memory) = allocations::measure(|| exercise(loss));
        println!("{label}: {report:?}");
        println!(
            "  allocator-requested bytes: before={}, after={}, peak={}; allocation/reallocation calls={}",
            memory.live_before, memory.live_after, memory.peak_live, memory.allocation_calls,
        );
    }
    println!(
        "PASS: connect/listen/accept, exact bidirectional bytes, bounded writes, half-close/EOF, close, timer-driven loss recovery"
    );
    println!(
        "Harness bounds: 2 endpoints, 1 listener + 2 streams, {BUFFER_BYTES} payload admission bytes per TCP direction, {MAX_PACKETS} packets x {MAX_FRAME_BYTES} bytes, {MAX_TIMERS_PER_ENDPOINT} observed timers per endpoint, {MAX_STEPS} event steps"
    );
    println!(
        "Limits: upstream fake bindings; no native syscalls, hard allocator quota, graceful OOM, or performance claim"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tcp_baseline() {
        exercise(LossPlan::default());
    }
    #[test]
    fn lost_syn_retransmits() {
        exercise(LossPlan {
            syn: true,
            data: false,
        });
    }
    #[test]
    fn lost_data_retransmits() {
        exercise(LossPlan {
            syn: false,
            data: true,
        });
    }
    #[test]
    fn deterministic_replay() {
        assert_eq!(
            exercise(LossPlan {
                syn: true,
                data: true
            }),
            exercise(LossPlan {
                syn: true,
                data: true
            })
        );
    }
    #[test]
    fn packet_frontdoor_rejects_overflow_and_oversize() {
        let mut queue = PacketQueue::new();
        assert!(!queue.offer(Host::Server, &[0; MAX_FRAME_BYTES + 1]));
        for _ in 0..MAX_PACKETS {
            assert!(queue.offer(Host::Server, b"packet"));
        }
        assert!(!queue.offer(Host::Server, b"overflow"));
        assert_eq!(queue.packets.len(), MAX_PACKETS);
        assert_eq!(queue.high_water, MAX_PACKETS);
        assert_eq!(queue.rejected, 2);
    }
}
