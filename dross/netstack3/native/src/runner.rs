//! Finite two-endpoint native proof over simulated Ethernet links.
//! Logical time is deterministic; random values come from Oscar secure entropy.
use crate::{
    Bindings, DeviceState, Identifier, TxAllocator,
    buffers::{ClientBuffers, SocketExtra, TCP_BYTES},
    time::Instant,
};
use alloc::vec;
use alloc::vec::Vec;
use core::num::{NonZeroU16, NonZeroU64, NonZeroUsize};
use net_types::ethernet::Mac;
use net_types::ip::{AddrSubnet, Ipv4, Ipv4Addr, Mtu, Subnet};
use net_types::{SpecifiedAddr, UnicastAddr, ZonedAddr};
use netstack3_base::{RemoveResourceResult, RngContext as _, TcpSocketState};
use netstack3_core::ip::IidSecret;
use netstack3_core::socket::ShutdownType;
use netstack3_core::{CoreApi, StackState, StackStateBuilder};
use netstack3_device::ethernet::{
    EthernetCreationProperties, EthernetDeviceId, EthernetLinkDevice, MaxEthernetFrameSize,
    RecvEthernetFrameMeta,
};
use netstack3_device::{DeviceId, WeakDeviceId};
use netstack3_ip::device::IpDeviceConfigurationUpdate;
use netstack3_ip::{Entry, Generation, Metric, RawMetric};
use netstack3_tcp::{AcceptError, Buffer, TcpSocketId};
use packet::Buf;

type Socket = TcpSocketId<Ipv4, WeakDeviceId<Bindings>, Bindings>;
const PORT: NonZeroU16 = NonZeroU16::new(22222).unwrap();
const MAX_STEPS: usize = 16_384;
const SECOND: u64 = 1_000_000_000;

struct Endpoint {
    core: StackState<Bindings>,
    bindings: Bindings,
    device: Option<EthernetDeviceId<Bindings>>,
}
impl Endpoint {
    fn new(index: u8) -> Self {
        let mut bindings = Bindings::new();
        let mut builder = StackStateBuilder::default();
        // Production core initializes IPv6 state even in this IPv4-only proof.
        // Each isolated endpoint gets a secret from the native entropy syscall.
        builder
            .ipv6_builder()
            .slaac_stable_secret_key(IidSecret::new_random(&mut bindings.rng()));
        let core = builder.build_with_ctx(&mut bindings);
        let mac = UnicastAddr::new(Mac::new([2, 0, 0, 0, 0, index + 1])).unwrap();
        let peer_mac = UnicastAddr::new(Mac::new([2, 0, 0, 0, 0, 2 - index])).unwrap();
        let local = Ipv4Addr::new([192, 0, 2, index + 1]);
        let peer = Ipv4Addr::new([192, 0, 2, 2 - index]);
        let device = core
            .api(&mut bindings)
            .device::<EthernetLinkDevice>()
            .add_device(
                Identifier(NonZeroU64::new(u64::from(index) + 1).unwrap()),
                EthernetCreationProperties {
                    mac,
                    max_frame_size: MaxEthernetFrameSize::from_mtu(Mtu::new(1500)).unwrap(),
                    tx_offload_spec: Default::default(),
                },
                RawMetric(100),
                DeviceState,
                TxAllocator,
            );
        let any_device: DeviceId<Bindings> = device.clone().into();
        core.api(&mut bindings)
            .device_ip::<Ipv4>()
            .update_configuration(
                &any_device,
                IpDeviceConfigurationUpdate {
                    ip_enabled: Some(true),
                    ..Default::default()
                }
                .into(),
            )
            .unwrap();
        core.api(&mut bindings)
            .device_ip::<Ipv4>()
            .add_ip_addr_subnet(&any_device, AddrSubnet::new(local, 24).unwrap())
            .unwrap();
        core.api(&mut bindings)
            .neighbor::<Ipv4, EthernetLinkDevice>()
            .insert_static_entry(&device, peer, peer_mac)
            .unwrap();
        let table = core.api(&mut bindings).routes::<Ipv4>().main_table_id();
        core.api(&mut bindings).routes::<Ipv4>().set_routes(
            &table,
            vec![
                Entry {
                    subnet: Subnet::new(Ipv4Addr::new([192, 0, 2, 0]), 24).unwrap(),
                    device: any_device,
                    gateway: None,
                    metric: Metric::ExplicitMetric(RawMetric(0)),
                    route_preference: Default::default(),
                }
                .with_generation(Generation::initial()),
            ],
        );
        Self {
            core,
            bindings,
            device: Some(device),
        }
    }
    fn api(&mut self) -> CoreApi<'_, &mut Bindings> {
        self.core.api(&mut self.bindings)
    }
    fn cleanup(mut self) {
        let table = self.api().routes::<Ipv4>().main_table_id();
        self.api().routes::<Ipv4>().set_routes(&table, Vec::new());
        drop(table);
        let device = self.device.take().unwrap();
        let removed = self
            .api()
            .device::<EthernetLinkDevice>()
            .remove_device(device);
        match removed {
            RemoveResourceResult::Removed(_) => {}
            RemoveResourceResult::Deferred(receiver) => {
                netstack3_base::DeferredResourceRemovalContext::defer_removal::<DeviceState>(
                    &mut self.bindings,
                    receiver,
                );
            }
        }
        drop(self.core);
        assert_eq!(
            self.bindings.timer_count(),
            0,
            "timers retained after dropping native core"
        );
        assert_eq!(
            self.bindings.deferred_pending(),
            0,
            "native resources retained"
        );
    }
}

#[derive(Clone, Copy, Default)]
pub struct Loss {
    pub syn: bool,
    pub data: bool,
}
#[derive(Debug, Default)]
pub struct Report {
    pub frames: usize,
    pub timers: usize,
    pub client_syns: usize,
    pub dropped_syn: usize,
    pub dropped_data: usize,
    pub data_retransmissions: u64,
    pub simulated_ns: u64,
    pub frame_high_water: u64,
    pub timer_high_water: usize,
    pub ring_wraps: [usize; 4],
}
struct Network {
    endpoints: [Endpoint; 2],
    now: Instant,
    steps: usize,
    loss: Loss,
    report: Report,
}
impl Network {
    fn new(loss: Loss) -> Self {
        Self {
            endpoints: [Endpoint::new(0), Endpoint::new(1)],
            now: Instant(0),
            steps: 0,
            loss,
            report: Report::default(),
        }
    }
    fn state(&mut self, endpoint: usize, socket: &Socket) -> TcpSocketState {
        self.endpoints[endpoint]
            .api()
            .tcp::<Ipv4>()
            .get_tcp_info(socket)
            .state
    }
    fn step(&mut self, deadline: Instant) -> bool {
        self.steps += 1;
        assert!(
            self.steps <= MAX_STEPS,
            "native proof event budget exceeded"
        );
        for sender in 0..2 {
            if let Some(frame) = self.endpoints[sender].bindings.pop_frame() {
                self.report.frames += 1;
                let tcp = tcp_segment(&frame.0);
                if sender == 0 && tcp.is_some_and(|(syn, _)| syn) {
                    self.report.client_syns += 1;
                }
                if sender == 0 && self.loss.syn && tcp.is_some_and(|(syn, _)| syn) {
                    self.loss.syn = false;
                    self.report.dropped_syn += 1;
                    return true;
                }
                if sender == 0 && self.loss.data && tcp.is_some_and(|(_, data)| data) {
                    self.loss.data = false;
                    self.report.dropped_data += 1;
                    return true;
                }
                let receiver = 1 - sender;
                let device = self.endpoints[receiver].device.as_ref().unwrap().clone();
                self.endpoints[receiver]
                    .api()
                    .device::<EthernetLinkDevice>()
                    .receive_frame(
                        RecvEthernetFrameMeta {
                            device_id: device,
                            parsing_context: Default::default(),
                            gso_info: None,
                        },
                        Buf::new(frame.0, ..),
                    );
                return true;
            }
        }
        let next = (0..2)
            .filter_map(|index| {
                self.endpoints[index]
                    .bindings
                    .next_deadline()
                    .map(|at| (at, index))
            })
            .min_by_key(|(at, index)| (*at, *index));
        let Some((at, endpoint)) = next else {
            return false;
        };
        if at > deadline {
            return false;
        }
        assert!(at >= self.now, "native timer moved backwards");
        self.now = at;
        for endpoint in &mut self.endpoints {
            endpoint.bindings.set_now(at);
        }
        let (dispatch, unique) = self.endpoints[endpoint].bindings.take_due_timer().unwrap();
        self.endpoints[endpoint]
            .api()
            .handle_timer(dispatch, unique);
        self.report.timers += 1;
        true
    }
    fn until(&mut self, seconds: u64, mut done: impl FnMut(&mut Self) -> bool) {
        let deadline = Instant(self.now.0.checked_add(seconds * SECOND).unwrap());
        for _ in 0..MAX_STEPS {
            if done(self) {
                return;
            }
            assert!(self.step(deadline), "native proof condition timed out");
        }
        panic!("native proof exceeded event budget");
    }
    fn finish(mut self) -> Report {
        while self
            .endpoints
            .iter()
            .any(|endpoint| endpoint.bindings.frame_count() > 0)
        {
            assert!(self.step(self.now));
        }
        for endpoint in &self.endpoints {
            let (frames, timers) = endpoint.bindings.high_water();
            self.report.frame_high_water = self.report.frame_high_water.max(frames);
            self.report.timer_high_water = self.report.timer_high_water.max(timers);
        }
        self.report.simulated_ns = self.now.0;
        for endpoint in self.endpoints {
            endpoint.cleanup();
        }
        self.report
    }
}

fn tcp_segment(frame: &[u8]) -> Option<(bool, bool)> {
    if frame.len() < 34 || frame[12..14] != [8, 0] || frame[23] != 6 {
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

struct Stream {
    endpoint: usize,
    socket: Socket,
    buffers: ClientBuffers,
    send_closed: bool,
}
impl Stream {
    fn write(&self, net: &mut Network, bytes: &[u8]) -> Result<usize, ()> {
        if self.send_closed {
            return Err(());
        }
        let written = net.endpoints[self.endpoint]
            .api()
            .tcp::<Ipv4>()
            .with_send_buffer(&self.socket, |buffer| buffer.enqueue(bytes))
            .ok_or(())?;
        net.endpoints[self.endpoint]
            .api()
            .tcp::<Ipv4>()
            .do_send(&self.socket);
        Ok(written)
    }
    fn read(&self, net: &mut Network, destination: &mut [u8]) -> Option<usize> {
        let count = self.buffers.read(destination);
        if count > 0 {
            net.endpoints[self.endpoint]
                .api()
                .tcp::<Ipv4>()
                .on_receive_buffer_read(&self.socket);
            return Some(count);
        }
        assert!(
            net.endpoints[self.endpoint]
                .api()
                .tcp::<Ipv4>()
                .get_socket_error(&self.socket)
                .is_none(),
            "native TCP error"
        );
        match net.state(self.endpoint, &self.socket) {
            TcpSocketState::CloseWait
            | TcpSocketState::LastAck
            | TcpSocketState::TimeWait
            | TcpSocketState::Closing
            | TcpSocketState::Close => Some(0),
            _ => None,
        }
    }
    fn shutdown_send(&mut self, net: &mut Network) {
        assert_eq!(
            net.endpoints[self.endpoint]
                .api()
                .tcp::<Ipv4>()
                .shutdown(&self.socket, ShutdownType::Send),
            Ok(true)
        );
        self.send_closed = true;
    }
    fn close(self, net: &mut Network) {
        net.endpoints[self.endpoint]
            .api()
            .tcp::<Ipv4>()
            .close(self.socket);
    }
}
fn receive_exact(net: &mut Network, stream: &Stream, expected: &[u8]) {
    let mut offset = 0;
    let mut scratch = [0; TCP_BYTES];
    net.until(30, |net| {
        match stream.read(net, &mut scratch) {
            Some(0) => panic!("native EOF before exact payload"),
            Some(count) => {
                assert!(offset + count <= expected.len());
                assert_eq!(&scratch[..count], &expected[offset..offset + count]);
                offset += count;
            }
            None => {}
        }
        offset == expected.len()
    });
}
fn eof(net: &mut Network, stream: &Stream) -> bool {
    let mut scratch = [0; TCP_BYTES];
    match stream.read(net, &mut scratch) {
        Some(0) => true,
        None => false,
        Some(_) => panic!("unexpected native payload before EOF"),
    }
}

pub fn run(loss: Loss) -> Report {
    let mut net = Network::new(loss);
    let listener = net.endpoints[1]
        .api()
        .tcp::<Ipv4>()
        .create(SocketExtra::default());
    net.endpoints[1]
        .api()
        .tcp::<Ipv4>()
        .bind(&listener, None, Some(PORT))
        .unwrap();
    net.endpoints[1]
        .api()
        .tcp::<Ipv4>()
        .listen(&listener, NonZeroUsize::new(1).unwrap())
        .unwrap();
    assert!(matches!(
        net.endpoints[1].api().tcp::<Ipv4>().accept(&listener),
        Err(AcceptError::WouldBlock)
    ));
    let active_extra = SocketExtra::default();
    let client = net.endpoints[0]
        .api()
        .tcp::<Ipv4>()
        .create(active_extra.clone());
    net.endpoints[0]
        .api()
        .tcp::<Ipv4>()
        .connect(
            &client,
            Some(ZonedAddr::Unzoned(
                SpecifiedAddr::new(Ipv4Addr::new([192, 0, 2, 2])).unwrap(),
            )),
            PORT,
        )
        .unwrap();
    let mut accepted = None;
    net.until(30, |net| {
        match net.endpoints[1].api().tcp::<Ipv4>().accept(&listener) {
            Ok(value) => {
                accepted = Some(value);
                true
            }
            Err(AcceptError::WouldBlock) => false,
            Err(_) => panic!("native accept failed"),
        }
    });
    assert_eq!(net.state(0, &client), TcpSocketState::Established);
    if loss.syn {
        assert_eq!(net.report.dropped_syn, 1);
        assert!(net.report.client_syns >= 2);
        assert!(net.report.timers > 0);
    }
    let (server, peer, server_buffers) = accepted.unwrap();
    assert_eq!(*peer.ip.addr(), Ipv4Addr::new([192, 0, 2, 1]));
    let mut client = Stream {
        endpoint: 0,
        socket: client,
        buffers: active_extra.take().unwrap(),
        send_closed: false,
    };
    let mut server = Stream {
        endpoint: 1,
        socket: server,
        buffers: server_buffers,
        send_closed: false,
    };
    assert!(!eof(&mut net, &client));
    assert!(!eof(&mut net, &server));
    let request: Vec<u8> = (0..if loss.data { 512 } else { 9217 })
        .map(|n| (n * 37 + 11) as u8)
        .collect();
    let response: Vec<u8> = (0..4099).map(|n| (n * 19 + 3) as u8).collect();
    let retransmits_before = net.endpoints[0]
        .api()
        .tcp::<Ipv4>()
        .get_tcp_info(&client.socket)
        .retransmits;
    let timers_before = net.report.timers;
    assert_eq!(client.write(&mut net, &request), Ok(request.len()));
    assert_eq!(server.write(&mut net, &response), Ok(response.len()));
    receive_exact(&mut net, &server, &request);
    receive_exact(&mut net, &client, &response);
    let retransmits_after = net.endpoints[0]
        .api()
        .tcp::<Ipv4>()
        .get_tcp_info(&client.socket)
        .retransmits;
    net.report.data_retransmissions = retransmits_after - retransmits_before;
    if loss.data {
        assert_eq!(net.report.dropped_data, 1);
        assert!(net.report.data_retransmissions > 0);
        assert!(net.report.timers > timers_before);
    }
    // Drain acknowledgements before refilling the fixed rings. A16001-byte
    // second transfer wraps even after the512-byte loss-recovery request.
    net.until(30, |net| {
        net.endpoints[0]
            .api()
            .tcp::<Ipv4>()
            .with_send_buffer(&client.socket, |buffer| buffer.limits().len == 0)
            == Some(true)
            && net.endpoints[1]
                .api()
                .tcp::<Ipv4>()
                .with_send_buffer(&server.socket, |buffer| buffer.limits().len == 0)
                == Some(true)
    });
    let wrapped_request: Vec<u8> = (0..16001).map(|n| (n * 23 + 7) as u8).collect();
    let wrapped_response: Vec<u8> = (0..16001).map(|n| (n * 41 + 5) as u8).collect();
    assert_eq!(
        client.write(&mut net, &wrapped_request),
        Ok(wrapped_request.len())
    );
    assert_eq!(
        server.write(&mut net, &wrapped_response),
        Ok(wrapped_response.len())
    );
    receive_exact(&mut net, &server, &wrapped_request);
    receive_exact(&mut net, &client, &wrapped_response);
    net.report.ring_wraps = [
        net.endpoints[0]
            .api()
            .tcp::<Ipv4>()
            .with_send_buffer(&client.socket, |buffer| buffer.wraps())
            .unwrap(),
        client.buffers.receive.0.lock().wraps(),
        net.endpoints[1]
            .api()
            .tcp::<Ipv4>()
            .with_send_buffer(&server.socket, |buffer| buffer.wraps())
            .unwrap(),
        server.buffers.receive.0.lock().wraps(),
    ];
    assert!(
        net.report.ring_wraps.iter().all(|count| *count > 0),
        "all native send/receive rings must wrap"
    );

    client.shutdown_send(&mut net);
    assert_eq!(client.write(&mut net, b"closed"), Err(()));
    net.until(30, |net| eof(net, &server));
    assert_eq!(net.state(1, &server.socket), TcpSocketState::CloseWait);
    let after_fin = b"native reply after peer half-close\0exact";
    assert_eq!(server.write(&mut net, after_fin), Ok(after_fin.len()));
    receive_exact(&mut net, &client, after_fin);
    server.shutdown_send(&mut net);
    net.until(30, |net| eof(net, &client));
    assert!(eof(&mut net, &client));
    assert!(eof(&mut net, &server));
    // Production socket IDs deliberately hide their weak-reference helper.
    // Observe the public state API through TIME-WAIT expiry before consuming
    // the owned IDs; this also verifies that the close timer completes.
    net.until(240, |net| {
        net.state(0, &client.socket) == TcpSocketState::Close
            && net.state(1, &server.socket) == TcpSocketState::Close
    });
    client.close(&mut net);
    server.close(&mut net);
    net.endpoints[1].api().tcp::<Ipv4>().close(listener);
    net.finish()
}
