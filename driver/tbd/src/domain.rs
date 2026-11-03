//! One Thunderbolt domain: the NHI, its control channel, the host router, and the hosts connected to its ports.
//!
//! Threads: `tbd-ctl-rx` receives control frames; `tbd-events` answers the other hosts' XDomain requests and
//! acknowledges plug events (it never waits on a request of its own); `tbd-cm` handles plugs and unplugs; each
//! peer gets a discovery thread, and its services (ThunderboltIP, streams) threads of their own.

use crate::ctl::{self, Ctl, Event, Frame, PDF_ERROR, PDF_EVENT, PDF_XDOMAIN_REQ, PDF_XDOMAIN_RESP};
use crate::net;
use crate::nhi::{self, Nhi};
use crate::property::Dir;
use crate::stream::{self, StreamSpec};
use crate::switch::{PathConfig, Router, STATE_UNPLUGGED, WIDTH_DUAL, WIDTH_SINGLE};
use crate::uuid::Uuid;
use crate::xdomain::{self as xd, LinkState, Remote};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

const RETRIES: u32 = 10;
const XD_TIMEOUT: Duration = Duration::from_millis(1000);
const BONDING_TIMEOUT: Duration = Duration::from_secs(10);
/// Link width values of `LANE_ADP_CS_1` current width (`enum tb_link_width`).
const LINK_WIDTH_SINGLE: u32 = 1;
const LINK_WIDTH_DUAL: u32 = 2;

/// What this host offers to the hosts it connects to.
#[derive(Clone)]
pub struct Config {
    /// The NHI's register BAR, for stream clients.
    pub bar: Option<Bar>,
    pub hostname: String,
    pub network: Option<net::NetConfig>,
    pub streams: Vec<StreamSpec>,
    /// Bring every connected lane down and up at start, so the other hosts forget the driver that ran before.
    pub bounce: bool,
    pub bonding: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Discovering,
    /// We have the higher UUID and wait for the other host to ask for a link width.
    BondingHigh,
    Enumerated,
    Failed,
    Gone,
}

pub struct PeerState {
    pub phase: Phase,
    pub remote_uuid: Option<Uuid>,
    pub remote: Option<Remote>,
    /// Width the other host asked for in a link state change request (BondingHigh).
    pub target_width: Option<u8>,
    pub bonded: bool,
    pub in_hops: BTreeSet<u32>,
    pub out_hops: BTreeSet<u32>,
    /// Set by a properties changed request: discovery reads the properties again.
    pub reread: bool,
}

/// Another host on one of our ports.
pub struct Peer {
    pub port: u32,
    pub route: u64,
    pub state: Mutex<PeerState>,
    pub cv: Condvar,
}

impl Peer {
    pub fn gone(&self) -> bool {
        self.state.lock().unwrap().phase == Phase::Gone
    }

    pub fn remote_uuid(&self) -> Uuid {
        self.state.lock().unwrap().remote_uuid.unwrap_or_default()
    }

    pub fn name(&self) -> String {
        let s = self.state.lock().unwrap();
        s.remote.as_ref().and_then(|r| r.device_name.clone()).unwrap_or_else(|| format!("port{}", self.port))
    }

    /// Takes a HopID from 8..=max (`tb_xdomain_alloc_in_hopid`/`_out_hopid`); `want` asks for that one.
    pub fn alloc_hop(&self, input: bool, want: Option<u32>, max: u32) -> Result<u32> {
        let mut s = self.state.lock().unwrap();
        let set = if input { &mut s.in_hops } else { &mut s.out_hops };
        let hop = match want {
            Some(h) => {
                ensure!((xd::MIN_HOPID..=max).contains(&h), "HopID {h} outside {}..={max}", xd::MIN_HOPID);
                ensure!(!set.contains(&h), "HopID {h} is taken");
                h
            }
            None => (xd::MIN_HOPID..=max).find(|h| !set.contains(h)).context("no free HopID")?,
        };
        set.insert(hop);
        Ok(hop)
    }

    pub fn free_hop(&self, input: bool, hop: u32) {
        let mut s = self.state.lock().unwrap();
        if input { s.in_hops.remove(&hop) } else { s.out_hops.remove(&hop) };
    }
}

pub struct Domain {
    pub nhi: Arc<Nhi>,
    pub ctl: Arc<Ctl>,
    pub router: RwLock<Router>,
    pub local_uuid: Uuid,
    pub config: Config,
    peers: Mutex<BTreeMap<u32, Arc<Peer>>>,
    /// Bumped whenever the property block changes; the other hosts only take newer generations.
    generation: Mutex<u32>,
    /// Per-peer extra properties of services (the stream directories with their HopIDs).
    service_props: Mutex<BTreeMap<(u32, String), Dir>>,
    plug_tx: Mutex<Sender<(u32, bool)>>,
}

/// Packets the control channel hands to the event thread instead of to a waiting request.
fn is_event(f: &Frame) -> bool {
    match f.pdf {
        PDF_EVENT => true,
        // Asynchronous notifications (`tb_async_error`).
        PDF_ERROR => f.words.len() >= 3 && matches!(f.words[2] & 0xff, 1 | 12 | 13 | 32..=39),
        PDF_XDOMAIN_REQ => true,
        PDF_XDOMAIN_RESP => net::is_request(f),
        _ => false,
    }
}

impl Domain {
    /// Takes over the router behind `nhi`: control channel up, router enumerated with plug events on.
    pub fn start(nhi: Arc<Nhi>, irqs: Irqs, config: Config) -> Result<Arc<Domain>> {
        irqs.attach(&nhi);
        let (ctl, events) = Ctl::start(&nhi, is_event)?;
        let router = Router::open(ctl.clone(), 0).context("host router")?;
        println!(
            "host router {:04x}:{:04x}, {} ports, upstream port {}",
            router.vendor, router.device, router.max_port, router.upstream_port
        );
        router.configure().context("configuring the host router")?;
        let local_uuid = router.uuid()?;
        println!("host UUID {local_uuid}");
        let (plug_tx, plug_rx) = mpsc::channel();
        let gen0 = Uuid::random().words()[0] & 0x7fff_ffff;
        let d = Arc::new(Domain {
            nhi,
            ctl,
            router: RwLock::new(router),
            local_uuid,
            config,
            peers: Mutex::new(BTreeMap::new()),
            generation: Mutex::new(gen0),
            service_props: Mutex::new(BTreeMap::new()),
            plug_tx: Mutex::new(plug_tx),
        });
        let e = d.clone();
        std::thread::Builder::new().name("tbd-events".into()).spawn(move || e.event_loop(events))?;
        let c = d.clone();
        std::thread::Builder::new().name("tbd-cm".into()).spawn(move || c.cm_loop(plug_rx))?;
        Ok(d)
    }

    fn peer_by_route(&self, route: u64) -> Option<Arc<Peer>> {
        self.peers.lock().unwrap().get(&((route & 0x3f) as u32)).filter(|p| p.route == route).cloned()
    }

    // ---- events ----

    fn event_loop(self: Arc<Self>, events: Receiver<Event>) {
        for f in events {
            let route = f.route();
            match f.pdf {
                PDF_EVENT => {
                    let port = f.words[2] & 0x3f;
                    let unplug = f.words[2] >> 31 != 0;
                    if let Err(e) = self.ctl.ack_plug(route, port, unplug) {
                        eprintln!("tbd: acking plug event on {route:x}:{port}: {e:#}");
                    }
                    if route == 0 {
                        let _ = self.plug_tx.lock().unwrap().send((port, unplug));
                    }
                }
                PDF_ERROR => {
                    eprintln!("tbd: notification {:#x} from {route:x}:{}", f.words[2] & 0xff, (f.words[2] >> 8) & 0x3f);
                    let _ = self.ctl.ack_notification(route);
                }
                PDF_XDOMAIN_REQ if xd::xdp_type(&f).is_some() => {
                    if let Err(e) = self.handle_xdp(&f) {
                        eprintln!("tbd: XDomain request from {route:x}: {e:#}");
                    }
                }
                _ if net::is_request(&f) => match self.peer_by_route(route) {
                    Some(p) => net::handle_request(&self, &p, &f),
                    None => eprintln!("tbd: ThunderboltIP request from unknown route {route:x}"),
                },
                _ => {
                    if ctl::trace() {
                        eprintln!("tbd: unhandled event pdf {} from {route:x}", f.pdf);
                    }
                }
            }
        }
    }

    /// Answers a request of the XDomain discovery protocol (`tb_xdp_handle_request`).
    fn handle_xdp(self: &Arc<Self>, f: &Frame) -> Result<()> {
        let route = f.route();
        let seq = xd::seq_of(f);
        let ty = xd::xdp_type(f).unwrap();
        let peer = self.peer_by_route(route);
        // The packet's length field must match its size.
        if (f.words[2] & 0x3f) as usize != f.words.len() - 3 {
            return Ok(());
        }
        match ty {
            xd::UUID_REQUEST | xd::UUID_REQUEST_OLD => {
                self.ctl.send(&xd::uuid_response(route, seq, self.local_uuid))?;
                if let Some(p) = peer {
                    let failed = p.state.lock().unwrap().phase == Phase::Failed;
                    if failed {
                        println!("{}: UUID request after a failed discovery, starting again", p.port);
                        self.spawn_discovery(p);
                    }
                }
            }
            xd::PROPERTIES_REQUEST if f.words.len() >= 17 => {
                let dst = Uuid::from_words(&f.words[12..16]);
                let src = Uuid::from_words(&f.words[8..12]);
                if dst != self.local_uuid {
                    self.ctl.send(&xd::error_response(route, seq, xd::ERROR_UNKNOWN_DOMAIN))?;
                    return Ok(());
                }
                let (block, generation) = self.property_block((route & 0x3f) as u32);
                let offset = (f.words[16] & 0xffff) as usize;
                if offset < block.len() {
                    self.ctl.send(&xd::properties_response(route, seq, self.local_uuid, src, offset, &block, generation))?;
                }
            }
            xd::PROPERTIES_CHANGED_REQUEST => {
                self.ctl.send(&xd::properties_changed_response(route, seq))?;
                if let Some(p) = peer {
                    let mut s = p.state.lock().unwrap();
                    s.reread = true;
                    p.cv.notify_all();
                }
            }
            xd::LINK_STATE_STATUS_REQUEST => {
                let Some(p) = peer else {
                    return self.ctl.send(&xd::error_response(route, seq, xd::ERROR_NOT_READY));
                };
                let ls = self.link_state(p.port)?;
                let (phase, target) = {
                    let s = p.state.lock().unwrap();
                    (s.phase, s.target_width)
                };
                if phase == Phase::BondingHigh && target.is_some_and(|t| t != ls.tlw) {
                    self.ctl.send(&xd::error_response(route, seq, xd::ERROR_NOT_READY))?;
                } else {
                    self.ctl.send(&xd::link_state_status_response(route, seq, ls))?;
                }
            }
            xd::LINK_STATE_CHANGE_REQUEST => {
                let accepted = peer.as_ref().is_some_and(|p| {
                    let mut s = p.state.lock().unwrap();
                    if s.phase == Phase::BondingHigh && f.words.len() >= 9 {
                        s.target_width = Some(f.words[8] as u8);
                        p.cv.notify_all();
                        true
                    } else {
                        false
                    }
                });
                if accepted {
                    self.ctl.send(&xd::link_state_change_response(route, seq, 0))?;
                } else {
                    self.ctl.send(&xd::error_response(route, seq, xd::ERROR_NOT_READY))?;
                }
            }
            _ => self.ctl.send(&xd::error_response(route, seq, xd::ERROR_NOT_SUPPORTED))?,
        }
        Ok(())
    }

    fn link_state(&self, port: u32) -> Result<LinkState> {
        let r = self.router.read().unwrap();
        let [cs0, cs1] = r.lane_cs(r.port(port))?;
        Ok(LinkState {
            slw: ((cs0 >> 20) & 0x3f) as u8,
            sls: ((cs0 >> 16) & 0xf) as u8,
            tls: (cs1 & 0xf) as u8,
            tlw: ((cs1 >> 4) & 3) as u8,
        })
    }

    // ---- local properties ----

    /// Our property block for the host on `port`, and its generation (`update_property_block`): Linux's
    /// standard entries, then the service directories, then the host name and our max HopID.
    pub fn property_block(&self, port: u32) -> (Vec<u32>, u32) {
        let mut root = Dir::new(None);
        root.add_immediate("vendorid", 0x1d6b).add_text("vendorid", "tbd").add_immediate("deviceid", 0x0004);
        root.add_immediate("devicerv", 0x8000_0100);
        if self.config.network.is_some() {
            root.add_dir("network", net::property_dir());
        }
        if !self.config.streams.is_empty() {
            let mut dir = stream::property_dir();
            for ((p, _), sub) in self.service_props.lock().unwrap().iter() {
                if *p == port {
                    dir.props.extend(sub.props.iter().cloned());
                }
            }
            root.add_dir("stream", dir);
        }
        let max_in = {
            let r = self.router.read().unwrap();
            r.ports.get(port as usize - 1).map(|p| p.max_in_hop).unwrap_or(xd::DEFAULT_MAX_HOPID)
        };
        root.add_text("deviceid", &self.config.hostname).add_immediate("maxhopid", max_in);
        (root.format(), *self.generation.lock().unwrap())
    }

    /// Sets a service's own entries in its directory for one peer (a stream's HopIDs) and tells the peer.
    pub fn set_service_props(self: &Arc<Self>, peer: &Arc<Peer>, key: &str, entries: Dir) {
        self.service_props.lock().unwrap().insert((peer.port, key.to_owned()), entries);
        *self.generation.lock().unwrap() += 1;
        let (d, p) = (self.clone(), peer.clone());
        std::thread::spawn(move || d.notify_properties_changed(&p));
    }

    fn notify_properties_changed(&self, peer: &Peer) {
        for retry in 0..RETRIES {
            if peer.gone() {
                return;
            }
            let req = xd::properties_changed_request(peer.route, retry % 4, self.local_uuid);
            match self.ctl.request(&req, xd::xdp_matcher(peer.route, retry % 4, xd::PROPERTIES_CHANGED_RESPONSE), XD_TIMEOUT) {
                Ok(f) if xd::xdp_error(&f).is_none() => return,
                _ => std::thread::sleep(XD_TIMEOUT),
            }
        }
        eprintln!("{}: properties changed notification not acknowledged", peer.port);
    }

    // ---- connection manager ----

    fn cm_loop(self: Arc<Self>, plugs: Receiver<(u32, bool)>) {
        let lanes: Vec<u32> = {
            let r = self.router.read().unwrap();
            r.ports.iter().filter(|p| p.is_lane() && p.primary() && p.cap_phy != 0).map(|p| p.number).collect()
        };
        if self.config.bounce {
            for &n in &lanes {
                if let Err(e) = self.bounce(n) {
                    eprintln!("tbd: bouncing port {n}: {e:#}");
                }
            }
        }
        for &n in &lanes {
            self.scan(n);
        }
        for (port, unplug) in plugs {
            let primary = {
                let r = self.router.read().unwrap();
                r.ports.get(port as usize - 1).filter(|p| p.is_lane()).map(|p| p.primary())
            };
            match primary {
                Some(true) if unplug => self.unplugged(port),
                Some(true) => self.scan(port),
                _ => {}
            }
        }
    }

    /// Takes the lane down and up again, so the host on the other end sees an unplug.
    fn bounce(&self, n: u32) -> Result<()> {
        let r = self.router.read().unwrap();
        let port = r.port(n).clone();
        if r.port_state(&port)? == STATE_UNPLUGGED {
            return Ok(());
        }
        println!("port {n}: taking the link down and up");
        let dual = r.port(port.dual()).clone();
        r.set_lane_enabled(&dual, false)?;
        r.set_lane_enabled(&port, false)?;
        std::thread::sleep(Duration::from_millis(500));
        r.set_lane_enabled(&port, true)?;
        r.set_lane_enabled(&dual, true)?;
        Ok(())
    }

    /// `tb_scan_port` for a lane of the host router: a host on the other end becomes a peer.
    fn scan(self: &Arc<Self>, n: u32) {
        if self.peers.lock().unwrap().contains_key(&n) {
            return;
        }
        let up = {
            let r = self.router.read().unwrap();
            r.wait_for_port(r.port(n))
        };
        match up {
            Ok(true) => {}
            Ok(false) => return,
            Err(e) => {
                eprintln!("tbd: port {n}: {e:#}");
                return;
            }
        }
        let route = n as u64;
        // A router of our own domain would answer; another host's doesn't (`tb_switch_alloc` fails).
        match self.ctl.read(route, 0, ctl::SPACE_SWITCH, 0, 1) {
            Ok(_) => {
                println!("port {n}: a device, not a host; tbd connects only to other hosts");
                return;
            }
            Err(e) => {
                if ctl::trace() {
                    eprintln!("tbd: port {n}: switch read: {e}");
                }
            }
        }
        {
            let r = self.router.read().unwrap();
            if let Err(e) = r.set_xdomain(r.port(n), true) {
                eprintln!("tbd: port {n}: configuring the link for XDomain: {e:#}");
            }
        }
        let peer = Arc::new(Peer {
            port: n,
            route,
            state: Mutex::new(PeerState {
                phase: Phase::Discovering,
                remote_uuid: None,
                remote: None,
                target_width: None,
                bonded: false,
                in_hops: BTreeSet::new(),
                out_hops: BTreeSet::new(),
                reread: false,
            }),
            cv: Condvar::new(),
        });
        self.peers.lock().unwrap().insert(n, peer.clone());
        println!("port {n}: another host, starting discovery");
        self.spawn_discovery(peer);
    }

    fn unplugged(&self, n: u32) {
        let Some(peer) = self.peers.lock().unwrap().remove(&n) else { return };
        println!("port {n}: {} unplugged", peer.name());
        self.release(&peer);
    }

    /// Ends everything on a peer's port: its services stop (they see the peer gone and take their paths down),
    /// lane bonding is undone and the lane is no longer marked as connected to another domain
    /// (`tb_xdomain_link_exit`, `tb_lc_unconfigure_xdomain`).
    fn release(&self, peer: &Peer) {
        let n = peer.port;
        {
            let mut s = peer.state.lock().unwrap();
            s.phase = Phase::Gone;
            peer.cv.notify_all();
        }
        let r = self.router.read().unwrap();
        let port = r.port(n).clone();
        let _ = r.set_xdomain(&port, false);
        let bonded = peer.state.lock().unwrap().bonded;
        let dual = r.port(port.dual()).clone();
        if bonded {
            let _ = r.set_lane_bonding(&port, false);
            let _ = r.set_target_width(&dual, WIDTH_SINGLE);
            let _ = r.set_target_width(&port, WIDTH_SINGLE);
        }
        let _ = r.set_lane_enabled(&dual, true);
    }

    /// Leaves the router for the next driver: every peer released, plug events off, and each connected link taken
    /// down and up once, so the other hosts see an unplug and discover whoever runs next from scratch.
    pub fn shutdown(&self) {
        let peers: Vec<_> = std::mem::take(&mut *self.peers.lock().unwrap()).into_values().collect();
        for p in &peers {
            net::logout(self, p);
        }
        for p in &peers {
            self.release(p);
        }
        // Give the service threads a moment to take their paths down.
        std::thread::sleep(Duration::from_millis(300));
        let r = self.router.read().unwrap();
        if r.cap_plug_events != 0
            && let Ok(v) = r.read1(0, ctl::SPACE_SWITCH, r.cap_plug_events + 1)
        {
            let _ = r.write(0, ctl::SPACE_SWITCH, r.cap_plug_events + 1, &[v | 0x7c]);
        }
        drop(r);
        for p in &peers {
            if let Err(e) = self.bounce(p.port) {
                eprintln!("tbd: port {}: {e:#}", p.port);
            }
        }
    }

    fn spawn_discovery(self: &Arc<Self>, peer: Arc<Peer>) {
        peer.state.lock().unwrap().phase = Phase::Discovering;
        let d = self.clone();
        let name = format!("tbd-xd{}", peer.port);
        std::thread::Builder::new()
            .name(name)
            .spawn(move || {
                if let Err(e) = d.discover(&peer) {
                    eprintln!("port {}: discovery failed: {e:#}", peer.port);
                    let mut s = peer.state.lock().unwrap();
                    if s.phase != Phase::Gone {
                        s.phase = Phase::Failed;
                    }
                }
            })
            .expect("spawn discovery");
    }

    /// Sends an XDP request with Linux's retries (1 s apart); `None` if the host keeps failing or is gone.
    fn xdp_request(&self, peer: &Peer, make: impl Fn(u32) -> Frame, response: u32) -> Result<Frame> {
        let mut last = String::new();
        for retry in 0..RETRIES {
            if peer.gone() {
                bail!("unplugged");
            }
            let seq = retry % 4;
            match self.ctl.request(&make(seq), xd::xdp_matcher(peer.route, seq, response), XD_TIMEOUT) {
                Ok(f) => match xd::xdp_error(&f) {
                    None => return Ok(f),
                    Some(xd::ERROR_NOT_SUPPORTED) => bail!("not supported by the other host"),
                    Some(e) => last = format!("error response {e}"),
                },
                Err(e) => last = e.to_string(),
            }
            std::thread::sleep(XD_TIMEOUT);
        }
        bail!("no answer after {RETRIES} tries ({last})")
    }

    /// `tb_xdomain_state_work`, start to finish.
    fn discover(self: &Arc<Self>, peer: &Arc<Peer>) -> Result<()> {
        std::thread::sleep(Duration::from_millis(100));
        let f = self.xdp_request(peer, |s| xd::uuid_request(peer.route, s), xd::UUID_RESPONSE).context("UUID")?;
        ensure!(f.words.len() >= 14, "short UUID response");
        let remote = Uuid::from_words(&f.words[8..12]);
        println!("port {}: remote UUID {remote}", peer.port);
        let looped = remote == self.local_uuid;
        {
            let mut s = peer.state.lock().unwrap();
            if s.remote_uuid.is_some_and(|u| u != remote) {
                bail!("a different host is on the port now");
            }
            s.remote_uuid = Some(remote);
        }
        {
            let (d, p) = (self.clone(), peer.clone());
            std::thread::spawn(move || d.notify_properties_changed(&p));
        }
        let bonding_possible = self.config.bonding && !looped;
        if bonding_possible {
            std::thread::sleep(Duration::from_millis(100));
            if let Err(e) = self.bond(peer, remote) {
                println!("port {}: lanes not bonded: {e:#}", peer.port);
            }
        }
        self.read_properties(peer)?;
        if bonding_possible && !peer.state.lock().unwrap().bonded {
            // Linux disables the second lane when the link stays single.
            let r = self.router.read().unwrap();
            let dual = r.port(r.port(peer.port).dual()).clone();
            let _ = r.set_lane_enabled(&dual, false);
        }
        self.enumerated(peer);
        // Read the properties again whenever the other host says they changed.
        loop {
            let mut s = peer.state.lock().unwrap();
            while !s.reread && s.phase != Phase::Gone {
                s = peer.cv.wait(s).unwrap();
            }
            if s.phase == Phase::Gone {
                return Ok(());
            }
            s.reread = false;
            drop(s);
            std::thread::sleep(Duration::from_millis(100));
            if let Err(e) = self.read_properties(peer) {
                eprintln!("port {}: reading changed properties: {e:#}", peer.port);
            }
        }
    }

    /// Lane bonding over XDomain: the host with the lower UUID asks for dual width and bonds; the other sets its
    /// target width when asked and waits for the link.
    fn bond(&self, peer: &Peer, remote: Uuid) -> Result<()> {
        let route = peer.route;
        let status = |what: &str| -> Result<LinkState> {
            let f = self.xdp_request(peer, |s| xd::link_state_status_request(route, s), xd::LINK_STATE_STATUS_RESPONSE)
                .with_context(|| format!("link state status ({what})"))?;
            let (st, ls) = xd::parse_link_state_status(&f).context("short link state status")?;
            ensure!(st == 0, "link state status {st}");
            Ok(ls)
        };
        let ls = status("first")?;
        ensure!(ls.slw as u32 >= 2, "the other host's lane is single only (width {:#x})", ls.slw);
        let (port, dual) = {
            let r = self.router.read().unwrap();
            let p = r.port(peer.port).clone();
            (p.clone(), r.port(p.dual()).clone())
        };
        if self.local_uuid > remote {
            peer.state.lock().unwrap().phase = Phase::BondingHigh;
            let deadline = Instant::now() + XD_TIMEOUT * RETRIES;
            let width = {
                let mut s = peer.state.lock().unwrap();
                while s.target_width.is_none() && s.phase == Phase::BondingHigh && Instant::now() < deadline {
                    s = peer.cv.wait_timeout(s, Duration::from_millis(100)).unwrap().0;
                }
                s.target_width.take()
            };
            peer.state.lock().unwrap().phase = Phase::Discovering;
            let (width, mask) = match width.map(u32::from) {
                Some(WIDTH_SINGLE) => (WIDTH_SINGLE, LINK_WIDTH_SINGLE),
                Some(WIDTH_DUAL) => (WIDTH_DUAL, LINK_WIDTH_DUAL | 4 | 8),
                Some(w) => bail!("asked for link width {w:#x}"),
                None => bail!("no link state change request"),
            };
            let r = self.router.read().unwrap();
            r.set_target_width(&dual, width)?;
            r.set_target_width(&port, width)?;
            ensure!(r.wait_for_width(&port, mask, BONDING_TIMEOUT)?, "link did not reach width {mask:#x}");
        } else {
            let tls = {
                let r = self.router.read().unwrap();
                (r.lane_cs(&port)?[1] & 0xf) as u8
            };
            std::thread::sleep(Duration::from_millis(100));
            let f = self.xdp_request(
                peer,
                |s| xd::link_state_change_request(route, s, WIDTH_DUAL as u8, tls),
                xd::LINK_STATE_CHANGE_RESPONSE,
            )?;
            ensure!(f.words.get(8) == Some(&0), "link state change refused ({:?})", f.words.get(8));
            status("after the change")?;
            // tb_xdomain_lane_bonding_enable.
            let r = self.router.read().unwrap();
            r.set_lane_enabled(&dual, true)?;
            let mut up = false;
            for _ in 0..10 {
                if (2..=6).contains(&r.port_state(&dual)?) {
                    up = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            ensure!(up, "second lane did not come up");
            if r.link_width(&port)? == LINK_WIDTH_SINGLE {
                r.set_target_width(&port, WIDTH_DUAL)?;
            }
            let dual_single = r.link_width(&dual)? == LINK_WIDTH_SINGLE;
            if dual_single {
                r.set_target_width(&dual, WIDTH_DUAL)?;
                r.set_lane_bonding(&port, true)?;
            }
            ensure!(r.wait_for_width(&port, LINK_WIDTH_DUAL | 4 | 8, BONDING_TIMEOUT)?, "lanes did not bond");
        }
        let mut r = self.router.write().unwrap();
        r.update_credits(port.number)?;
        r.update_credits(dual.number)?;
        peer.state.lock().unwrap().bonded = true;
        println!("port {}: lanes bonded", peer.port);
        Ok(())
    }

    /// Reads the other host's whole property block, page by page (`tb_xdp_properties_request`).
    fn read_properties(&self, peer: &Peer) -> Result<()> {
        let remote = peer.remote_uuid();
        let mut block: Vec<u32> = Vec::new();
        let mut total = None;
        let mut generation = 0;
        while total.is_none_or(|t| block.len() < t) {
            let off = block.len() as u16;
            let f = self
                .xdp_request(peer, |s| xd::properties_request(peer.route, s, self.local_uuid, remote, off), xd::PROPERTIES_RESPONSE)
                .context("properties")?;
            let (o, t, g, data) = xd::parse_properties_response(&f).context("malformed properties response")?;
            ensure!(o == off as usize, "properties at offset {o}, asked for {off}");
            ensure!(t <= xd::PROPERTIES_MAX_LENGTH && t > 0, "property block of {t} dwords");
            ensure!(!data.is_empty() || block.len() >= t, "empty properties page");
            let t = *total.get_or_insert(t);
            let take = data.len().min(t - block.len());
            block.extend_from_slice(&data[..take]);
            generation = g;
        }
        let mut s = peer.state.lock().unwrap();
        if s.remote.as_ref().is_some_and(|r| generation <= r.generation) {
            return Ok(());
        }
        let dir = Dir::parse(&block).context("parsing the other host's properties")?;
        let r = Remote::from_dir(dir, generation).context("properties without vendorid/deviceid")?;
        let first = s.remote.is_none();
        let keys = |r: &Remote| r.dir.dirs().map(|(k, _)| k.to_owned()).collect::<Vec<_>>();
        let changed = s.remote.as_ref().is_none_or(|old| keys(old) != keys(&r));
        if changed {
            println!(
            "port {}: {} {} (vendor {:#x}, device {:#x}, max HopID {}), services: {}",
            peer.port,
            r.vendor_name.as_deref().unwrap_or("?"),
            r.device_name.as_deref().unwrap_or("?"),
            r.vendor,
            r.device,
            r.max_hopid,
            r.dir.dirs().map(|(k, _)| k).collect::<Vec<_>>().join(", ")
            );
        }
        s.remote = Some(r);
        if !first {
            peer.cv.notify_all();
        }
        Ok(())
    }

    fn enumerated(self: &Arc<Self>, peer: &Arc<Peer>) {
        let remote = {
            let mut s = peer.state.lock().unwrap();
            s.phase = Phase::Enumerated;
            peer.cv.notify_all();
            s.remote.clone().unwrap()
        };
        if let Some(cfg) = &self.config.network
            && remote.service("network", 1).is_some()
        {
            net::spawn(self.clone(), peer.clone(), cfg.clone());
        }
        if remote.dir.dir("stream").is_some() {
            for spec in &self.config.streams {
                stream::spawn(self.clone(), peer.clone(), spec.clone());
            }
        }
    }

    // ---- DMA paths ----

    /// Routes a DMA tunnel through the host router (`tb_tunnel_alloc_dma` + activate): frames from NHI ring
    /// `tx_ring` leave the lane with HopID `tx_path`, and frames arriving with HopID `rx_path` go to NHI ring
    /// `rx_ring`.
    pub fn enable_paths(&self, peer: &Peer, tx_path: u32, tx_ring: u32, rx_path: u32, rx_ring: u32) -> Result<()> {
        let r = self.router.read().unwrap();
        let lane = r.port(peer.port).clone();
        let nhi = r.nhi_port().context("no NHI port")?.clone();
        // RX: lane -> NHI. Flow control on ingress only (`tb_dma_init_rx_path`).
        let rx = PathConfig { credits: r.rx_credits(&lane), ingress_fc: true, egress_fc: false };
        r.activate_hop(lane.number, rx_path, nhi.number, rx_ring, &rx).context("DMA RX path")?;
        // TX: NHI -> lane, flow control both ways.
        let tx = PathConfig { credits: r.tx_credits(&nhi), ingress_fc: true, egress_fc: true };
        if let Err(e) = r.activate_hop(nhi.number, tx_ring, lane.number, tx_path, &tx) {
            let _ = r.deactivate_hop(lane.number, rx_path);
            return Err(e.context("DMA TX path"));
        }
        Ok(())
    }

    pub fn disable_paths(&self, peer: &Peer, tx_ring: u32, rx_path: u32) {
        let r = self.router.read().unwrap();
        let lane = peer.port;
        let Some(nhi) = r.nhi_port().map(|p| p.number) else { return };
        if let Err(e) = r.deactivate_hop(lane, rx_path) {
            eprintln!("tbd: RX path {rx_path} on port {lane}: {e:#}");
        }
        if let Err(e) = r.deactivate_hop(nhi, tx_ring) {
            eprintln!("tbd: TX path from ring {tx_ring}: {e:#}");
        }
    }
}

/// The NHI's register BAR as a client maps it: a descriptor and the offset in it.
#[derive(Clone)]
pub struct Bar {
    pub fd: Arc<std::os::fd::OwnedFd>,
    pub offset: u64,
    pub size: u64,
}

/// The NHI's interrupts: a vector per ring (MSI-X), or one MSI for all of them.
pub enum Irqs {
    Msix(Vec<gpud_platform::IrqEvent>),
    Msi(gpud_platform::IrqEvent),
}

impl Irqs {
    pub fn attach(self, nhi: &Arc<Nhi>) {
        match self {
            Irqs::Msix(v) => nhi.use_msix(v),
            Irqs::Msi(e) => nhi.spawn_irq_thread(e),
        }
    }
}

/// Opens the NHI at `addr` (VFIO on Linux, pcid on dross) and returns it with its interrupts (MSI-X if the platform routes it,
/// else MSI; `TBD_MSI=1` forces MSI).
#[cfg(target_os = "linux")]
pub fn open_nhi(addr: &str) -> Result<(Arc<Nhi>, Irqs, Box<dyn std::any::Any>, Bar)> {
    use gpud_platform::pci::{COMMAND_BUS_MASTER, COMMAND_MEMORY, CONFIG_COMMAND, CONFIG_DEVICE_ID, CONFIG_VENDOR_ID};
    use gpud_platform::Region;
    let (container, dev) = gpud_platform::open(addr.parse()?, 1 << 20..1 << 40)?;
    let (vendor, device) = (dev.config_u16(CONFIG_VENDOR_ID)?, dev.config_u16(CONFIG_DEVICE_ID)?);
    nhi::check_supported(vendor, device)?;
    let cmd = dev.config_u16(CONFIG_COMMAND)?;
    dev.set_config_u16(CONFIG_COMMAND, cmd | COMMAND_MEMORY | COMMAND_BUS_MASTER)?;
    let mmio = dev.map(Region::Bar(0))?;
    let (fd, offset, size) = dev.export_region(Region::Bar(0))?;
    let bar = Bar { fd: Arc::new(fd), offset, size };
    let dma: Arc<dyn gpud_platform::DmaSpace> = Arc::from(container);
    let nhi = Nhi::new(mmio, dma, nhi::is_falcon_ridge(vendor, device))?;
    let msix = if std::env::var_os("TBD_MSI").is_some() { Err(gpud_platform::Error::Unsupported("TBD_MSI".into())) } else { dev.enable_msix(16) };
    let irqs = match msix {
        Ok(v) => Irqs::Msix(v),
        Err(e) => {
            eprintln!("tbd: no MSI-X ({e}), one MSI for all rings");
            Irqs::Msi(dev.enable_msi()?)
        }
    };
    Ok((nhi, irqs, Box::new(dev), bar))
}
