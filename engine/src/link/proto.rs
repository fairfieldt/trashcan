//! The link protocol: one ordered, reliable byte stream each way between two processes, carried by the cable's
//! USB4STREAM when both ends have it and by TCP otherwise (see `mod.rs` for the overview).
//!
//! Every byte has a sequence number (its offset in the stream), and every record says which bytes it carries and
//! which of the other side's bytes have arrived (`ack`). The receiver takes bytes in order from whichever path
//! brings them first and drops what it already has. A link with a stream at both ends carries its data on the stream
//! only (until an epoch is up the data waits; TCP carries the control records); once the stream has carried data, a
//! stream that fails ends the link with an error, and so does one that is not up within `stream_wait`.
//!
//! On the stream the sender keeps within the other side's receive ring (credits, counted in frames: `tx`/`rx`), so
//! the ring cannot overflow; a gap (lost or damaged frame) is answered with a NAK and the sender goes back to the
//! first byte the receiver lacks (go-back-N). Without progress for `rto` it goes back by itself; without progress
//! for `dead` the stream is given up. Each attempt to use a stream is an epoch, opened over TCP (START) by the
//! connecting side and confirmed on the stream itself (PROBE both ways); records of other sessions or epochs are
//! ignored, so frames that sat in a ring or a closed device from earlier never get mixed in.
//!
//! The hold. On the Mac Pros a direction of a cable can keep the last frames written to it from the reader until
//! more frames follow: 0, 1, 2 or 3 of them, the same for every frame of an open (docs/kernel.md). So a write on the
//! stream ends with that many more frames, acknowledgements ("pushes"). The connecting side measures the hold of its
//! sending direction once the epoch is up (hold probes: a ping written alone, then one more frame at a time until
//! the other side reports it read) and again with every keepalive of an idle link; the accepting side reads the hold
//! of its direction from those pings, which say how many of its frames have been read. Until a side has measured
//! it sends `PUSH` frames. Between probes the estimate follows the frame counters every record carries
//! (`hold_sample`): the frames written here less the frames the other side says it has read are never fewer than
//! the hold, so a smaller number lowers it; and an acknowledgement says for how long its sender had found nothing
//! to read, so the frames written here before that and still unread were held, and their number raises it. A wrong
//! estimate costs time, never data: a side whose data stays unacknowledged while nothing moves writes one more
//! frame, a ping (a nudge, after `Tuning::nudge`, long before the retransmission timer). The estimate goes up only
//! on a second sign of held frames soon after a first: a frame that sticks once in ten thousand (interrupt-driven
//! stream devices do that, docs/kernel.md) is cheaper to nudge than to push every write for. What was not measured
//! by a probe is tried again without, after `HOLD_STICK` and then at ever longer intervals.
use super::{
    device::{close_later, Frames, Found},
    record::{crc, crcs, find_magic, kind, Header, FRAME, HEADER, PING, REPORT, STREAM_PAYLOAD, TCP_PAYLOAD},
};
use std::{
    io::{self, ErrorKind, IoSlice, Read, Write},
    net::{SocketAddr, TcpStream},
    os::unix::io::AsRawFd,
    time::{Duration, Instant},
};

/// Timing and sizes of the protocol. The defaults are the product's; tests shorten the timers.
#[derive(Clone, Debug)]
pub struct Tuning {
    /// Go back to the first unacknowledged byte on the stream after this long without progress.
    pub rto: Duration,
    /// Give the stream up (and end the link) after this long without progress.
    pub dead: Duration,
    /// Keepalive on an idle stream.
    pub ping: Duration,
    /// Give the stream up after this long without a frame.
    pub idle_dead: Duration,
    /// Give a stream epoch up if both directions are not confirmed within this.
    pub probe_timeout: Duration,
    /// Wait between attempts to open a stream epoch, doubling up to `backoff_max`.
    pub backoff_min: Duration,
    pub backoff_max: Duration,
    /// Give a TCP path up after this long with unacknowledged data, if another path remains.
    pub tcp_dead: Duration,
    /// Acknowledge received data after at most this long.
    pub ack_delay: Duration,
    /// Return credits (report frames read) at least every this many frames.
    pub credit_frames: u32,
    /// Frames of the other side's ring kept free of data, for control records and stragglers.
    pub margin: u32,
    /// Received bytes held for the application before reading stops (the other side is told: FULL).
    pub inq_limit: usize,
    /// Fault injection for tests and `trashcan link-test`: send this many frames past the credits.
    pub overdrive: u32,
    /// A link with a stream at both ends whose stream is not up this long after it started (or after an epoch that
    /// had not carried data yet failed) ends.
    pub stream_wait: Duration,
    /// With data unacknowledged (or a hold probe unanswered), nothing written and nothing heard for this long, write
    /// one more frame: the other side's receiver may be holding the last ones back.
    pub nudge: Duration,
    /// Frames to write after every write on the stream, instead of measuring the hold (tests and
    /// `trashcan link-test --push`).
    pub push: Option<u32>,
    /// Time the device's reads and writes (`Stats::read_ns` and `write_ns`; `trashcan link-test --profile`).
    pub profile: bool,
}

impl Tuning {
    /// Data frames that may be in flight to a ring of `ring` frames.
    pub fn window(&self, ring: u32) -> u32 {
        ring.saturating_sub(self.margin).max(1) + self.overdrive
    }
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            rto: Duration::from_millis(5),
            // Long enough for a peer that does not run its link for a while: a node thread that waits for the disk
            // while the model loads held its link for about a second, and the nodes' SSDs answered nothing for up
            // to 8 s after their page cache was dropped (results/deploy/link-fallback-20261026). Losses are repaired
            // by the go-back after `rto`; these only give up a stream that no longer works, which ends the link.
            dead: Duration::from_secs(10),
            ping: Duration::from_secs(1),
            idle_dead: Duration::from_secs(10),
            probe_timeout: Duration::from_secs(3),
            backoff_min: Duration::from_secs(1),
            backoff_max: Duration::from_secs(30),
            tcp_dead: Duration::from_secs(2),
            ack_delay: Duration::from_micros(200),
            credit_frames: 32,
            margin: 32,
            inq_limit: 32 << 20,
            overdrive: 0,
            stream_wait: Duration::from_secs(20),
            // Longer than a peer takes to answer when only its maintenance thread turns the link (1 ms): a nudge
            // that was not needed costs a frame each way, and one that was costs this wait once, after which the
            // estimate covers it.
            nudge: Duration::from_micros(1500),
            push: None,
            profile: false,
        }
    }
}

/// Frames written after a write on the stream while the hold of its direction is not known (while an epoch opens,
/// and until the first hold probe is answered), so that its last frames are read: the receiving NHIs here read a
/// frame only once up to three more have arrived behind it (docs/kernel.md).
const PUSH: usize = 3;

/// The largest hold the link assumes (and so the most frames it writes after a write). Three is the most measured.
const HOLD_MAX: u32 = 4;

/// Hold probes the connecting side runs when its epoch comes up, before it calls its direction measured; the
/// estimate is the smallest of what they found (a frame in flight while the answer was written makes one read high,
/// never low).
const HOLD_PROBES: u32 = 3;

/// Nudges per stall (they stop when the other side shows progress): enough to release `HOLD_MAX` held frames.
const NUDGE_MAX: u32 = HOLD_MAX;

/// A hold probe starts when this side has written nothing for this long, so that its ping arrives alone and the
/// answers to earlier frames are in.
const QUIET: Duration = Duration::from_micros(150);

/// A frame of ours counts as held by the other side when it was written this much or more before the other side
/// found nothing to read (by its record's `age`, see `stream_control`): more than a frame takes to get there, with
/// room for the jitter of the path.
const HELD_MARGIN: Duration = Duration::from_micros(100);

/// Writes on the stream remembered (their first frame and when they began and ended), for `held`.
const TX_LOG: usize = 64;

/// An estimate that went up without a hold probe behind it stays for this long (lower readings are ignored
/// meanwhile) and is then tried one lower; if it goes up again it stays twice as long, up to `HOLD_STICK_MAX`. A
/// hold that is there costs two stalls each time it is tried without, an unneeded push costs microseconds on every
/// write.
const HOLD_STICK: Duration = Duration::from_secs(1);
const HOLD_STICK_MAX: Duration = Duration::from_secs(256);

/// Signs of held frames (`hold_sample`), each within `SIGN_FRAMES` frames written here of the one before, that raise
/// the estimate: a hold shows at every write that is followed by a pause, while one sign alone can be a frame that
/// stuck once (a missed interrupt at the receiver), a peer that was late, or a thread that lost its core.
const SIGNS: u32 = 2;
const SIGN_FRAMES: u32 = 64;

/// Flag of a stream record written without a read just before (`send_now`) when the last read was longer ago than
/// `STALE_AFTER`: its `rx` may miss frames that have arrived since, so it says nothing about what is held.
pub const STALE: u8 = 8;
const STALE_AFTER: Duration = Duration::from_micros(200);

/// Flag of a stream record: its sender has measured the hold of its direction (`Link::settle` waits for both).
pub const MEASURED: u8 = 4;

/// Stream probes go every 2 ms for the first `PROBE_FAST` of an epoch (the other end's device may open late: an open
/// waits for an earlier close of it), then every 50 ms.
const PROBE_FAST: Duration = Duration::from_millis(500);

/// An epoch that takes longer than this to come up (or to fail) is described in its message.
const SLOW_EPOCH: Duration = Duration::from_millis(100);

/// How long a link whose other side closed its stream device waits for word over TCP of why (`peer_closed_stream`).
const EOF_WAIT: Duration = Duration::from_millis(200);

/// A pause of the link's turns this long, or a turn this long, is said on stderr (`State::slow`).
const SLOW_TURN: Duration = Duration::from_millis(20);

/// An epoch that carries one way only is given up on evidence that the other side is going on without us, so that
/// the next epoch can open (the one-cable churn and the main suite saw epochs where one side's frames stopped
/// arriving), from its reports over TCP: either `EVIDENCE` of them, received `ONE_WAY` or more after our first
/// frame, all show none of our frames read; or they show frames written to us that have not arrived, for `ONE_WAY`
/// and over `EVIDENCE` reports. Its stream records are no such evidence (a receiver may hold records back until more
/// follow, so an old one can arrive late), and silence is none: a peer that is busy or stalled says nothing, and its
/// epoch waits for `probe_timeout`. Probes go every 2 ms with pushes and reports every `REPORT_EVERY`, so a working
/// direction shows within a few milliseconds.
const ONE_WAY: Duration = Duration::from_millis(50);
const EVIDENCE: u32 = 3;
/// While an epoch opens (and, once up, while the other side still probes), each side reports its frames of the epoch
/// read and written over TCP this often.
const REPORT_EVERY: Duration = Duration::from_millis(10);

/// A link message, written to stderr by a thread of its own: on the node Mac Pros a write to the log directory stalls
/// for 0.3-1.3 s several times a minute (results/kernel/usb4stream-busypoll-20261014/logcheck), and a link that waited
/// for one while it held its state answered nothing meanwhile (the main suite's one-way epoch at 17:12: the accepting
/// side's "stream up" line, then about 0.6 s of silence). A write that took over 50 ms says so in the next message.
pub fn say(msg: String) {
    static LOG: std::sync::OnceLock<std::sync::mpsc::Sender<String>> = std::sync::OnceLock::new();
    let tx = LOG.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let spawned = std::thread::Builder::new().name("link-log".into()).spawn(move || {
            let mut slow = None;
            for m in rx {
                let t = Instant::now();
                match slow.take() {
                    Some(d) => eprintln!("{m} (the previous link message took {d:.0?} to write)"),
                    None => eprintln!("{m}"),
                }
                let took = t.elapsed();
                if took > Duration::from_millis(50) {
                    slow = Some(took);
                }
            }
        });
        if spawned.is_err() {
            // No thread to write from: the message goes out here, as before.
            let (tx, _) = std::sync::mpsc::channel();
            return tx;
        }
        tx
    });
    if let Err(e) = tx.send(msg) {
        eprintln!("{}", e.0);
    }
}

/// The time of day, for link messages (to line them up across machines).
pub fn clock() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs() as libc::time_t;
    // SAFETY: localtime_r writes the broken-down time into `tm`, which outlives the call.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::localtime_r(&secs, &mut tm) };
    format!("{:02}:{:02}:{:02}.{:03}", tm.tm_hour, tm.tm_min, tm.tm_sec, now.subsec_millis())
}

/// Counters of one link, since it opened.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Stats {
    /// Payload bytes sent on the stream and over TCP, retransmissions included.
    pub sent_stream: u64,
    pub sent_tcp: u64,
    /// Payload bytes taken in order from each path.
    pub got_stream: u64,
    pub got_tcp: u64,
    pub frames: u64,
    /// Times the stream sender went back (NAK or timeout).
    pub rewinds: u64,
    pub naks_sent: u64,
    pub naks_got: u64,
    /// Stream records that failed their payload CRC, bytes skipped to find the next record, records of other
    /// sessions or epochs, and data that had already arrived.
    pub crc_errors: u64,
    pub resync_bytes: u64,
    pub stale: u64,
    pub duplicates: u64,
    pub gaps: u64,
    /// Stream epochs opened, confirmed (up), lost while up, and given up before they were up.
    pub epochs: u64,
    pub ups: u64,
    pub fallbacks: u64,
    pub failed_epochs: u64,
    /// Milliseconds from the link's start to its stream's first epoch up (none: never up).
    pub first_up_ms: Option<f64>,
    /// TCP paths given up while others remained.
    pub tcp_failovers: u64,
    /// Frames written after a write on the stream so that its last frame is read (`send`).
    pub pushes: u64,
    /// The hold of this side's sending direction as now estimated, in frames (none: not measured, `PUSH` frames
    /// follow every write), and the hold probes of this side that were answered.
    pub hold: Option<u32>,
    pub hold_probes: u64,
    /// Times the estimate went up after the first measurement, and frames written because data stayed
    /// unacknowledged or a hold probe unanswered (`Tuning::nudge`).
    pub hold_raises: u64,
    pub nudges: u64,
    /// With `Tuning::profile`: calls of the device's read and write, reads that found nothing, and the time in them.
    pub reads: u64,
    pub reads_empty: u64,
    pub read_ns: u64,
    pub writes: u64,
    pub write_ns: u64,
    /// Acknowledgements and credits sent over TCP because the stream had no room for them.
    pub tcp_reports: u64,
}

/// A byte buffer filled at the end and consumed at the front.
#[derive(Default)]
pub struct Buf {
    data: Vec<u8>,
    start: usize,
    end: usize,
}

impl Buf {
    pub fn new(capacity: usize) -> Self {
        Self { data: vec![0; capacity], start: 0, end: 0 }
    }
    pub fn bytes(&self) -> &[u8] {
        &self.data[self.start..self.end]
    }
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
    pub fn consume(&mut self, n: usize) {
        self.start += n;
        if self.start == self.end {
            self.start = 0;
            self.end = 0;
        }
    }
    /// Free space at the end, moving the contents to the front when that helps.
    pub fn space(&mut self) -> &mut [u8] {
        if self.start > 0 && (self.end == self.data.len() || self.start >= self.data.len() / 2) {
            self.data.copy_within(self.start..self.end, 0);
            self.end -= self.start;
            self.start = 0;
        }
        &mut self.data[self.end..]
    }
    pub fn filled(&mut self, n: usize) {
        self.end += n;
    }
    pub fn push(&mut self, b: &[u8]) {
        if self.data.len() - self.end < b.len() {
            self.space();
            if self.data.len() - self.end < b.len() {
                let want = (self.end + b.len()).max(2 * self.data.len()).max(1 << 16);
                self.data.resize(want, 0);
            }
        }
        self.data[self.end..self.end + b.len()].copy_from_slice(b);
        self.end += b.len();
    }
    /// Move up to `out.len()` bytes out.
    pub fn take(&mut self, out: &mut [u8]) -> usize {
        let n = out.len().min(self.len());
        out[..n].copy_from_slice(&self.data[self.start..self.start + n]);
        self.consume(n);
        n
    }
    pub fn clear(&mut self) {
        self.start = 0;
        self.end = 0;
    }
}

/// Where in-order payload goes while the application waits for it; the rest waits in the link's queue.
pub struct Sink<'a> {
    pub buf: &'a mut [u8],
    pub filled: usize,
}

impl Sink<'_> {
    pub fn none() -> Sink<'static> {
        Sink { buf: &mut [], filled: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Off,
    /// A device open for this epoch is due (`job`).
    Opening,
    Probing,
    Up,
}

/// Work done outside the lock (it can take seconds): open the stream device for an epoch.
pub struct Job {
    pub epoch: u32,
    /// When the epoch asked for its device.
    pub at: Instant,
    pub local: SocketAddr,
    /// On the accepting side, the connecting side's controller UUIDs (its `local`, `peer`), which must be the other
    /// way round here: the same cable.
    pub expect: Option<(String, String)>,
}

pub struct TcpPath {
    pub sock: TcpStream,
    rbuf: Buf,
    wbuf: Vec<u8>,
    wpos: usize,
    pub alive: bool,
    pub name: String,
}

impl TcpPath {
    pub fn new(sock: TcpStream) -> Self {
        let name = match (sock.local_addr(), sock.peer_addr()) {
            (Ok(l), Ok(p)) => format!("{l} -> {p}"),
            _ => "tcp".into(),
        };
        Self { sock, rbuf: Buf::new(256 << 10), wbuf: Vec::new(), wpos: 0, alive: true, name }
    }
    fn pending(&self) -> usize {
        self.wbuf.len() - self.wpos
    }
    /// Write what is pending; false if the socket failed.
    fn flush(&mut self) -> io::Result<bool> {
        let mut progress = false;
        while self.wpos < self.wbuf.len() {
            match self.sock.write(&self.wbuf[self.wpos..]) {
                Ok(0) => return Err(ErrorKind::WriteZero.into()),
                Ok(n) => {
                    self.wpos += n;
                    progress = true;
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        if self.wpos == self.wbuf.len() {
            self.wbuf.clear();
            self.wpos = 0;
        }
        Ok(progress)
    }
}

pub struct StreamPath {
    pub phase: Phase,
    pub epoch: u32,
    pub dev: Option<Box<dyn Frames>>,
    pub found: Option<Found>,
    /// The other side's receive ring, in frames.
    peer_ring: u32,
    /// Frames written in this epoch; the other side's report of how many of them left its ring.
    tx: u32,
    peer_rx: u32,
    /// One more than the number of the last frame read in this epoch, and the value last reported.
    rx_last: u32,
    rx_reported: u32,
    rbuf: Buf,
    /// Next byte to send on the stream.
    next: u64,
    probe_start: Instant,
    last_probe: Instant,
    probes: u32,
    probe_reply: bool,
    /// Frames of this epoch have arrived; the other side has read one of ours.
    heard: bool,
    peer_heard: bool,
    last_rx: Instant,
    last_ping: Instant,
    nak: Option<(u64, Instant)>,
    nak_due: bool,
    rewind: Option<(u64, Instant)>,
    stalled_since: Option<Instant>,
    last_rto: Instant,
    retry_at: Option<Instant>,
    backoff: Duration,
    up_since: Instant,
    pub job: Option<Job>,
    /// When the epoch began (START sent or received), its device opened and its first record arrived; records and
    /// ends of file read in it (for messages about epochs that fail or come up late).
    opened_at: Instant,
    dev_at: Option<Instant>,
    heard_at: Option<Instant>,
    records: u64,
    eofs: u64,
    /// Epochs failed since the stream was last up.
    failed_since_up: u32,
    /// Evidence about the epoch's directions (see `ONE_WAY`): when our first frame of it went; the other side's
    /// reports since `ONE_WAY` after that which showed none of our frames read; its last report (our
    /// frames read, its frames written), how many came, and since when (and over how many reports) its frames
    /// written to us have not arrived.
    first_tx_at: Option<Instant>,
    zero_late: u32,
    rep_rx: u32,
    rep_tx: u32,
    reports: u32,
    rep_new: bool,
    behind_since: Option<Instant>,
    behind_reports: u32,
    /// Our reports: when the last went, and whether a probe of the other side has come since (once up: it still
    /// probes, so it has not heard us yet).
    last_report: Instant,
    probe_seen: bool,
    /// The longest time between two turns of the link (by any thread) while the epoch opened: a busy or stalled
    /// process shows here.
    max_gap: Duration,
    hdrs: Vec<[u8; HEADER]>,
    lens: Vec<usize>,
    crcs: Vec<u32>,
    spans: Vec<(usize, Header)>,
    sums: Vec<u32>,
    /// The last reason logged for having no stream.
    why: String,
    /// The hold of this side's sending direction in this epoch (see the module documentation): the estimate (none
    /// until the first hold probe is answered), when it last went up, and the hold probes answered.
    hold: Option<u32>,
    hold_raised: Option<Instant>,
    hold_stick: Duration,
    hold_probes: u32,
    /// What the last exact measurement of the hold found (a hold probe answered, or the two bounds of
    /// `hold_sample` agreeing): an estimate above it is tried one lower when `hold_stick` is over.
    hold_probed: Option<u32>,
    /// Signs of held frames close to one another (`SIGNS`): how many, the fewest frames any of them showed held,
    /// and the frames written when the last came. The first record that acknowledged the data of the stall under
    /// way (its `rx`, and whether it was stale), and whether the estimate has gone up before in this epoch.
    signs: u32,
    sign_held: u32,
    sign_tx: u32,
    resolved: Option<(u32, bool)>,
    hold_went_up: bool,
    /// When the device was last read (a turn began), whether the frames of this turn are written without a read
    /// after a pause (`STALE`), and how late a frame may show on this device (`Frames::late`).
    turn_at: Instant,
    stale: bool,
    late: Duration,
    /// A hold probe under way: the other side's `rx` that shows its ping read, and whether another frame of ours
    /// followed the ping at once (then the reading says nothing about a frame written alone).
    probe: Option<u32>,
    probe_spoiled: bool,
    probe_at: Instant,
    /// When this side last wrote a frame of the epoch, and the frames written since the other side last showed
    /// progress because nothing moved (`Tuning::nudge`).
    last_tx: Instant,
    nudges: u32,
    /// The bytes queued when the first of those was written (the stall is over once they are acknowledged) and the
    /// frames written when the last was.
    nudge_mark: u64,
    nudge_tx: u32,
    /// From the acknowledgements among the records just read that said how long their sender had read nothing
    /// (`hold_sample`): the most frames of ours any of them shows held, and that number from the last of them that
    /// left no other frame of ours unread (the hold exactly).
    held_seen: Option<u32>,
    exact_seen: Option<u32>,
    /// This turn (begun then) read the device until it had nothing more, so an acknowledgement written now can say
    /// for how long nothing has been readable: since `read_at`, when a read last returned frames.
    dry: Option<Instant>,
    read_at: Instant,
    /// The last writes of the epoch: the number of their first frame, when the write began and when it returned.
    txlog: std::collections::VecDeque<(u32, Instant, Instant)>,
    /// The other side has measured its direction.
    pub peer_measured: bool,
}

impl StreamPath {
    fn new(now: Instant, backoff: Duration, epoch: u32) -> Self {
        Self {
            phase: Phase::Off,
            epoch,
            dev: None,
            found: None,
            peer_ring: 256,
            tx: 0,
            peer_rx: 0,
            rx_last: 0,
            rx_reported: 0,
            rbuf: Buf::new(2 << 20),
            next: 0,
            probe_start: now,
            last_probe: now,
            probes: 0,
            probe_reply: false,
            heard: false,
            peer_heard: false,
            last_rx: now,
            last_ping: now,
            nak: None,
            nak_due: false,
            rewind: None,
            stalled_since: None,
            last_rto: now,
            retry_at: None,
            backoff,
            up_since: now,
            job: None,
            opened_at: now,
            dev_at: None,
            heard_at: None,
            records: 0,
            eofs: 0,
            failed_since_up: 0,
            first_tx_at: None,
            zero_late: 0,
            rep_rx: 0,
            rep_tx: 0,
            reports: 0,
            rep_new: false,
            behind_since: None,
            behind_reports: 0,
            last_report: now,
            probe_seen: false,
            max_gap: Duration::ZERO,
            hdrs: Vec::new(),
            lens: Vec::new(),
            crcs: Vec::new(),
            spans: Vec::new(),
            sums: Vec::new(),
            why: String::new(),
            hold: None,
            hold_raised: None,
            hold_stick: HOLD_STICK,
            hold_probes: 0,
            hold_probed: None,
            signs: 0,
            sign_held: 0,
            sign_tx: 0,
            resolved: None,
            hold_went_up: false,
            turn_at: now,
            stale: false,
            late: Duration::ZERO,
            probe: None,
            probe_spoiled: false,
            probe_at: now,
            last_tx: now,
            nudges: 0,
            nudge_mark: 0,
            nudge_tx: 0,
            held_seen: None,
            exact_seen: None,
            dry: None,
            read_at: now,
            txlog: std::collections::VecDeque::with_capacity(TX_LOG),
            peer_measured: false,
        }
    }

    /// A new epoch begins (its START sent or received).
    fn begin(&mut self, now: Instant) {
        self.opened_at = now;
        self.dev_at = None;
        self.heard_at = None;
        self.records = 0;
        self.eofs = 0;
        self.first_tx_at = None;
        self.zero_late = 0;
        self.rep_rx = 0;
        self.rep_tx = 0;
        self.reports = 0;
        self.rep_new = false;
        self.behind_since = None;
        self.behind_reports = 0;
        self.probe_seen = false;
        self.max_gap = Duration::ZERO;
        self.hold = None;
        self.hold_raised = None;
        self.hold_stick = HOLD_STICK;
        self.hold_probes = 0;
        self.hold_probed = None;
        self.signs = 0;
        self.sign_held = 0;
        self.sign_tx = 0;
        self.resolved = None;
        self.hold_went_up = false;
        self.turn_at = now;
        self.stale = false;
        self.probe = None;
        self.probe_spoiled = false;
        self.probe_at = now;
        self.last_tx = now;
        self.nudges = 0;
        self.held_seen = None;
        self.exact_seen = None;
        self.dry = None;
        self.read_at = now;
        self.txlog.clear();
        self.peer_measured = false;
    }

    /// Remember a write of frames from number `first` on.
    fn wrote(&mut self, first: u32, began: Instant) {
        if self.txlog.len() == TX_LOG {
            self.txlog.pop_front();
        }
        self.txlog.push_back((first, began, Instant::now()));
        self.last_tx = began;
    }

    /// Of our frames from number `rx` on, how many the other side must have had in front of it, unreadable, when it
    /// wrote a record saying that it had read frames below `rx` and then nothing for `age`: those whose write had
    /// returned `HELD_MARGIN` or more before that time was up, counted from when the write of frame `rx - 1` began
    /// (the earliest the other side can have read it). None when that write is no longer remembered.
    fn held(&self, rx: u32, age: Duration) -> Option<u32> {
        let last = rx.checked_sub(1)?;
        let base = self.txlog.iter().rev().find(|w| w.0 <= last)?.1;
        let limit = (base + age).checked_sub(HELD_MARGIN)?;
        // Frames from `rx` up to the first write that returned too late.
        let end = self.txlog.iter().find(|w| w.2 > limit).map_or(self.tx, |w| w.0);
        Some(end.saturating_sub(rx))
    }

    /// Probes sent in this epoch and whether the other side has been heard (tests).
    #[cfg(test)]
    pub fn progress(&self) -> (u32, bool) {
        (self.probes, self.heard)
    }

    /// How far the epoch got, for messages: when its device opened and its first record arrived (from its start),
    /// what was sent and read.
    fn detail(&self, now: Instant) -> String {
        let at = |t: Option<Instant>| t.map_or("never".to_owned(), |t| format!("{:.0} ms", (t - self.opened_at).as_secs_f64() * 1e3));
        format!(
            "{:.0} ms into the epoch: device open at {}, first record at {}, {} probes and {} frames sent, {} records \
             and {} ends of file read, ours {}read by the other side; {} of its reports since {} ms after our first \
             frame showed none of ours read; {} reports from it (of ours read {}, written to us {}); longest pause of \
             this side {:.0} ms",
            (now - self.opened_at).as_secs_f64() * 1e3,
            at(self.dev_at),
            at(self.heard_at),
            self.probes,
            self.tx,
            self.records,
            self.eofs,
            if self.peer_heard { "" } else { "not " },
            self.zero_late,
            ONE_WAY.as_millis(),
            self.reports,
            self.rep_rx,
            self.rep_tx,
            self.max_gap.as_secs_f64() * 1e3
        )
    }
}

pub struct State {
    pub session: u64,
    /// The side that opened the TCP connection; it opens stream epochs.
    pub connector: bool,
    /// The other machine, for messages.
    pub label: String,
    pub tune: Tuning,
    pub local: SocketAddr,
    closed: Option<(ErrorKind, String)>,
    /// The other side said BYE.
    bye: bool,
    // Sending: bytes from `sbase` on, of which those below `acked` have arrived.
    sbuf: Vec<u8>,
    sbase: u64,
    acked: u64,
    /// Next byte to send over TCP.
    tcp_next: u64,
    tcp_stalled_since: Option<Instant>,
    tcp_read_at: Instant,
    /// The link's last turn (`pump`), by any thread.
    last_pump: Instant,
    // Receiving.
    expected: u64,
    pub inq: Buf,
    ack_sent: u64,
    ack_due: Option<Instant>,
    pong_due: bool,
    ping_due: bool,
    full: bool,
    peer_full: bool,
    pub tcp: Vec<TcpPath>,
    pub stream: StreamPath,
    pub stats: Stats,
    /// When the application last drove the link (the maintenance thread leaves it alone meanwhile).
    pub last_app: Instant,
    /// The other machine's host name, from the handshake.
    pub peer: String,
    /// Both ends have a stream on this cable, so the link should be on it.
    pub expects_stream: bool,
    born: Instant,
    /// Why the last epoch failed, and since when the stream has been down (the link's start, or the last failed
    /// epoch).
    last_fail: String,
    down_since: Instant,
    /// When a pause of the link was last said (`slow`).
    slow_said: Option<Instant>,
    /// The other side closed its stream device; it said (over TCP) that its stream had failed.
    stream_eof: bool,
    peer_failed: bool,
}

fn err(kind: ErrorKind, msg: &str) -> io::Error {
    io::Error::new(kind, msg.to_owned())
}

impl State {
    pub fn new(session: u64, connector: bool, label: String, tune: Tuning, sock: TcpStream, epoch: u32) -> io::Result<Self> {
        let now = Instant::now();
        let local = sock.local_addr()?;
        let backoff = tune.backoff_min;
        Ok(Self {
            session,
            connector,
            label,
            tune,
            local,
            closed: None,
            bye: false,
            sbuf: Vec::new(),
            sbase: 0,
            acked: 0,
            tcp_next: 0,
            tcp_stalled_since: None,
            tcp_read_at: now,
            last_pump: now,
            expected: 0,
            inq: Buf::new(1 << 16),
            ack_sent: 0,
            ack_due: None,
            pong_due: false,
            ping_due: false,
            full: false,
            peer_full: false,
            tcp: vec![TcpPath::new(sock)],
            stream: StreamPath::new(now, backoff, epoch),
            stats: Stats::default(),
            last_app: now,
            peer: String::new(),
            expects_stream: false,
            born: now,
            last_fail: String::new(),
            down_since: now,
            slow_said: None,
            stream_eof: false,
            peer_failed: false,
        })
    }

    /// The stream has carried data (either way): from then on a stream that fails ends the link.
    fn carried(&self) -> bool {
        self.stats.sent_stream > 0 || self.stats.got_stream > 0
    }

    fn sid(&self) -> u32 {
        self.session as u32
    }

    /// Total bytes the application has queued.
    pub fn written(&self) -> u64 {
        self.sbase + self.sbuf.len() as u64
    }

    /// Queued bytes not yet handed to a path.
    pub fn unsent(&self) -> u64 {
        let next = if self.stream_carries() { self.stream.next.max(self.acked) } else { self.tcp_next };
        self.written() - next.min(self.written())
    }

    pub fn path(&self) -> &'static str {
        if self.stream.phase == Phase::Up {
            "stream"
        } else {
            "tcp"
        }
    }

    pub fn error(&self) -> Option<io::Error> {
        self.closed.as_ref().map(|(k, m)| err(*k, m))
    }

    pub fn is_closed(&self) -> bool {
        self.closed.is_some()
    }

    fn close(&mut self, kind: ErrorKind, msg: String) {
        if self.closed.is_none() {
            let now = Instant::now();
            if matches!(self.stream.phase, Phase::Opening | Phase::Probing) && now - self.stream.opened_at > SLOW_EPOCH {
                say(format!(
                    "link {}: closed with stream epoch {:08x} still opening ({}) [{}]",
                    self.label,
                    self.stream.epoch,
                    self.stream.detail(now),
                    clock()
                ));
            }
            self.closed = Some((kind, msg));
            if let Some(d) = self.stream.dev.take() {
                close_later(d);
            }
            self.stream.phase = Phase::Off;
            for p in &self.tcp {
                let _ = p.sock.shutdown(std::net::Shutdown::Both);
            }
        }
    }

    /// Queue bytes for sending.
    pub fn queue(&mut self, data: &[u8]) -> io::Result<()> {
        if let Some(e) = self.error() {
            return Err(e);
        }
        // Drop acknowledged bytes once they are worth moving.
        let done = (self.acked - self.sbase) as usize;
        if done >= 1 << 20 && done * 2 >= self.sbuf.len() {
            self.sbuf.drain(..done);
            self.sbase = self.acked;
        }
        self.sbuf.extend_from_slice(data);
        Ok(())
    }

    /// The stream carries new data: it is up and everything sent over TCP has arrived.
    fn stream_carries(&self) -> bool {
        self.stream.phase == Phase::Up && self.acked >= self.tcp_next
    }

    /// Move received data along, send what is due and run the timers. Returns whether anything moved.
    pub fn pump(&mut self, now: Instant, sink: &mut Sink) -> io::Result<bool> {
        if let Some(e) = self.error() {
            return Err(e);
        }
        let gap = now.saturating_duration_since(self.last_pump);
        if gap > self.stream.max_gap {
            self.stream.max_gap = gap;
            // A long pause while an epoch opens (or just after it came up) is when the other side hears nothing.
            let s = &self.stream;
            let opening = matches!(s.phase, Phase::Opening | Phase::Probing);
            if gap > Duration::from_millis(100) && opening {
                say(format!(
                    "link {}: no turn for {gap:.0?} while stream epoch {:08x} was opening [{}]",
                    self.label,
                    s.epoch,
                    clock()
                ));
            }
        }
        // On the stream the link turns every millisecond or more often (`service.rs`): a longer pause is the
        // process not running it, which the other side sees as a stream that acknowledges nothing.
        if self.stream.phase == Phase::Up && gap > SLOW_TURN {
            self.slow(now, format!("no turn for {gap:.0?}"));
        }
        self.last_pump = self.last_pump.max(now);
        let r = self.turn(now, sink);
        let took = now.elapsed();
        if took > SLOW_TURN && !self.is_closed() {
            self.slow(now, format!("a turn took {took:.0?}"));
        }
        r
    }

    /// Send what is queued without reading first, for a caller that goes on to wait for the answer (`pump` follows):
    /// the read a turn starts with costs two system calls, and on an exchange the other side waits for these frames.
    /// The acknowledgement and credits the frames carry are those of the last turn.
    pub fn send_now(&mut self, now: Instant) -> io::Result<()> {
        if let Some(e) = self.error() {
            return Err(e);
        }
        if self.stream.phase == Phase::Up {
            // Nothing was read in this turn: its records do not say that nothing was readable, and after a pause
            // their `rx` is old.
            self.stream.dry = None;
            self.stream.stale = now.saturating_duration_since(self.stream.turn_at) > STALE_AFTER;
            self.send(now);
            if let Some(e) = self.error() {
                return Err(e);
            }
        }
        Ok(())
    }

    /// Say that the link was held up (at most once a second), with this thread's page faults and involuntary context
    /// switches so far: a thread that waits for the disk (its code paged out by a model load) shows major faults.
    fn slow(&mut self, now: Instant, what: String) {
        if self.slow_said.is_some_and(|t| now - t < Duration::from_secs(1)) {
            return;
        }
        self.slow_said = Some(now);
        // SAFETY: getrusage fills `ru`, which outlives the call.
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        #[cfg(target_os = "linux")]
        unsafe {
            libc::getrusage(libc::RUSAGE_THREAD, &mut ru)
        };
        let thread = std::thread::current();
        say(format!(
            "link {}: {what} on the stream (thread {}: {} major and {} minor faults, {} involuntary switches so far) [{}]",
            self.label,
            thread.name().unwrap_or("?"),
            ru.ru_majflt,
            ru.ru_minflt,
            ru.ru_nivcsw,
            clock()
        ));
    }

    fn turn(&mut self, now: Instant, sink: &mut Sink) -> io::Result<bool> {
        // While the stream carries the data, TCP brings only rare control records: a read every 100 µs keeps them
        // timely without a socket call in every turn of a spinning exchange.
        let mut progress = false;
        if self.stream.phase != Phase::Up || now - self.tcp_read_at >= Duration::from_micros(100) {
            self.tcp_read_at = now;
            progress |= self.read_tcp(now, sink);
        }
        progress |= self.read_stream(now, sink);
        if let Some(e) = self.error() {
            return Err(e);
        }
        if self.full && self.inq.len() <= self.tune.inq_limit / 2 {
            self.full = false;
            self.tcp_control(kind::OPEN, 0, &[]);
            // Nothing was read meanwhile: the stream's timers start again.
            self.stream.last_rx = now;
            self.stream.stalled_since = None;
        }
        self.timers(now);
        if let Some(e) = self.error() {
            return Err(e);
        }
        progress |= self.send(now);
        self.check_tcp(now);
        if let Some(e) = self.error() {
            return Err(e);
        }
        Ok(progress)
    }

    fn deliver(&mut self, payload: &[u8], sink: &mut Sink) {
        let mut n = 0;
        if self.inq.is_empty() {
            n = payload.len().min(sink.buf.len() - sink.filled);
            sink.buf[sink.filled..sink.filled + n].copy_from_slice(&payload[..n]);
            sink.filled += n;
        }
        if n < payload.len() {
            self.inq.push(&payload[n..]);
            if !self.full && self.inq.len() > self.tune.inq_limit {
                self.full = true;
                self.tcp_control(kind::FULL, 0, &[]);
            }
        }
    }

    fn on_ack(&mut self, ack: u64) {
        if ack > self.acked && ack <= self.written() {
            self.acked = ack;
            self.stream.stalled_since = None;
            self.tcp_stalled_since = None;
        }
    }

    fn on_data(&mut self, seq: u64, payload: &[u8], stream: bool, now: Instant, sink: &mut Sink) {
        let end = seq + payload.len() as u64;
        if seq <= self.expected && end > self.expected {
            let skip = (self.expected - seq) as usize;
            self.deliver(&payload[skip..], sink);
            let n = end - self.expected;
            if stream {
                self.stats.got_stream += n;
            } else {
                self.stats.got_tcp += n;
            }
            self.expected = end;
            self.ack_due.get_or_insert(now);
            self.stream.nak = None;
        } else if end <= self.expected {
            self.stats.duplicates += 1;
            self.ack_due.get_or_insert(now);
        } else {
            self.stats.gaps += 1;
            if stream {
                let repeat = match self.stream.nak {
                    Some((s, t)) => s != self.expected || now - t > self.tune.rto,
                    None => true,
                };
                if repeat {
                    self.stream.nak_due = true;
                }
            }
        }
    }

    fn read_tcp(&mut self, now: Instant, sink: &mut Sink) -> bool {
        let mut progress = false;
        for i in 0..self.tcp.len() {
            if self.full || !self.tcp[i].alive || self.closed.is_some() {
                continue;
            }
            for _ in 0..8 {
                let r = {
                    let p = &mut self.tcp[i];
                    let space = p.rbuf.space();
                    p.sock.read(space)
                };
                match r {
                    Ok(0) => {
                        // The other side's process closed this path. Without a goodbye, while other paths
                        // remain, only this path is gone (its cable, say); otherwise the link ends.
                        let others = self.tcp.iter().enumerate().any(|(j, p)| j != i && p.alive) || self.stream.phase == Phase::Up;
                        if !self.bye && others {
                            self.tcp_failed(i, "closed", ErrorKind::UnexpectedEof);
                            break;
                        }
                        let why = if self.bye { "closed the link" } else { "closed the link without saying goodbye" };
                        self.close(ErrorKind::UnexpectedEof, format!("{} {why}", self.label));
                        return progress;
                    }
                    Ok(n) => {
                        self.tcp[i].rbuf.filled(n);
                        progress = true;
                        if let Err(e) = self.parse_tcp(i, now, sink) {
                            self.close(ErrorKind::InvalidData, format!("{}: {e}", self.label));
                            return progress;
                        }
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(e) => {
                        self.tcp_failed(i, &format!("read: {e}"), e.kind());
                        break;
                    }
                }
            }
        }
        progress
    }

    fn parse_tcp(&mut self, i: usize, now: Instant, sink: &mut Sink) -> Result<(), String> {
        let mut rbuf = std::mem::take(&mut self.tcp[i].rbuf);
        let r = (|| {
            loop {
                let b = rbuf.bytes();
                if b.len() < HEADER {
                    return Ok(());
                }
                let h = Header::decode(b).ok_or("damaged record over TCP")?;
                let total = HEADER + h.len as usize;
                if b.len() < total {
                    return Ok(());
                }
                let payload = &b[HEADER..total];
                if h.len > 0 && crc(payload) != h.pcrc {
                    return Err("damaged payload over TCP".to_owned());
                }
                if h.session != self.sid() {
                    return Err(format!("record of another session over TCP ({:08x})", h.session));
                }
                self.on_tcp_record(h, payload, now, sink)?;
                rbuf.consume(total);
            }
        })();
        self.tcp[i].rbuf = rbuf;
        r
    }

    fn on_tcp_record(&mut self, h: Header, payload: &[u8], now: Instant, sink: &mut Sink) -> Result<(), String> {
        self.on_ack(h.ack);
        match h.kind {
            kind::DATA => self.on_data(h.seq, payload, false, now, sink),
            kind::ACK => {
                if h.flags & PING != 0 {
                    self.pong_due = true;
                }
                if h.flags & REPORT != 0 {
                    self.on_report(h, now);
                }
            }
            kind::START if !self.connector => {
                if self.stream.phase != Phase::Off {
                    self.stream_down(now, "the other side opened a new epoch", false);
                }
                let info: serde_json::Value = serde_json::from_slice(payload).map_err(|e| e.to_string())?;
                let s = |k: &str| info[k].as_str().unwrap_or("").to_owned();
                self.stream.epoch = h.epoch;
                self.stream.peer_ring = info["ring"].as_u64().unwrap_or(256) as u32;
                self.stream.phase = Phase::Opening;
                self.stream.begin(now);
                self.stream.job = Some(Job { epoch: h.epoch, at: now, local: self.local, expect: Some((s("local"), s("peer"))) });
            }
            kind::FAIL => {
                if h.epoch == self.stream.epoch && self.stream.phase != Phase::Off {
                    self.stream_down(now, "the other side gave the stream up", false);
                } else if h.epoch == self.stream.epoch && self.stream_eof {
                    self.peer_failed = true;
                }
            }
            kind::FULL => self.peer_full = true,
            kind::OPEN => {
                self.peer_full = false;
                self.stream.stalled_since = None;
                self.stream.last_rx = now;
                self.tcp_stalled_since = None;
            }
            kind::BYE => self.bye = true,
            k => return Err(format!("unexpected record {k} over TCP")),
        }
        Ok(())
    }

    /// The other side's report of its frames of the epoch read (ours) and written (to us), over TCP: evidence for
    /// giving a one-way epoch up (see `ONE_WAY`).
    fn on_report(&mut self, h: Header, now: Instant) {
        let s = &mut self.stream;
        if h.epoch != s.epoch || !matches!(s.phase, Phase::Probing | Phase::Up) {
            return;
        }
        s.reports += 1;
        // Up, it is a credit report: sent over TCP when the stream had no room for one (`send`).
        if s.phase == Phase::Up && h.rx > s.peer_rx {
            s.peer_rx = h.rx.min(s.tx);
        }
        s.rep_rx = s.rep_rx.max(h.rx);
        s.rep_tx = s.rep_tx.max(h.tx);
        if h.rx == 0 && !s.peer_heard && s.first_tx_at.is_some_and(|t| now - t >= ONE_WAY) {
            s.zero_late += 1;
        }
        // Whether its frames arrive is judged in `timers`, after this turn has read the stream.
        s.rep_new = true;
    }

    /// Our frames of the epoch read and written, over TCP (see `on_report`).
    fn tcp_report(&mut self, now: Instant, flags: u8) {
        let Some(p) = self.tcp.iter_mut().find(|p| p.alive) else { return };
        let s = &mut self.stream;
        let h = Header {
            kind: kind::ACK,
            flags: REPORT | flags,
            session: self.session as u32,
            epoch: s.epoch,
            ack: self.expected,
            tx: s.tx,
            rx: s.rx_last,
            ..Default::default()
        };
        p.wbuf.extend_from_slice(&h.encode());
        self.ack_sent = self.expected;
        s.last_report = now;
        s.probe_seen = false;
    }

    fn read_stream(&mut self, now: Instant, sink: &mut Sink) -> bool {
        self.stream.dry = None;
        self.stream.stale = false;
        self.stream.turn_at = now;
        if self.full || !matches!(self.stream.phase, Phase::Probing | Phase::Up) {
            return false;
        }
        let mut progress = false;
        let records = self.stream.records;
        for _ in 0..8 {
            let r = {
                let s = &mut self.stream;
                let Some(dev) = s.dev.as_mut() else { return progress };
                let space = s.rbuf.space();
                if self.tune.profile {
                    let t = Instant::now();
                    let r = dev.read(space);
                    self.stats.reads += 1;
                    self.stats.reads_empty += r.is_err() as u64;
                    self.stats.read_ns += t.elapsed().as_nanos() as u64;
                    r
                } else {
                    dev.read(space)
                }
            };
            match r {
                Ok(0) => {
                    // A CLOSE. While the epoch opens it can be a stale one (a CLOSE sent to a closed device waits
                    // and arrives when the device opens again); once up, the other side has given the stream up.
                    if self.stream.phase == Phase::Probing {
                        self.stats.stale += 1;
                        self.stream.eofs += 1;
                        continue;
                    }
                    self.peer_closed_stream(now, sink);
                    return true;
                }
                Ok(n) => {
                    self.stream.rbuf.filled(n);
                    self.stream.read_at = Instant::now();
                    progress = true;
                    self.parse_stream(now, sink);
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    // (Dry since this turn began at the latest, if nothing was read in it.)
                    self.stream.dry = Some(now);
                    break;
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    self.stream_down(now, &format!("read: {e}"), true);
                    return progress;
                }
            }
            if self.stream.phase == Phase::Off {
                break;
            }
        }
        if self.stream.records != records {
            self.hold_sample(now);
        }
        progress
    }

    /// What the records just read say about the hold of this side's sending direction. The other side reads a frame
    /// of ours only once the hold's worth of later frames have reached it, so:
    /// - the frames written here less the frames it says it has read (`rx`, in every record) are never fewer than
    ///   the hold: a smaller number lowers the estimate;
    /// - an acknowledgement written in a turn that found nothing more to read says for how long that had been so
    ///   since the last frame read (its `seq`, less what its device may be late by): our frames after that one that
    ///   were written within that time (less `HELD_MARGIN`) were in front of the other side, unreadable (`held`).
    ///   With a hold that number is not more than the hold, so a larger one is a sign that the estimate is low;
    /// - the answer to a hold probe (a ping written alone and nothing else until it is answered) is the hold
    ///   exactly, and so is a record for which the two numbers above agree: the accepting side's first estimate
    ///   comes from the connecting side's probe pings that way;
    /// - when data of ours was acknowledged only after nudges, by a record that still did not show the last nudge
    ///   read (and was not `STALE`), the frames it had not read up to that nudge are what was held: a sign too.
    ///
    /// `SIGNS` signs close to one another raise the estimate to the fewest frames any of them showed held.
    fn hold_sample(&mut self, now: Instant) {
        let s = &mut self.stream;
        let (held, exact, resolved) = (s.held_seen.take(), s.exact_seen.take(), s.resolved.take());
        if s.phase != Phase::Up || s.peer_rx == 0 || self.tune.push.is_some() {
            return;
        }
        let unread = s.tx.wrapping_sub(s.peer_rx).min(HOLD_MAX);
        let answered = s.probe.is_some_and(|p| s.peer_rx >= p);
        let was = s.hold;
        // Exact: a probe's answer, or the two bounds agreeing where that does not raise the estimate (a rise from
        // the bounds is a sign like any other).
        let measured = if answered && !s.probe_spoiled { Some(unread) } else { exact.filter(|&n| was.is_none_or(|e| n <= e)) };
        if let Some(n) = measured {
            s.hold = Some(n);
            s.hold_probed = Some(n);
            s.hold_raised = None;
            s.signs = 0;
        } else if let Some(e) = s.hold {
            // What this batch shows held beyond the estimate, if anything.
            let mut sign = held.filter(|&n| n > e);
            if let Some((rx, stale)) = resolved {
                // A stall is over: its data is acknowledged. Did that take the nudges?
                let behind = s.nudge_tx.saturating_sub(rx).min(e + s.nudges).min(HOLD_MAX);
                if !stale && behind > e {
                    sign = Some(sign.map_or(behind, |n| n.max(behind)));
                }
            }
            let mut to = e;
            if let Some(n) = sign {
                let near = s.signs > 0 && s.tx.wrapping_sub(s.sign_tx) <= SIGN_FRAMES;
                s.sign_held = if near { s.sign_held.min(n) } else { n };
                s.signs = if near { s.signs + 1 } else { 1 };
                s.sign_tx = s.tx;
                if s.signs >= SIGNS {
                    to = s.sign_held;
                }
            }
            if to > e {
                s.hold = Some(to);
                if std::mem::replace(&mut s.hold_went_up, true) {
                    s.hold_stick = (s.hold_stick * 2).min(HOLD_STICK_MAX);
                }
                s.hold_raised = Some(now);
                s.signs = 0;
            } else if unread < e && s.hold_raised.is_none_or(|t| now.saturating_duration_since(t) >= s.hold_stick) {
                s.hold = Some(unread);
            }
        }
        if s.hold > was && was.is_some() {
            self.stats.hold_raises += 1;
        }
        if answered {
            if !s.probe_spoiled {
                s.hold_probes += 1;
                self.stats.hold_probes += 1;
            }
            s.probe = None;
            s.probe_at = now;
        }
        if resolved.is_some() || answered {
            s.nudges = 0;
        }
        self.stats.hold = s.hold;
    }

    /// An estimate above the last exact measurement that has stood for `hold_stick` is tried one lower (see
    /// `HOLD_STICK`): if the hold is there, the stalls that follow put it back for twice as long.
    fn hold_decay(&mut self, now: Instant) {
        let s = &mut self.stream;
        let (Some(e), Some(at)) = (s.hold, s.hold_raised) else { return };
        if e > s.hold_probed.unwrap_or(0) && now.saturating_duration_since(at) >= s.hold_stick {
            s.hold = Some(e - 1);
            // The next step down waits as long again.
            s.hold_raised = (e - 1 > s.hold_probed.unwrap_or(0)).then_some(now);
            self.stats.hold = s.hold;
        }
    }

    /// Frames to write after a write on the stream so that the other side reads its last frame.
    fn pushes(&self) -> usize {
        match self.tune.push {
            Some(n) => n as usize,
            None => self.stream.hold.map_or(PUSH, |h| h as usize),
        }
    }

    /// This side has measured the hold of its sending direction in this epoch (or does not measure): the connecting
    /// side with its hold probes, the accepting side from the first of their pings.
    fn measured(&self) -> bool {
        self.tune.push.is_some() || if self.connector { self.stream.hold_probes >= HOLD_PROBES } else { self.stream.hold.is_some() }
    }

    /// Whether this side has measured its direction, and the other side its own (`Link::settle`).
    pub fn hold_measured(&self) -> (bool, bool) {
        (self.measured(), self.stream.peer_measured)
    }

    /// The other side closed its stream device while the stream was up. Its process ended, it closed the link (a
    /// goodbye on the stream ahead of the CLOSE, or over TCP) or its stream failed (FAIL over TCP); what it said over
    /// TCP may still be on the way, so wait for that up to `EOF_WAIT`. Only a failure (or silence) is a lost stream.
    fn peer_closed_stream(&mut self, now: Instant, sink: &mut Sink) {
        if let Some(d) = self.stream.dev.take() {
            close_later(d);
        }
        self.stream.phase = Phase::Off;
        self.stream.rbuf.clear();
        self.stream_eof = true;
        let until = Instant::now() + EOF_WAIT;
        loop {
            if self.closed.is_some() {
                return;
            }
            if self.bye {
                self.close(ErrorKind::UnexpectedEof, format!("{} closed the link", self.label));
                return;
            }
            let failed = self.peer_failed;
            if (failed || Instant::now() > until) && !self.carried() {
                // Before the stream carried data: a failed epoch, tried again (`stream_down` from the state it had).
                self.stream_eof = false;
                self.peer_failed = false;
                self.stream.phase = Phase::Up;
                self.stream_down(now, "the other side closed its stream", !failed);
                return;
            }
            if failed || Instant::now() > until {
                self.stats.fallbacks += 1;
                let why = if failed { "the other side gave the stream up" } else { "the other side closed its stream and said nothing" };
                say(format!("link {}: stream lost ({why}); the link ends [{}]", self.label, clock()));
                self.close(ErrorKind::ConnectionAborted, format!("{}: the stream failed ({why})", self.label));
                return;
            }
            // A TCP end of file here closes the link: its process ended.
            self.read_tcp(now, sink);
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    fn parse_stream(&mut self, now: Instant, sink: &mut Sink) {
        let mut rbuf = std::mem::take(&mut self.stream.rbuf);
        let mut spans = std::mem::take(&mut self.stream.spans);
        let mut sums = std::mem::take(&mut self.stream.sums);
        'parse: loop {
            let b = rbuf.bytes();
            // The complete records at the front whose headers check out, up to 48, so that their payload CRCs are
            // computed together.
            spans.clear();
            let mut at = 0;
            while spans.len() < 48 {
                match Header::decode(&b[at..]) {
                    Some(h) if h.len as usize <= STREAM_PAYLOAD && b.len() - at >= HEADER + h.len as usize => {
                        spans.push((at, h));
                        at += HEADER + h.len as usize;
                    }
                    _ => break,
                }
            }
            if spans.is_empty() {
                if b.len() < HEADER {
                    break;
                }
                if matches!(Header::decode(b), Some(h) if h.len as usize <= STREAM_PAYLOAD) {
                    break; // A record still arriving.
                }
                let skip = 1 + find_magic(&b[1..]);
                self.stats.resync_bytes += skip as u64;
                rbuf.consume(skip);
                continue;
            }
            {
                let parts: Vec<&[u8]> = spans.iter().map(|(at, h)| &b[at + HEADER..at + HEADER + h.len as usize]).collect();
                crcs(&parts, &mut sums);
            }
            let mut done = 0;
            for (k, &(at, h)) in spans.iter().enumerate() {
                if h.len > 0 && sums[k] != h.pcrc {
                    // A damaged record: look for the next one from the byte after its start.
                    self.stats.crc_errors += 1;
                    let skip = 1 + find_magic(&b[at + 1..]);
                    self.stats.resync_bytes += skip as u64;
                    done = at + skip;
                    break;
                }
                self.on_stream_record(h, &b[at + HEADER..at + HEADER + h.len as usize], now, sink);
                done = at + HEADER + h.len as usize;
                if self.stream.phase == Phase::Off {
                    rbuf.consume(done);
                    break 'parse;
                }
            }
            rbuf.consume(done);
        }
        if self.stream.phase == Phase::Off {
            rbuf.clear();
        }
        self.stream.rbuf = rbuf;
        self.stream.spans = spans;
        self.stream.sums = sums;
    }

    fn on_stream_record(&mut self, h: Header, payload: &[u8], now: Instant, sink: &mut Sink) {
        if h.session != self.sid() || h.epoch != self.stream.epoch {
            self.stats.stale += 1;
            return;
        }
        let s = &mut self.stream;
        s.last_rx = now;
        s.heard_at.get_or_insert(now);
        s.records += 1;
        s.rx_last = s.rx_last.max(h.tx.wrapping_add(1));
        if h.rx > s.peer_rx {
            s.peer_rx = h.rx.min(s.tx);
        }
        s.heard = true;
        // (A record here showing none of ours read is no evidence: it may have been written long before and held
        // in the receiver until more followed; the other side's reports over TCP are counted instead.)
        if h.rx > 0 {
            s.peer_heard = true;
        }
        if h.kind == kind::PROBE && s.phase == Phase::Up {
            s.probe_seen = true;
        }
        if h.flags & MEASURED != 0 {
            s.peer_measured = true;
        }
        self.on_ack(h.ack);
        let s = &mut self.stream;
        if s.nudges > 0 && s.probe.is_none() && s.resolved.is_none() && self.acked >= s.nudge_mark {
            s.resolved = Some((h.rx, h.flags & STALE != 0));
        }
        match h.kind {
            kind::DATA => self.on_data(h.seq, payload, true, now, sink),
            kind::ACK => {
                if h.flags & PING != 0 {
                    self.pong_due = true;
                }
                // An acknowledgement's `seq`: one more than the microseconds since its sender last read a frame,
                // when its turn had found nothing more to read (0: it does not say).
                let s = &mut self.stream;
                if h.seq > 0 && s.phase == Phase::Up && h.rx <= s.tx {
                    if let Some(n) = s.held(h.rx, Duration::from_micros(h.seq - 1)) {
                        let n = n.min(HOLD_MAX);
                        s.held_seen = Some(s.held_seen.map_or(n, |m| m.max(n)));
                        if n == s.tx - h.rx {
                            s.exact_seen = Some(n);
                        }
                    }
                }
            }
            kind::NAK => {
                self.stats.naks_got += 1;
                self.on_ack(h.seq);
                let s = &mut self.stream;
                if s.phase == Phase::Up && h.seq < s.next && h.seq >= self.acked {
                    let again = matches!(s.rewind, Some((q, t)) if q == h.seq && now - t < self.tune.rto);
                    if !again {
                        s.next = h.seq;
                        s.rewind = Some((h.seq, now));
                        self.stats.rewinds += 1;
                    }
                }
            }
            kind::BYE => self.bye = true,
            kind::PROBE => {
                // A probe's `seq` is its sender's receive ring size.
                if (32..=65536).contains(&h.seq) {
                    self.stream.peer_ring = h.seq as u32;
                }
                self.stream.probe_reply = true;
            }
            _ => self.stats.stale += 1,
        }
        if self.stream.phase == Phase::Probing && self.stream.heard && self.stream.peer_heard {
            self.stream_up(now);
        }
    }

    fn stream_up(&mut self, now: Instant) {
        let mut after = String::new();
        if self.stats.first_up_ms.is_none() {
            let ms = (now - self.born).as_secs_f64() * 1e3;
            self.stats.first_up_ms = Some(ms);
            after = format!(", {ms:.1} ms after the link started");
        }
        if now - self.stream.opened_at > SLOW_EPOCH {
            after = format!("{after} ({})", self.stream.detail(now));
        }
        let s = &mut self.stream;
        s.phase = Phase::Up;
        s.up_since = now;
        s.failed_since_up = 0;
        s.next = self.acked;
        s.last_rx = now;
        s.stalled_since = None;
        s.probe_reply = true;
        self.stats.ups += 1;
        let label = s.found.as_ref().map_or(String::new(), |f| format!(" ({})", f.label));
        say(format!("link {}: stream up{label}, epoch {:08x}{after} [{}]", self.label, s.epoch, clock()));
    }

    /// Give the current stream epoch up: close the device off this thread and tell the other side (unless it gave up
    /// first). Once the stream has carried data this ends the link: it does not go on over TCP. Before that the
    /// connecting side tries another epoch (at once the first time, then backing off) until `stream_wait`.
    pub fn stream_down(&mut self, now: Instant, why: &str, tell: bool) {
        if self.stream.phase == Phase::Off {
            return;
        }
        let detail = self.stream.detail(now);
        let was = self.stream.phase;
        let epoch = self.stream.epoch;
        let brief = now - self.stream.up_since < Duration::from_secs(1);
        {
            let s = &mut self.stream;
            if let Some(d) = s.dev.take() {
                close_later(d);
            }
            s.phase = Phase::Off;
            s.job = None;
            s.rbuf.clear();
            s.nak = None;
            s.nak_due = false;
            s.probe_reply = false;
        }
        if was == Phase::Up {
            self.down_since = now;
        }
        if was == Phase::Up && self.carried() {
            self.stats.fallbacks += 1;
            let more = if brief { format!(" ({detail})") } else { String::new() };
            say(format!("link {}: stream lost ({why}); the link ends{more} [{}]", self.label, clock()));
            if tell {
                self.tcp_control(kind::FAIL, epoch, &[]);
                for p in self.tcp.iter_mut().filter(|p| p.alive) {
                    let _ = p.flush();
                }
            }
            self.close(ErrorKind::ConnectionAborted, format!("{}: the stream failed ({why})", self.label));
            return;
        }
        // An epoch that never came up, or came up and failed before it carried any data (the other side gave up on
        // it while it still probed): the connecting side tries again, until `stream_wait`.
        let s = &mut self.stream;
        if self.connector {
            // The first failed epoch is tried again at once: the new open waits for this epoch's close anyway.
            if s.failed_since_up == 0 {
                s.retry_at = Some(now);
            } else {
                s.retry_at = Some(now + s.backoff);
                s.backoff = (s.backoff * 2).min(self.tune.backoff_max);
            }
        }
        s.failed_since_up += 1;
        self.stats.failed_epochs += 1;
        if !why.is_empty() {
            self.last_fail = why.to_owned();
            let what = if was == Phase::Up { "lost before it carried data" } else { "not opened" };
            say(format!("link {}: stream epoch {epoch:08x} {what} ({why}; {detail}) [{}]", self.label, clock()));
        }
        if tell {
            self.tcp_control(kind::FAIL, epoch, &[]);
        }
    }

    /// The device for epoch `epoch` opened (or not), outside the lock.
    pub fn install(&mut self, epoch: u32, r: io::Result<(Box<dyn Frames>, Found)>, now: Instant) {
        if self.stream.phase != Phase::Opening || self.stream.epoch != epoch || self.closed.is_some() {
            if let Ok((d, _)) = r {
                close_later(d);
            }
            return;
        }
        match r {
            Err(e) => {
                self.stream.phase = Phase::Probing;
                self.stream_down(now, &format!("{e}"), !self.connector);
            }
            Ok((d, found)) => {
                let s = &mut self.stream;
                s.late = d.late();
                s.dev = Some(d);
                s.dev_at = Some(now);
                if self.connector {
                    s.peer_ring = found.ring;
                }
                let start = serde_json::json!({"ring": found.ring, "local": found.local, "peer": found.peer});
                s.found = Some(found);
                s.tx = 0;
                s.peer_rx = 0;
                s.rx_last = 0;
                s.rx_reported = 0;
                s.rbuf.clear();
                s.heard = false;
                s.peer_heard = false;
                s.probes = 0;
                s.probe_start = now;
                s.last_probe = now - Duration::from_secs(1);
                s.probe_reply = false;
                s.nak = None;
                s.nak_due = false;
                s.rewind = None;
                s.phase = Phase::Probing;
                self.stats.epochs += 1;
                if self.connector {
                    let payload = serde_json::to_vec(&start).unwrap();
                    self.tcp_control(kind::START, epoch, &payload);
                }
            }
        }
    }

    /// No usable stream here for epoch `epoch` (`why` is empty when the link is not a Thunderbolt one): the
    /// accepting side says so, the connecting side looks again after `backoff_max`. A reason is logged once.
    pub fn no_stream(&mut self, epoch: u32, why: &str, now: Instant) {
        if self.stream.phase != Phase::Opening || self.stream.epoch != epoch {
            return;
        }
        if !why.is_empty() && self.stream.why != why {
            say(format!("link {}: no stream ({why}) [{}]", self.label, clock()));
            self.stream.why = why.to_owned();
        }
        let s = &mut self.stream;
        s.phase = Phase::Off;
        s.job = None;
        if self.connector {
            s.retry_at = Some(now + self.tune.backoff_max);
        } else {
            self.tcp_control(kind::FAIL, epoch, &[]);
        }
    }

    /// The connecting side's next epoch, when one is due.
    fn maybe_retry(&mut self, now: Instant) {
        let s = &mut self.stream;
        if self.connector && s.phase == Phase::Off && s.job.is_none() && s.retry_at.is_some_and(|t| now >= t) {
            s.retry_at = None;
            s.epoch = s.epoch.wrapping_add(1);
            s.phase = Phase::Opening;
            s.begin(now);
            s.job = Some(Job { epoch: s.epoch, at: now, local: self.local, expect: None });
        }
    }

    /// Start the first epoch now (the connecting side, at session start).
    pub fn first_epoch(&mut self, now: Instant) {
        if self.connector {
            self.stream.retry_at = Some(now);
            self.maybe_retry(now);
        }
    }

    fn timers(&mut self, now: Instant) {
        self.maybe_retry(now);
        // The connecting side opens the epochs and says why none came up; the other side waits twice as long.
        let wait = if self.connector { self.tune.stream_wait } else { self.tune.stream_wait * 2 };
        if self.expects_stream && self.stream.phase != Phase::Up && self.closed.is_none() && now - self.down_since > wait {
            let last = if self.last_fail.is_empty() { String::new() } else { format!(" (last attempt: {})", self.last_fail) };
            let why = format!("the stream was not up within {wait:.0?}{last}");
            say(format!("link {}: {why}; the link ends [{}]", self.label, clock()));
            if let Some(d) = self.stream.dev.take() {
                close_later(d);
            }
            self.stream.phase = Phase::Off;
            self.close(ErrorKind::TimedOut, format!("{}: {why}", self.label));
            return;
        }
        let waiting = self.full || self.peer_full;
        match self.stream.phase {
            Phase::Opening => {
                if now - self.stream.opened_at > self.tune.probe_timeout + Duration::from_secs(2) {
                    self.stream.phase = Phase::Probing;
                    self.stream_down(now, "the device did not open", true);
                }
            }
            Phase::Probing => {
                let s = &mut self.stream;
                if std::mem::take(&mut s.rep_new) {
                    // More of its frames missing here than the receiver holds back (the last PUSH and one in flight).
                    if s.rep_tx > s.rx_last + PUSH as u32 + 1 {
                        s.behind_since.get_or_insert(now);
                        s.behind_reports += 1;
                    } else {
                        s.behind_since = None;
                        s.behind_reports = 0;
                    }
                }
                if !s.peer_heard && s.zero_late >= EVIDENCE {
                    self.stream_down(now, "the other side reads none of our frames", true);
                } else if s.behind_reports >= EVIDENCE && s.behind_since.is_some_and(|t| now - t >= ONE_WAY) {
                    self.stream_down(now, "the other side's frames do not arrive", true);
                } else if now - s.probe_start > self.tune.probe_timeout {
                    self.stream_down(now, "no answer on the stream", true);
                }
            }
            Phase::Up => {
                self.hold_decay(now);
                let window = self.window();
                let written = self.written();
                let s = &mut self.stream;
                // Bytes the other side has not acknowledged, and no acknowledgement for a while: go back (data in
                // flight) or ask for credits (none left); with no progress for `dead` the stream is given up. Only
                // an acknowledgement resets this clock, so a stream that keeps losing the same record falls back.
                if self.acked < written && !waiting {
                    let since = *s.stalled_since.get_or_insert(now);
                    if now - since > self.tune.dead {
                        let why = format!("nothing acknowledged for {:?}", self.tune.dead);
                        self.stream_down(now, &why, true);
                        return;
                    }
                    if now - since > self.tune.rto && now - s.last_rto > self.tune.rto {
                        s.last_rto = now;
                        if s.next > self.acked {
                            s.next = self.acked;
                            self.stats.rewinds += 1;
                        }
                        if s.tx.wrapping_sub(s.peer_rx) + 1 >= window {
                            // The credit report may have been lost: a ping brings a fresh one.
                            self.ping_due = true;
                        }
                    }
                } else {
                    s.stalled_since = None;
                }
                if !waiting {
                    if now - s.last_rx > self.tune.idle_dead {
                        let why = format!("no frame for {:?}", self.tune.idle_dead);
                        self.stream_down(now, &why, true);
                        return;
                    }
                    // The connecting side's keepalives are hold probes, and the accepting side reads its own
                    // direction's hold from them; its own keepalive comes later, when those stop arriving.
                    let ping = if self.connector { self.tune.ping } else { self.tune.ping * 3 / 2 };
                    if now - s.last_rx > ping && now - s.last_ping > ping {
                        self.ping_due = true;
                    }
                }
            }
            Phase::Off => {}
        }
    }

    /// Give TCP path `i` up; the link goes on over the others (or the stream), or closes.
    fn tcp_failed(&mut self, i: usize, why: &str, kind: ErrorKind) {
        if !self.tcp[i].alive {
            return;
        }
        self.tcp[i].alive = false;
        let _ = self.tcp[i].sock.shutdown(std::net::Shutdown::Both);
        let others = self.tcp.iter().any(|p| p.alive);
        if others || self.stream.phase == Phase::Up {
            self.stats.tcp_failovers += 1;
            say(format!("link {}: TCP path {} failed ({why}); continuing without it [{}]", self.label, self.tcp[i].name, clock()));
            // Its unsent and unacknowledged bytes go again over the next path.
            self.tcp_next = self.acked;
            self.tcp[i].wbuf.clear();
            self.tcp[i].wpos = 0;
        } else {
            self.close(kind, format!("{}: {} ({why})", self.label, self.tcp[i].name));
        }
    }

    fn check_tcp(&mut self, now: Instant) {
        if self.closed.is_some() {
            return;
        }
        let waiting = self.full || self.peer_full;
        let active = self.tcp.iter().position(|p| p.alive);
        match active {
            Some(i) if self.tcp_next > self.acked && !waiting => {
                let since = *self.tcp_stalled_since.get_or_insert(now);
                if now - since > self.tune.tcp_dead && self.tcp.iter().filter(|p| p.alive).count() > 1 {
                    let why = format!("nothing acknowledged for {:?}", self.tune.tcp_dead);
                    self.tcp_failed(i, &why, ErrorKind::TimedOut);
                    self.tcp_stalled_since = None;
                }
            }
            Some(_) => self.tcp_stalled_since = None,
            None if self.stream.phase != Phase::Up => {
                self.close(ErrorKind::ConnectionAborted, format!("{}: every path failed", self.label));
            }
            None => {}
        }
    }

    /// Queue a control record on the first live TCP path.
    pub fn tcp_control(&mut self, k: u8, epoch: u32, payload: &[u8]) {
        let Some(p) = self.tcp.iter_mut().find(|p| p.alive) else { return };
        let h = Header {
            kind: k,
            flags: 0,
            len: payload.len() as u16,
            session: self.session as u32,
            epoch,
            seq: 0,
            ack: self.expected,
            tx: 0,
            rx: 0,
            pcrc: if payload.is_empty() { 0 } else { crc(payload) },
        };
        p.wbuf.extend_from_slice(&h.encode());
        p.wbuf.extend_from_slice(payload);
        self.ack_sent = self.expected;
    }

    fn send(&mut self, now: Instant) -> bool {
        let tx0 = self.stream.tx;
        let mut progress = false;
        if self.stream.phase == Phase::Probing {
            self.send_probe(now);
        }
        if self.stream.nak_due && matches!(self.stream.phase, Phase::Up | Phase::Probing) && self.stream_control(kind::NAK, 0, self.expected) {
            self.stream.nak_due = false;
            self.stream.nak = Some((self.expected, now));
            self.stats.naks_sent += 1;
        }
        if self.stream.phase == Phase::Up && self.acked >= self.tcp_next {
            if self.stream.next < self.acked {
                self.stream.next = self.acked;
            }
            progress |= self.send_stream_data(now);
        } else if self.stream.phase != Phase::Up && !self.expects_stream {
            // A link with a stream at both ends sends its data on the stream only: before it is up, the data waits.
            progress |= self.send_tcp_data();
        }
        // Acknowledgements, credits, keepalives and answers not carried by data.
        let credits = self.stream.phase == Phase::Up && self.stream.rx_last.wrapping_sub(self.stream.rx_reported) >= self.tune.credit_frames;
        let ack = self.expected > self.ack_sent
            && (self.expected - self.ack_sent >= 256 << 10 || self.ack_due.is_some_and(|t| now - t >= self.tune.ack_delay));
        let reply = self.stream.probe_reply && self.stream.phase == Phase::Up;
        // The keepalive of an idle link goes as a hold probe (below): a ping alone, which measures the hold again.
        let keepalive = self.ping_due && !(self.pong_due || credits || ack || reply) && self.stream.tx == tx0 && self.hold_probe_fits(now);
        if keepalive {
            self.hold_probe(now);
        } else if self.pong_due || self.ping_due || credits || ack || reply {
            let flags = if self.ping_due { PING } else { 0 };
            let sent = if self.stream.phase == Phase::Up {
                self.stream_control(kind::ACK, flags, 0)
            } else {
                self.tcp_ack(flags);
                true
            };
            if sent {
                if self.ping_due {
                    self.stream.last_ping = now;
                }
                self.pong_due = false;
                self.ping_due = false;
                self.ack_due = None;
                self.stream.probe_reply = false;
            } else if self.stream.phase == Phase::Up && now - self.stream.last_report >= self.tune.rto {
                // No room on the stream for an acknowledgement: when both sides have filled the other's ring and the
                // frames that would return credits were lost, neither could send another. Acknowledgement and credits
                // go over TCP instead (a ping too, answered on the stream).
                self.tcp_report(now, flags);
                self.stats.tcp_reports += 1;
                if self.ping_due {
                    self.stream.last_ping = now;
                }
                self.pong_due = false;
                self.ping_due = false;
                self.ack_due = None;
            }
        }
        if self.stream.phase == Phase::Up {
            if keepalive {
                // Alone: nothing follows it.
            } else if self.stream.tx != tx0 {
                // The last frames to arrive are not read on these NHIs until as many more as the direction holds
                // follow them (the module documentation): every write on the stream ends with that many more,
                // acknowledgements.
                if self.stream.probe.is_some() {
                    self.stream.probe_spoiled = true;
                }
                for _ in 0..self.pushes() {
                    if !self.stream_control(kind::ACK, 0, 0) {
                        break;
                    }
                    self.stats.pushes += 1;
                }
            } else if !self.nudge(now) && self.connector && self.stream.hold_probes < HOLD_PROBES && self.hold_probe_fits(now) {
                // Nothing to send: measure the hold of this direction while the link is quiet. (One side probes:
                // the answer to a ping would follow the other side's own ping, which would no longer be alone.)
                self.hold_probe(now);
            }
        }
        // Our frames of the epoch read and written, over TCP, while the epoch opens and, once up, while the other
        // side still probes: what the other side needs to tell a direction that carries nothing from a slow peer.
        let s = &self.stream;
        let report = match s.phase {
            Phase::Probing => true,
            Phase::Up => s.probe_seen,
            _ => false,
        };
        if report && s.dev.is_some() && now - s.last_report >= REPORT_EVERY {
            self.tcp_report(now, 0);
        }
        for i in 0..self.tcp.len() {
            if !self.tcp[i].alive {
                continue;
            }
            match self.tcp[i].flush() {
                Ok(p) => progress |= p,
                Err(e) => self.tcp_failed(i, &format!("write: {e}"), e.kind()),
            }
        }
        progress
    }

    /// A hold probe can go now: the epoch is up, the link measures, none is under way, and this direction is quiet
    /// (nothing unacknowledged, nothing written for `QUIET`, and `QUIET` since the last probe was answered), so that
    /// its ping arrives alone.
    fn hold_probe_fits(&self, now: Instant) -> bool {
        let s = &self.stream;
        s.phase == Phase::Up
            && self.tune.push.is_none()
            && s.probe.is_none()
            && self.acked >= self.written()
            && now.saturating_duration_since(s.last_tx) >= QUIET
            && now.saturating_duration_since(s.probe_at) >= QUIET
    }

    /// Start a hold probe: a ping with nothing behind it. The other side answers a ping at once, and its answer
    /// says how many of our frames it has read: with a hold the ping stays unread until nudges follow it, and the
    /// frames it took are the hold (`hold_sample`).
    fn hold_probe(&mut self, now: Instant) {
        if self.stream_control(kind::ACK, PING, 0) {
            let s = &mut self.stream;
            s.probe = Some(s.tx);
            s.probe_spoiled = false;
            s.probe_at = now;
            s.nudges = 0;
            s.last_ping = now;
            self.ping_due = false;
        }
    }

    /// With data unacknowledged (or a hold probe unanswered), nothing written and nothing heard for `Tuning::nudge`:
    /// one more frame, a ping, in case the other side's receiver holds our last frames back (the estimate of the
    /// hold is too low, or the hold changed) or ours holds its acknowledgement (its answer brings it). At most
    /// `NUDGE_MAX` per stall; the retransmission timer goes on as before. Returns whether a frame was written.
    fn nudge(&mut self, now: Instant) -> bool {
        let written = self.written();
        let s = &mut self.stream;
        if self.tune.push.is_some() || self.full || self.peer_full {
            return false;
        }
        if self.acked >= written && s.probe.is_none() {
            // (Acknowledged over TCP, or nothing to wait for.)
            s.nudges = 0;
            s.resolved = None;
            return false;
        }
        let wait = self.tune.nudge;
        if s.nudges >= NUDGE_MAX || now.saturating_duration_since(s.last_tx) < wait || now.saturating_duration_since(s.last_rx) < wait {
            return false;
        }
        let first = s.nudges == 0;
        if !self.stream_control(kind::ACK, PING, 0) {
            return false;
        }
        let s = &mut self.stream;
        if first {
            s.nudge_mark = written;
        }
        s.nudges += 1;
        s.nudge_tx = s.tx;
        self.stats.nudges += 1;
        true
    }

    fn send_probe(&mut self, now: Instant) {
        let s = &self.stream;
        // The connecting side opens its device first and the other side only on START, so it waits for the other
        // side's first record before it writes: nothing is sent toward an end that is not open yet.
        if self.connector && !s.heard {
            return;
        }
        let interval = if now - s.probe_start < PROBE_FAST { Duration::from_millis(2) } else { Duration::from_millis(50) };
        let due = now - s.last_probe >= interval || s.probe_reply;
        let ring = self.stream.found.as_ref().map_or(256, |f| f.ring) as u64;
        if due && self.stream_control(kind::PROBE, 0, ring) {
            let s = &mut self.stream;
            s.probes += 1;
            s.last_probe = now;
            s.probe_reply = false;
            // A lone frame waits in the receiver until more arrive behind it (see PUSH), so the probe is followed by
            // PUSH frames too, as data is; they go only while the other side's ring has room for them.
            for _ in 0..PUSH {
                if !self.stream_control(kind::ACK, 0, 0) {
                    break;
                }
                self.stats.pushes += 1;
            }
        }
    }

    /// One header-only record on the stream; false if it could not go now.
    fn stream_control(&mut self, k: u8, flags: u8, seq: u64) -> bool {
        let measured = if self.measured() { MEASURED } else { 0 };
        let s = &mut self.stream;
        let Some(dev) = s.dev.as_mut() else { return false };
        if s.tx.wrapping_sub(s.peer_rx) + 4 > s.peer_ring && k != kind::PROBE {
            return false;
        }
        let t = Instant::now();
        // An acknowledgement written in a turn that read the device dry says for how long nothing had been readable
        // when the turn began: since the last frames were read (the other side's `hold_sample`).
        let seq = match s.dry {
            Some(dry) if k == kind::ACK && s.rx_last > 0 => 1 + dry.saturating_duration_since(s.read_at).saturating_sub(s.late).as_micros() as u64,
            _ => seq,
        };
        let stale = if s.stale { STALE } else { 0 };
        let h = Header {
            kind: k,
            flags: flags | measured | stale,
            len: 0,
            session: self.session as u32,
            epoch: s.epoch,
            seq,
            ack: self.expected,
            tx: s.tx,
            rx: s.rx_last,
            pcrc: 0,
        };
        let r = dev.writev(&[IoSlice::new(&h.encode())]);
        if self.tune.profile {
            self.stats.writes += 1;
            self.stats.write_ns += t.elapsed().as_nanos() as u64;
        }
        match r {
            Ok(_) => {
                s.wrote(s.tx, t);
                s.tx += 1;
                s.first_tx_at.get_or_insert(t);
                s.rx_reported = s.rx_last;
                self.ack_sent = self.expected;
                self.stats.frames += 1;
                true
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => false,
            Err(e) => {
                let why = format!("write: {e}");
                self.stream_down(Instant::now(), &why, true);
                false
            }
        }
    }

    fn send_stream_data(&mut self, now: Instant) -> bool {
        let mut progress = false;
        let written = self.written();
        let measured = if self.measured() { MEASURED } else { 0 } | if self.stream.stale { STALE } else { 0 };
        while self.stream.next < written && self.stream.phase == Phase::Up && !self.peer_full {
            let s = &mut self.stream;
            let window = self.tune.window(s.peer_ring);
            let inflight = s.tx.wrapping_sub(s.peer_rx);
            if inflight >= window {
                break;
            }
            let budget = ((window - inflight) as usize).min(256);
            s.hdrs.clear();
            s.lens.clear();
            let mut seq = s.next;
            while s.lens.len() < budget && seq < written {
                let len = (written - seq).min(STREAM_PAYLOAD as u64) as usize;
                s.lens.push(len);
                seq += len as u64;
                if len < STREAM_PAYLOAD {
                    // A short record must end the write: the driver cuts frames every 4 KiB of the write.
                    break;
                }
            }
            {
                let mut at = (s.next - self.sbase) as usize;
                let parts: Vec<&[u8]> = s
                    .lens
                    .iter()
                    .map(|&len| {
                        let p = &self.sbuf[at..at + len];
                        at += len;
                        p
                    })
                    .collect();
                crcs(&parts, &mut s.crcs);
            }
            let mut seq = s.next;
            for (k, &len) in s.lens.iter().enumerate() {
                let h = Header {
                    kind: kind::DATA,
                    flags: measured,
                    len: len as u16,
                    session: self.session as u32,
                    epoch: s.epoch,
                    seq,
                    ack: self.expected,
                    tx: s.tx + k as u32,
                    rx: s.rx_last,
                    pcrc: s.crcs[k],
                };
                s.hdrs.push(h.encode());
                seq += len as u64;
            }
            let began;
            let r = {
                let mut iov = Vec::with_capacity(2 * s.lens.len());
                let mut at = s.next;
                for (h, &len) in s.hdrs.iter().zip(&s.lens) {
                    let off = (at - self.sbase) as usize;
                    iov.push(IoSlice::new(h));
                    iov.push(IoSlice::new(&self.sbuf[off..off + len]));
                    at += len as u64;
                }
                began = Instant::now();
                let r = s.dev.as_mut().map(|d| d.writev(&iov));
                if self.tune.profile {
                    self.stats.writes += 1;
                    self.stats.write_ns += began.elapsed().as_nanos() as u64;
                }
                r
            };
            match r {
                Some(Ok(mut n)) => {
                    let mut records = 0usize;
                    let mut bytes = 0u64;
                    for &len in &s.lens {
                        let rec = HEADER + len;
                        if n < rec {
                            break;
                        }
                        n -= rec;
                        records += 1;
                        bytes += len as u64;
                    }
                    debug_assert!(n == 0 || n < FRAME, "a record was cut");
                    if records > 0 {
                        s.wrote(s.tx, began);
                    }
                    s.tx += records as u32;
                    s.next += bytes;
                    s.rx_reported = s.rx_last;
                    self.ack_sent = self.expected;
                    self.ack_due = None;
                    self.stats.frames += records as u64;
                    self.stats.sent_stream += bytes;
                    progress |= records > 0;
                    if records < s.lens.len() {
                        break;
                    }
                }
                Some(Err(e)) if e.kind() == ErrorKind::WouldBlock => break,
                Some(Err(e)) => {
                    let why = format!("write: {e}");
                    self.stream_down(now, &why, true);
                    break;
                }
                None => break,
            }
        }
        progress
    }

    fn tcp_ack(&mut self, flags: u8) {
        let Some(p) = self.tcp.iter_mut().find(|p| p.alive) else { return };
        let h = Header { kind: kind::ACK, flags, session: self.session as u32, ack: self.expected, ..Default::default() };
        p.wbuf.extend_from_slice(&h.encode());
        self.ack_sent = self.expected;
    }

    fn send_tcp_data(&mut self) -> bool {
        let written = self.written();
        let mut progress = false;
        let Some(i) = self.tcp.iter().position(|p| p.alive) else { return false };
        if self.tcp_next < self.acked {
            self.tcp_next = self.acked;
        }
        while self.tcp_next < written && self.tcp[i].pending() < 256 << 10 {
            let len = (written - self.tcp_next).min(TCP_PAYLOAD as u64) as usize;
            let at = (self.tcp_next - self.sbase) as usize;
            let payload = &self.sbuf[at..at + len];
            let h = Header {
                kind: kind::DATA,
                flags: 0,
                len: len as u16,
                session: self.session as u32,
                epoch: 0,
                seq: self.tcp_next,
                ack: self.expected,
                tx: 0,
                rx: 0,
                pcrc: crc(payload),
            };
            let p = &mut self.tcp[i];
            p.wbuf.extend_from_slice(&h.encode());
            p.wbuf.extend_from_slice(payload);
            self.tcp_next += len as u64;
            self.ack_sent = self.expected;
            self.ack_due = None;
            self.stats.sent_tcp += len as u64;
            progress = true;
        }
        progress
    }

    /// File descriptors to wait on and their events. Returns whether the stream device is not among them though it
    /// may bring something (its rings are busy-polled: nothing wakes a `poll()` on it), so that the caller sleeps
    /// only briefly.
    pub fn poll_fds(&self, out: &mut Vec<libc::pollfd>) -> bool {
        out.clear();
        let read = if self.full { 0 } else { libc::POLLIN };
        for p in self.tcp.iter().filter(|p| p.alive) {
            let ev = read | if p.pending() > 0 { libc::POLLOUT } else { 0 };
            out.push(libc::pollfd { fd: p.sock.as_raw_fd(), events: ev, revents: 0 });
        }
        match self.stream.dev.as_ref().map(|d| d.fd()) {
            Some(Some(fd)) => {
                out.push(libc::pollfd { fd, events: read, revents: 0 });
                false
            }
            Some(None) => !self.full && matches!(self.stream.phase, Phase::Probing | Phase::Up),
            None => false,
        }
    }

    /// Everything queued has arrived at the other side.
    pub fn delivered(&self) -> bool {
        self.acked >= self.written()
    }

    /// Say goodbye and close everything (the device off this thread).
    pub fn shutdown(&mut self) {
        if self.closed.is_some() {
            return;
        }
        let deadline = Instant::now() + Duration::from_millis(200);
        // On the stream too, ahead of the CLOSE that closing the device sends, so that the other side does not take
        // that for a failed stream (with the frames that make the receiver read it).
        if self.stream.phase == Phase::Up && self.stream_control(kind::BYE, 0, 0) {
            for _ in 0..self.pushes() {
                if !self.stream_control(kind::ACK, 0, 0) {
                    break;
                }
            }
        }
        for i in 0..self.tcp.len() {
            if !self.tcp[i].alive {
                continue;
            }
            let h = Header { kind: kind::BYE, session: self.session as u32, ack: self.expected, ..Default::default() };
            self.tcp[i].wbuf.extend_from_slice(&h.encode());
            while self.tcp[i].pending() > 0 && Instant::now() < deadline {
                match self.tcp[i].flush() {
                    Ok(_) => std::hint::spin_loop(),
                    Err(_) => break,
                }
            }
        }
        self.close(ErrorKind::NotConnected, format!("the link to {} is closed", self.label));
    }

    fn window(&self) -> u32 {
        self.tune.window(self.stream.peer_ring)
    }

    /// The protocol state, for messages about a link that stopped moving.
    pub fn describe(&self) -> String {
        let s = &self.stream;
        format!(
            "{}: {:?} epoch {:08x}, written {} acked {} stream next {} tcp next {}, expected {} ack sent {}, frames tx {} peer rx {} \
             rx {} reported {} ring {}, inq {}, full {} peer full {}, tcp paths {}/{}, {:?}",
            self.label,
            s.phase,
            s.epoch,
            self.written(),
            self.acked,
            s.next,
            self.tcp_next,
            self.expected,
            self.ack_sent,
            s.tx,
            s.peer_rx,
            s.rx_last,
            s.rx_reported,
            s.peer_ring,
            self.inq.len(),
            self.full,
            self.peer_full,
            self.tcp.iter().filter(|p| p.alive).count(),
            self.tcp.len(),
            self.stats
        )
    }
}
