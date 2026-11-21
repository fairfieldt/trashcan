//! Links over localhost TCP, with and without a simulated stream (sim.rs), under the faults the stream and the
//! cables show: every byte delivered must be the byte sent.
use super::{sim::*, *};
use std::thread;

fn fast() -> Tuning {
    Tuning {
        rto: Duration::from_millis(5),
        dead: Duration::from_millis(200),
        ping: Duration::from_millis(200),
        idle_dead: Duration::from_millis(1000),
        probe_timeout: Duration::from_millis(1000),
        backoff_min: Duration::from_millis(30),
        backoff_max: Duration::from_millis(300),
        tcp_dead: Duration::from_millis(300),
        ..Tuning::default()
    }
}

fn opts(streams: Arc<dyn Streams>, tune: Tuning) -> Options {
    Options { streams, tune, connect_for: Duration::from_secs(5) }
}

/// A connected pair over localhost; with `pair`, a simulated stream with a ring of `ring` frames.
fn link_pair(pair: Option<&Arc<Pair>>, ring: u32, tune: Tuning) -> (Link, Link) {
    link_pair_with(pair, ring, tune, |_| {})
}

/// `link_pair`, calling `accepted` with the accepting side's link as soon as it exists (while the connecting side
/// still opens its first epoch).
fn link_pair_with(pair: Option<&Arc<Pair>>, ring: u32, tune: Tuning, accepted: impl FnOnce(&Link) + Send + 'static) -> (Link, Link) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let (sa, sb): (Arc<dyn Streams>, Arc<dyn Streams>) = match pair {
        Some(p) => (Arc::new(SimStreams { pair: p.clone(), side: 0, ring }), Arc::new(SimStreams { pair: p.clone(), side: 1, ring })),
        None => (Arc::new(Sysfs), Arc::new(Sysfs)),
    };
    let ob = opts(sb, tune.clone());
    let accepting = thread::spawn(move || {
        let (s, _) = listener.accept().unwrap();
        match Link::accept_with(s, &ob).unwrap() {
            Accepted::Link(l, tag) => {
                assert_eq!(tag, "test");
                accepted(&l);
                l
            }
            Accepted::Joined(_) => panic!("join"),
        }
    });
    let a = Link::connect_with(&addr, "test", &opts(sa, tune)).unwrap();
    let b = accepting.join().unwrap();
    (a, b)
}

/// Deterministic bytes of stream `who`, from offset `at`.
fn pattern(who: u64, at: u64, len: usize) -> Vec<u8> {
    (0..len as u64).map(|i| ((at + i).wrapping_mul(0x9e37_79b9) ^ who.wrapping_mul(0x85eb_ca6b)) as u8 ^ ((at + i) >> 13) as u8).collect()
}

/// Both sides send `total` bytes in messages of varying sizes and check every byte received.
fn both_ways(a: &Link, b: &Link, total: usize) {
    let sizes = [1usize, 7, 4048, 4049, 20 << 10, 40 << 10, 300 << 10, 3, 5 << 20, 64];
    thread::scope(|s| {
        for (me, other, who) in [(a, b, 0u64), (b, a, 1u64)] {
            s.spawn(move || {
                let mut at = 0u64;
                let mut k = 0;
                while (at as usize) < total {
                    let n = sizes[k % sizes.len()].min(total - at as usize);
                    me.send(&[&pattern(who, at, n)]).unwrap();
                    at += n as u64;
                    k += 1;
                }
            });
            s.spawn(move || {
                let mut at = 0u64;
                let mut buf = vec![0u8; 1 << 20];
                while (at as usize) < total {
                    let n = buf.len().min(total - at as usize);
                    if let Err(e) = me.read_exact(&mut buf[..n], Wait::Stall(Duration::from_secs(20))) {
                        panic!("{e}\nother side: {}", other.describe());
                    }
                    assert!(buf[..n] == pattern(1 - who, at, n)[..], "bytes {at}..{} differ", at + n as u64);
                    at += n as u64;
                }
            });
        }
    });
}

fn wait_for(what: &str, limit: Duration, mut f: impl FnMut() -> bool) {
    let start = Instant::now();
    while !f() {
        assert!(start.elapsed() < limit, "{what} not within {limit:?}");
        thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn tcp_only_link_moves_data() {
    let (a, b) = link_pair(None, 256, fast());
    assert_eq!(a.path(), "tcp");
    both_ways(&a, &b, 12 << 20);
}

#[test]
fn stream_link_moves_data_on_the_stream() {
    let p = Pair::new(256);
    let (a, b) = link_pair(Some(&p), 256, fast());
    assert_eq!(a.path(), "stream");
    wait_for("the accepting side on the stream", Duration::from_secs(2), || b.path() == "stream");
    both_ways(&a, &b, 24 << 20);
    let (sa, sb) = (a.stats(), b.stats());
    assert!(sa.got_stream > 20 << 20 && sb.got_stream > 20 << 20, "{sa:?} {sb:?}");
    assert_eq!(p.counts().overrun, 0, "credits must keep the ring from overflowing");
    assert_eq!(sa.fallbacks + sb.fallbacks, 0);
}

#[test]
fn lost_damaged_and_cut_frames_are_sent_again() {
    for f in [
        Faults { drop_every: 37, ..Default::default() },
        Faults { corrupt_every: 29, ..Default::default() },
        Faults { truncate_every: 31, ..Default::default() },
        Faults { drop_every: 53, corrupt_every: 41, truncate_every: 67, ..Default::default() },
    ] {
        let p = Pair::new(256);
        let (a, b) = link_pair(Some(&p), 256, fast());
        assert_eq!(a.path(), "stream");
        p.set_faults(f);
        both_ways(&a, &b, 6 << 20);
        let (sa, sb) = (a.stats(), b.stats());
        assert!(sa.rewinds + sb.rewinds > 0, "{f:?}: {sa:?} {sb:?}");
        assert_eq!(sa.fallbacks + sb.fallbacks, 0, "{f:?}: {sa:?} {sb:?}");
        assert_eq!(sa.sent_tcp + sb.sent_tcp, 0, "{f:?}: {sa:?} {sb:?}");
    }
}

#[test]
fn credits_hold_with_a_small_ring() {
    let p = Pair::new(32);
    let (a, b) = link_pair(Some(&p), 32, Tuning { margin: 4, ..fast() });
    both_ways(&a, &b, 4 << 20);
    assert_eq!(p.counts().overrun, 0);
}

#[test]
fn a_stream_that_dies_ends_the_link_and_delivers_nothing_wrong() {
    // The stream loses everything mid-transfer: both sides end the link within about `dead`, with an error to every
    // reader and writer, and every byte read before that is the byte sent. Nothing goes on over TCP.
    for how in 0..4 {
        let p = Pair::new(256);
        p.set_faults(Faults { hold: 3, ..Default::default() });
        let (a, b) = link_pair(Some(&p), 256, fast());
        wait_for("both on the stream", Duration::from_secs(2), || b.path() == "stream");
        let start = Instant::now();
        let (ea, eb) = thread::scope(|s| {
            let ra = s.spawn(|| until_error(&a, 0));
            let rb = s.spawn(|| until_error(&b, 1));
            thread::sleep(Duration::from_millis(30));
            match how {
                0 => p.set_faults(Faults { dead: true, ..Default::default() }),
                1 => p.close_from(1),
                2 => a.give_up_stream("test"),
                _ => b.give_up_stream("test"),
            }
            (ra.join().unwrap(), rb.join().unwrap())
        });
        assert!(start.elapsed() < Duration::from_secs(2), "case {how}: {:?}", start.elapsed());
        for e in [&ea, &eb] {
            assert!(!e.contains("differ"), "case {how}: {e}");
        }
        let (sa, sb) = (a.stats(), b.stats());
        assert!(sa.fallbacks + sb.fallbacks >= 1, "case {how}: {sa:?} {sb:?}");
        assert_eq!(sa.sent_tcp + sb.sent_tcp, 0, "case {how}: no data over TCP: {sa:?} {sb:?}");
    }
}

/// Send and read pattern bytes both ways until the link fails; the error, or "differ" if a byte read was wrong.
fn until_error(me: &Link, who: u64) -> String {
    thread::scope(|s| {
        let w = s.spawn(|| {
            let mut at = 0u64;
            loop {
                let n = 40 << 10;
                if let Err(e) = me.send(&[&pattern(who, at, n)]) {
                    return e.to_string();
                }
                at += n as u64;
            }
        });
        let mut at = 0u64;
        let mut buf = vec![0u8; 64 << 10];
        let r = loop {
            match me.read_exact(&mut buf, Wait::Stall(Duration::from_secs(5))) {
                Ok(()) => {
                    if buf != pattern(1 - who, at, buf.len()) {
                        break format!("bytes {at}.. differ");
                    }
                    at += buf.len() as u64;
                }
                Err(e) => break e.to_string(),
            }
        };
        let wr = w.join().unwrap();
        format!("{r}; {wr}")
    })
}

#[test]
fn closing_a_link_on_the_stream_is_a_clean_end() {
    // The other side says goodbye and closes its stream device: this side reads what was sent, then the end of the
    // link, with no stream failure on either side.
    for _ in 0..20 {
        let p = Pair::new(256);
        p.set_faults(Faults { hold: 3, ..Default::default() });
        let (a, b) = link_pair(Some(&p), 256, fast());
        wait_for("both on the stream", Duration::from_secs(2), || b.path() == "stream");
        a.send(&[&pattern(0, 0, 100_000)]).unwrap();
        drop(a);
        let mut buf = vec![0u8; 100_000];
        b.read_exact(&mut buf, Wait::Stall(Duration::from_secs(2))).unwrap();
        assert!(buf == pattern(0, 0, 100_000));
        let e = b.read_exact(&mut [0u8; 1], Wait::Stall(Duration::from_secs(2))).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::UnexpectedEof, "{e}");
        assert_eq!(b.stats().fallbacks, 0, "{:?}", b.stats());
    }
}

#[test]
fn a_stream_that_does_not_come_up_fails_the_connection() {
    // Both ends have the stream but its device will not open: no link (rather than one over TCP).
    let p = Pair::new(256);
    p.set_faults(Faults { refuse_open: true, ..Default::default() });
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let tune = Tuning { stream_wait: Duration::from_millis(500), ..fast() };
    let ob = opts(Arc::new(SimStreams { pair: p.clone(), side: 1, ring: 256 }), tune.clone());
    let accepting = thread::spawn(move || {
        let (s, _) = listener.accept().unwrap();
        let Ok(Accepted::Link(l, _)) = Link::accept_with(s, &ob) else { panic!("accept") };
        l.read_exact(&mut [0u8; 1], Wait::Stall(Duration::from_secs(3))).unwrap_err()
    });
    let start = Instant::now();
    let e = Link::connect_with(&addr, "test", &opts(Arc::new(SimStreams { pair: p.clone(), side: 0, ring: 256 }), tune))
        .err()
        .expect("the connection must fail");
    assert!(format!("{e:#}").contains("stream was not up"), "{e:#}");
    assert!(start.elapsed() < Duration::from_secs(2), "{:?}", start.elapsed());
    accepting.join().unwrap();
    // A first epoch that fails is tried again at once: the link comes up on the stream.
    let p = Pair::new(256);
    p.set_faults(Faults { deaf_once: Some(1), hold: 3, ..Default::default() });
    let (a, b) = link_pair(Some(&p), 256, fast());
    assert_eq!(a.path(), "stream");
    both_ways(&a, &b, 1 << 20);
    assert_eq!(a.report()["transport"], "stream");
    // (Two when the next open finds the device still closing: the sim's close runs on a thread of its own.)
    assert!((1..=2).contains(&a.stats().failed_epochs), "{:?}", a.stats());
}

#[test]
fn stale_frames_of_an_earlier_session_are_ignored() {
    let p = Pair::new(256);
    // An earlier session's records (and garbage) wait for side 1.
    let (old_a, old_b) = link_pair(Some(&p), 256, fast());
    both_ways(&old_a, &old_b, 1 << 20);
    drop(old_b);
    for _ in 0..50 {
        let _ = old_a.send(&[&pattern(9, 0, 4000)]);
    }
    drop(old_a);
    p.stale(1, &pattern(7, 0, 64 << 10));
    let (a, b) = link_pair(Some(&p), 256, fast());
    wait_for("both on the stream", Duration::from_secs(3), || a.path() == "stream" && b.path() == "stream");
    both_ways(&a, &b, 4 << 20);
}

#[test]
fn the_other_side_going_away_is_an_error_at_once() {
    for stream in [false, true] {
        let p = Pair::new(256);
        let (a, b) = link_pair(stream.then_some(&p), 256, fast());
        drop(b);
        let start = Instant::now();
        let e = a.read_exact(&mut [0u8; 4], Wait::Idle { spin: Duration::ZERO }).unwrap_err();
        assert!(start.elapsed() < Duration::from_secs(2), "{e}");
        assert!(matches!(e.kind(), ErrorKind::UnexpectedEof | ErrorKind::ConnectionAborted | ErrorKind::ConnectionReset), "{e:?}");
    }
}

#[test]
fn a_full_receiver_holds_the_sender_without_a_fallback() {
    let p = Pair::new(64);
    let (a, b) = link_pair(Some(&p), 64, Tuning { inq_limit: 1 << 20, ..fast() });
    wait_for("both on the stream", Duration::from_secs(2), || b.path() == "stream");
    let total = 24 << 20;
    thread::scope(|s| {
        s.spawn(|| {
            for at in (0..total).step_by(1 << 20) {
                a.send(&[&pattern(0, at as u64, 1 << 20)]).unwrap();
            }
        });
        // Nobody reads for longer than every stream timer.
        thread::sleep(Duration::from_millis(1500));
        let mut buf = vec![0u8; 1 << 20];
        for at in (0..total).step_by(1 << 20) {
            b.read_exact(&mut buf, Wait::Stall(Duration::from_secs(10))).unwrap();
            assert!(buf == pattern(0, at as u64, 1 << 20));
        }
    });
    let (sa, sb) = (a.stats(), b.stats());
    assert_eq!(sa.fallbacks + sb.fallbacks, 0, "{sa:?} {sb:?}");
}

#[test]
fn a_reader_and_a_writer_thread_share_a_link() {
    let p = Pair::new(256);
    let (a, b) = link_pair(Some(&p), 256, fast());
    // Requests one way, replies the other, each side reading on a thread of its own (the chain node's pattern).
    let n = 300;
    thread::scope(|s| {
        let b2 = b.clone();
        let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
        s.spawn(move || {
            for _ in 0..n {
                let c = Conn { link: b2.clone() };
                let (h, blob) = c.recv().unwrap();
                assert_eq!(blob, pattern(h["i"].as_u64().unwrap(), 0, blob.len()));
                tx.send(blob).unwrap();
            }
        });
        s.spawn(move || {
            let c = Conn { link: b.clone() };
            for (i, blob) in rx.iter().enumerate() {
                c.send(&json!({"i": i, "n": blob.len()}), &blob[..blob.len() / 4]).unwrap();
            }
        });
        let c = Conn { link: a.clone() };
        for i in 0..n {
            let len = [40 << 10, 10 << 10, 4, 5 << 20][i % 4];
            c.send(&json!({"i": i}), &pattern(i as u64, 0, len)).unwrap();
            let (h, blob) = c.recv().unwrap();
            assert_eq!(h["i"].as_u64().unwrap() as usize, i);
            assert_eq!(blob, pattern(i as u64, 0, len / 4));
        }
    });
}

fn lane_pair(count: usize, tune: Tuning) -> (Lanes, Lanes) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let o = opts(Arc::new(Sysfs), tune);
    let o2 = o.clone();
    let accepted = thread::spawn(move || Lanes::accept_groups_with(&listener, 1, &o2).unwrap().pop().unwrap().1);
    let a = Lanes::connect_with(&vec![addr; count], "test", &o).unwrap();
    (a, accepted.join().unwrap())
}

fn payload(len: usize, side: usize, iteration: usize) -> Vec<u8> {
    (0..len).map(|i| (i.wrapping_mul(13) + side * 97 + iteration * 31) as u8).collect()
}

#[test]
fn exchanges_and_control_messages() {
    // Empty lane slices, unequal slices, decode/verification shapes and multi-MiB prefill; sizes change within a
    // session and control messages follow each exchange.
    let lengths = [0, 4, 8, 20, 5120 * 4, 3 * 5120 * 4, 3 * (256 << 10), 3 * (256 << 10) + 4, 256 * 5120 * 4, 4];
    for count in [1, 3] {
        let (mut a, mut b) = lane_pair(count, fast());
        assert_eq!(a.count(), count);
        thread::scope(|s| {
            let peer = s.spawn(|| {
                for (iteration, len) in lengths.into_iter().enumerate() {
                    let out = payload(len, 1, iteration);
                    let mut inp = vec![0; len];
                    b.exchange(&out, &mut inp).unwrap();
                    assert_eq!(inp, payload(len, 0, iteration));
                    assert_eq!(b.recv_message().unwrap(), iteration.to_le_bytes());
                    b.send_message(&iteration.to_le_bytes()).unwrap();
                }
            });
            for (iteration, len) in lengths.into_iter().enumerate() {
                let out = payload(len, 0, iteration);
                let mut inp = vec![0; len];
                a.exchange(&out, &mut inp).unwrap();
                assert_eq!(inp, payload(len, 1, iteration));
                a.send_message(&iteration.to_le_bytes()).unwrap();
                assert_eq!(a.recv_message().unwrap(), iteration.to_le_bytes());
            }
            peer.join().unwrap();
        });
    }
}

#[test]
fn a_lane_whose_cable_goes_away_carries_on_over_another() {
    let (mut a, mut b) = lane_pair(3, fast());
    let rounds = 40;
    thread::scope(|s| {
        let peer = s.spawn(|| {
            for i in 0..rounds {
                let mut inp = vec![0; 3 * 40960];
                b.exchange(&payload(3 * 40960, 1, i), &mut inp).unwrap();
                assert_eq!(inp, payload(3 * 40960, 0, i));
            }
        });
        for i in 0..rounds {
            if i == 10 {
                // Lane 1's own TCP connection goes away (its cable, as far as the link can tell).
                a.lanes[1].kill_path(0);
            }
            let mut inp = vec![0; 3 * 40960];
            a.exchange(&payload(3 * 40960, 0, i), &mut inp).unwrap();
            assert_eq!(inp, payload(3 * 40960, 1, i));
        }
        peer.join().unwrap();
    });
    assert!(a.lanes[1].stats().tcp_failovers + b.lanes[1].stats().tcp_failovers >= 1);
}

#[test]
fn message_waits() {
    let (mut a, mut b) = lane_pair(1, fast());
    // A delayed message within the first-byte limit, its framing kept.
    let peer = thread::spawn(move || {
        thread::sleep(Duration::from_millis(75));
        b.send_message(br#"{"ok":true}"#).unwrap();
        b.send_message(b"next").unwrap();
        thread::sleep(BUSY + Duration::from_millis(25));
        b.send_message(b"request").unwrap();
        b
    });
    assert_eq!(a.recv_message_first_within(Duration::from_secs(2)).unwrap(), br#"{"ok":true}"#);
    assert_eq!(a.recv_message().unwrap(), b"next");
    assert_eq!(a.recv_message_idle().unwrap(), b"request");
    let mut b = peer.join().unwrap();
    // A timeout consumes nothing.
    let start = Instant::now();
    let e = a.recv_message_first_within(Duration::from_millis(20)).unwrap_err();
    assert!(start.elapsed() >= Duration::from_millis(20) && start.elapsed() < Duration::from_secs(2), "{e:#}");
    b.send_message(b"ready").unwrap();
    assert_eq!(a.recv_message_first_within(Duration::from_secs(2)).unwrap(), b"ready");
    // The end of the link is an error at once, whatever the deadline.
    drop(b);
    let start = Instant::now();
    assert!(a.recv_message_first_within(Duration::from_secs(300)).is_err());
    assert!(start.elapsed() < Duration::from_secs(2));
}

#[test]
fn a_lone_frame_does_not_wait_for_a_retransmission() {
    // The NHIs do not read the last frames to arrive until up to three more follow them (`hold`). Every write ends
    // with three more frames, so a ping-pong runs at the stream's speed instead of the retransmission timer's.
    let p = Pair::new(256);
    p.set_faults(Faults { hold: 3, ..Default::default() });
    let tune = Tuning { rto: Duration::from_millis(50), ..fast() };
    let (a, b) = link_pair(Some(&p), 256, tune);
    assert_eq!(a.path(), "stream");
    wait_for("the accepting side on the stream", Duration::from_secs(2), || b.path() == "stream");
    let start = Instant::now();
    let peer = thread::spawn(move || {
        let mut buf = [0u8; 64];
        for _ in 0..200 {
            b.read_exact(&mut buf, Wait::Stall(Duration::from_secs(5))).unwrap();
            b.send(&[&buf]).unwrap();
        }
        b
    });
    let mut buf = [0u8; 64];
    for i in 0..200u64 {
        let m = pattern(0, i * 64, 64);
        a.send(&[&m]).unwrap();
        a.read_exact(&mut buf, Wait::Stall(Duration::from_secs(5))).unwrap();
        assert!(buf[..] == m[..], "round {i}");
    }
    let b = peer.join().unwrap();
    let (took, sa, sb) = (start.elapsed(), a.stats(), b.stats());
    // Waiting for retransmissions would take 200 x 50 ms.
    assert!(took < Duration::from_secs(2), "200 round trips took {took:?}\n{sa:?}\n{sb:?}");
    assert!(sa.rewinds + sb.rewinds < 5 && sa.pushes > 0 && sb.pushes > 0, "{sa:?}\n{sb:?}");
    assert_eq!((a.path(), b.path()), ("stream", "stream"));
}

#[test]
fn exchange_errors() {
    let (mut a, b) = lane_pair(3, fast());
    assert!(a.exchange(&[0; 3], &mut [0; 3]).is_err());
    assert!(a.exchange(&[0; 4], &mut [0; 8]).is_err());
    let error = a.exchange_within(&[0; 4], &mut [0; 4], Duration::from_millis(10)).unwrap_err();
    assert!(error.to_string().contains("stalled"), "{error:#}");
    drop(b);
    assert!(a.exchange(&[0; 4], &mut [0; 4]).is_err());
}

#[test]
fn read_and_write_traits_frame_like_a_socket() {
    use std::io::{BufReader, IoSlice, Read, Write};
    let p = Pair::new(256);
    let (mut a, b) = link_pair(Some(&p), 256, fast());
    let peer = thread::spawn(move || {
        let mut r = BufReader::with_capacity(1 << 20, b.clone());
        for i in 0..200u64 {
            let mut len = [0u8; 12];
            r.read_exact(&mut len).unwrap();
            let n = u64::from_le_bytes(len[4..].try_into().unwrap()) as usize;
            let mut blob = vec![0u8; n];
            r.read_exact(&mut blob).unwrap();
            assert_eq!(blob, pattern(i, 0, n));
        }
        // The end of the link reads as 0.
        let mut rest = Vec::new();
        r.read_to_end(&mut rest).unwrap();
        assert!(rest.is_empty());
    });
    for i in 0..200u64 {
        let n = [12, 40 << 10, 5 << 20, 0][i as usize % 4];
        let mut head = [0u8; 12];
        head[4..].copy_from_slice(&(n as u64).to_le_bytes());
        let blob = pattern(i, 0, n);
        assert_eq!(a.write_vectored(&[IoSlice::new(&head), IoSlice::new(&blob)]).unwrap(), 12 + n);
    }
    a.flush().unwrap();
    drop(a);
    peer.join().unwrap();
}

#[test]
fn an_epoch_comes_up_soon_after_a_late_open() {
    // The accepting side's device opens 700 ms late (past the fast probes), and every receiver holds its last three
    // frames: probes and their pushes must bring the epoch up within a few ms of the open.
    let p = Pair::new(256);
    p.set_faults(Faults { hold: 3, slow_open: Some((1, Duration::from_millis(700))), ..Default::default() });
    let (a, b) = link_pair(Some(&p), 256, Tuning { probe_timeout: Duration::from_secs(3), ..fast() });
    wait_for("both on the stream", Duration::from_secs(3), || a.path() == "stream" && b.path() == "stream");
    let (ua, ub) = (a.stats().first_up_ms.unwrap(), b.stats().first_up_ms.unwrap());
    assert!(ua > 690. && ua < 760., "the connecting side up after {ua} ms");
    assert!(ub < 760., "the accepting side up after {ub} ms");
    assert_eq!(a.stats().failed_epochs + b.stats().failed_epochs, 0);
}

#[test]
fn an_epoch_that_carries_one_way_is_given_up_at_once() {
    // The accepting side's first device reads nothing the connecting side writes (a whole epoch dead one way, as the
    // one-cable churn saw): the connecting side gives the epoch up after ONE_WAY and its next epoch, opened at once,
    // comes up.
    let p = Pair::new(256);
    p.set_faults(Faults { deaf_once: Some(1), hold: 3, ..Default::default() });
    let (a, b) = link_pair(Some(&p), 256, Tuning { probe_timeout: Duration::from_secs(3), ..fast() });
    wait_for("both on the stream", Duration::from_secs(2), || a.path() == "stream" && b.path() == "stream");
    let s = a.stats();
    assert_eq!(s.failed_epochs, 1, "{s:?}");
    assert!(s.first_up_ms.is_some_and(|ms| (50. ..250.).contains(&ms)), "{s:?}");
}

#[test]
fn a_busy_peer_does_not_fail_its_epoch() {
    // The accepting side stops turning for 300 ms (its link's lock held, as on a machine too busy to run it) right
    // after its first probe, before it has read the connecting side's: silence is no evidence of a dead direction, so
    // the epoch waits and comes up once the accepting side runs again, with no failed epoch. (Before, the connecting
    // side gave it up 50 ms after first hearing the other side.)
    let mut frozen = 0;
    for _ in 0..8 {
        let p = Pair::new(256);
        p.set_faults(Faults { hold: 3, ..Default::default() });
        let caught = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let c = caught.clone();
        let (a, b) = link_pair_with(Some(&p), 256, Tuning { probe_timeout: Duration::from_secs(3), ..fast() }, move |l| {
            let l = l.clone();
            thread::spawn(move || {
                let start = Instant::now();
                while start.elapsed() < Duration::from_secs(2) {
                    if let Ok(st) = l.inner.st.try_lock() {
                        let (probes, heard) = st.stream.progress();
                        if st.stream.phase == Phase::Probing && probes >= 1 {
                            if !heard {
                                c.store(true, std::sync::atomic::Ordering::SeqCst);
                                thread::sleep(Duration::from_millis(300));
                            }
                            return;
                        }
                    }
                    std::hint::spin_loop();
                }
            });
        });
        wait_for("both on the stream", Duration::from_secs(3), || a.path() == "stream" && b.path() == "stream");
        if !caught.load(std::sync::atomic::Ordering::SeqCst) {
            continue;
        }
        frozen += 1;
        let s = a.stats();
        assert_eq!(s.failed_epochs + b.stats().failed_epochs, 0, "{s:?}");
        assert!(s.first_up_ms.is_some_and(|ms| ms > 290.), "{s:?}");
    }
    assert!(frozen >= 2, "the accepting side was caught between its first probe and the other side's only {frozen} times");
}

#[test]
fn an_epoch_whose_other_side_is_not_heard_is_given_up_on_its_reports() {
    // The accepting side's frames stop arriving after its first probe and pushes (as in the main suite at 17:12),
    // while it reads the connecting side's and goes on: its reports over TCP show frames written that do not arrive,
    // so the connecting side gives the epoch up after ONE_WAY and the next one comes up.
    let p = Pair::new(256);
    p.set_faults(Faults { mute_once: Some((1, 4)), hold: 3, ..Default::default() });
    let (a, b) = link_pair(Some(&p), 256, Tuning { probe_timeout: Duration::from_secs(3), ..fast() });
    wait_for("both on the stream", Duration::from_secs(2), || a.path() == "stream" && b.path() == "stream");
    let s = a.stats();
    // (Two when the retry finds the device still closing: the sim's close runs on a thread of its own.)
    assert!((1..=2).contains(&s.failed_epochs), "{s:?}");
    assert!(s.first_up_ms.is_some_and(|ms| (50. ..300.).contains(&ms)), "{s:?}");
}

#[test]
fn reports_say_how_a_link_carried_its_data() {
    let p = Pair::new(256);
    let (a, b) = link_pair(Some(&p), 256, fast());
    wait_for("both on the stream", Duration::from_secs(2), || b.path() == "stream");
    for r in [a.report(), b.report()] {
        assert_eq!(r["transport"], "stream", "{r}");
        assert_eq!(r["stream_at_both_ends"], true, "{r}");
        assert!(r["first_up_ms"].as_f64().is_some_and(|ms| ms < 1000.), "{r}");
    }
    // No stream at both ends: TCP.
    let (a, _b) = link_pair(None, 256, fast());
    let r = a.report();
    assert_eq!(r["transport"], "tcp");
    assert_eq!(r["stream_at_both_ends"], false);
}

#[test]
fn a_peer_that_stops_for_a_second_does_not_lose_its_stream() {
    // The accepting side does not turn its link for a second while data flows both ways (its thread holds the link,
    // as a node thread waiting for the disk did): the stream waits for it, and every byte arrives.
    let p = Pair::new(256);
    p.set_faults(Faults { hold: 3, ..Default::default() });
    let d = Tuning::default();
    let (a, b) = link_pair(Some(&p), 256, Tuning { dead: d.dead, idle_dead: d.idle_dead, ..fast() });
    wait_for("both on the stream", Duration::from_secs(2), || b.path() == "stream");
    thread::scope(|s| {
        s.spawn(|| both_ways(&a, &b, 16 << 20));
        thread::sleep(Duration::from_millis(20));
        let st = b.inner.lock();
        thread::sleep(Duration::from_secs(1));
        drop(st);
    });
    let (sa, sb) = (a.stats(), b.stats());
    assert_eq!(sa.fallbacks + sb.fallbacks, 0, "{sa:?} {sb:?}");
}

/// `rounds` ping-pongs of 64 bytes, `a` first; the time they took.
fn ping_pongs(a: &Link, b: &Link, rounds: u64) -> Duration {
    let start = Instant::now();
    thread::scope(|s| {
        s.spawn(|| {
            let mut buf = [0u8; 64];
            for _ in 0..rounds {
                b.read_exact(&mut buf, Wait::Stall(Duration::from_secs(5))).unwrap();
                b.send(&[&buf]).unwrap();
            }
        });
        let mut buf = [0u8; 64];
        for i in 0..rounds {
            let m = pattern(0, i * 64, 64);
            a.send(&[&m]).unwrap();
            a.read_exact(&mut buf, Wait::Stall(Duration::from_secs(5))).unwrap();
            assert!(buf[..] == m[..], "round {i}");
        }
    });
    start.elapsed()
}

/// `rounds` exchanges of `len` bytes between the two ends of one link, checked, `a` waiting `gap` before each (as a
/// machine that computes between two all-reduces: nothing of the next exchange pushes the last one's frames along);
/// the time they took.
fn exchanges(a: &Link, b: &Link, rounds: usize, len: usize, gap: Duration) -> Duration {
    let (mut la, mut lb) = (Lanes::from_links(vec![a.clone()]), Lanes::from_links(vec![b.clone()]));
    let start = Instant::now();
    thread::scope(|s| {
        s.spawn(|| {
            for i in 0..rounds {
                let mut inp = vec![0; len];
                lb.exchange(&payload(len, 1, i), &mut inp).unwrap();
                assert_eq!(inp, payload(len, 0, i));
            }
        });
        for i in 0..rounds {
            if !gap.is_zero() {
                thread::sleep(gap);
            }
            let mut inp = vec![0; len];
            la.exchange(&payload(len, 0, i), &mut inp).unwrap();
            assert_eq!(inp, payload(len, 1, i));
        }
    });
    start.elapsed()
}

#[test]
fn each_direction_gets_the_pushes_its_hold_needs() {
    // The hold belongs to a direction of the cable. Both sides have measured theirs when `connect` returns, a write
    // is followed by that many frames and no more, and nothing waits for a nudge or a retransmission.
    for hold_at in [[0, 0], [0, 1], [3, 0], [2, 1], [3, 3]] {
        let p = Pair::new(256);
        p.set_faults(Faults { hold_at, ..Default::default() });
        let tune = Tuning { rto: Duration::from_millis(50), ..fast() };
        let (a, b) = link_pair(Some(&p), 256, tune);
        assert_eq!((a.path(), b.path()), ("stream", "stream"));
        // The connecting side (end 0) sends toward end 1. A reading can be one high (a frame in flight), never low.
        let (ha, hb) = (a.stats().hold.expect("measured"), b.stats().hold.expect("measured"));
        assert!(ha >= hold_at[1] as u32 && hb >= hold_at[0] as u32, "{hold_at:?}: {ha} {hb}");
        wait_for("exact estimates", Duration::from_secs(3), || {
            (a.stats().hold, b.stats().hold) == (Some(hold_at[1] as u32), Some(hold_at[0] as u32))
        });
        let (pa, pb) = (a.stats().pushes, b.stats().pushes);
        let rounds = 300;
        let took = ping_pongs(&a, &b, rounds);
        let (sa, sb) = (a.stats(), b.stats());
        assert!(took < Duration::from_secs(2), "{hold_at:?}: {rounds} round trips took {took:?}\n{sa:?}\n{sb:?}");
        assert_eq!(sa.rewinds + sb.rewinds, 0, "{hold_at:?}: {sa:?}\n{sb:?}");
        // One write a round, each followed by the direction's hold in frames (an acknowledgement on its own now
        // and then adds a few).
        for (s, before, hold) in [(&sa, pa, hold_at[1]), (&sb, pb, hold_at[0])] {
            let pushed = s.pushes - before;
            assert!(pushed >= rounds * hold as u64 && pushed <= (rounds + 20) * hold as u64, "{hold_at:?}: {pushed} pushes\n{s:?}");
        }
        // Exchanges with a pause between them. (A thread of a busy test machine that loses its core for a
        // millisecond brings a nudge from the other side; the estimates must not follow those.)
        let took = exchanges(&a, &b, 200, 9216, Duration::from_micros(300));
        let (sa, sb) = (a.stats(), b.stats());
        assert!(took < Duration::from_secs(2), "{hold_at:?}: 200 exchanges took {took:?}\n{sa:?}\n{sb:?}");
        assert_eq!(sa.rewinds + sb.rewinds, 0, "{hold_at:?}: {sa:?}\n{sb:?}");
        assert_eq!((sa.hold, sb.hold), (Some(hold_at[1] as u32), Some(hold_at[0] as u32)), "{hold_at:?}: {sa:?}\n{sb:?}");
    }
}

#[test]
fn a_hold_that_grows_costs_a_nudge_and_is_learnt() {
    // The link measured no hold; then the receivers start to hold frames back (the hold of a direction has differed
    // between opens of a device; within one it has not been seen to change, but nothing promises that). The first
    // messages wait for nudges (1 ms each here, not the 50 ms retransmission timer) or for the acknowledgements
    // that say what is held, the estimates go up, and the rest run as before. Ping-pongs first, where a message is
    // one frame and only nudges get it through, then exchanges with a pause between them, where both sides wait at
    // once.
    let gap = Duration::from_micros(300);
    for exchange in [false, true] {
        let p = Pair::new(256);
        let tune = Tuning { rto: Duration::from_millis(50), nudge: Duration::from_millis(1), ..fast() };
        let (a, b) = link_pair(Some(&p), 256, tune);
        wait_for("estimates", Duration::from_secs(3), || (a.stats().hold, b.stats().hold) == (Some(0), Some(0)));
        p.set_faults(Faults { hold_at: [2, 3], ..Default::default() });
        let took = if exchange { exchanges(&a, &b, 300, 9216, gap) } else { ping_pongs(&a, &b, 300) };
        let (sa, sb) = (a.stats(), b.stats());
        assert!(took < Duration::from_secs(2), "exchange {exchange}: 300 rounds took {took:?}\n{sa:?}\n{sb:?}");
        assert!(sa.hold_raises > 0 && sb.hold_raises > 0, "exchange {exchange}: {sa:?}\n{sb:?}");
        assert!(sa.hold >= Some(3) && sb.hold >= Some(2), "exchange {exchange}: {sa:?}\n{sb:?}");
        assert!(sa.rewinds + sb.rewinds <= 2, "exchange {exchange}: {sa:?}\n{sb:?}");
        // Once learnt, nothing waits, and the estimates stay.
        let raises = sa.hold_raises + sb.hold_raises;
        let took = if exchange { exchanges(&a, &b, 300, 9216, gap) } else { ping_pongs(&a, &b, 300) };
        let (sa, sb) = (a.stats(), b.stats());
        assert!(took < Duration::from_millis(600), "exchange {exchange}: 300 more rounds took {took:?}\n{sa:?}\n{sb:?}");
        assert_eq!(sa.hold_raises + sb.hold_raises, raises, "exchange {exchange}: {sa:?}\n{sb:?}");
        assert_eq!((a.path(), b.path()), ("stream", "stream"));
    }
}

#[test]
fn an_idle_link_follows_its_hold_up_and_down() {
    // The connecting side's keepalives are hold probes: an idle link notices that its directions hold more, or
    // less, within a few keepalives, and the accepting side learns its own direction's hold from the same pings.
    let p = Pair::new(256);
    p.set_faults(Faults { hold_at: [3, 2], ..Default::default() });
    let (a, b) = link_pair(Some(&p), 256, fast());
    wait_for("estimates", Duration::from_secs(3), || (a.stats().hold, b.stats().hold) == (Some(2), Some(3)));
    p.set_faults(Faults::default());
    wait_for("lower estimates", Duration::from_secs(5), || (a.stats().hold, b.stats().hold) == (Some(0), Some(0)));
    p.set_faults(Faults { hold_at: [1, 2], ..Default::default() });
    wait_for("higher estimates", Duration::from_secs(5), || (a.stats().hold, b.stats().hold) == (Some(2), Some(1)));
    let (sa, sb) = (a.stats(), b.stats());
    assert_eq!(sa.fallbacks + sb.fallbacks + sa.rewinds + sb.rewinds, 0, "{sa:?}\n{sb:?}");
    both_ways(&a, &b, 2 << 20);
    assert_eq!((a.path(), b.path()), ("stream", "stream"));
}

#[test]
fn a_fixed_number_of_pushes_is_kept() {
    // `Tuning::push` (link-test's `--push`): no measuring, no nudges.
    let p = Pair::new(256);
    p.set_faults(Faults { hold: 2, ..Default::default() });
    let tune = Tuning { push: Some(2), rto: Duration::from_millis(50), ..fast() };
    let (a, b) = link_pair(Some(&p), 256, tune);
    let took = ping_pongs(&a, &b, 200);
    let (sa, sb) = (a.stats(), b.stats());
    assert!(took < Duration::from_secs(2), "{took:?}\n{sa:?}\n{sb:?}");
    assert_eq!((sa.hold, sa.hold_probes, sa.nudges, sb.nudges), (None, 0, 0, 0), "{sa:?}\n{sb:?}");
    assert!(sa.pushes >= 400 && sa.pushes < 480, "{sa:?}");
}
