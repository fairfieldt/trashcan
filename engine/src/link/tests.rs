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
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let (sa, sb): (Arc<dyn Streams>, Arc<dyn Streams>) = match pair {
        Some(p) => (Arc::new(SimStreams { pair: p.clone(), side: 0, ring }), Arc::new(SimStreams { pair: p.clone(), side: 1, ring })),
        None => (Arc::new(Sysfs), Arc::new(Sysfs)),
    };
    let ob = opts(sb, tune.clone());
    let accepted = thread::spawn(move || {
        let (s, _) = listener.accept().unwrap();
        match Link::accept_with(s, &ob).unwrap() {
            Accepted::Link(l, tag) => {
                assert_eq!(tag, "test");
                l
            }
            Accepted::Joined(_) => panic!("join"),
        }
    });
    let a = Link::connect_with(&addr, "test", &opts(sa, tune)).unwrap();
    let b = accepted.join().unwrap();
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
        // With faults this dense (up to 1 frame in 15), a record the go-back can lose again and again sends the
        // stream back to TCP after `dead`; the data stays intact either way.
        assert!(sa.got_stream + sb.got_stream > 1 << 20, "{f:?}: on the stream: {sa:?} {sb:?}");
    }
}

#[test]
fn credits_hold_with_a_small_ring_and_overdrive_recovers() {
    let p = Pair::new(32);
    let (a, b) = link_pair(Some(&p), 32, Tuning { margin: 4, ..fast() });
    both_ways(&a, &b, 4 << 20);
    assert_eq!(p.counts().overrun, 0);
    // Sending past the credits overruns the ring; the link recovers every byte.
    let p = Pair::new(32);
    let (a, b) = link_pair(Some(&p), 32, Tuning { overdrive: 64, ..fast() });
    both_ways(&a, &b, 4 << 20);
    assert!(p.counts().overrun > 0);
}

#[test]
fn dead_stream_falls_back_to_tcp_and_returns() {
    let p = Pair::new(256);
    let (a, b) = link_pair(Some(&p), 256, fast());
    wait_for("both on the stream", Duration::from_secs(2), || b.path() == "stream");
    thread::scope(|s| {
        s.spawn(|| both_ways(&a, &b, 16 << 20));
        thread::sleep(Duration::from_millis(30));
        p.set_faults(Faults { dead: true, ..Default::default() });
        wait_for("fallback", Duration::from_secs(3), || a.path() == "tcp" && b.path() == "tcp");
        thread::sleep(Duration::from_millis(50));
        p.set_faults(Faults::default());
    });
    wait_for("the stream back", Duration::from_secs(5), || a.path() == "stream" && b.path() == "stream");
    both_ways(&a, &b, 2 << 20);
    let (sa, sb) = (a.stats(), b.stats());
    assert!(sa.fallbacks + sb.fallbacks >= 1 && sa.ups >= 2, "{sa:?} {sb:?}");
}

#[test]
fn closed_stream_falls_back_and_returns() {
    let p = Pair::new(256);
    let (a, b) = link_pair(Some(&p), 256, fast());
    wait_for("both on the stream", Duration::from_secs(2), || b.path() == "stream");
    both_ways(&a, &b, 1 << 20);
    p.close_from(1);
    wait_for("fallback", Duration::from_secs(2), || b.path() == "tcp");
    both_ways(&a, &b, 3 << 20);
    wait_for("the stream back", Duration::from_secs(5), || a.path() == "stream" && b.path() == "stream");
    both_ways(&a, &b, 1 << 20);
}

#[test]
fn a_device_that_will_not_open_leaves_tcp() {
    let p = Pair::new(256);
    p.set_faults(Faults { refuse_open: true, ..Default::default() });
    let (a, b) = link_pair(Some(&p), 256, fast());
    assert_eq!(a.path(), "tcp");
    both_ways(&a, &b, 2 << 20);
    p.set_faults(Faults::default());
    wait_for("the stream once the device opens", Duration::from_secs(5), || a.path() == "stream" && b.path() == "stream");
    both_ways(&a, &b, 2 << 20);
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
fn reports_say_how_a_link_carried_its_data() {
    // On the stream from its first epoch.
    let p = Pair::new(256);
    let (a, b) = link_pair(Some(&p), 256, fast());
    wait_for("both on the stream", Duration::from_secs(2), || b.path() == "stream");
    for r in [a.report(), b.report()] {
        assert_eq!(r["transport"], "stream", "{r}");
        assert_eq!(r["stream_at_both_ends"], true, "{r}");
        assert!(r["first_up_ms"].as_f64().is_some_and(|ms| ms < 1000.), "{r}");
    }
    // A device that does not open at first: on TCP, said so after `tcp_warn`, then on the stream after failed
    // epochs.
    let p = Pair::new(256);
    p.set_faults(Faults { refuse_open: true, ..Default::default() });
    let (a, b) = link_pair(Some(&p), 256, Tuning { tcp_warn: Duration::from_millis(100), ..fast() });
    assert_eq!(a.report()["transport"], "tcp");
    assert!(a.report()["first_up_ms"].is_null());
    wait_for("the warning", Duration::from_secs(2), || a.stats().tcp_warnings == 1);
    p.set_faults(Faults::default());
    wait_for("the stream", Duration::from_secs(5), || a.path() == "stream" && b.path() == "stream");
    let r = a.report();
    assert_eq!(r["transport"], "retried", "{r}");
    assert!(r["failed_epochs"].as_u64().is_some_and(|n| n >= 1), "{r}");
    assert_eq!(a.stats().tcp_warnings, 1, "one warning per stretch on TCP");
    // No stream at both ends: TCP, and no warning.
    let (a, _b) = link_pair(None, 256, Tuning { tcp_warn: Duration::from_millis(50), ..fast() });
    thread::sleep(Duration::from_millis(150));
    let r = a.report();
    assert_eq!(r["transport"], "tcp");
    assert_eq!(r["stream_at_both_ends"], false);
    assert_eq!(a.stats().tcp_warnings, 0);
}
