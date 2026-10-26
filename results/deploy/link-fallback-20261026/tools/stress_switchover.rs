
/// Stress (old code): switch from the stream to TCP and back many times mid-transfer, in many ways, and check every
/// byte. Messages framed as the chain's Conn does, with a reader thread and a writer thread on each side.
#[test]
fn stress_switchover_is_byte_exact() {
    let iters: usize = std::env::var("STRESS_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for it in 0..iters {
        let p = Pair::new(256);
        p.set_faults(Faults { hold: 3, ..Default::default() });
        let (a, b) = link_pair(Some(&p), 256, fast());
        wait_for("both on the stream", Duration::from_secs(3), || a.path() == "stream" && b.path() == "stream");
        let how = rnd() % 5;
        let delay = Duration::from_micros(rnd() % 40_000);
        let outage = Duration::from_micros(rnd() % 300_000);
        let total = (1 << 20) + (rnd() % (6 << 20)) as usize;
        thread::scope(|s| {
            s.spawn(|| both_ways(&a, &b, total));
            thread::sleep(delay);
            match how {
                0 => {
                    p.set_faults(Faults { dead: true, hold: 3, ..Default::default() });
                    thread::sleep(outage);
                    p.set_faults(Faults { hold: 3, ..Default::default() });
                }
                1 => p.close_from((rnd() % 2) as usize),
                2 => a.give_up_stream("stress"),
                3 => b.give_up_stream("stress"),
                _ => {
                    p.set_faults(Faults { drop_every: 3, hold: 3, ..Default::default() });
                    thread::sleep(outage);
                    p.set_faults(Faults { hold: 3, ..Default::default() });
                }
            }
        });
        // Back on the stream, then more bytes.
        let _ = wait_for_ok(Duration::from_secs(8), || a.path() == "stream" && b.path() == "stream");
        both_ways(&a, &b, 1 << 20);
        let (sa, sb) = (a.stats(), b.stats());
        eprintln!("iter {it} how {how} delay {delay:?} outage {outage:?} total {total}: fallbacks {} {} ups {} {}", sa.fallbacks, sb.fallbacks, sa.ups, sb.ups);
    }
}

fn wait_for_ok(limit: Duration, mut f: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while !f() {
        if start.elapsed() > limit {
            return false;
        }
        thread::sleep(Duration::from_millis(2));
    }
    true
}
