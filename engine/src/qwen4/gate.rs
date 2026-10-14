//! The gate of graphs submitted ahead of their input (`GpuStage::presubmit`, `GpuStage::head_presubmit`). Such a
//! graph starts with `qwen4_gate`: one wave polls a word of host memory until the host writes the submission's ticket
//! there (`Gate::release`, once the input is written), evicting the word's line from the GPU's L2 between polls by
//! reading device-local memory (on Tahiti the line otherwise stays in the L2 and the wave never sees the write;
//! results/qwen4/gate-shader-20261014).
//!
//! The wait is bounded on the GPU, so no job waits anywhere near amdgpu's 2 s timeout whatever the host does, and a
//! process killed with gates armed leaves nothing waiting for it:
//! - by shader-clock ticks (GFX6's shader clock runs at the current DPM level's clock): `BOUND_S` at the top level,
//!   where the clock holder (`crate::clocks`) keeps the GPUs while they decode, `MAX_WAIT_S` at the lowest;
//! - by a poll count: `BOUND_S` over the fastest a sweep reads VRAM, so the ticks end a wait unless the clock stops.
//!
//! The sweep must read VRAM: from memory amdgpu evicted to system memory it is a stream of PCIe reads (on alight, two
//! processes overcommitting one GPU's VRAM, with gates sweeping evicted weights, preceded a PCIe completion timeout
//! and the loss of the GPU; results/qwen4/decode-gate-20261014). Gates are off for the whole process (`disable`) when
//! the startup check finds a sweep slow, when amdgpu reports any of the process's VRAM evicted (`resident`, before
//! every submission ahead), and when a gate times a sweep slower than VRAM (it gives up at once).
//!
//! A gate that gives up (or is abandoned, `Gate::cancel`) writes a predicate that skips every later dispatch of its
//! graph (conditional rendering, 0.01 us per dispatch, `qwen4 gate-skip-check`); the host then runs the stage again
//! without a gate.
use crate::vulkan::{Buffer, Device, Push};
use anyhow::{bail, Result};
use ash::vk;
use std::rc::Rc;
use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};

/// The longest a gate waits at the top DPM level.
pub const BOUND_S: f64 = 0.1;
/// The shader clocks of the top and the lowest DPM level.
const TOP_HZ: f64 = 850e6;
const MIN_HZ: f64 = 300e6;
/// The longest a gate waits at the lowest DPM level (0.28 s).
pub const MAX_WAIT_S: f64 = BOUND_S * TOP_HZ / MIN_HZ;
/// amdgpu's job timeout on the Macs' kernels.
const KERNEL_TIMEOUT_S: f64 = 2.0;
const _: () = assert!(MAX_WAIT_S * 4.0 < KERNEL_TIMEOUT_S, "a gate must end long before amdgpu resets the GPU");
/// Sweeps the startup check tries, cheapest first: bytes of device-local memory read between polls (of memory the
/// stage holds anyway, its first layer's expert weights), a line every `STRIDE` bytes. 8 MiB evicted the gate's line in
/// all 21 placements tried in results/qwen4/gate-shader-20261014; some gate pages need 16 MiB.
const SWEEPS: [usize; 3] = [8 << 20, 16 << 20, 32 << 20];
const STRIDE: usize = 4096;
/// The fastest a sweep reads a line of VRAM at the top DPM level (6.1-6.8 ns measured): the poll bound is `BOUND_S`
/// over it, so the tick bound ends a wait first.
const MIN_LINE_S: f64 = 4e-9;
/// The most shader-clock ticks a sweep may take per line: from VRAM about 5 at the top DPM level, from evicted memory
/// 180-700. A gate whose sweep takes longer gives up at once.
const MAX_LINE_TICKS: u32 = 32;
/// A poll load of fewer shader-clock ticks than this hit the L2 (230-260 ticks at the top and the lowest DPM level),
/// instead of reading host memory over PCIe (550-800).
const MISS_TICKS: u32 = 400;
/// Polls of the startup check.
const PROBE_POLLS: u32 = 24;
/// Pages the startup check tries per gate with each sweep: a sweep evicts the lines of some pages and not of others
/// (on alight, 8 MiB at 4 KiB missed one gate page of four, which 16 MiB at 4 KiB evicted).
const PAGES: usize = 8;
/// Words of a gate's page: the gate, the ticket it waits for, how its wait ended.
const WORD: usize = 0;
const TICKET: usize = 16;
const RESULT: usize = 32;
/// Written with a ticket into the gate word: abandon the submission (its graph skips its work).
const CANCEL: u32 = 1 << 31;

/// Gates that gave up since the process started; their stages were redone.
pub static TIMEOUTS: AtomicUsize = AtomicUsize::new(0);
/// Gates are off for the rest of the process (`disable`).
static OFF: AtomicBool = AtomicBool::new(false);

/// Turn gates off for the rest of the process, logging why (once).
pub fn disable(reason: &str) {
    if !OFF.swap(true, Ordering::SeqCst) {
        eprintln!("qwen4: graphs are no longer submitted ahead: {reason}");
    }
}

/// Whether gates are on for this process.
pub fn enabled() -> bool {
    !OFF.load(Ordering::SeqCst)
}

/// KiB of this process's VRAM buffers that amdgpu keeps outside VRAM (`amd-evicted-vram` in the fdinfo of its render
/// nodes, summed); `None` when no open file reports it. With `rescan` every open file is looked at, else the files
/// found last time (a few microseconds each), and all of them again when one of those no longer reports it.
fn evicted_vram_kib(rescan: bool) -> Option<u64> {
    static FILES: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());
    // The last look at every open file found none that reports it.
    static UNREPORTED: AtomicBool = AtomicBool::new(false);
    let mut files = FILES.lock().unwrap_or_else(|e| e.into_inner());
    if !rescan {
        if UNREPORTED.load(Ordering::Relaxed) {
            return None;
        }
        if let Some(kib) = files.iter().map(|p| amdgpu_evicted(p)).sum::<Option<u64>>().filter(|_| !files.is_empty()) {
            return Some(kib);
        }
    }
    files.clear();
    let mut total = None;
    for e in std::fs::read_dir("/proc/self/fdinfo").ok()?.flatten() {
        if let Some(kib) = amdgpu_evicted(&e.path()) {
            files.push(e.path());
            total = Some(total.unwrap_or(0) + kib);
        }
    }
    UNREPORTED.store(total.is_none(), Ordering::Relaxed);
    total
}

/// `amd-evicted-vram` of the open file whose fdinfo is `path`, when it is an amdgpu render node.
fn amdgpu_evicted(path: &std::path::Path) -> Option<u64> {
    let s = std::fs::read_to_string(path).ok()?;
    if !s.lines().any(|l| l.starts_with("drm-driver:") && l.ends_with("amdgpu")) {
        return None;
    }
    parse_evicted(&s)
}

/// The `amd-evicted-vram` KiB of one render node's fdinfo.
fn parse_evicted(fdinfo: &str) -> Option<u64> {
    let v = fdinfo.lines().find_map(|l| l.strip_prefix("amd-evicted-vram:"))?;
    v.trim().strip_suffix("KiB")?.trim().parse().ok()
}

/// Whether all of this process's VRAM buffers are in VRAM, by amdgpu's fdinfo; when any is not, gates go off for the
/// process. Before every submission ahead (while its graph would wait anyway). True when the kernel does not report it
/// (the gates' own sweep timing still applies).
pub fn resident() -> bool {
    match evicted_vram_kib(false) {
        Some(kib) if kib > 0 => {
            disable(&format!(
                "amdgpu moved {kib} KiB of this process's VRAM to system memory, where a gate's sweep would read it"
            ));
            false
        }
        _ => true,
    }
}

/// Shader-clock ticks (in units of 65536, as the gate reads them) of `seconds` at the top DPM level's clock.
fn ticks(seconds: f64) -> u32 {
    (seconds * TOP_HZ) as u32 >> 16
}

/// A GPU's gate parameters, from the startup check (`probe`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Device-local bytes read between polls, a line every `STRIDE` bytes.
    pub bytes: usize,
    /// Bounds: polls, and shader-clock ticks in units of 65536.
    pub polls: u32,
    pub ticks: u32,
}

impl Params {
    /// A sweep of `bytes` with the gates' bounds: `BOUND_S` at the top clock, and as many polls as the fastest sweep of
    /// VRAM fits in that time.
    fn new(bytes: usize) -> Self {
        let lines = (bytes / STRIDE) as f64;
        Self { bytes, polls: (BOUND_S / (lines * MIN_LINE_S)).ceil() as u32, ticks: ticks(BOUND_S) }
    }

    fn lines(&self) -> u32 {
        (self.bytes / STRIDE) as u32
    }

    fn push(&self, gate: &Gate, sweep: &Buffer) -> Push {
        let mut p = Push::default();
        (p.ptr[0], p.ptr[1], p.ptr[2]) = (gate.page.address, sweep.address, gate.pred.address);
        (p.u[0], p.u[1], p.u[2], p.u[3]) = (self.polls, self.ticks, self.lines(), (STRIDE / 4) as u32);
        p.u[4] = self.lines() * MAX_LINE_TICKS;
        p
    }
}

/// How a gate's last wait ended.
#[derive(Clone, Copy, Debug, Default)]
pub struct Outcome {
    /// 1 opened, 2 out of polls, 3 out of time, 4 abandoned, 5 a sweep slower than VRAM.
    pub how: u32,
    /// Polls before the one that saw the ticket (0: open at the first).
    pub polls: u32,
    pub ticks: u64,
    pub load_ticks: u64,
    pub fewest: u32,
    pub most: u32,
    /// The last word the gate read and the ticket it read (it opens when they match).
    pub last: u32,
    pub ticket: u32,
    /// The most ticks one sweep took.
    pub slowest_sweep: u32,
}

impl Outcome {
    pub fn opened(&self) -> bool {
        self.how == 1
    }
}

/// A gate: a page of host-cached memory (the gate word, the ticket, how the wait ended) and the device-local
/// predicate its graph's dispatches run on.
pub struct Gate {
    pub page: Buffer,
    pub pred: Buffer,
    ticket: u32,
}

impl Gate {
    pub fn new(gpu: &Rc<Device>) -> Result<Self> {
        let page = gpu.buffer(4096, true, false, None)?;
        page.write(&[0u8; 4096])?;
        let pred = if gpu.conditional() { gpu.predicate_buffer(64)? } else { gpu.buffer(64, false, false, None)? };
        Ok(Self { page, pred, ticket: 0 })
    }

    fn write(&self, word: usize, v: u32) {
        unsafe { std::ptr::write_volatile((self.page.ptr as *mut u32).add(word), v) };
        // A full fence (x86 mfence), so that the write leaves this core before anything after it.
        fence(Ordering::SeqCst);
    }

    fn read(&self, word: usize) -> u32 {
        unsafe { std::ptr::read_volatile((self.page.ptr as *const u32).add(word)) }
    }

    /// A new ticket, written where the gate of the next submission reads it; the gate opens once `release` writes it
    /// into the gate word. Before the submission.
    pub fn arm(&mut self) {
        self.ticket = self.ticket % (CANCEL - 1) + 1;
        self.write(TICKET, self.ticket);
    }

    /// The ticket of the armed submission.
    pub fn ticket(&self) -> u32 {
        self.ticket
    }

    /// Let the armed submission through: its input is written.
    pub fn release(&self) {
        self.write(WORD, self.ticket);
    }

    /// Abandon the armed submission: its gate ends at its next poll and its graph skips its work.
    pub fn cancel(&self) {
        self.write(WORD, self.ticket | CANCEL);
    }

    /// How the last wait ended (once its graph ended).
    pub fn outcome(&self) -> Outcome {
        let r = |i: usize| self.read(RESULT + i);
        Outcome {
            how: r(0),
            polls: r(1),
            ticks: r(2) as u64 | (r(3) as u64) << 32,
            load_ticks: r(4) as u64 | (r(5) as u64) << 32,
            fewest: r(6),
            most: r(7),
            last: r(8),
            ticket: r(9),
            slowest_sweep: r(10),
        }
    }
}

/// Record a gate at the start of a graph: timestamps around it in `queries` at `slot` and `slot + 1`, then the
/// barrier that hands its predicate to the command processor, the visibility of the host's input, and conditional
/// rendering on the predicate (the caller ends it with `Device::end_conditional` before the graph's end).
pub fn record(gpu: &Device, cmd: vk::CommandBuffer, params: &Params, gate: &Gate, sweep: &Buffer, queries: vk::QueryPool, slot: u32) -> Result<()> {
    unsafe { gpu.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, slot) };
    gpu.dispatch(cmd, "qwen4_gate", params.push(gate, sweep), 1, 1);
    unsafe {
        gpu.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, slot + 1);
        let b = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT)];
        gpu.dev.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT,
            vk::DependencyFlags::empty(),
            &b,
            &[],
            &[],
        );
    }
    gpu.host_barrier(cmd, false);
    gpu.begin_conditional(cmd, &gate.pred, 0)
}

/// The shader clock (MHz) over a gate's wait of `gpu_us` by the GPU's timestamps; `None` for waits too short to tell.
fn clock_mhz(o: &Outcome, gpu_us: f64) -> Option<f64> {
    (gpu_us >= 50.0).then(|| o.ticks as f64 / gpu_us)
}

/// Check a gate's wait against the GPU timestamps around it: within its bound, and a shader clock at a DPM level's
/// rate. An error turns gates off (the run itself ended by its bound).
pub fn check(o: &Outcome, gpu_us: f64) -> Result<()> {
    anyhow::ensure!(gpu_us <= MAX_WAIT_S * 1.1e6, "a gate waited {gpu_us:.0} us, past its {MAX_WAIT_S:.2} s bound");
    if let Some(mhz) = clock_mhz(o, gpu_us) {
        anyhow::ensure!((250.0..=900.0).contains(&mhz), "a gate's shader clock ran at {mhz:.0} MHz over {gpu_us:.0} us");
    }
    Ok(())
}

/// The startup check of a GPU's gates (`label` names the stage in log lines): each of `gates` waits, never released,
/// for `PROBE_POLLS` polls with each sweep of `SWEEPS` in turn, on its page and if need be on up to `PAGES - 1` fresh
/// pages, until every poll read host memory (none hit the L2); a gate keeps the page that passed. The first sweep under
/// which every gate found such a page sets the parameters.
/// `None`, with a log line, when the gates cannot work here: no sweep evicts the gate's line, the shader clock
/// disagrees with the GPU timestamps, the device cannot bound or skip (no shader clock, no conditional rendering), the
/// check fails, or `TRASHCAN_Q4_GATE=0`; graphs are then submitted as their input arrives. When the sweep memory is not
/// in VRAM (not device-local, amdgpu reports VRAM evicted, or a sweep is slow), gates go off for the whole process.
pub fn probe(gpu: &Rc<Device>, label: &str, memory: &Buffer, mut gates: [&mut Gate; 2]) -> Option<Params> {
    if std::env::var("TRASHCAN_Q4_GATE").is_ok_and(|v| v == "0") || !enabled() {
        return None;
    }
    let no = |why: &str| -> Option<Params> {
        eprintln!("qwen4: {label}: {why}; graphs are not submitted ahead");
        None
    };
    if !gpu.shader_clock || !gpu.conditional() {
        return no("no shader clock or conditional rendering");
    }
    if !memory.device_local {
        return no("the memory gates read between polls is not device-local");
    }
    let vram = match evicted_vram_kib(true) {
        None => "amdgpu does not report evicted VRAM: the sweeps' timing alone guards it",
        Some(0) => "none of the process's VRAM evicted",
        Some(kib) => {
            disable(&format!("{label}: amdgpu moved {kib} KiB of this process's VRAM to system memory"));
            return None;
        }
    };
    match probe_sweeps(gpu, label, memory, &mut gates, vram) {
        Ok(p) => p,
        Err(e) => no(&format!("the gate check failed: {e:#}")),
    }
}

fn probe_sweeps(
    gpu: &Rc<Device>,
    label: &str,
    memory: &Buffer,
    gates: &mut [&mut Gate; 2],
    vram: &str,
) -> Result<Option<Params>> {
    let q = unsafe {
        gpu.dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None)?
    };
    let result = (|| -> Result<Option<Params>> {
        'sweeps: for bytes in SWEEPS.into_iter().filter(|&b| b <= memory.size) {
            let trial = Params { polls: PROBE_POLLS, ticks: ticks(0.05), ..Params::new(bytes) };
            for (i, gate) in gates.iter_mut().enumerate() {
                let mut tries = 0;
                loop {
                    tries += 1;
                    match check_page(gpu, label, q, &trial, gate, memory, i)? {
                        Check::Miss => break,
                        Check::Clock => return Ok(None),
                        Check::Slow(why) => {
                            disable(&format!("{label}: {why}"));
                            return Ok(None);
                        }
                        Check::Hit if tries < PAGES => **gate = Gate::new(gpu)?,
                        Check::Hit => continue 'sweeps,
                    }
                }
            }
            let p = Params::new(bytes);
            eprintln!(
                "qwen4: {label}: gates read {} MiB every {STRIDE} B between polls and give up after {BOUND_S} s at the \
                 top clock or {} polls ({vram})",
                bytes >> 20,
                p.polls
            );
            return Ok(Some(p));
        }
        eprintln!("qwen4: {label}: the gate's polls hit the L2 with every sweep tried; graphs are not submitted ahead");
        Ok(None)
    })();
    unsafe { gpu.dev.destroy_query_pool(q, None) };
    result
}

/// How one page of a gate fared in the startup check.
enum Check {
    /// Every poll read host memory.
    Miss,
    Hit,
    /// The shader clock disagrees with the GPU timestamps.
    Clock,
    /// The sweep is slower than VRAM: why.
    Slow(String),
}

/// One gate's page under one sweep in the startup check (`probe`): the gate polls `PROBE_POLLS` times, never released.
fn check_page(gpu: &Rc<Device>, label: &str, q: vk::QueryPool, trial: &Params, gate: &mut Gate, memory: &Buffer, i: usize) -> Result<Check> {
    let cmd = gpu.begin()?;
    unsafe {
        gpu.dev.cmd_reset_query_pool(cmd, q, 0, 2);
        gpu.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0);
    }
    gpu.dispatch(cmd, "qwen4_gate", trial.push(gate, memory), 1, 1);
    unsafe { gpu.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1) };
    gpu.host_barrier(cmd, true);
    gpu.finish(cmd)?;
    gate.arm();
    gpu.run(cmd)?;
    let mut t = [0u64; 2];
    unsafe { gpu.dev.get_query_pool_results(q, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
    let gpu_us = t[1].wrapping_sub(t[0]) as f64 * gpu.period as f64 / 1e3;
    let o = gate.outcome();
    let per_line = o.slowest_sweep / trial.lines();
    match o.how {
        5 => {
            return Ok(Check::Slow(format!(
                "a sweep of the gates' {} MiB took {} shader-clock ticks ({per_line} per line; from VRAM about 5): \
                 the memory is not in VRAM",
                trial.bytes >> 20,
                o.slowest_sweep
            )))
        }
        3 => return Ok(Check::Slow(format!("the gate check ran out of time after {} polls ({gpu_us:.0} us)", o.polls))),
        2 if o.polls == PROBE_POLLS => {}
        how => bail!("the gate check ended with how {how} after {} polls", o.polls),
    }
    let Some(mhz) = clock_mhz(&o, gpu_us).filter(|m| (250.0..=900.0).contains(m)) else {
        eprintln!(
            "qwen4: {label}: the gate's shader clock ({} ticks) disagrees with the GPU timestamps ({gpu_us:.0} us); \
             graphs are not submitted ahead",
            o.ticks
        );
        return Ok(Check::Clock);
    };
    let miss = o.fewest >= MISS_TICKS;
    eprintln!(
        "qwen4: {label}: gate {i} check, {} MiB read every {STRIDE} B between polls: poll loads {}-{} ticks, sweeps up to \
         {} ticks ({per_line} per line) at {mhz:.0} MHz, {:.1} us per poll{}",
        trial.bytes >> 20,
        o.fewest,
        o.most,
        o.slowest_sweep,
        gpu_us / PROBE_POLLS as f64,
        if miss { "" } else { ": hits in the L2" }
    );
    Ok(if miss { Check::Miss } else { Check::Hit })
}

/// Percentiles of `v` (nearest rank): count, min, p10, p50, p90, p99, max.
fn spread(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| s[((s.len() as f64 * q) as usize).min(s.len() - 1)];
    serde_json::json!({"n": s.len(), "min": s[0], "p10": at(0.1), "p50": at(0.5), "p90": at(0.9), "p99": at(0.99),
        "max": s[s.len() - 1]})
}

/// What a stage's gates did (forward and head scoring together).
#[derive(Default, Debug)]
pub struct Stats {
    /// Graphs released by the host, and how many of them found their gate open at the first poll.
    pub released: usize,
    pub open_at_start: usize,
    /// Gates that gave up at a bound, gates abandoned, and gates that found a sweep slower than VRAM.
    pub timeouts: usize,
    pub abandoned: usize,
    pub slow_sweeps: usize,
    /// Host write of the ticket to the GPU timestamp after the gate, us (calibrated timestamps only).
    pub release_us: Vec<f64>,
    /// GPU us of a gate that opened at its first poll (its cost per graph).
    pub open_us: Vec<f64>,
    /// GPU ms a gate waited, when it did, the mean shader-clock ticks of its poll loads (550-800: every poll read host
    /// memory), and the most ticks of one of its sweeps.
    pub wait_ms: Vec<f64>,
    pub poll_load_ticks: Vec<f64>,
    pub sweep_ticks: Vec<f64>,
}

impl Stats {
    /// Record a released run: its outcome, its GPU us by the timestamps around the gate, and the release's latency.
    pub fn add(&mut self, o: &Outcome, gpu_us: f64, release_us: Option<f64>) {
        self.released += 1;
        match o.how {
            1 if o.polls == 0 => {
                self.open_at_start += 1;
                self.open_us.push(gpu_us);
            }
            1 => {
                self.wait_ms.push(gpu_us / 1e3);
                self.poll_load_ticks.push(o.load_ticks as f64 / (o.polls + 1) as f64);
                self.sweep_ticks.push(o.slowest_sweep as f64);
            }
            4 => self.abandoned += 1,
            5 => self.slow_sweeps += 1,
            _ => self.timeouts += 1,
        }
        self.release_us.extend(release_us);
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"released": self.released, "open_at_start": self.open_at_start, "timeouts": self.timeouts,
            "abandoned": self.abandoned, "slow_sweeps": self.slow_sweeps,
            "release_to_past_gate_us": spread(&self.release_us), "open_gate_gpu_us": spread(&self.open_us),
            "wait_ms": spread(&self.wait_ms), "poll_load_ticks": spread(&self.poll_load_ticks),
            "slowest_sweep_ticks": spread(&self.sweep_ticks)})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_end_far_inside_the_kernel_timeout_at_every_dpm_level() {
        // The tick bound at the top level (850 MHz, measured 821-841) and at the lowest (300 MHz, measured 297).
        let t = (ticks(BOUND_S) as f64) * 65536.0;
        assert!((t / TOP_HZ - BOUND_S).abs() < 1e-3);
        assert!((t / MIN_HZ - MAX_WAIT_S).abs() < 1e-3);
        assert!(t / 821e6 < 0.11 && t / 296.9e6 < 0.29);
    }

    #[test]
    fn the_poll_bound_comes_after_the_tick_bound_at_the_fastest_sweep_measured() {
        for bytes in SWEEPS {
            let p = Params::new(bytes);
            // 6.1 ns per line, the fastest measured: the polls outlast BOUND_S.
            let polls_s = p.polls as f64 * p.lines() as f64 * 6.1e-9;
            assert!(polls_s > BOUND_S * 1.4 && polls_s < MAX_WAIT_S, "{bytes}: {polls_s}");
            // From VRAM (about 5 ticks per line) a sweep is far under its limit, from system memory (180+) far over.
            assert!(5 * p.lines() * 4 < p.lines() * MAX_LINE_TICKS && 180 * p.lines() > p.lines() * MAX_LINE_TICKS * 4);
        }
    }

    #[test]
    fn every_sweep_reads_whole_waves_of_lines() {
        // The gate reads 512 lines per round of its sweep loop.
        for bytes in SWEEPS {
            assert_eq!(bytes % STRIDE, 0);
            assert_eq!((bytes / STRIDE) % 512, 0, "{bytes}");
        }
    }

    #[test]
    fn fdinfo_reports_evicted_vram() {
        let fdinfo = "pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\ndrm-pdev:\t0000:06:00.0\n\
                      drm-memory-vram:\t4775936 KiB\namd-evicted-vram:\t1024 KiB\namd-requested-vram:\t4776960 KiB\n";
        assert_eq!(parse_evicted(fdinfo), Some(1024));
        assert_eq!(parse_evicted("drm-driver:\tamdgpu\namd-evicted-vram:\t0 KiB\n"), Some(0));
        assert_eq!(parse_evicted("drm-driver:\tamdgpu\ndrm-memory-vram:\t1 KiB\n"), None);
    }

    #[test]
    fn stats_count_how_gates_ended() {
        let mut s = Stats::default();
        let o = |how, polls| Outcome { how, polls, ..Default::default() };
        s.add(&o(1, 0), 4.5, Some(9.0));
        s.add(&o(1, 30), 400.0, Some(12.0));
        s.add(&o(3, 9000), 100000.0, None);
        s.add(&o(4, 2), 30.0, None);
        s.add(&o(5, 0), 1800.0, None);
        assert_eq!((s.released, s.open_at_start, s.timeouts, s.abandoned, s.slow_sweeps), (5, 1, 1, 1, 1));
        assert_eq!(s.wait_ms, vec![0.4]);
        assert_eq!(s.release_us, vec![9.0, 12.0]);
        // 400 us at the top level's 841 MHz.
        assert!(check(&Outcome { ticks: 336_400, ..o(1, 30) }, 400.0).is_ok());
        // 400 us at 2000 ticks per us: a shader clock no DPM level has.
        assert!(check(&Outcome { ticks: 800_000, ..o(1, 30) }, 400.0).is_err());
        assert!(check(&o(3, 9000), 700_000.0).is_err());
    }
}
