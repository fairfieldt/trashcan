//! Cross-machine all-reduce benchmark for tensor parallelism over Thunderbolt (`trashcan allreduce-bench`).
//!
//! Every GPU of both machines runs `steps` segments per token, as a 4-way tensor-parallel decode of Qwen3.8-27B
//! runs 128 all-reduces (two per layer). A segment optionally streams VRAM like a projection, then writes its
//! rank's partial vector to host memory. The host waits for its GPUs, adds their partials, exchanges the sum with
//! the other machine over its lanes, writes the total back for every local GPU and releases the next segment,
//! which checks every element of the total before using it. A GCN 1.0 shader cannot wait on memory written from
//! outside its GPU, so every all-reduce is a submission boundary; `--release` and `--notify` choose how the host
//! crosses it.
use crate::{
    gpu::Instance,
    link::Lanes,
    vulkan::{Buffer, Device, Push},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    net::TcpListener,
    rc::Rc,
    time::{Duration, Instant},
};

const TIMEOUT: Duration = Duration::from_secs(30);
/// Workgroups of the VRAM streaming kernel.
const WORK_GROUPS: u32 = 256;
/// VRAM streamed per submission while warming the clocks up.
const WARM_MIB: f64 = 64.;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Release {
    /// Submit the next segment once the total is written.
    Submit,
    /// Submit every segment up front, each waiting on a timeline semaphore that the host signals.
    Signal,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Notify {
    /// Block in vkWaitSemaphores.
    Wait,
    /// Spin on the semaphore counter.
    Poll,
}

/// What both machines must agree on; the side that starts a session sends it to the other.
#[derive(Clone, Debug, Serialize, Deserialize, clap::Args)]
pub struct Config {
    /// All-reduces per token (two per layer: 128 for Qwen3.8-27B).
    #[arg(long, default_value_t = 128)]
    pub steps: usize,
    /// Tokens to run; the first is a warm-up and is not counted.
    #[arg(long, default_value_t = 21)]
    pub tokens: usize,
    /// Token columns per all-reduce: 1 for decode, 512 for a prefill chunk.
    #[arg(long, default_value_t = 1)]
    pub cols: usize,
    /// Hidden size.
    #[arg(long, default_value_t = 5120)]
    pub width: usize,
    /// MiB of VRAM each segment streams before writing its partial, like a projection (0: none).
    #[arg(long, default_value_t = 0.)]
    pub work_mib: f64,
    #[arg(long, value_enum, default_value = "submit")]
    pub release: Release,
    #[arg(long, value_enum, default_value = "wait")]
    pub notify: Notify,
    /// Run each token's segments back to back without the host in between: GPU time only, nothing checked.
    #[arg(long)]
    pub free: bool,
    /// No GPUs: exchange host vectors only, to time the lanes.
    #[arg(long)]
    pub host_only: bool,
    /// Keep the GPUs busy this long first, so their clocks have ramped up.
    #[arg(long, default_value_t = 500)]
    pub warmup_ms: u64,
}

#[derive(clap::Args)]
pub struct Options {
    #[command(flatten)]
    pub config: Config,
    /// Local GPUs.
    #[arg(long, value_delimiter = ',', default_value = "0,1")]
    pub gpus: Vec<usize>,
    /// Start a session with the other machine: one TCP address per lane, e.g. invite.tb0:7071,invite.tb1:7071.
    #[arg(long, value_delimiter = ',')]
    pub connect: Vec<String>,
    /// Serve sessions started by the other machine, accepting TCP lanes on this address.
    #[arg(long)]
    pub listen: Option<String>,
    /// USB4STREAM devices instead of TCP, one per cable, in the same cable order on both machines.
    #[arg(long, value_delimiter = ',')]
    pub stream: Vec<String>,
    /// With --stream: serve sessions instead of starting one.
    #[arg(long)]
    pub serve: bool,
    /// Run several configurations over the same lanes: a file of JSON objects, one per line, each overriding
    /// the command-line configuration. One result line is printed per configuration.
    #[arg(long)]
    pub plan: Option<std::path::PathBuf>,
}

/// Without lanes the benchmark runs on this machine alone and the all-reduce covers its GPUs only.
pub fn run(o: &Options) -> Result<()> {
    if let Some(addr) = &o.listen {
        let listener = TcpListener::bind(addr).with_context(|| format!("listening on {addr}"))?;
        eprintln!("allreduce: serving on {addr}");
        loop {
            if let Err(e) = Lanes::accept(&listener).and_then(|mut lanes| serve(o, &mut lanes)) {
                eprintln!("allreduce: session failed: {e:#}");
            }
        }
    }
    if o.serve {
        ensure!(!o.stream.is_empty(), "--serve needs --stream; TCP sessions are served with --listen");
        loop {
            if let Err(e) = Lanes::streams(&o.stream, true).and_then(|mut lanes| serve(o, &mut lanes)) {
                eprintln!("allreduce: session failed: {e:#}");
            }
            // Let the other machine close its ends before the streams reopen.
            std::thread::sleep(Duration::from_millis(500));
        }
    }
    let mut configs = vec![o.config.clone()];
    if let Some(plan) = &o.plan {
        let text = std::fs::read_to_string(plan).with_context(|| format!("reading {}", plan.display()))?;
        configs = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let mut c = serde_json::to_value(&o.config)?;
                let changes: Value = serde_json::from_str(l).with_context(|| format!("plan line {l}"))?;
                for (k, v) in changes.as_object().context("plan lines are JSON objects")? {
                    c[k] = v.clone();
                }
                Ok(serde_json::from_value(c)?)
            })
            .collect::<Result<Vec<Config>>>()?;
    }
    let mut lanes = if !o.connect.is_empty() {
        Some(Lanes::connect(&o.connect)?)
    } else if !o.stream.is_empty() {
        Some(Lanes::streams(&o.stream, false)?)
    } else {
        None
    };
    for c in &configs {
        let ranks = local_ranks(c, &o.gpus);
        if let Some(l) = lanes.as_mut() {
            l.send_message(&serde_json::to_vec(&json!({"config": c, "ranks": ranks}))?)?;
            let reply: Value = serde_json::from_slice(&l.recv_message()?)?;
            ensure!(reply["ranks"] == json!(ranks), "the other machine has {} ranks, this one {ranks}", reply["ranks"]);
        }
        let machines = if lanes.is_some() { 2 } else { 1 };
        let mine = bench(c, &o.gpus, 0, machines, lanes.as_mut())?;
        let peer: Value = match lanes.as_mut() {
            Some(l) => serde_json::from_slice(&l.recv_message()?)?,
            None => Value::Null,
        };
        let link = lanes.as_ref().map(|l| json!({"kind": l.kind, "lanes": l.count()}));
        println!("{}", json!({"config": c, "link": link, "machine0": mine, "machine1": peer}));
    }
    if let Some(l) = lanes.as_mut() {
        l.send_message(&serde_json::to_vec(&json!({"bye": true}))?)?;
    }
    Ok(())
}

fn local_ranks(c: &Config, gpus: &[usize]) -> usize {
    if c.host_only {
        1
    } else {
        gpus.len()
    }
}

/// Configurations from the machine that started the session, run until it says goodbye.
fn serve(o: &Options, lanes: &mut Lanes) -> Result<()> {
    loop {
        let request: Value = serde_json::from_slice(&lanes.recv_message()?)?;
        if request.get("bye").is_some() {
            return Ok(());
        }
        let c: Config = serde_json::from_value(request["config"].clone())?;
        let ranks = local_ranks(&c, &o.gpus);
        lanes.send_message(&serde_json::to_vec(&json!({"ranks": ranks}))?)?;
        ensure!(request["ranks"] == json!(ranks), "the other machine has {} ranks, this one {ranks}", request["ranks"]);
        let stats = bench(&c, &o.gpus, 1, 2, Some(lanes))?;
        lanes.send_message(&serde_json::to_vec(&stats)?)?;
        eprintln!("allreduce: configuration done, token_ms {}", stats["token_ms"]);
    }
}

/// Rank `rank`'s partial for `step`, as `allreduce_partial.comp` computes it.
fn pattern(rank: usize, step: usize, i: usize) -> f32 {
    ((i * 7 + step * 13 + rank * 29) % 61) as f32 - 30.
}

fn push(ptr: [u64; 3], n: usize, k: usize, m: usize, offset: usize) -> Push {
    let mut p = Push { ptr: [ptr[0], ptr[1], ptr[2], 0, 0, 0, 0, 0], ..Push::default() };
    p.u[0] = n as u32;
    p.u[1] = k as u32;
    p.u[2] = m as u32;
    p.u[10] = offset as u32;
    p
}

fn barrier(d: &Device, cmd: vk::CommandBuffer, src: vk::PipelineStageFlags, dst: vk::PipelineStageFlags, write: vk::AccessFlags, read: vk::AccessFlags) {
    let b = [vk::MemoryBarrier::default().src_access_mask(write).dst_access_mask(read)];
    unsafe { d.dev.cmd_pipeline_barrier(cmd, src, dst, vk::DependencyFlags::empty(), &b, &[], &[]) };
}

/// One local GPU: a command buffer per segment and the host buffers it exchanges through.
struct Gpu {
    d: Rc<Device>,
    /// This rank's partial, written by the GPU.
    out: Buffer,
    /// The total, written by the host.
    inp: Buffer,
    err: Buffer,
    _keep: [Buffer; 3],
    query: vk::QueryPool,
    done: vk::Semaphore,
    release: vk::Semaphore,
    cmds: Vec<vk::CommandBuffer>,
    warm: vk::CommandBuffer,
}

impl Gpu {
    fn new(inst: &Rc<Instance>, pd: vk::PhysicalDevice, c: &Config, rank: usize, ranks: usize) -> Result<Self> {
        let d = Device::new(inst.clone(), pd)?;
        let n = c.cols * c.width;
        let out = d.buffer(n * 4, true, false, None)?;
        let inp = d.buffer(n * 4, true, false, None)?;
        let x = d.upload(&vec![0u8; n * 4])?;
        let err = d.upload(bytemuck::cast_slice(&[0u32, u32::MAX]))?;
        // Two windows of vec4s, alternated by step, so consecutive segments never find their data in L2.
        let window = (c.work_mib * (1 << 16) as f64) as usize;
        let warm_window = (WARM_MIB * (1 << 16) as f64) as usize;
        let span = window.max(warm_window);
        let work = d.buffer(2 * span * 16, false, false, None)?;
        let sink = d.buffer(WORK_GROUPS as usize * 16, false, false, None)?;
        let steps = c.steps as u32;
        let query = unsafe {
            d.dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2 * steps),
                None,
            )?
        };
        let groups = n.div_ceil(256) as u32;
        let mut cmds = Vec::new();
        for s in 0..c.steps {
            let cmd = d.begin()?;
            unsafe {
                if s == 0 {
                    d.dev.cmd_reset_query_pool(cmd, query, 0, 2 * steps);
                }
                d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, query, 2 * s as u32);
            }
            // The host wrote the total before this submission, or before signalling the semaphore it waits on.
            barrier(&d, cmd, vk::PipelineStageFlags::HOST, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::HOST_WRITE, vk::AccessFlags::SHADER_READ);
            if s > 0 && !c.free {
                d.dispatch(cmd, "allreduce_check", push([x.address, inp.address, err.address], n, s - 1, ranks, 0), groups, 1);
            }
            if window > 0 {
                let p = push([work.address, 0, sink.address], window, 0, 0, (s % 2) * span);
                d.dispatch(cmd, "allreduce_work", p, WORK_GROUPS, 1);
            }
            d.dispatch(cmd, "allreduce_partial", push([out.address, 0, 0], n, s, rank, 0), groups, 1);
            barrier(&d, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::HOST_READ);
            unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query, 2 * s as u32 + 1) };
            d.finish(cmd)?;
            cmds.push(cmd);
        }
        let warm = d.begin()?;
        d.dispatch(warm, "allreduce_work", push([work.address, 0, sink.address], warm_window, 0, 0, 0), WORK_GROUPS, 1);
        d.finish(warm)?;
        let done = d.timeline(false)?;
        let release = d.timeline(false)?;
        Ok(Self { d, out, inp, err, _keep: [x, work, sink], query, done, release, cmds, warm })
    }

    /// Submit segment `step`, optionally waiting for the release semaphore to reach `wait`; it signals `done`.
    fn submit(&self, step: usize, wait: Option<u64>, done: u64) -> Result<()> {
        self.d.submit(self.cmds[step], wait.map(|v| (self.release, v)), Some((self.done, done)), vk::Fence::null())
    }

    fn signal(&self, value: u64) -> Result<()> {
        unsafe { self.d.dev.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(self.release).value(value))? };
        Ok(())
    }

    fn wait(&self, value: u64, notify: Notify) -> Result<()> {
        match notify {
            Notify::Wait => unsafe {
                let semaphores = [self.done];
                let values = [value];
                let info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
                self.d.dev.wait_semaphores(&info, TIMEOUT.as_nanos() as u64).context("GPU segment")?;
            },
            Notify::Poll => {
                let start = Instant::now();
                while unsafe { self.d.dev.get_semaphore_counter_value(self.done)? } < value {
                    ensure!(start.elapsed() < TIMEOUT, "GPU segment timed out");
                    std::hint::spin_loop();
                }
            }
        }
        Ok(())
    }

    fn partial(&self, n: usize) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.out.ptr as *const f32, n) }
    }

    fn write_total(&self, total: &[f32]) {
        unsafe { std::ptr::copy_nonoverlapping(total.as_ptr(), self.inp.ptr as *mut f32, total.len()) };
    }

    /// Execution time of every segment of the latest token, from its timestamps.
    fn exec_us(&self, steps: usize) -> Result<Vec<f64>> {
        let mut ticks = vec![0u64; 2 * steps];
        unsafe {
            self.d.dev.get_query_pool_results(self.query, 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?;
        }
        Ok(ticks.chunks(2).map(|t| t[1].wrapping_sub(t[0]) as f64 * self.d.period as f64 / 1e3).collect())
    }

    /// Mismatching elements the check kernel counted, and the lowest mismatching index.
    fn errors(&self) -> Result<(u32, u32)> {
        let e: Vec<u32> = bytemuck::cast_slice(&self.err.download()?).to_vec();
        Ok((e[0], e[1]))
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.d.dev.device_wait_idle();
            self.d.dev.free_command_buffers(self.d.pool, &self.cmds);
            self.d.dev.free_command_buffers(self.d.pool, &[self.warm]);
            self.d.dev.destroy_query_pool(self.query, None);
            self.d.dev.destroy_semaphore(self.done, None);
            self.d.dev.destroy_semaphore(self.release, None);
        }
    }
}

/// Microseconds per step (and milliseconds per token) of every token after the warm-up.
#[derive(Default)]
struct Timings {
    token_ms: Vec<f64>,
    step: Vec<f64>,
    /// From releasing a segment until the host saw every local GPU finish it.
    gpu: Vec<f64>,
    /// The slowest local GPU's execution of the segment, from its timestamps.
    exec: Vec<f64>,
    /// `gpu - exec`: submission or release latency plus completion notice.
    launch_notify: Vec<f64>,
    sum: Vec<f64>,
    exchange: Vec<f64>,
    /// Writing the total for the local GPUs and releasing the next segment.
    write_release: Vec<f64>,
}

fn percentiles(v: &[f64]) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let at = |q: f64| s[((s.len() - 1) as f64 * q).round() as usize];
    let r = |x: f64| (x * 100.).round() / 100.;
    json!({"p50": r(at(0.5)), "p90": r(at(0.9)), "p99": r(at(0.99)), "max": r(at(1.)), "mean": r(s.iter().sum::<f64>() / s.len() as f64)})
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// This machine's side of the benchmark; `machine` is its index among `machines`.
fn bench(c: &Config, gpu_ids: &[usize], machine: usize, machines: usize, mut lanes: Option<&mut Lanes>) -> Result<Value> {
    ensure!(c.tokens >= 2 && c.steps >= 1, "need two tokens (the first is a warm-up) and a step");
    ensure!(!(c.host_only && c.free), "--free times the GPUs; --host-only has none");
    let n = c.cols * c.width;
    let per_machine = local_ranks(c, gpu_ids);
    let ranks = per_machine * machines;
    let gpus = if c.host_only {
        Vec::new()
    } else {
        let inst = Rc::new(Instance::new()?);
        let all = inst.amd_gpus()?;
        gpu_ids
            .iter()
            .enumerate()
            .map(|(i, &g)| Gpu::new(&inst, *all.get(g).with_context(|| format!("GPU {g} not found"))?, c, machine * per_machine + i, ranks))
            .collect::<Result<Vec<_>>>()?
    };
    let warm_until = Instant::now() + Duration::from_millis(c.warmup_ms);
    while !gpus.is_empty() && Instant::now() < warm_until {
        for g in &gpus {
            g.d.submit(g.warm, None, None, vk::Fence::null())?;
        }
        for g in &gpus {
            unsafe { g.d.dev.queue_wait_idle(g.d.queue)? };
        }
    }
    if let Some(l) = lanes.as_deref_mut() {
        l.exchange(&[0; 4], &mut [0; 4])?;
    }
    let mut local = vec![0f32; n];
    let mut remote = vec![0f32; n];
    let mut t = Timings::default();
    let mut mismatches = 0usize;
    let mut check = |total: &[f32], step: usize| {
        mismatches += total.iter().enumerate().filter(|(i, v)| **v != (0..ranks).map(|r| pattern(r, step, *i)).sum::<f32>()).count();
    };
    for token in 0..c.tokens {
        // Timeline values only grow, and start at 1 so the first wait is not already satisfied.
        let base = (1 + token * (c.steps + 1)) as u64;
        let counted = token > 0;
        if !c.free && c.release == Release::Signal {
            for g in &gpus {
                for s in 0..c.steps {
                    g.submit(s, Some(base + s as u64), base + s as u64 + 1)?;
                }
            }
        }
        let start = Instant::now();
        if c.free {
            for g in &gpus {
                for s in 0..c.steps {
                    g.submit(s, None, base + s as u64 + 1)?;
                }
            }
            for g in &gpus {
                g.wait(base + c.steps as u64, c.notify)?;
            }
        } else {
            for g in &gpus {
                match c.release {
                    Release::Submit => g.submit(0, None, base + 1)?,
                    Release::Signal => g.signal(base)?,
                }
            }
            let mut released = Instant::now();
            for s in 0..c.steps {
                let value = base + s as u64 + 1;
                if c.host_only {
                    for (i, v) in local.iter_mut().enumerate() {
                        *v = pattern(machine, s, i);
                    }
                    released = Instant::now();
                }
                for g in &gpus {
                    g.wait(value, c.notify)?;
                }
                let done = Instant::now();
                if let Some((first, rest)) = gpus.split_first() {
                    local.copy_from_slice(first.partial(n));
                    for g in rest {
                        for (a, b) in local.iter_mut().zip(g.partial(n)) {
                            *a += b;
                        }
                    }
                }
                let summed = Instant::now();
                if let Some(l) = lanes.as_deref_mut() {
                    l.exchange(bytemuck::cast_slice(&local), bytemuck::cast_slice_mut(&mut remote))?;
                    for (a, b) in local.iter_mut().zip(&remote) {
                        *a += b;
                    }
                }
                let exchanged = Instant::now();
                for g in &gpus {
                    g.write_total(&local);
                }
                if s + 1 < c.steps {
                    for g in &gpus {
                        match c.release {
                            Release::Submit => g.submit(s + 1, None, value + 1)?,
                            Release::Signal => g.signal(value)?,
                        }
                    }
                }
                let next = Instant::now();
                if counted {
                    t.gpu.push(us(done - released));
                    t.sum.push(us(summed - done));
                    t.exchange.push(us(exchanged - summed));
                    t.write_release.push(us(next - exchanged));
                    t.step.push(us(next - released));
                }
                // The GPUs check every total but the token's last; check that one (and host-only totals) here.
                if c.host_only || s + 1 == c.steps {
                    check(&local, s);
                }
                released = next;
            }
        }
        let token_ms = start.elapsed().as_secs_f64() * 1e3;
        if counted {
            t.token_ms.push(token_ms);
            if !gpus.is_empty() {
                let per_gpu = gpus.iter().map(|g| g.exec_us(c.steps)).collect::<Result<Vec<_>>>()?;
                let first = t.gpu.len() - if c.free { 0 } else { c.steps };
                for s in 0..c.steps {
                    let exec = per_gpu.iter().map(|e| e[s]).fold(0., f64::max);
                    t.exec.push(exec);
                    if !c.free {
                        t.launch_notify.push(t.gpu[first + s] - exec);
                    }
                }
            }
        }
    }
    let mut errors = 0u64;
    let mut first_error = None;
    for g in &gpus {
        let (count, index) = g.errors()?;
        errors += count as u64;
        if count > 0 {
            first_error = Some(index);
        }
    }
    Ok(json!({
        "machine": machine,
        "ranks": ranks,
        "gpu_mismatches": errors,
        "first_gpu_mismatch": first_error,
        "host_mismatches": mismatches,
        "token_ms": percentiles(&t.token_ms),
        "step_us": percentiles(&t.step),
        "gpu_us": percentiles(&t.gpu),
        "exec_us": percentiles(&t.exec),
        "launch_notify_us": percentiles(&t.launch_notify),
        "sum_us": percentiles(&t.sum),
        "exchange_us": percentiles(&t.exchange),
        "write_release_us": percentiles(&t.write_release),
    }))
}
