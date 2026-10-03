//! Native BF16 Gemma4 drafter using the target's last sliding/global KV caches.
use crate::{
    generation::Sampler,
    model::{AttentionPartitions, Engine},
    vulkan::{attention_shaders, Buffer, Device, Push},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use memmap2::Mmap;
use serde::Deserialize;
use std::{collections::HashMap, fs::File, path::Path, rc::Rc};
type DraftOutput = (Vec<(f32, u32)>, Vec<f32>);
#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}
struct Checkpoint {
    map: Mmap,
    tensors: HashMap<String, Tensor>,
    start: usize,
}
impl Checkpoint {
    fn open(path: &Path) -> Result<Self> {
        let map = unsafe { Mmap::map(&File::open(path)?)? };
        ensure!(map.len() >= 8, "truncated safetensors");
        let n = usize::try_from(u64::from_le_bytes(map[..8].try_into()?))?;
        let start = 8usize.checked_add(n).context("header overflow")?;
        ensure!(start <= map.len(), "truncated header");
        let mut h: serde_json::Value = serde_json::from_slice(&map[8..start])?;
        h.as_object_mut()
            .context("invalid header")?
            .remove("__metadata__");
        let tensors: HashMap<String, Tensor> = serde_json::from_value(h)?;
        for (name, t) in &tensors {
            let end = start
                .checked_add(t.data_offsets[1])
                .context("offset overflow")?;
            ensure!(
                t.dtype == "BF16" && t.data_offsets[0] <= t.data_offsets[1] && end <= map.len(),
                "invalid tensor {name}"
            );
            let size = t
                .shape
                .iter()
                .try_fold(2usize, |s, d| s.checked_mul(*d))
                .context("shape overflow")?;
            ensure!(
                t.data_offsets[1] - t.data_offsets[0] == size,
                "tensor shape/size mismatch: {name}"
            );
        }
        Ok(Self {
            map,
            tensors,
            start,
        })
    }
    fn data(&self, name: &str) -> Result<(&Tensor, &[u8])> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("missing assistant tensor {name}"))?;
        Ok((
            t,
            &self.map[self.start + t.data_offsets[0]..self.start + t.data_offsets[1]],
        ))
    }
    fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let (_, b) = self.data(name)?;
        Ok(b.chunks_exact(2)
            .map(|s| f32::from_bits((u16::from_le_bytes(s.try_into().unwrap()) as u32) << 16))
            .collect())
    }
    fn validate(&self) -> Result<()> {
        let check = |name: &str, shape: &[usize]| -> Result<()> {
            ensure!(
                self.data(name)?.0.shape == shape,
                "unsupported assistant shape {name}"
            );
            Ok(())
        };
        check("model.embed_tokens.weight", &[262144, 1024])?;
        check("model.norm.weight", &[1024])?;
        check("pre_projection.weight", &[1024, 7680])?;
        check("post_projection.weight", &[3840, 1024])?;
        for il in 0..4 {
            let d = if il < 3 { 256 } else { 512 };
            let p = format!("model.layers.{il}");
            for (name, n, k) in [
                ("self_attn.q_proj.weight", 16 * d, 1024),
                ("self_attn.o_proj.weight", 1024, 16 * d),
                ("mlp.gate_proj.weight", 8192, 1024),
                ("mlp.up_proj.weight", 8192, 1024),
                ("mlp.down_proj.weight", 1024, 8192),
            ] {
                check(&format!("{p}.{name}"), &[n, k])?;
            }
            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "pre_feedforward_layernorm.weight",
            ] {
                check(&format!("{p}.{name}"), &[1024])?;
            }
            check(&format!("{p}.self_attn.q_norm.weight"), &[d])?;
            check(&format!("{p}.layer_scalar"), &[1])?;
        }
        Ok(())
    }
    fn matrix(&self, d: &Rc<Device>, name: &str, g: usize, split: u8) -> Result<Buffer> {
        let (t, b) = self.data(name)?;
        ensure!(t.shape.len() == 2, "{name}: matrix expected");
        let (n, k) = (t.shape[0], t.shape[1]);
        let (row, rows, col, cols) = match split {
            1 => (g * n / 2, n / 2, 0, k),
            2 => (0, n, g * k / 2, k / 2),
            _ => (0, n, 0, k),
        };
        let mut shard = Vec::with_capacity(rows * cols * 2);
        for r in row..row + rows {
            shard.extend_from_slice(&b[(r * k + col) * 2..(r * k + col + cols) * 2]);
        }
        d.upload_matrix(&shard)
    }
}
struct DraftCard {
    gpu: Rc<Device>,
    w: HashMap<String, Buffer>,
    input: Buffer,
    params: Buffer,
    params_host: Buffer,
    x: Buffer,
    h: Buffer,
    q: Buffer,
    nq: Buffer,
    rq: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    geglu: Buffer,
    partial: Buffer,
    post: Buffer,
    recv: Vec<Buffer>,
    dummy_recv: Buffer,
    peer: Vec<Buffer>,
    logits: Buffer,
    top_scratch: Buffer,
    top: Buffer,
    next_h: Buffer,
    cmds: HashMap<(AttentionPartitions, usize), Vec<vk::CommandBuffer>>,
    own: vk::Semaphore,
    other: vk::Semaphore,
    fence: vk::Fence,
    scales: Vec<f32>,
}
pub struct Assistant {
    cards: Vec<DraftCard>,
    counter: u64,
    pub proposed: usize,
    pub accepted: usize,
    pub rounds: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pending_h: Option<Vec<f32>>,
    top_k: usize,
}
impl Assistant {
    pub fn new(path: &Path, target: &Engine, top_k: usize) -> Result<Self> {
        ensure!((1..=64).contains(&top_k), "draft top-k must be 1..64");
        let ck = Checkpoint::open(path)?;
        ck.validate()?;
        let mut cards = Vec::new();
        for g in 0..2 {
            let gpu = target.cards[g].gpu.clone();
            let mut w = HashMap::new();
            let mut scales = Vec::new();
            for il in 0..4 {
                let prefix = format!("model.layers.{il}");
                for (short, name, split) in [
                    ("q", "self_attn.q_proj.weight", 1),
                    ("o", "self_attn.o_proj.weight", 2),
                    ("gate", "mlp.gate_proj.weight", 1),
                    ("up", "mlp.up_proj.weight", 1),
                    ("down", "mlp.down_proj.weight", 2),
                ] {
                    w.insert(
                        format!("{il}.{short}"),
                        ck.matrix(&gpu, &format!("{prefix}.{name}"), g, split)?,
                    );
                }
                for (short, name) in [
                    ("an", "input_layernorm.weight"),
                    ("qn", "self_attn.q_norm.weight"),
                    ("pn", "post_attention_layernorm.weight"),
                    ("fn", "pre_feedforward_layernorm.weight"),
                    ("dn", "post_feedforward_layernorm.weight"),
                ] {
                    let v = ck.floats(&format!("{prefix}.{name}"))?;
                    w.insert(
                        format!("{il}.{short}"),
                        gpu.upload(bytemuck::cast_slice(&v))?,
                    );
                }
                scales.push(ck.floats(&format!("{prefix}.layer_scalar"))?[0]);
            }
            for (short, name, split) in [
                ("pre", "pre_projection.weight", 0),
                ("post", "post_projection.weight", 0),
                ("head", "model.embed_tokens.weight", 1),
            ] {
                w.insert(short.into(), ck.matrix(&gpu, name, g, split)?);
            }
            let norm = ck.floats("model.norm.weight")?;
            w.insert("norm".into(), gpu.upload(bytemuck::cast_slice(&norm))?);
            let alloc = |n: usize| gpu.buffer(n * 4, false, false, None);
            let own = gpu.timeline(true)?;
            let other = gpu.timeline(false)?;
            let fence = unsafe {
                gpu.dev
                    .create_fence(&vk::FenceCreateInfo::default(), None)?
            };
            cards.push(DraftCard {
                w,
                input: gpu.buffer(7680 * 4, true, false, None)?,
                params: gpu.buffer(16, false, false, None)?,
                params_host: gpu.buffer(16, true, false, None)?,
                x: alloc(1024)?,
                h: alloc(1024)?,
                q: alloc(4096)?,
                nq: alloc(4096)?,
                rq: alloc(4096)?,
                o: alloc(4096)?,
                gate: alloc(4096)?,
                up: alloc(4096)?,
                geglu: alloc(4096)?,
                partial: alloc(1024)?,
                post: alloc(1024)?,
                recv: (0..8)
                    .map(|_| gpu.external_buffer(1024 * 4, None))
                    .collect::<Result<Vec<_>>>()?,
                dummy_recv: gpu.buffer(4, false, false, None)?,
                peer: Vec::new(),
                logits: alloc(131072)?,
                top_scratch: gpu.buffer(128 * 64 * 8, false, false, None)?,
                top: gpu.buffer(64 * 8, true, false, None)?,
                next_h: gpu.buffer(3840 * 4, true, false, None)?,
                cmds: HashMap::new(),
                own,
                other,
                fence,
                scales,
                gpu: gpu.clone(),
            });
        }
        for g in 0..2 {
            let c = &cards[g];
            let other = &cards[1 - g];
            let imported = other
                .recv
                .iter()
                .map(|b| c.gpu.external_buffer(b.size, Some(b.export()?)))
                .collect::<Result<Vec<_>>>()?;
            let ex =
                ash::khr::external_semaphore_fd::Device::new(&c.gpu.inst.instance, &other.gpu.dev);
            let fd = unsafe {
                ex.get_semaphore_fd(
                    &vk::SemaphoreGetFdInfoKHR::default()
                        .semaphore(other.own)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
                )?
            };
            let im = ash::khr::external_semaphore_fd::Device::new(&c.gpu.inst.instance, &c.gpu.dev);
            unsafe {
                im.import_semaphore_fd(
                    &vk::ImportSemaphoreFdInfoKHR::default()
                        .semaphore(c.other)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                        .fd(fd),
                )?;
            }
            cards[g].peer = imported;
        }
        let this = Self {
            cards,
            counter: 0,
            proposed: 0,
            accepted: 0,
            rounds: 0,
            draft_ms: 0.,
            verify_ms: 0.,
            pending_h: None,
            top_k,
        };
        eprintln!(
            "MTP loaded: native BF16; full vocabulary head (checkpoint has no centroid tensors)"
        );
        for (g, c) in this.cards.iter().enumerate() {
            eprintln!(
                "GPU{g}: target + MTP allocations {:.1} MiB",
                c.gpu.device_bytes.get() as f64 / 1048576.
            );
        }
        Ok(this)
    }
    fn record(
        &self,
        g: usize,
        target: &Engine,
        attention: AttentionPartitions,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let c = &self.cards[g];
        let gpu = &c.gpu;
        let mut commands = Vec::new();
        let mut cmd = gpu.begin()?;
        let push = |ptr: [u64; 6], u: [u32; 16]| Push {
            ptr: [
                ptr[0],
                ptr[1],
                ptr[2],
                ptr[3],
                ptr[4],
                ptr[5],
                c.params.address,
                0,
            ],
            u,
        };
        let matrix = |cmd, w: &Buffer, x: &Buffer, y: &Buffer, k: usize, n: usize, defer: bool| {
            let mut u = [0; 16];
            u[0] = n as u32;
            u[1] = k as u32;
            u[9] = 2;
            gpu.dispatch_matrix(
                cmd,
                if n == 131072 { "headvec" } else { "matvec" },
                push([0, x.address, y.address, 0, 0, 0], u).defer_barrier(defer),
                n.div_ceil(if n == 131072 { 16 } else { 4 }) as u32,
                1,
                w,
            )
        };
        let norm = |cmd,
                    x: &Buffer,
                    w: &Buffer,
                    y: &Buffer,
                    heads: usize,
                    dim: usize,
                    recv: Option<&Buffer>,
                    residual: bool,
                    scale: f32| {
            let mut u = [0; 16];
            u[0] = dim as u32;
            u[3] = (heads * dim) as u32;
            u[11] = u32::from(recv.is_some()) | if residual { 2 } else { 0 };
            u[12] = scale.to_bits();
            gpu.dispatch_norm(
                cmd,
                push(
                    [
                        x.address,
                        w.address,
                        y.address,
                        0,
                        if residual { c.x.address } else { 0 },
                        0,
                    ],
                    u,
                ),
                heads as u32,
                1,
                recv.unwrap_or(&c.dummy_recv),
            )
        };
        let end = |cmd, i, commands: &mut Vec<vk::CommandBuffer>| -> Result<vk::CommandBuffer> {
            gpu.copy(cmd, &c.partial, 0, &c.peer[i], 0, 1024 * 4);
            gpu.finish(cmd)?;
            commands.push(cmd);
            let next = gpu.begin()?;
            gpu.barrier(next);
            Ok(next)
        };
        gpu.copy(cmd, &c.params_host, 0, &c.params, 0, 16);
        matrix(cmd, &c.w["pre"], &c.input, &c.x, 7680, 1024, false)?;
        for il in 0..4 {
            let w = |name: &str| &c.w[&format!("{il}.{name}")];
            let layer = &target.cards[g].layers[if il < 3 { 46 } else { 47 }];
            let dim = layer.dim;
            if il > 0 {
                norm(
                    cmd,
                    &c.partial,
                    &c.w[&format!("{}.dn", il - 1)],
                    &c.post,
                    1,
                    1024,
                    Some(&c.recv[2 * il - 1]),
                    true,
                    c.scales[il - 1],
                )?;
            }
            norm(cmd, &c.x, w("an"), &c.h, 1, 1024, None, false, 1.)?;
            matrix(cmd, w("q"), &c.h, &c.q, 1024, 8 * dim, false)?;
            norm(cmd, &c.q, w("qn"), &c.nq, 8, dim, None, false, 1.)?;
            let mut u = [0; 16];
            u[0] = (8 * dim) as u32;
            u[6] = dim as u32;
            u[12] = if il < 3 { 10000f32 } else { 1000000f32 }.to_bits();
            gpu.dispatch(
                cmd,
                "rope",
                push(
                    [
                        c.nq.address,
                        if il < 3 {
                            0
                        } else {
                            target.cards[g].rope.address
                        },
                        c.rq.address,
                        0,
                        0,
                        0,
                    ],
                    u,
                ),
                (8 * dim).div_ceil(256) as u32,
                1,
            );
            let mut u = [0; 16];
            u[4] = 8;
            u[5] = layer.kv as u32;
            u[6] = dim as u32;
            u[7] = layer.cap as u32;
            u[8] = if il < 3 { 1024 } else { 0 };
            let parts = if layer.sliding {
                attention.sliding
            } else {
                attention.global
            };
            let (partial_shader, merge_shader) = attention_shaders(parts)?;
            gpu.dispatch(
                cmd,
                partial_shader,
                push(
                    [
                        c.rq.address,
                        layer.k.address,
                        layer.v.address,
                        c.o.address,
                        target.cards[g].attn_scratch.address,
                        0,
                    ],
                    u,
                ),
                8 * parts,
                1,
            );
            gpu.dispatch(
                cmd,
                merge_shader,
                push(
                    [
                        0,
                        0,
                        0,
                        c.o.address,
                        target.cards[g].attn_scratch.address,
                        0,
                    ],
                    u,
                ),
                8,
                1,
            );
            matrix(cmd, w("o"), &c.o, &c.partial, 8 * dim, 1024, true)?;
            cmd = end(cmd, 2 * il, &mut commands)?;
            norm(
                cmd,
                &c.partial,
                w("pn"),
                &c.post,
                1,
                1024,
                Some(&c.recv[2 * il]),
                true,
                1.,
            )?;
            norm(cmd, &c.x, w("fn"), &c.h, 1, 1024, None, false, 1.)?;
            matrix(cmd, w("gate"), &c.h, &c.gate, 1024, 4096, true)?;
            matrix(cmd, w("up"), &c.h, &c.up, 1024, 4096, false)?;
            let mut u = [0; 16];
            u[0] = 4096;
            gpu.dispatch(
                cmd,
                "geglu",
                push([c.gate.address, c.up.address, c.geglu.address, 0, 0, 0], u),
                16,
                1,
            );
            matrix(cmd, w("down"), &c.geglu, &c.partial, 4096, 1024, true)?;
            cmd = end(cmd, 2 * il + 1, &mut commands)?;
        }
        norm(
            cmd,
            &c.partial,
            &c.w["3.dn"],
            &c.post,
            1,
            1024,
            Some(&c.recv[7]),
            true,
            c.scales[3],
        )?;
        norm(cmd, &c.x, &c.w["norm"], &c.h, 1, 1024, None, false, 1.)?;
        matrix(cmd, &c.w["head"], &c.h, &c.logits, 1024, 131072, true)?;
        matrix(cmd, &c.w["post"], &c.h, &c.next_h, 1024, 3840, false)?;
        let mut u = [0; 16];
        u[0] = 131072;
        u[1] = self.top_k as u32;
        u[2] = (128 * self.top_k) as u32;
        u[10] = (g * 131072) as u32;
        u[11] = 2; // Same suppressed tokens as the target.
        u[12] = 258883;
        u[13] = 258882;
        gpu.dispatch(
            cmd,
            "topk",
            push([c.logits.address, 0, c.top_scratch.address, 0, 0, 0], u),
            128,
            1,
        );
        let mut u = [0; 16];
        u[0] = (128 * self.top_k) as u32;
        u[1] = self.top_k as u32;
        gpu.dispatch(
            cmd,
            "merge",
            push([c.top_scratch.address, 0, c.top.address, 0, 0, 0], u),
            1,
            1,
        );
        gpu.finish(cmd)?;
        commands.push(cmd);
        Ok(commands)
    }
    pub fn draft(&mut self, target: &Engine, token: u32, h: &[f32]) -> Result<DraftOutput> {
        let started = std::time::Instant::now();
        ensure!(h.len() == 3840 && target.pos > 0, "invalid draft input");
        // Draft queries use position `target.pos`, but can only read the
        // `target.pos` keys already cached by the target. Reuse its policy for
        // that visible key count, with independent sliding/global graph keys.
        let attention = AttentionPartitions::for_batch(target.pos - 1, 1);
        let key = (attention, self.top_k);
        for g in 0..self.cards.len() {
            if !self.cards[g].cmds.contains_key(&key) {
                let commands = self.record(g, target, attention)?;
                self.cards[g].cmds.insert(key, commands);
            }
        }
        let mut input = target.embedding(token)?;
        input.extend_from_slice(h);
        for c in &self.cards {
            c.input.write(bytemuck::cast_slice(&input))?;
            c.params_host.write(bytemuck::cast_slice(&[
                target.pos as u32,
                1,
                target.pos as u32,
                0,
            ]))?;
            unsafe {
                c.gpu.dev.reset_fences(&[c.fence])?;
            }
        }
        for c in &self.cards {
            c.gpu
                .submit_sequence(&c.cmds[&key], c.other, c.own, self.counter, c.fence)?;
        }
        for c in &self.cards {
            unsafe {
                c.gpu
                    .dev
                    .wait_for_fences(&[c.fence], true, 60_000_000_000)?;
            }
        }
        self.counter += 8;
        let mut candidates = Vec::new();
        for c in &self.cards {
            for s in c.top.read(self.top_k * 8)?.chunks_exact(8) {
                let x = (
                    f32::from_le_bytes(s[..4].try_into()?),
                    u32::from_le_bytes(s[4..].try_into()?),
                );
                candidates.push(x);
            }
        }
        let next = self.cards[0]
            .next_h
            .read(3840 * 4)?
            .chunks_exact(4)
            .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
            .collect();
        self.draft_ms += started.elapsed().as_secs_f64() * 1e3;
        candidates.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        candidates.truncate(self.top_k);
        Ok((candidates, next))
    }
    pub fn set_top_k(&mut self, k: usize) -> Result<()> {
        ensure!((1..=64).contains(&k), "draft top-k must be 1..64");
        self.top_k = k;
        Ok(())
    }
    pub fn reset(&mut self) {
        self.pending_h = None;
    }
    pub fn step(
        &mut self,
        e: &mut Engine,
        anchor: u32,
        sampler: &mut Sampler,
        k: usize,
    ) -> Result<(Vec<u32>, u32)> {
        ensure!(k > 0 && k <= 7, "draft length must be 1..7");
        let start = e.pos;
        let mut h = if let Some(h) = self.pending_h.take() {
            h
        } else {
            e.hidden()?
        };
        let mut tokens = vec![anchor];
        let mut token = anchor;
        let mut proposals = Vec::new();
        for _ in 0..k {
            let (draft_logits, nh) = self.draft(e, token, &h)?;
            let q = sampler.probabilities(&draft_logits);
            let next = sampler.draw(&q);
            proposals.push(q);
            tokens.push(next);
            token = next;
            h = nh;
            if matches!(next, 1 | 50 | 106) {
                break;
            }
        }
        self.proposed += tokens.len() - 1;
        self.rounds += 1;
        let started = std::time::Instant::now();
        let logits = e.forward(&tokens, None, true)?;
        self.verify_ms += started.elapsed().as_secs_f64() * 1e3;
        let (accepted, pending) = verify_prefix(&tokens, &logits, &proposals, sampler);
        self.accepted += accepted;
        self.pending_h = Some(e.hidden_at(accepted)?);
        e.rewind(start + accepted + 1)?;
        tokens.truncate(accepted + 1);
        Ok((tokens, pending))
    }
}
impl Drop for Assistant {
    fn drop(&mut self) {
        for c in &self.cards {
            unsafe {
                let _ = c.gpu.dev.device_wait_idle();
                for commands in c.cmds.values() {
                    c.gpu.dev.free_command_buffers(c.gpu.pool, commands);
                }
                c.gpu.dev.destroy_semaphore(c.own, None);
                c.gpu.dev.destroy_semaphore(c.other, None);
                c.gpu.dev.destroy_fence(c.fence, None);
            }
        }
    }
}

/// Rejection sampling preserves the target law using the union of the truncated distributions.
/// Greedy distributions are point masses, so this reduces to accepting the matching prefix.
fn verify_prefix(
    tokens: &[u32],
    logits: &[Vec<(f32, u32)>],
    q: &[Vec<(f64, u32)>],
    sampler: &mut Sampler,
) -> (usize, u32) {
    for (i, proposal) in q.iter().enumerate() {
        let p = sampler.probabilities(&logits[i]);
        let id = tokens[i + 1];
        let pp = p.iter().find(|v| v.1 == id).map_or(0., |v| v.0);
        let qp = proposal
            .iter()
            .find(|v| v.1 == id)
            .expect("draft token has proposal probability")
            .0;
        let acceptance = (pp / qp).min(1.);
        if acceptance == 1. || (acceptance > 0. && sampler.uniform() < acceptance) {
            continue;
        }
        let mut residual: Vec<_> = p
            .iter()
            .map(|&(prob, id)| {
                (
                    (prob - proposal.iter().find(|v| v.1 == id).map_or(0., |v| v.0)).max(0.),
                    id,
                )
            })
            .filter(|v| v.0 > 0.)
            .collect();
        let sum: f64 = residual.iter().map(|p| p.0).sum();
        for p in &mut residual {
            p.0 /= sum;
        }
        return (i, sampler.draw(&residual));
    }
    (q.len(), sampler.sample(&logits[q.len()]))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn q(tokens: &[u32]) -> Vec<Vec<(f64, u32)>> {
        tokens[1..].iter().map(|id| vec![(1., *id)]).collect()
    }
    #[test]
    fn acceptance_and_correction() {
        let logits = vec![vec![(4., 2)], vec![(4., 3)], vec![(4., 4)]];
        let mut sampler = Sampler::new(0., 64, 0.95, 1).unwrap();
        for (tokens, expected) in [
            (vec![1, 2, 3], (2, 4)),
            (vec![1, 2, 9], (1, 3)),
            (vec![1, 9, 9], (0, 2)),
        ] {
            assert_eq!(
                verify_prefix(&tokens, &logits, &q(&tokens), &mut sampler),
                expected
            );
        }
    }
    #[test]
    fn rejection_consumes_only_committed_samples() {
        let logits = vec![vec![(4., 2), (3., 3)], vec![(4., 4), (3., 5)]];
        let mut a = Sampler::new(1., 64, 0.95, 7).unwrap();
        let mut b = Sampler::new(1., 64, 0.95, 7).unwrap();
        let expected = b.sample(&logits[0]);
        assert_eq!(
            verify_prefix(&[1, 999], &logits, &q(&[1, 999]), &mut a),
            (0, expected)
        );
        assert_eq!(a.sample(&logits[1]), b.sample(&logits[1]));
    }
    #[test]
    fn stochastic_verification_preserves_target_mass() {
        let p = vec![(0.6f32.ln(), 1), (0.4f32.ln(), 2)];
        let logits = vec![p.clone(), p];
        let proposal = vec![(0.2, 1), (0.8, 2)];
        let mut s = Sampler::new(1., 2, 1., 123).unwrap();
        let mut first = 0;
        let mut accepted = 0;
        for _ in 0..50000 {
            let id = s.draw(&proposal);
            let (a, next) =
                verify_prefix(&[0, id], &logits, std::slice::from_ref(&proposal), &mut s);
            accepted += a;
            let result = if a == 1 { id } else { next };
            first += usize::from(result == 1);
        }
        assert!((first as f64 / 50000. - 0.6).abs() < 0.012);
        assert!((accepted as f64 / 50000. - 0.6).abs() < 0.012);
    }
}
