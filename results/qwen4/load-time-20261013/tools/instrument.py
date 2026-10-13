#!/usr/bin/env python3
"""Add the TEMPORARY load-time counters (loadprof.rs) to the new load path of a trashcan checkout.
usage: instrument.py CHECKOUT      (revert with git checkout -- engine/src; rm engine/src/loadprof.rs)"""
import os, shutil, sys

root = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, 'loadprof.rs'), os.path.join(root, 'engine/src/loadprof.rs'))


def indent(text):
    return '\n'.join(('    ' + l) if l.startswith('                    ') else l for l in text.split('\n'))


def edit(path, pairs, optional=False):
    p = os.path.join(root, path)
    s = open(p).read()
    for old, new in pairs:
        if s.count(old) != 1 and s.count(indent(old)) == 1:
            old, new = indent(old), indent(new)
        if optional and s.count(old) == 0:
            continue
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    open(p, 'w').write(s)


edit('engine/src/main.rs', [
    ("mod link;\n", "mod link;\npub mod loadprof;\n"),
    ("fn main() -> Result<()> {\n", "fn main() -> Result<()> {\n    loadprof::start();\n"),
])
edit('engine/src/vulkan.rs', [
    ("        repack(s.buf.ptr, s.buf.size, std::mem::take(&mut self.jobs), self.threads);",
     "        crate::loadprof::time(14, || repack(s.buf.ptr, s.buf.size, std::mem::take(&mut self.jobs), self.threads));"),
    ("                unsafe {\n                    self.gpu.dev.wait_for_fences(&[s.fence], true, FENCE_NS)?;",
     "                let _t = std::time::Instant::now();\n                unsafe {\n                    self.gpu.dev.wait_for_fences(&[s.fence], true, FENCE_NS)?;\n                    crate::loadprof::add(15, _t.elapsed().as_nanos() as u64);"),
    ("            self.gpu.dev.queue_submit(self.gpu.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], s.fence)?;",
     "            crate::loadprof::time(16, || self.gpu.dev.queue_submit(self.gpu.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], s.fence))?;\n            crate::loadprof::add(20, 1);"),
    ("                unsafe { self.gpu.dev.wait_for_fences(&[s.fence], true, FENCE_NS)? };\n                s.busy = false;",
     "                crate::loadprof::time(17, || unsafe { self.gpu.dev.wait_for_fences(&[s.fence], true, FENCE_NS) })?;\n                s.busy = false;"),
    ("        let dst = self.gpu.buffer(data.len(), false, false, None)?;\n        // The buffer is rounded",
     "        crate::loadprof::add(12, data.len() as u64);\n        crate::loadprof::add(13, 1);\n        let _t = std::time::Instant::now();\n        let dst = crate::loadprof::time(0, || self.gpu.buffer(data.len(), false, false, None))?;\n        // The buffer is rounded"),
    ("            self.copy(&dst, done, at, n);\n            done += n;\n        }\n        Ok(dst)",
     "            self.copy(&dst, done, at, n);\n            done += n;\n        }\n        crate::loadprof::add(2, _t.elapsed().as_nanos() as u64);\n        Ok(dst)"),
    ("        let dst = self.gpu.buffer(p.size(), false, false, None)?;",
     "        crate::loadprof::add(12, p.size() as u64);\n        crate::loadprof::add(13, 1);\n        let dst = crate::loadprof::time(0, || self.gpu.buffer(p.size(), false, false, None))?;"),
    ("    pub fn new(gpu: &Rc<Device>) -> Result<Self> {\n        let mut up = Self {",
     "    pub fn new(gpu: &Rc<Device>) -> Result<Self> {\n        let _t = std::time::Instant::now();\n        let mut up = Self {"),
    ("            up.slots.push(Staging { buf, cmd, fence, open: false, busy: false });\n        }\n        Ok(up)",
     "            up.slots.push(Staging { buf, cmd, fence, open: false, busy: false });\n        }\n        crate::loadprof::add(1, _t.elapsed().as_nanos() as u64);\n        Ok(up)"),
])
edit('engine/src/qwen4/gpu.rs', [
    ("        let f32s = |v: &[f32]| up.borrow_mut().bytes(bytemuck::cast_slice(v));",
     "        let rows_f32 = |g: &Gguf, t: &TensorInfo| crate::loadprof::time(7, || rows_f32(g, t));\n"
     "        let vector = |g: &Gguf, t: &TensorInfo| crate::loadprof::time(7, || vector(g, t));\n"
     "        let f32s = |v: &[f32]| up.borrow_mut().bytes(bytemuck::cast_slice(v));"),
])
edit('engine/src/qwen4/pipeline.rs', [
    ("        let gguf = Gguf::open(path)?;\n",
     "        crate::loadprof::mark(\"Local::new\");\n        let gguf = Gguf::open(path)?;\n        crate::loadprof::mark(\"gguf open\");\n"),
    ("                let all = inst.amd_gpus()?;\n",
     "                let all = inst.amd_gpus()?;\n                crate::loadprof::mark(\"instance\");\n"),
    ("                    let gpu = Device::new(inst.clone(), pd)?;\n                    let stage = GpuStage::new(gpu, &gguf, &c, first, last, heads[i], context, max_batch, kv)?;",
     "                    let _t0 = std::time::Instant::now();\n"
     "                    let gpu = crate::loadprof::time(10, || Device::new(inst.clone(), pd))?;\n"
     "                    let stage = crate::loadprof::time(11, || GpuStage::new(gpu, &gguf, &c, first, last, heads[i], context, max_batch, kv))?;\n"
     "                    crate::loadprof::report(&format!(\"GPU{}\", indices[i]), _t0.elapsed().as_secs_f64());"),
    ("        Ok(Self {\n            gguf,\n            c,\n",
     "        crate::loadprof::mark(\"Local::new done\");\n        Ok(Self {\n            gguf,\n            c,\n"),
])
edit('engine/src/qwen4/bench.rs', [
    ("    run(local, &stream[..batch], 0)?;\n    let mut pre = Vec::new();",
     "    crate::loadprof::mark(\"stage_bench\");\n    run(local, &stream[..batch], 0)?;\n    crate::loadprof::mark(\"warm-up chunk done\");\n    let mut pre = Vec::new();"),
    ("    let t0 = std::time::Instant::now();\n    for i in 0..decode {",
     "    crate::loadprof::mark(\"first decode step\");\n    let t0 = std::time::Instant::now();\n    for i in 0..decode {"),
])
edit('engine/src/qwen4/pipeline.rs', [
    ("        unsafe { libc::madvise(start as *mut libc::c_void, len, libc::MADV_WILLNEED) };\n    }\n}",
     "        unsafe { libc::madvise(start as *mut libc::c_void, len, libc::MADV_WILLNEED) };\n    }\n    crate::loadprof::mark(\"prefetch done\");\n}"),
], optional=True)
print('instrumented', root)
