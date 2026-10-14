#!/usr/bin/env python3
# r21q (diagnostic binary, never committed): `stage-bench --tokens-file --out` (bench::stage_dump) takes
# TRASHCAN_Q4_DUMP_CHUNK=N (chunks of N tokens with the stages' --batch capacity, as the chain's chunk bands do) and
# TRASHCAN_Q4_DUMP_QUEUED=1 (every chunk but the last through Local::push, so each GPU stage has the next chunk queued
# behind the running one, as on the chain's nodes; drained before the last chunk, which runs with captures).
p = '/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/engine/src/qwen4/bench.rs'
s = open(p).read()
old = '''    let mut caps = super::cpu::Captures::new();
    let chunks: Vec<_> = tokens.chunks(batch).collect();
    let mut pos = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        let x = super::embed(&local.gguf, &local.c, chunk)?;
        let ple = super::ple_rows(&local.gguf, &local.c, &tokens[..pos], chunk)?;
        let last = i + 1 == chunks.len();
        let out = local.forward(x, ple.as_deref(), pos, chunk.len(), last.then_some(&mut caps))?;
        if let (true, super::gpu::Output::Final(h)) = (last, out) {
            local.head_slices(&h, Some(&mut caps))?;
        }
        pos += chunk.len();
    }'''
new = '''    let mut caps = super::cpu::Captures::new();
    let size = std::env::var("TRASHCAN_Q4_DUMP_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(batch);
    let queued = std::env::var_os("TRASHCAN_Q4_DUMP_QUEUED").is_some();
    eprintln!("stage_dump: {} tokens in chunks of {size}, {}", tokens.len(), if queued { "queued" } else { "one at a time" });
    let chunks: Vec<_> = tokens.chunks(size).collect();
    let mut pos = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        let x = super::embed(&local.gguf, &local.c, chunk)?;
        let ple = super::ple_rows(&local.gguf, &local.c, &tokens[..pos], chunk)?;
        let last = i + 1 == chunks.len();
        if queued && !last {
            local.push(x, ple, pos, chunk.len())?;
        } else {
            if queued {
                local.drain(|_| Ok(()))?;
            }
            let out = local.forward(x, ple.as_deref(), pos, chunk.len(), last.then_some(&mut caps))?;
            if let (true, super::gpu::Output::Final(h)) = (last, out) {
                local.head_slices(&h, Some(&mut caps))?;
            }
        }
        pos += chunk.len();
    }'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w').write(s)
print('ok')
