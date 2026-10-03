#!/usr/bin/env python3
"""Measure staged batched shader variants sequentially on incite."""
import json
import os
from pathlib import Path
import subprocess
import sys

out = Path(__file__).resolve().parent
root = out.parents[2]
os.chdir(root)
env = dict(os.environ, PATH=str(Path.home() / '.cargo/bin') + ':' + os.environ['PATH'])
model = Path.home() / 'models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf'
kind = sys.argv[1]
shader = root / 'engine/shaders' / ('gemm.comp' if kind == 'gemm' else 'multivec_body.glsl')
saved = shader.read_bytes()
batches = [64, 128, 512] if kind == 'gemm' else [2, 4, 8]
try:
    for name in sys.argv[2:]:
        shader.write_bytes(saved if name == 'baseline' else (out / 'shader-candidates' / name).read_bytes())
        label = kind + '-' + name.split('.')[0]
        with (out / (label + '-build.log')).open('w') as log:
            subprocess.run(['cargo', 'build', '--release'], env=env, stdout=log, stderr=log, check=True)
        for batch in batches:
            for tensor in ['ffn_gate', 'ffn_down', 'attn_q']:
                base = out / (label + '-b' + str(batch) + '-' + tensor)
                command = ['sudo', '-n', 'python3', 'scripts/latency-run.py', 'target/release/trashcan',
                           'matvec-bench', str(model), '--iterations', '100', '--batch', str(batch),
                           '--kernel', 'gemm' if kind == 'gemm' else 'multivec4',
                           '--tensor', 'blk.0.' + tensor + '.weight']
                with base.with_suffix('.jsonl').open('w') as result, base.with_suffix('.log').open('w') as log:
                    subprocess.run(command, env=env, stdout=result, stderr=log, check=True)
                rows = [json.loads(line) for line in base.with_suffix('.jsonl').read_text().splitlines()]
                print(label, batch, tensor, sum(row['milliseconds'] for row in rows) / len(rows), flush=True)
finally:
    shader.write_bytes(saved)
