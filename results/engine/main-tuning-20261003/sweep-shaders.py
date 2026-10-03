#!/usr/bin/env python3
"""Run sequential candidate builds on incite, restoring production source afterwards."""
import json
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
os.chdir(root)
out = Path(__file__).resolve().parent
model = Path.home() / 'models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf'
env = dict(os.environ, PATH=str(Path.home() / '.cargo/bin') + ':' + os.environ['PATH'])
shader = root / 'engine/shaders/matvecq4.comp'
saved = shader.read_bytes()
try:
    for candidate in sys.argv[1:]:
        source = out / 'shader-candidates' / (candidate + '.comp')
        shader.write_bytes(source.read_bytes())
        with (out / (candidate + '-build.log')).open('w') as log:
            subprocess.run(['cargo', 'build', '--release'], env=env, stdout=log, stderr=log, check=True)
        for projection in ['ffn_gate', 'ffn_down', 'attn_q', 'attn_output', 'attn_k']:
            name = candidate + '-' + projection
            command = ['sudo', '-n', 'python3', 'scripts/latency-run.py', 'target/release/trashcan',
                       'matvec-bench', str(model), '--kernel', 'matvecq4', '--iterations', '1000', '--tensor',
                       'blk.0.' + projection + '.weight']
            with (out / (name + '.jsonl')).open('w') as result, (out / (name + '.log')).open('w') as log:
                subprocess.run(command, env=env, stdout=result, stderr=log, check=True)
            rows = [json.loads(line) for line in (out / (name + '.jsonl')).read_text().splitlines()]
            print(name, sum(r['original_GB_s'] for r in rows) / len(rows), flush=True)
finally:
    shader.write_bytes(saved)
