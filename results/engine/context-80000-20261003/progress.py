#!/usr/bin/env python3
"""Read-only progress snapshot; run from ~/trashcan on incite."""
import json
from pathlib import Path

root = Path('results/engine/context-80000-20261003')
path = root / 'ctx80000.log'
records = [json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]
if records:
    last = records[-1]
    print(json.dumps(last))
    if last['phase'] == 'prefill':
        recent = records[-4:]
        print(f"Prefill cumulative {last['position'] / last['elapsed_seconds']:.2f} tok/s; latest four chunks {sum(d['chunk_tokens'] for d in recent) / sum(d['chunk_seconds'] for d in recent):.2f} tok/s")
else:
    print(path.read_text().splitlines()[-1])
for name in ['card0', 'card1']:
    gpu = Path('/sys/class/drm') / name / 'device'
    print(name, {p: round(int((gpu / ('mem_info_' + p)).read_text()) / 1048576, 2) for p in ['vram_used', 'vis_vram_used', 'gtt_used']})
