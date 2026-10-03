#!/usr/bin/env python3
"""Audit and summarize the complete 4K/80K benchmark evidence."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
report = {'base_commit': '5925623', 'synthetic_input_token_id': 9259, 'batch': 512,
          'cpu_latency_request_us': 0, 'runs': {},
          'validation_scope': 'finite top-1 scores and sequential positions; no independent 80K reference logits or retrieval-quality check'}
sha = (root / 'binary.sha256').read_text().split()[0]
assert sha == (root / 'tested-binary.sha256').read_text().split()[0]
report['binary_sha256'] = sha
changes = []
for line in (root / 'source.sha256').read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    content = subprocess.check_output(['git', 'show', f'5925623:{name}'], cwd=root.parents[2])
    if hashlib.sha256(content).hexdigest() != digest:
        changes.append(name)
assert changes == ['engine/src/main.rs'], changes
report['source_changes_from_base'] = changes
for pp in [4096, 80000]:
    measured = json.loads((root / f'ctx{pp}.json').read_text())
    assert measured['pp'] == pp and measured['tg'] == 128
    records = [json.loads(line) for line in (root / f'ctx{pp}.log').read_text().splitlines() if line.startswith('{')]
    prefill = [d for d in records if d['phase'] == 'prefill']
    decode = [d for d in records if d['phase'] == 'decode']
    assert len(prefill) == math.ceil(pp / 512) and len(decode) == 128
    pos = 0
    for d in prefill:
        pos += d['chunk_tokens']
        assert d['position'] == pos and d['chunk_seconds'] > 0 and math.isfinite(d['top1_score'])
    assert pos == pp
    for step, d in enumerate(decode, 1):
        assert d['position'] == pp + step and d['tokens'] == step and math.isfinite(d['top1_score'])
    measured['finite_prefill_steps'] = len(prefill)
    measured['finite_decode_steps'] = len(decode)
    measured['final_position'] = decode[-1]['position']
    measured['decode_first_step_seconds'] = decode[0]['elapsed_seconds']
    measured['decode_after_first_tok_s'] = 127 / (decode[-1]['elapsed_seconds'] - decode[0]['elapsed_seconds'])
    measured['decode_last64_tok_s'] = 64 / (decode[-1]['elapsed_seconds'] - decode[63]['elapsed_seconds'])
    for phase, data in [('prefill', prefill), ('decode', decode)]:
        with (root / f'ctx{pp}.{phase}-progress.tsv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]), delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    intervals = []
    previous = 0
    for end in ([4096] if pp == 4096 else [16384, 32768, 49152, 65536, 80000]):
        rows = [d for d in prefill if previous < d['position'] <= end]
        seconds = sum(d['chunk_seconds'] for d in rows)
        tokens = sum(d['chunk_tokens'] for d in rows)
        assert tokens == end - previous
        intervals.append({'start_position': previous, 'end_position': end, 'tokens': tokens,
                          'seconds': seconds, 'tok_s': tokens / seconds})
        previous = end
    measured['prefill_intervals'] = intervals
    measured['last_full_prefill_chunk_tok_s'] = 512 / next(d for d in reversed(prefill) if d['chunk_tokens'] == 512)['chunk_seconds']
    for phase, suffix in [('last_prefill_chunk', '.prefill.tsv'), ('last_decode_step', '.tsv')]:
        rows = list(csv.DictReader((root / f'ctx{pp}{suffix}').open(), delimiter='\t'))
        measured[phase] = {'gpu_work_ms': [sum(float(d[f'gpu{g}_ms']) for d in rows) for g in [0, 1]],
                          'gaps_ms': [sum(float(d[f'gap{g}_ms']) for d in rows) for g in [0, 1]],
                          'note': 'the two GPUs run concurrently; final 80K prefill chunk has 128 tokens'}
        if phase == 'last_decode_step':
            global_segments = [d for d in rows if int(d['segment']) in [10, 22, 34, 46, 58, 70, 82, 94]]
            measured[phase]['global_attention_segments_ms'] = [sum(float(d[f'gpu{g}_ms']) for d in global_segments) for g in [0, 1]]
    report['runs'][str(pp)] = measured
clocks = list(csv.DictReader((root / 'clocks.txt').open(), delimiter=' '))
report['clocks'] = {'samples': len(clocks),
                    'per_gpu_clock_samples': [dict(Counter(f'{d[f"sclk{g}"]}/{d[f"mclk{g}"]}' for d in clocks)) for g in [0, 1]],
                    'max_temperature_c': [max(int(d[f'temp{g}']) for d in clocks) for g in [0, 1]],
                    'note': '0.25-second nominal sampling across loading, 4K and 80K phases'}
memory = list(csv.DictReader((root / 'late-memory.csv').open()))
report['late_memory_max_mib'] = {
    field: max(int(d[field]) for d in memory) / 1048576
    for field in memory[0] if field != 't'
}
report['late_memory_note'] = '1-second samples from around 64K through process exit; early snapshots are in memory-snapshots.txt'
assert (root / 'resets-before.txt').read_bytes() == (root / 'resets-after.txt').read_bytes()
report['new_gpu_resets'] = False
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: 4K and 80K runs, finite scores, sequential positions, unchanged model kernels, and no new GPU reset.')
for name, d in report['runs'].items():
    print(f"{name}: prefill {d['pp_seconds']:.2f}s / {d['pp_tok_s']:.2f} tok/s; decode {d['tg_seconds']:.2f}s / {d['tg_tok_s']:.2f} tok/s")
