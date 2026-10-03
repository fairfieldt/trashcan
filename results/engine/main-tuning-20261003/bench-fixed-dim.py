#!/usr/bin/env python3
"""Validate and measure fixed-dimension attention against the accepted shared-KV build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

here = Path(__file__).resolve().parent
root = here.parents[2]
out = here / 'fixed-dim'
os.chdir(root)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('phases', nargs='*', default=['validate', 'decode', 'chat', 'long', 'checks'])
args = parser.parse_args()
out.mkdir(exist_ok=True)
model = Path.home() / 'models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf'
assistant = Path.home() / 'models/gemma-4-12b-assistant/model.safetensors'
binaries = [('shared', 'target/release/trashcan-shared-accepted'), ('fixed', 'target/release/trashcan')]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert digest(binaries[0][1]) == json.loads((here / 'shared-kv/build.json').read_text())['binaries']['shared']['sha256'], 'Shared baseline binary differs'
manifest = {'note': 'Shared baseline is checkpoint cc3f50f; fixed-dimension source is identified by exact hashes. Model comment corrected without changing behavior.',
            'driver_sha256': digest(__file__),
            'binaries': {label: {'path': path, 'sha256': digest(path)} for label, path in binaries},
            'source_sha256': {str(p): digest(p) for p in sorted(
                [Path('Cargo.toml'), Path('Cargo.lock'), Path('engine/Cargo.toml'), Path('engine/build.rs')]
                + list(Path('engine/src').glob('*.rs')) + list(Path('engine/shaders').glob('*')))
                if p.is_file()},
            'invocation': args.phases}
(out / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')

def run(binary, name, command, suffix='json'):
    result, log = out / (name + '.' + suffix), out / (name + '.log')
    if result.exists() or log.exists():
        raise RuntimeError(f'Refusing to overwrite {name}; preserve the earlier run first')
    command = ['sudo', '-n', 'python3', 'scripts/latency-run.py', binary, *map(str, command)]
    (out / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    started = time.time()
    with result.open('w') as stdout, log.open('w') as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    (out / (name + '.window.json')).write_text(json.dumps({
        'command_epoch_started': started, 'command_epoch_finished': time.time(),
        'returncode': completed.returncode,
        'note': 'Whole command window includes model loading and cleanup, not just inference.'
    }, indent=2) + '\n')
    completed.check_returncode()
    print(name + ' completed', flush=True)

if 'micro' in args.phases:
    for tokens, parts in [(1, 8), (128, 16), (768, 32), (1536, 64), (2048, 64), (8192, 64)]:
        for kernel in ('scalar', 'selected'):
            run(binaries[1][1], f'micro-{kernel}-{tokens}-p{parts}',
                ['attention-bench', '--tokens', tokens, '--partitions', parts,
                 '--iterations', 100, '--kernel', kernel], 'jsonl')
if 'validate' in args.phases:
    with (out / 'validation.log').open('x') as log:
        subprocess.run(['sudo', '-n', 'python3', 'scripts/latency-run.py',
                        'scripts/test-engine.sh', str(model), str(assistant)],
                       stdout=log, stderr=log, check=True)
    print('Full validation completed', flush=True)
if 'decode' in args.phases:
    for trial in (1, 2):
        for label, binary in binaries:
            name = f'decode-{label}-{trial}'
            run(binary, name, ['bench', model, '--pp', 512, '--tg', 256, '--batch', 512,
                               '--profile', out / (name + '.tsv')])
if 'chat' in args.phases:
    prompts = [
        ('transformer', 'Explain how a transformer works, step by step, in English.'),
        ('photosynthesis', 'Explain photosynthesis to a ten-year-old in three short paragraphs.'),
        ('csv', 'Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.'),
    ]
    parity = []
    for name, prompt in prompts:
        for label, binary in binaries:
            for draft in (False, True):
                base = f'chat-{label}-{name}' + ('-mtp' if draft else '')
                command = ['chat', model, prompt, '--batch', 128, '--temperature', 0,
                           '--max-tokens', 128, '--profile', out / (base + '.tsv')]
                if draft:
                    command += ['--draft', assistant]
                run(binary, base, command, 'txt')
                reference = (out / f'chat-shared-{name}.txt').read_bytes()
                exact = (out / (base + '.txt')).read_bytes() == reference
                parity.append({'run': base, 'matches_shared_ordinary': exact})
                (out / 'chat-parity.json').write_text(json.dumps(parity, indent=2) + '\n')
                if not exact:
                    raise RuntimeError(base + ': output differs; inspect before claiming a speedup')
if 'long' in args.phases:
    for label, binary in binaries:
        name = 'long-' + label
        run(binary, name, ['bench', model, '--pp', 4096, '--tg', 128, '--batch', 512,
                           '--profile', out / (name + '.tsv')])
if 'checks' in args.phases:
    for tokens, batch in [(250, 128), (2300, 512)]:
        run(binaries[1][1], f'check-{tokens}', ['check-batches', model, '--tokens', tokens,
                                             '--batch', batch, '--draft', assistant])
