#!/usr/bin/env python3
"""Compare final and saved merged-main binaries sequentially on the same GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys

out = Path(__file__).resolve().parent
os.chdir(out.parents[2])
model = Path.home() / 'models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf'
assistant = Path.home() / 'models/gemma-4-12b-assistant/model.safetensors'

def run(binary, name, command, suffix='json'):
    with (out / (name + '.' + suffix)).open('w') as result, (out / (name + '.log')).open('w') as log:
        subprocess.run(['sudo', '-n', 'python3', 'scripts/latency-run.py', binary,
                        *command], stdout=result, stderr=log, check=True)
    print(name, 'completed', flush=True)

phases = sys.argv[1:] or ['decode', 'chat', 'long', 'boundary']
if 'decode' in phases:
    for trial in [1, 2]:
        run('target/release/trashcan', f'final-{trial}',
            ['bench', str(model), '--pp', '512', '--tg', '256', '--batch', '512',
             '--profile', str(out / f'final-{trial}.tsv')])
if 'chat' in phases:
    prompts = [
        ('transformer', 'Explain how a transformer works, step by step, in English.'),
        ('photosynthesis', 'Explain photosynthesis to a ten-year-old in three short paragraphs.'),
        ('csv', 'Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.'),
    ]
    parity = []
    for label, binary in [('baseline', 'target/release/trashcan-main-baseline'), ('final', 'target/release/trashcan')]:
        for name, prompt in prompts:
            for draft in [False, True]:
                base = f'chat-{label}-{name}' + ('-mtp' if draft else '')
                command = ['chat', str(model), prompt, '--batch', '128', '--temperature', '0', '--max-tokens', '128',
                           '--profile', str(out / (base + '.tsv'))]
                if draft:
                    command += ['--draft', str(assistant)]
                run(binary, base, command, 'txt')
                reference = (out / f'chat-baseline-{name}.txt').read_bytes()
                parity.append({'run': base, 'matches_baseline': (out / (base + '.txt')).read_bytes() == reference})
                if draft:
                    assert (out / (base + '.txt')).read_bytes() == (out / f'chat-{label}-{name}.txt').read_bytes(), base + ': MTP output differs'
                (out / 'chat-parity.json').write_text(json.dumps(parity, indent=2) + '\n')
if 'long' in phases:
    for label, binary in [('baseline', 'target/release/trashcan-main-baseline'), ('final', 'target/release/trashcan')]:
        run(binary, f'long-{label}', ['bench', str(model), '--pp', '4096', '--tg', '128', '--batch', '512',
                                    '--profile', str(out / f'long-{label}.tsv')])
if 'boundary' in phases:
    run('target/release/trashcan', 'boundary-check', ['check-batches', str(model), '--tokens', '250',
                                                   '--batch', '128', '--draft', str(assistant)])
