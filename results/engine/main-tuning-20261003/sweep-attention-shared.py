#!/usr/bin/env python3
"""Sequential attention experiments; restore and rebuild production source on exit."""
import hashlib
import os
from pathlib import Path
import subprocess

out = Path(__file__).resolve().parent
os.chdir(out.parents[2])
source = Path('engine/shaders/attention_partial.comp')
original = source.read_bytes()
(out / 'shader-candidates' / 'attention_partial-adaptive-baseline.comp').write_bytes(original)
cases = [(1, 8), (128, 16), (768, 32), (1536, 64), (2048, 64), (8192, 64)]

def build(label):
    with (out / f'attention-shared-{label}-build.log').open('w') as log:
        subprocess.run(['cargo', 'build', '--release'], stdout=log, stderr=log, check=True)

try:
    for label in ['baseline', 'heads2', 'heads4', 'heads8']:
        data = original if label == 'baseline' else (out / 'shader-candidates' / f'attention_partial-{label}.comp').read_bytes()
        source.write_bytes(data)
        build(label)
        # Newer builds also contain independent shared-KV modules. This sweep
        # specifically replaces attention_partial.comp, so select that module.
        # The original recorded core revision predates the --kernel option.
        help_text = subprocess.check_output(
            ['target/release/trashcan', 'attention-bench', '--help'], text=True)
        kernel_args = ['--kernel', 'scalar'] if '--kernel' in help_text else []
        (out / f'attention-shared-{label}-source.sha256').write_text(hashlib.sha256(data).hexdigest() + '  engine/shaders/attention_partial.comp\n')
        for tokens, parts in cases:
            name = f'attention-shared-{label}-{tokens}-p{parts}'
            with (out / (name + '.jsonl')).open('w') as result, (out / (name + '.log')).open('w') as log:
                subprocess.run(['sudo', '-n', 'python3', 'scripts/latency-run.py',
                                'target/release/trashcan', 'attention-bench', '--tokens', str(tokens),
                                '--partitions', str(parts), '--iterations', '100', *kernel_args],
                               stdout=result, stderr=log, check=True)
            print(name + ' completed', flush=True)
finally:
    source.write_bytes(original)
    build('restored')
    (out / 'attention-shared-restored-binary.sha256').write_text(
        hashlib.sha256(Path('target/release/trashcan').read_bytes()).hexdigest() + '  target/release/trashcan\n')
    assert source.read_bytes() == original
