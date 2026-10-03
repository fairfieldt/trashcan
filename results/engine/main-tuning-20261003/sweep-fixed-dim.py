#!/usr/bin/env python3
"""Last isolated attention probe; restore production sources and binary on exit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

here = Path(__file__).resolve().parent
root = here.parents[2]
os.chdir(root)
candidate = here / 'shader-candidates/fixed-dim'
out = here / 'fixed-dim-probe'
out.mkdir(exist_ok=False)
pinned = json.loads((candidate / 'source.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, expected in pinned['baseline_sha256'].items():
    assert sha(name) == expected, name + ': production baseline differs'
source_names = ['attention_shared_body.glsl', 'attention_partial_heads2.comp', 'attention_partial_heads8.comp']
original = {name: (Path('engine/shaders') / name).read_bytes() for name in source_names}
baseline_binary = Path('target/release/trashcan-shared-accepted')
fixed_binary = Path('target/release/trashcan-fixed-dim')
expected_binary = json.loads((here / 'shared-kv/build.json').read_text())['binaries']['shared']['sha256']
assert sha('target/release/trashcan') == expected_binary
shutil.copy2('target/release/trashcan', baseline_binary)
report = {'baseline_build': '../shared-kv/build.json', 'baseline_binary_sha256': expected_binary,
          'candidate_manifest': '../shader-candidates/fixed-dim/source.json', 'runs': []}

def save():
    (out / 'probe.json').write_text(json.dumps(report, indent=2) + '\n')

def build(label):
    with (out / (label + '-build.log')).open('w') as log:
        subprocess.run(['cargo', 'build', '--release'], stdout=log, stderr=log, check=True)

save()
try:
    with (out / 'cpu-compile.json').open('w') as result, (out / 'cpu-compile.log').open('w') as log:
        subprocess.run(['python3', str(candidate / 'compile.py'), '--output-dir', str(out / 'spv')],
                       stdout=result, stderr=log, check=True)
    for name in source_names:
        (Path('engine/shaders') / name).write_bytes((candidate / name).read_bytes())
    build('fixed-dim')
    shutil.copy2('target/release/trashcan', fixed_binary)
    report['candidate_binary_sha256'] = sha(fixed_binary)
    report['candidate_shader_sha256'] = {name: sha(Path('engine/shaders') / name) for name in source_names}
    save()
    for trial in (1, 2):
        for tokens, parts in [(128, 16), (768, 32), (1536, 64), (8192, 64)]:
            for label, binary in [('baseline', baseline_binary), ('fixed-dim', fixed_binary)]:
                name = f'{label}-{trial}-{tokens}-p{parts}'
                command = ['sudo', '-n', 'python3', 'scripts/latency-run.py', str(binary),
                           'attention-bench', '--tokens', str(tokens), '--partitions', str(parts),
                           '--iterations', '100', '--kernel', 'selected']
                started = time.time()
                with (out / (name + '.jsonl')).open('w') as stdout, (out / (name + '.log')).open('w') as stderr:
                    completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
                report['runs'].append({'name': name, 'command': command, 'epoch_started': started,
                                       'epoch_finished': time.time(), 'returncode': completed.returncode})
                save()
                completed.check_returncode()
                rows = [json.loads(line) for line in (out / (name + '.jsonl')).read_text().splitlines()]
                assert len(rows) == 4 and all(row['pass'] for row in rows), name
                print(name + ' completed', flush=True)
    report['completed'] = True
finally:
    for name, data in original.items():
        (Path('engine/shaders') / name).write_bytes(data)
    build('restored')
    report['restored_binary_sha256'] = sha('target/release/trashcan')
    report['restored_binary_identical'] = report['restored_binary_sha256'] == expected_binary
    report['restored_source_exact'] = all((Path('engine/shaders') / name).read_bytes() == data
                                          for name, data in original.items())
    save()
    assert report['restored_source_exact'] and report['restored_binary_identical']
