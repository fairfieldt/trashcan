#!/usr/bin/env python3
"""Build an isolated instrumented server and run one sequential diagnostic case."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

out = Path(__file__).resolve().parent
os.chdir(out.parents[2])
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('label', choices=['cold', 'cached', 'cold-full', 'cached-full'])
parser.add_argument('--build', action='store_true')
args = parser.parse_args()
source = Path('engine/src/server.rs')
manifest = json.loads((out / 'server-diagnostic-base.json').read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(source) == manifest['sha256'], 'Production source differs from diagnostic base'
binary = Path('target/server-diagnostic/release/trashcan')
if args.build:
    patch = str(out / 'server-diagnostic.patch')
    subprocess.run(['git', 'apply', '--check', patch], check=True)
    subprocess.run(['git', 'apply', patch], check=True)
    try:
        assert digest(source) == manifest['patched_sha256']
        with (out / 'server-diagnostic-build.log').open('w') as log:
            subprocess.run(['cargo', 'build', '--release', '--target-dir', 'target/server-diagnostic',
                            '--manifest-path', 'engine/Cargo.toml'], stdout=log, stderr=log, check=True)
        (out / 'server-diagnostic-source.sha256').write_text(digest(source) + '  engine/src/server.rs\n')
        (out / 'server-diagnostic-binary.sha256').write_text(digest(binary) + '  ' + str(binary) + '\n')
    finally:
        subprocess.run(['git', 'apply', '--reverse', '--check', patch], check=True)
        subprocess.run(['git', 'apply', '--reverse', patch], check=True)
        assert digest(source) == manifest['sha256'], 'Production source was not restored'
assert binary.exists(), 'Build the isolated binary first with --build'
run = out / ('diagnostic-' + args.label)
run.mkdir(exist_ok=False)
port = 18160 + ['cold', 'cached', 'cold-full', 'cached-full'].index(args.label)
model = Path.home() / 'models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf'
command = ['sudo', '-n', 'python3', 'scripts/latency-run.py', 'env',
           'TRASHCAN_REQUEST_PROFILE=' + str(run), str(binary), 'serve', str(model),
           '--batch', '128', '--port', str(port)]
if args.label.startswith('cold'):
    command.append('--no-prefix-cache')
logfile = run / 'server.log'
with logfile.open('w') as log:
    process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
    try:
        deadline = time.monotonic() + 240
        while 'Responses server listening' not in logfile.read_text():
            if process.poll() is not None:
                raise RuntimeError(logfile.read_text())
            if time.monotonic() >= deadline:
                raise RuntimeError('server startup timed out')
            time.sleep(0.5)
        client = ['python3', str(out / 'server-diagnostic.py'), '--url', f'http://127.0.0.1:{port}',
                  '--output', str(run / 'client.json')]
        if args.label.endswith('-full'):
            label = 'cold' if args.label.startswith('cold') else 'cached'
            client += ['--replay-report', str(out / f'prefix-{label}.json')]
        with (run / 'client.log').open('w') as checklog:
            subprocess.run(client, stdout=checklog, stderr=checklog, check=True)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
print(args.label + ' diagnostic complete; production source unchanged', flush=True)
