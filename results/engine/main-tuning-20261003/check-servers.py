#!/usr/bin/env python3
"""Run cold/cached ordinary and speculative servers sequentially on incite."""
import contextlib
import os
from pathlib import Path
import signal
import subprocess
import time
import sys

out = Path(__file__).resolve().parent
os.chdir(out.parents[2])
model = Path.home() / 'models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf'
assistant = Path.home() / 'models/gemma-4-12b-assistant/model.safetensors'
for index, label in enumerate(sys.argv[1:] or ['cold', 'cached', 'cold-mtp', 'cached-mtp']):
    draft = label.endswith('-mtp')
    cold = label.startswith('cold')
    port = 18140 + index
    url = f'http://127.0.0.1:{port}'
    logfile = out / f'server-{label}.log'
    command = ['sudo', '-n', 'python3', 'scripts/latency-run.py', 'target/release/trashcan',
               'serve', str(model), '--batch', '128', '--port', str(port)]
    if cold:
        command.append('--no-prefix-cache')
    if draft:
        command += ['--draft', str(assistant)]
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
            print(label, 'ready', flush=True)
            check = ['python3', 'scripts/bench-prefix-cache.py', '--url', url,
                     '--batch', '128', '--output', str(out / f'prefix-{label}.json')]
            if cold:
                check.append('--cold')
            else:
                check += ['--compare-cold', str(out / ('prefix-cold-mtp.json' if draft else 'prefix-cold.json'))]
            if draft:
                check.append('--draft')
            with (out / f'prefix-{label}.log').open('w') as checklog:
                subprocess.run(check, stdout=checklog, stderr=checklog, check=True)
            if not cold:
                with (out / f'protocol-{label}.log').open('w') as checklog:
                    subprocess.run(['python3', 'scripts/test-server.py', '--url', url],
                                   stdout=checklog, stderr=checklog, check=True)
            print(label, 'passed', flush=True)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
