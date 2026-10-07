#!/usr/bin/env python3
"""GPU cancellation regression on one leased Mac Pro, with a model that fits TP2.

Run through gpuq run, with with-clocks.sh and latency-run.py around this script.
Tests signals, active event/timeline gates, failed peer I/O, and normal output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate', default='./candidate')
p.add_argument('--baseline', default='./baseline')
p.add_argument('--model', default='/home/fairfieldt/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf')
p.add_argument('--mtp', default='/home/fairfieldt/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf')
p.add_argument('--out', default='results')
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
candidate, baseline = str(Path(a.candidate).resolve()), str(Path(a.baseline).resolve())
events = []


def record(**values):
    values['unix'] = time.time()
    events.append(values)
    with (out / 'events.jsonl').open('a') as f:
        f.write(json.dumps(values) + '\n')
    print(json.dumps(values), flush=True)


def start(name, args, env=None):
    settings = os.environ.copy(); settings.update(env or {})
    stdout = (out / (name + '.jsonl')).open('w')
    stderr = (out / (name + '.log')).open('w')
    proc = subprocess.Popen(args, env=settings, stdout=stdout, stderr=stderr, start_new_session=True)
    stdout.close(); stderr.close()
    record(event='start', name=name, pid=proc.pid, command=args, env=env or {})
    return proc


def wait_for(proc, predicate, name, timeout=60):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert proc.poll() is None, f'{name} exited early: {proc.returncode}'
        assert time.monotonic() < deadline, f'{name} did not become ready'
        time.sleep(.01)


def text(name):
    return (out / (name + '.log')).read_text(errors='replace')


def stop(proc, name, sig=signal.SIGTERM, second=None):
    begin = time.monotonic()
    proc.send_signal(sig)
    if second:
        time.sleep(.001)
        if proc.poll() is None:
            proc.send_signal(second)
    code = proc.wait(timeout=5)
    elapsed = time.monotonic() - begin
    record(event='cancel', name=name, signal=sig, second=second, exit=code, seconds=elapsed)
    assert code == -sig, (name, code, sig)
    assert elapsed < 3, (name, elapsed)
    assert 'cleanup failed' not in text(name), name


def bench_args(binary, mtp=False, pp=512, tg=128, reps=2):
    return [binary, 'qwen-bench', a.model, '--tp', '--batch', '256', '--context', '8192',
            '--pp', str(pp), '--tg', str(tg), '--reps', str(reps)] + (['--mtp', a.mtp] if mtp else [])


def rows(name):
    return [json.loads(line) for line in (out / (name + '.jsonl')).read_text().splitlines()]


def profile_has(path, n):
    if not path.exists(): return False
    for line in path.read_text().splitlines():
        try:
            if json.loads(line).get('n') == n: return True
        except ValueError: pass  # A line may still be being appended.
    return False


def cancel_bench(name, sig, env=None, pp=128, mtp=False, n=1, loading=False, second=None):
    profile = out / (name + '.profile.jsonl')
    settings = dict(env or {}, TRASHCAN_TP_PROFILE=str(profile.resolve()))
    proc = start(name, bench_args(candidate, mtp, pp=pp, tg=4096, reps=1), settings)
    try:
        if loading:
            wait_for(proc, lambda: 'tp: 2 ranks' in text(name), name)
            time.sleep(.05)
        else:
            wait_for(proc, lambda: profile_has(profile, n), name)
            time.sleep(.005)  # Advance into the next operation, not the profile write.
        stop(proc, name, sig, second)
    finally:
        if proc.poll() is None: stop(proc, name)


def node(name):
    # Same-host TCP is solely a cancellation/peer-disconnect test, not a TP4 benchmark.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    proc = start(name, [candidate, 'qwen-tp-node', a.model, '--mtp', a.mtp,
                        '--gpus', '1', '--listen', f'127.0.0.1:{port}'])
    wait_for(proc, lambda: 'tp-node:' in text(name), name)
    return proc, port


def receive(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        assert chunk, 'peer closed unexpectedly'
        data.extend(chunk)
    return bytes(data)


def send_message(sock, value):
    data = json.dumps(value).encode(); sock.sendall(struct.pack('<I', len(data)) + data)


def message(sock):
    return json.loads(receive(sock, struct.unpack('<I', receive(sock, 4))[0]))


def failed_peer(name, proc, port, tokens, micro):
    previous = text(name).count('session ended')
    with socket.create_connection(('127.0.0.1', port), timeout=30) as sock:
        sock.sendall(struct.pack('<4I', 0x454e414c, 1, 0, 1))
        send_message(sock, {'setup': {'first': 1, 'ranks': 2, 'context': 512, 'batch': 128,
                       'verify': 3, 'mtp': True, 'draft_vocab': 65536}})
        assert message(sock).get('ok'), 'node setup failed'
        send_message(sock, {'forward': {'tokens': [1] * tokens, 'pos': 0,
                       'verify': False, 'dump': False, 'micro': micro, 'threads': 1}})
        receive(sock, 1)  # Its first GPU segment finished; remaining work is gated.
    begin = time.monotonic()
    wait_for(proc, lambda: text(name).count('session ended') > previous, name, timeout=5)
    elapsed = time.monotonic() - begin
    record(event='peer_closed', name=name, tokens=tokens, micro=micro, seconds=elapsed)
    assert elapsed < 3 and 'cleanup failed' not in text(name)


def main():
    (out / 'events.jsonl').write_text('')
    for label, path in [('baseline', baseline), ('candidate', candidate)]:
        record(event='binary', name=label, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    record(event='kernel', version=Path('/proc/sys/kernel/osrelease').read_text().strip(),
           boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    governors = {path: path.read_text() for path in Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_governor')}
    try:
        for path in governors:
            subprocess.run(['sudo', '-n', 'tee', str(path)], input=b'performance', stdout=subprocess.DEVNULL, check=True)
        # Normal inference and MTP exercise buffer reuse/rollback before signals.
        for label, binary, mtp in [('baseline-plain', baseline, False), ('candidate-plain', candidate, False),
                                   ('candidate-mtp', candidate, True), ('baseline-mtp', baseline, True)]:
            proc = start(label, bench_args(binary, mtp))
            assert proc.wait(timeout=120) == 0, label
            record(event='completed', name=label, rows=len(rows(label)))
        for kind in ['plain', 'mtp']:
            old, new = rows('baseline-' + kind), rows('candidate-' + kind)
            assert len(old) == len(new) == 2
            assert all(x['token_ids'] == y['token_ids'] for x, y in zip(old, new)), kind
        record(event='correctness', baseline_candidate_ids_match=True)

        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            name = 'idle-node-' + str(sig); proc, _ = node(name)
            stop(proc, name, sig)
        cancel_bench('loading', signal.SIGTERM, loading=True)
        cancel_bench('decode-int', signal.SIGINT)
        cancel_bench('decode-term', signal.SIGTERM)
        cancel_bench('decode-hup', signal.SIGHUP)
        cancel_bench('decode-repeat', signal.SIGINT, second=signal.SIGTERM)
        cancel_bench('timeline-decode', signal.SIGTERM, {'TRASHCAN_TP_EVENTS': '0'})
        cancel_bench('per-segment', signal.SIGTERM, {'TRASHCAN_TP_EVENTS': '0', 'TRASHCAN_TP_SUBMIT': '1'})
        cancel_bench('dual-prefill', signal.SIGTERM, pp=4096, n=128)
        cancel_bench('mtp-verify', signal.SIGTERM, mtp=True, n=3)

        name = 'peer-disconnect'; proc, port = node(name)
        try:
            failed_peer(name, proc, port, tokens=1, micro=1)
            failed_peer(name, proc, port, tokens=128, micro=2)
            # Reuse that same node for real inference after both failed sessions.
            driver = start('peer-reuse', [candidate, 'qwen-bench', a.model, '--gpus', '0',
                '--tp-connect', f'127.0.0.1:{port}', '--context', '512', '--batch', '128', '--pp', '128', '--tg', '16'])
            assert driver.wait(timeout=90) == 0
            wait_for(proc, lambda: text(name).count('session ended') >= 3, name)
            record(event='peer_reuse', success=True)
        finally:
            if proc.poll() is None: stop(proc, name)

        name = 'idle-http'
        proc = start(name, [candidate, 'serve', a.model, '--tp', '--context', '512',
                           '--batch', '128', '--port', '18011', '--model-id', 'qwen-shutdown-test'])
        try:
            wait_for(proc, lambda: 'Responses server listening' in text(name), name)
            stop(proc, name)
        finally:
            if proc.poll() is None: stop(proc, name)

        # A resident driver must not silently continue with only its own shard
        # after closing a lost peer connection.
        peer_name = 'http-lost-peer'; peer, port = node(peer_name)
        name = 'http-failed-session'
        proc = start(name, [candidate, 'serve', a.model, '--gpus', '0',
                           '--tp-connect', f'127.0.0.1:{port}', '--context', '512',
                           '--batch', '128', '--port', '18012', '--model-id', 'qwen-shutdown-test'])
        try:
            wait_for(proc, lambda: 'Responses server listening' in text(name), name)
            stop(peer, peer_name)
            payload = json.dumps(dict(model='qwen-shutdown-test', input='Say hello.', max_output_tokens=4)).encode()
            errors = []
            for _ in range(2):
                request = urllib.request.Request('http://127.0.0.1:18012/v1/responses', data=payload,
                                                  headers={'Content-Type': 'application/json'})
                try:
                    urllib.request.urlopen(request, timeout=10)
                    raise AssertionError('failed TP session returned successful inference')
                except urllib.error.HTTPError as error:
                    assert error.code == 500, error.code
                    errors.append(json.loads(error.read())['error']['message'])
            assert 'session failed; reload the model' in errors[1], errors
            record(event='failed_session_rejected', errors=errors)
        finally:
            if peer.poll() is None: stop(peer, peer_name)
            if proc.poll() is None: stop(proc, name)
        # Fresh GPU allocations and output after the interruption sequence.
        name = 'after-cancellation'; proc = start(name, bench_args(candidate, mtp=True))
        assert proc.wait(timeout=120) == 0
        assert all(x['token_ids'] == y['token_ids'] for x, y in zip(rows('candidate-mtp'), rows(name)))
        record(event='success', normal_inference_after_cancellation=True)
    finally:
        for path, value in governors.items():
            subprocess.run(['sudo', '-n', 'tee', str(path)], input=value.encode(), stdout=subprocess.DEVNULL, check=True)


if __name__ == '__main__': main()
