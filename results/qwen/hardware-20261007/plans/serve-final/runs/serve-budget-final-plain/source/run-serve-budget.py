#!/usr/bin/env python3
"""One short TP4 serving/cache regression under root-owned, already-granted leases.

Run plain, mtp and optionally adaptive as separate cooled cases. Never acquires or
releases leases. Reuses run-tp4's stream preflight and exact-PID node teardown,
scoped-run's governor/latency restoration, and cooled-plan's temperature reader.
Uncertain startup, HTTP, checkpoint or teardown leaves GPU processes untouched.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TP = load_module('tp4_serving_helpers', HERE / 'run-tp4.py')
COOL = load_module('tp4_serving_cooling', HERE / 'cooled-plan.py')
STOP_SERVER = '''import json, os, pathlib, signal, sys
root = pathlib.Path(sys.argv[1])
client = json.loads((root / 'client/client-summary.json').read_text())
scope = json.loads((root / 'scope.json').read_text())
log = (root / 'stderr.log').read_text(errors='replace')
assert 'request failed:' not in log and 'cleanup failed' not in log, 'server request/GPU cleanup failed; left untouched'
assert client.get('safe_to_stop') and client.get('server_identity_verified'), 'serial idle barrier not confirmed'
assert client['idle_barrier_unix'] >= scope['started_unix'], 'stale idle barrier'
p = pathlib.Path('/proc') / str(scope['pid'])
if p.exists():
    assert p.joinpath('stat').read_text().split()[21] == scope['process_start_ticks'], 'PID reused'
    actual = p.joinpath('cmdline').read_bytes().rstrip(b'\\0').split(b'\\0')
    assert actual == [v.encode() for v in scope['command']], 'server command changed'
    os.kill(scope['pid'], signal.SIGTERM)
'''


def verify_finished_scope(scope):
    # Popen.wait() records the signal as -15; the wrapper/SSH exposes 128+15=143.
    assert scope.get('finished_unix') and scope.get('returncode') in (0, -15, 143)
    assert scope.get('governors_before') and not scope.get('restore_errors')
    assert scope['governors_after'] == scope['governors_before']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--incite', type=int, required=True)
    p.add_argument('--invite', type=int, required=True)
    p.add_argument('--binary', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--mode', choices=('plain', 'mtp', 'adaptive'), required=True)
    p.add_argument('--no-prefix-cache', action='store_true', help='plain generation oracle without prefix reuse')
    p.add_argument('--fan-regime', help='fan policy label recorded in the manifest; does not change controls')
    p.add_argument('--root', default=TP.ROOT)
    p.add_argument('--model', default=TP.MODEL)
    p.add_argument('--mtp-model', default=TP.MTP)
    p.add_argument('--streams', default=TP.STREAM)
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--draft-tokens', type=int, default=2)
    p.add_argument('--budgets', default='1,2,3,4,5,6,7,8')
    p.add_argument('--env', default='')
    p.add_argument('--max-c', type=float, default=72)
    p.add_argument('--cooling-timeout', type=float, default=60)
    p.add_argument('--timeout', type=float, default=240, help='active jobs are never killed on timeout')
    p.add_argument('--startup-timeout', type=float, default=90)
    p.add_argument('--teardown-timeout', type=float, default=30)
    p.add_argument('--stop-probe', action='store_true')
    p.add_argument('--pause-seconds', type=float, default=2)
    p.add_argument('--reference', type=Path, help='completed plain client capture to compare after safe teardown')
    p.add_argument('--source-archive', type=Path, help='optional exact tested build source archive; hash recorded')
    p.add_argument('--out', type=Path, default=HERE.parent / 'serve-runs')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.name):
        p.error('use a simple unique run name')
    if a.no_prefix_cache and a.mode != 'plain':
        p.error('the no-prefix-cache oracle must use plain mode')
    if not a.root.startswith('/home/fairfieldt/trashcan-exp/') or '..' in Path(a.root).parts:
        p.error('use a workspace below /home/fairfieldt/trashcan-exp/')
    streams = a.streams.split(',')
    if not streams or any(not re.fullmatch(r'/dev/tbstream[0-9]+', value) for value in streams):
        p.error('use exact comma-separated Thunderbolt device paths')
    assignments = shlex.split(a.env)
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', value, re.S) for value in assignments):
        p.error('--env needs NAME=value assignments')
    if any(value.split('=', 1)[0] == 'TRASHCAN_QWEN_ADAPTIVE_MTP' for value in assignments):
        p.error('--mode controls TRASHCAN_QWEN_ADAPTIVE_MTP')
    try:
        budgets = [int(value) for value in a.budgets.split(',')]
        assert budgets and len(set(budgets)) == len(budgets) and all(1 <= n <= 8 for n in budgets)
        assert a.incite > 0 and a.invite > 0 and 1024 <= a.port <= 65535 and 1 <= a.draft_tokens <= 7
        assert 0 < a.max_c <= 72 and 0 <= a.pause_seconds <= 10 and 0 < a.cooling_timeout <= 120
        assert 0 < a.timeout <= 240 and 0 < a.startup_timeout <= 120 and 0 < a.teardown_timeout <= 60
    except (ValueError, AssertionError):
        p.error('invalid budget, ticket, port, draft depth, temperature or timeout')
    local = a.out / a.name
    local.mkdir(parents=True, exist_ok=False)
    remote = a.root + '/results/hardware-20261007/' + a.name
    binary = a.binary if a.binary.startswith('/') else a.root + '/' + a.binary
    model_id = 'budget-' + a.name
    helper = (HERE / 'scoped-run.py').read_text()
    client_source = (HERE / 'serve-budget-client.py').read_text()
    env = [*assignments, 'TRASHCAN_QWEN_ADAPTIVE_MTP=' + ('1' if a.mode == 'adaptive' else '0')]
    manifest = dict(name=a.name, mode=a.mode, root=a.root, remote_result_directory=remote,
                    budgets=budgets, binary=binary, model=a.model, mtp_model=a.mtp_model,
                    context=1024, batch=64, checkpoints=8, draft_tokens=a.draft_tokens,
                    dry_run=a.dry_run, started_unix=time.time(), hosts={}, sources={},
                    stop_probe=a.stop_probe, no_prefix_cache=a.no_prefix_cache, gpu_commands_started=False, lease_ownership='caller; server and peer remain inside granted leases; CPU-only client has no lease wrapper')
    if a.fan_regime is not None:
        manifest['fan_policy'] = dict(regime=a.fan_regime)
    for path in [Path(__file__), HERE / 'serve-budget-client.py', HERE / 'run-tp4.py', HERE / 'scoped-run.py', HERE / 'cooled-plan.py', HERE / 'cooled-serve-plan.py']:
        source = path.read_bytes()
        snapshot = local / 'source' / path.name
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_bytes(source)
        manifest['sources'][path.name] = hashlib.sha256(source).hexdigest()
    if a.source_archive:
        manifest['tested_source_archive'] = dict(path=str(a.source_archive.resolve()),
                                                sha256=hashlib.sha256(a.source_archive.read_bytes()).hexdigest())

    def save():
        temporary = local / 'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(local / 'manifest.json')

    def command(host, ticket, node):
        cmd = [binary, 'qwen-tp-node' if node else 'serve', a.model]
        if node:
            cmd += ['--stream', a.streams]
        else:
            cmd += ['--tp-stream', a.streams, '--host', '127.0.0.1', '--port', str(a.port),
                    '--model-id', model_id, '--context', '1024', '--batch', '64',
                    '--max-tokens', '8', '--checkpoints', '8', '--temperature', '0', '--top-k', '1']
            if a.no_prefix_cache:
                cmd.append('--no-prefix-cache')
        if a.mode != 'plain':
            cmd += ['--mtp', a.mtp_model]
            if not node:
                cmd += ['--draft-tokens', str(a.draft_tokens)]
        wrapped = ['scripts/with-clocks.sh', remote + '/clocks.txt', 'sudo', '-n', 'env', *env,
                   'python3', '-c', helper, remote + '/scope.json', *cmd]
        shell = 'cd ' + shlex.quote(a.root) + ' && ' + shlex.join(wrapped) + ' >' + shlex.quote(remote + '/stdout.log') + ' 2>' + shlex.quote(remote + '/stderr.log')
        argv = TP.SSH + [host, shlex.join(['gpuq', 'run', str(ticket), '--', 'bash', '-c', shell])]
        manifest['hosts'][host] = dict(ticket=ticket, command=cmd, environment=env, ssh_argv=argv,
                                       exact_shell_command=shlex.join(argv))
        return argv

    node_argv = command('invite', a.invite, True)
    server_argv = command('incite', a.incite, False)
    client_cmd = ['python3', '-c', client_source, '--url', f'http://127.0.0.1:{a.port}', '--model', model_id,
                  '--out', remote + '/client', '--budgets', a.budgets, '--pause-seconds', str(a.pause_seconds)]
    if a.stop_probe:
        client_cmd.append('--stop-probe')
    if a.no_prefix_cache:
        client_cmd.append('--no-prefix-cache')
    # The CPU-only HTTP client invokes the server already inside the granted
    # incite lease. A second gpuq run wrapper is neither needed nor requested.
    client_argv = TP.SSH + ['incite', shlex.join(client_cmd)]
    manifest['client_ssh_argv'] = client_argv
    save()
    if a.dry_run:
        manifest['state'] = 'dry-run; no remote operations'
        save()
        print(local / 'manifest.json')
        return 0

    def ssh(host, cmd, timeout=15):
        return subprocess.run(TP.SSH + [host, cmd], capture_output=True, timeout=timeout)

    def start(label, argv):
        with (local / (label + '.ssh.stdout.log')).open('wb') as stdout, (local / (label + '.ssh.stderr.log')).open('wb') as stderr:
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        manifest[label + '_local_ssh_pid'] = child.pid
        save()
        return child

    def checked(host, args):
        result = ssh(host, shlex.join(args))
        if result.returncode:
            raise RuntimeError(host + ': ' + result.stderr.decode(errors='replace'))
        return result

    def ready(host, child, marker, deadline):
        while True:
            if child.poll() is not None:
                raise RuntimeError(host + ' exited before readiness; inspect scope and GPU state')
            result = ssh(host, shlex.join(['grep', '-q', marker, remote + '/stderr.log']))
            if result.returncode == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(host + ' startup not confirmed; process left untouched')
            time.sleep(1)

    def wait_exit(child, seconds, label):
        deadline = time.monotonic() + seconds
        while child.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(label + ' timeout; process left untouched')
            time.sleep(1)
        return child.returncode

    def retrieve():
        for host in ('incite', 'invite'):
            try:
                archive = local / (host + '.tar')
                with archive.open('wb') as out:
                    result = subprocess.run(TP.SSH + [host, shlex.join(['tar', '-C', remote, '-cf', '-', '.'])],
                                            stdout=out, stderr=subprocess.PIPE, timeout=30)
                (local / (host + '.retrieval.stderr.log')).write_bytes(result.stderr)
                manifest['hosts'][host]['retrieval_returncode'] = result.returncode
                if result.returncode == 0:
                    dest = local / host
                    dest.mkdir(exist_ok=True)
                    with tarfile.open(archive) as tar:
                        tar.extractall(dest, filter='data')
            except Exception as error:
                manifest['hosts'][host]['retrieval_error'] = repr(error)
        save()

    code = 1
    try:
        for host in ('invite', 'incite'):
            result = ssh(host, 'mkdir -p ' + shlex.quote(str(Path(remote).parent)) + ' && mkdir ' + shlex.quote(remote))
            if result.returncode:
                raise RuntimeError(host + ' result directory setup failed: ' + result.stderr.decode())
            result = checked(host, ['python3', '-c', TP.PREFLIGHT_STREAMS, *streams])
            manifest['hosts'][host]['stream_preflight'] = json.loads(result.stdout)
        deadline = time.monotonic() + a.cooling_timeout
        manifest['thermal_samples'] = []
        while True:
            samples = {host: json.loads(checked(host, ['python3', '-c', COOL.READ_TEMPERATURES]).stdout)
                       for host in ('incite', 'invite')}
            maximum = max(row['celsius'] for data in samples.values() for row in data['gpus'])
            manifest['thermal_samples'].append(dict(unix=time.time(), maximum_c=maximum, hosts=samples))
            save()
            if maximum <= a.max_c:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('still cooling; no GPU commands started')
            time.sleep(5)
        wall_deadline = time.monotonic() + a.timeout
        manifest['gpu_commands_started'] = True
        save()
        node = start('invite', node_argv)
        ready('invite', node, 'tp-node:', min(wall_deadline, time.monotonic() + a.startup_timeout))
        server = start('incite', server_argv)
        ready('incite', server, 'Responses server listening', min(wall_deadline, time.monotonic() + a.startup_timeout))
        client = start('client', client_argv)
        client_code = wait_exit(client, max(0, wall_deadline - time.monotonic()), 'HTTP client')
        result = checked('incite', ['cat', remote + '/client/client-summary.json'])
        manifest['client_summary'] = json.loads(result.stdout)
        manifest['client_returncode'] = client_code
        save()
        # This checks a serial HTTP barrier AND exact PID/start time/argv. It is
        # never called merely because the local client command stopped or timed out.
        checked('incite', ['python3', '-c', STOP_SERVER, remote])
        manifest['hosts']['incite']['ssh_returncode'] = wait_exit(server, a.teardown_timeout, 'server exit')
        deadline = time.monotonic() + a.teardown_timeout
        while node.poll() is None:
            result = ssh('invite', shlex.join(['python3', '-c', TP.STOP_NODE, remote]))
            if result.returncode == 0:
                wait_exit(node, 20, 'node exit')
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('node teardown not confirmed; left untouched: ' + result.stderr.decode())
            time.sleep(1)
        manifest['hosts']['invite']['ssh_returncode'] = node.returncode
        manifest['state'] = 'completed'
        code = int(client_code != 0 or not manifest['client_summary'].get('passed')
                   or any(manifest['hosts'][host]['ssh_returncode'] not in (0, 143) for host in ('incite', 'invite')))
    except (Exception, KeyboardInterrupt) as error:
        manifest['state'] = 'incomplete; inspect jobs before releasing leases' if manifest['gpu_commands_started'] else 'not started; no GPU commands issued'
        manifest['error'] = repr(error)
        print(str(error), file=sys.stderr, flush=True)
    finally:
        manifest['finished_unix'] = time.time()
        save()
        retrieve()
    if manifest.get('state') == 'completed':
        for host in ('incite', 'invite'):
            try:
                scope = json.loads((local / host / 'scope.json').read_text())
                verify_finished_scope(scope)
                manifest['hosts'][host]['binary_sha256'] = scope['binary_sha256']
            except Exception as error:
                manifest['hosts'][host]['scope_verification_error'] = repr(error)
                code = 1
        if code == 0 and a.reference:
            try:
                client_module = load_module('completed_budget_comparison', HERE / 'serve-budget-client.py')
                manifest['comparison'] = client_module.compare(a.reference, local / 'incite/client')
                manifest['reference'] = str(a.reference.resolve())
            except Exception as error:
                manifest['comparison_error'] = repr(error)
                code = 1
        save()
    print(local / 'manifest.json', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
