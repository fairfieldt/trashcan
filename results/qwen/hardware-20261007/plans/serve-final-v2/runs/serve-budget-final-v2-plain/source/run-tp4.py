#!/usr/bin/env python3
"""Run a single TP4 case under two already-granted gpuq leases.

Requires scripts/with-clocks.sh, clockmon.sh, clocksummary.sh and the binary in
the remote workspace. This script never requests/releases leases. A wall timeout
leaves active jobs intact; inspect the manifest and remote logs before releasing.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = '/home/fairfieldt/trashcan-exp/tp4-offline-20261007'
MODEL = '/home/fairfieldt/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf'
MTP = '/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf'
STREAM = '/dev/tbstream0,/dev/tbstream1,/dev/tbstream2'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
       '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3']

# Stop only our exact recorded process, with the most recent session torn down.
# This runs as the same user as the node, never uses a broad pkill pattern.
STOP_NODE = '''import json, os, pathlib, signal, sys
root = pathlib.Path(sys.argv[1])
s = json.loads((root / 'scope.json').read_text())
log = (root / 'stderr.log').read_text(errors='replace')
last_start = log.rfind('tp-node: session over ')
last_end = log.rfind('tp-node: session ended')
# Failed stream opens produce an ended line before a session ever starts. They
# are safe to stop too. Compare ordering, not counts: pre-session failures must
# never make an actually active later session look finished.
assert last_end >= 0 and last_end > last_start, 'node GPU teardown not confirmed'
assert 'cleanup failed' not in log, 'GPU cleanup failure; node left untouched'
p = pathlib.Path('/proc') / str(s['pid'])
if p.exists():
    assert p.joinpath('stat').read_text().split()[21] == s['process_start_ticks'], 'PID reused'
    actual = p.joinpath('cmdline').read_bytes().rstrip(b'\\0').split(b'\\0')
    assert actual == [v.encode() for v in s['command']], 'node command changed'
    os.kill(s['pid'], signal.SIGTERM)
'''

PREFLIGHT_STREAMS = '''import json, os, sys
paths = sys.argv[1:]
status = {path: dict(exists=os.path.exists(path), readable=os.access(path, os.R_OK),
                     writable=os.access(path, os.W_OK)) for path in paths}
print(json.dumps(dict(uid=os.getuid(), streams=status)))
if any(not all(checks.values()) for checks in status.values()):
    sys.exit('Thunderbolt devices must exist and be readable/writable by the benchmark user')
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--incite', type=int, required=True)
    p.add_argument('--invite', type=int, required=True)
    p.add_argument('--binary', required=True, help='absolute path or workspace-relative filename')
    p.add_argument('--name', required=True)
    p.add_argument('--root', default=ROOT)
    p.add_argument('--model', default=MODEL)
    p.add_argument('--mtp-model', default=MTP)
    p.add_argument('--stream', '--streams', dest='stream', default=STREAM,
                   help='comma-separated Thunderbolt device paths; no TCP fallback')
    p.add_argument('--command', choices=['qwen-bench', 'qwen-check', 'qwen-validate'], default='qwen-bench')
    p.add_argument('--args', default='--pp 512 --tg 128 --reps 2')
    p.add_argument('--env', default='', help='shell-quoted NAME=value assignments, no shell expansion')
    p.add_argument('--mtp', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--fan-regime', help='fan policy label recorded in the manifest; does not change controls')
    p.add_argument('--timeout', type=float, default=900, help='driver wall seconds; active jobs are left intact')
    p.add_argument('--startup-timeout', type=float, default=30)
    p.add_argument('--teardown-timeout', type=float, default=60)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'runs')
    p.add_argument('--dry-run', action='store_true', help='write exact commands locally; make no SSH calls')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.name):
        p.error('name must contain only letters, digits, underscores or hyphens')
    if not a.root.startswith('/home/fairfieldt/trashcan-exp/') or '..' in Path(a.root).parts:
        p.error('root must be an absolute workspace below /home/fairfieldt/trashcan-exp/')
    assignments = shlex.split(a.env)
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', value, re.S) for value in assignments):
        p.error('--env must contain NAME=value assignments')
    if min(a.incite, a.invite, a.timeout, a.startup_timeout, a.teardown_timeout) <= 0:
        p.error('tickets and timeouts must be positive')
    streams = a.stream.split(',')
    if any(not re.fullmatch(r'/dev/tbstream[0-9]+', path) for path in streams):
        p.error('--streams must be comma-separated /dev/tbstreamN device paths')
    binary = a.binary if a.binary.startswith('/') else a.root + '/' + a.binary
    helper = Path(__file__).with_name('scoped-run.py').read_text()
    local = a.out / a.name
    local.mkdir(parents=True, exist_ok=False)
    remote = a.root + '/results/hardware-20261007/' + a.name
    manifest = dict(name=a.name, root=a.root, remote_result_directory=remote,
                    binary=binary, environment=assignments, started_unix=time.time(),
                    timeout_seconds=a.timeout, dry_run=a.dry_run, hosts={})
    if a.fan_regime is not None:
        manifest['fan_policy'] = dict(regime=a.fan_regime)

    def save():
        (local / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def command(host, ticket, node):
        env = list(assignments)
        if a.trace or a.profile:
            env += ['TRASHCAN_TP_PROFILE=' + remote + '/profile.jsonl']
        if a.profile:
            env += ['TRASHCAN_TP_PROFILE_DISPATCH=1']
        cmd = [binary, 'qwen-tp-node' if node else a.command, a.model]
        if node:
            cmd += ['--stream', a.stream]
        else:
            cmd += ['--tp-stream', a.stream] + shlex.split(a.args)
        if a.mtp:
            cmd += ['--mtp', a.mtp_model]
        wrapped = ['scripts/with-clocks.sh', remote + '/clocks.txt', 'sudo', '-n',
                   'env', *env, 'python3', '-c', helper, remote + '/scope.json', *cmd]
        shell = ('cd ' + shlex.quote(a.root) + ' && ' + shlex.join(wrapped)
                 + ' >' + shlex.quote(remote + '/stdout.log')
                 + ' 2>' + shlex.quote(remote + '/stderr.log'))
        leased = shlex.join(['gpuq', 'run', str(ticket), '--', 'bash', '-c', shell])
        argv = SSH + [host, leased]
        manifest['hosts'][host] = dict(ticket=ticket, command=cmd, environment=env,
                                       ssh_argv=argv, exact_shell_command=shlex.join(argv))
        return argv

    node_argv = command('invite', a.invite, True)
    driver_argv = command('incite', a.incite, False)
    save()
    if a.dry_run:
        print(local / 'manifest.json')
        return 0

    def ssh(host, shell, timeout=15):
        return subprocess.run(SSH + [host, shell], capture_output=True, timeout=timeout)

    def start(host, argv):
        with (local / (host + '.ssh.stdout.log')).open('wb') as stdout, \
             (local / (host + '.ssh.stderr.log')).open('wb') as stderr:
            child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        manifest['hosts'][host]['local_ssh_pid'] = child.pid
        save()
        return child

    def retrieve():
        for host in ['incite', 'invite']:
            dest = local / host
            dest.mkdir(exist_ok=True)
            archive = local / (host + '.tar')
            try:
                with archive.open('wb') as output:
                    proc = subprocess.run(SSH + [host, shlex.join(['tar', '-C', remote, '-cf', '-', '.'])],
                                          stdout=output, stderr=subprocess.PIPE, timeout=30)
                manifest['hosts'][host]['retrieval_returncode'] = proc.returncode
                (local / (host + '.retrieval.stderr.log')).write_bytes(proc.stderr)
                if proc.returncode == 0:
                    with tarfile.open(archive) as tar:
                        tar.extractall(dest, filter='data')
            except Exception as error:
                manifest['hosts'][host]['retrieval_error'] = repr(error)
        save()

    node = driver = None
    code = 1
    try:
        for host in ['invite', 'incite']:
            # Fail rather than overwrite a prior remote run, even if local logs moved.
            result = ssh(host, 'mkdir -p ' + shlex.quote(str(Path(remote).parent))
                         + ' && mkdir ' + shlex.quote(remote))
            if result.returncode:
                raise RuntimeError(f'{host} result-directory setup failed: {result.stderr.decode()}')
        # Check as the SSH/benchmark user before either host starts a GPU process.
        # os.access accounts for ACLs without opening or resetting the stream.
        for host in ['invite', 'incite']:
            result = ssh(host, shlex.join(['python3', '-c', PREFLIGHT_STREAMS, *streams]))
            manifest['hosts'][host]['stream_preflight'] = dict(
                returncode=result.returncode, stdout=result.stdout.decode(), stderr=result.stderr.decode())
            save()
            if result.returncode:
                raise RuntimeError(f'{host} stream preflight failed: {result.stdout.decode()} {result.stderr.decode()}')
        node = start('invite', node_argv)
        deadline = time.monotonic() + a.startup_timeout
        while True:
            result = ssh('invite', shlex.join(['grep', '-q', 'tp-node:', remote + '/stderr.log']))
            if result.returncode == 0:
                break
            if node.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('node did not become ready; no GPU process was killed')
            time.sleep(1)
        driver = start('incite', driver_argv)
        deadline = time.monotonic() + a.timeout
        while driver.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError('driver wall timeout: jobs left running; do not release leases while active')
            time.sleep(1)
        manifest['hosts']['incite']['ssh_returncode'] = driver.returncode
        deadline = time.monotonic() + a.teardown_timeout
        last = None
        while node.poll() is None:
            last = ssh('invite', shlex.join(['python3', '-c', STOP_NODE, remote]))
            if last.returncode == 0:
                node.wait(timeout=20)
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('node teardown not confirmed; left running: ' + last.stderr.decode())
            time.sleep(1)
        manifest['hosts']['invite']['ssh_returncode'] = node.returncode
        manifest['state'] = 'completed'
        code = driver.returncode or (0 if node.returncode in [0, 143] else 1)
    except (Exception, KeyboardInterrupt) as error:
        manifest['state'] = 'incomplete; inspect remote jobs before releasing leases'
        manifest['error'] = repr(error)
        print(str(error), file=sys.stderr)
    finally:
        manifest['finished_unix'] = time.time()
        save()
        retrieve()
    if manifest.get('state') == 'completed':
        for host in ['incite', 'invite']:
            try:
                scope = json.loads((local / host / 'scope.json').read_text())
                assert scope.get('finished_unix'), 'scope did not finish'
                assert not scope.get('restore_errors'), scope['restore_errors']
                assert scope['governors_after'] == scope['governors_before'], 'governors were not restored'
            except Exception as error:
                manifest['hosts'][host]['scope_verification_error'] = repr(error)
                code = 1
        save()
    print(local / 'manifest.json')
    return code


if __name__ == '__main__':
    sys.exit(main())
