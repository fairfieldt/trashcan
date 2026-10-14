#!/usr/bin/env python3
"""Run one qwen4exp chain measurement over leased Mac Pros.

Nodes start first, last machine of the chain first, each under its gpuq lease with clocks logged and the 0 us CPU
latency request; the driver starts as soon as every node's port is open. After the driver exits every node is stopped
(SIGTERM, then SIGKILL), all at once.

usage: q4run.py --name NAME --tickets HOST=ID,... --driver-host HOST --nodes HOST,HOST,... --driver 'CMD ARGS'
                [--binary B] [--env 'K=V ...'] [--node-args ARGS]

The driver command is `trashcan qwen4 CMD MODEL ARGS`; give it `--nodes` itself (the addresses are resolved on the
previous machine of the chain, e.g. invite.tb1:7071).
"""
import argparse, os, re, shlex, signal, socket, subprocess, sys, time

ROOT = os.environ.get('Q4_ROOT', '/home/fairfieldt/trashcan-exp/flashnext')
MODEL = os.environ.get('Q4_MODEL', '/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf')

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--tickets', required=True)
p.add_argument('--binary', default='trashcan')
p.add_argument('--env', default='')
p.add_argument('--driver-host', default='incite')
p.add_argument('--nodes', default='', help='node hosts in chain order')
p.add_argument('--node-args', default='--gpus 0,1')
p.add_argument('--driver', required=True)
p.add_argument('--raw', action='store_true', help='--driver is a whole command run in the workspace (e.g. a server test)')
p.add_argument('--model', default=MODEL)
p.add_argument('--dpm', default='', help='hold the GPUs at this amdgpu DPM level during the run (tools/dpm-run.sh)')
a = p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.name):
    p.error('bad name')
tickets = dict(t.split('=') for t in a.tickets.split(','))
nodes = [h for h in a.nodes.split(',') if h]


def leased(host, cmd):
    return ['ssh', host, f'gpuq run {tickets[host]} -- bash -c ' + shlex.quote(f'cd {ROOT}; mkdir -p results/clocks; {cmd}')]


def wrapped(clock, argv):
    dpm = ['scripts/dpm-run.sh', a.dpm, '--'] if a.dpm else []
    return shlex.join([*dpm, 'scripts/with-clocks.sh', f'results/clocks/{clock}.txt', 'sudo', '-n', 'env', *shlex.split(a.env),
                       'python3', 'scripts/latency-run.py', *argv])


start = time.time()
# Refuse to start when any engine binary from an earlier run is still running (whatever its name; it would take this
# run's place in the chain), or when a GPU refuses the high DPM level: amdgpu latches an overheated SI GPU in its
# thermal state (power level 1, 450 MHz) until a reboot, and that one stage then paces the whole chain.
PRECHECK = (f'pgrep -f "^{ROOT}/trashcan[^ ]* qwen4 " > /dev/null && exit 3; '
            'for c in 0 1; do f=/sys/class/drm/card$c/device/power_dpm_force_performance_level; '
            'echo high | sudo -n tee $f > /dev/null 2>&1 || exit $((10 + c)); echo auto | sudo -n tee $f > /dev/null 2>&1; '
            'done; exit 0')
checks = [(host, subprocess.Popen(['ssh', host, PRECHECK], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
          for host in [a.driver_host] + nodes]
for host, check in checks:
    rc = check.wait()
    if rc == 3:
        sys.exit(f'{host}: an engine process from an earlier run is still running; stop it first')
    if rc in (10, 11):
        sys.exit(f'{host}: GPU {rc - 10} refuses the high DPM level (thermal state latched after overheating); reboot it')


def listening(host):
    try:
        with socket.create_connection((f'{host}.tdf.sh', 7071), timeout=1):
            return True
    except OSError:
        return False


def launch(host):
    log = f'results/node-{a.name}-{host}.log'
    cmd = wrapped(f'node-{a.name}-{host}', [f'{ROOT}/{a.binary}', 'qwen4', 'node', a.model, '--listen', '0.0.0.0:7071',
                                             *shlex.split(a.node_args)]) + f' > {log} 2>&1'
    return subprocess.Popen(leased(host, cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


for host in nodes:
    if listening(host):
        sys.exit(f'{host}: port 7071 is already open before this run started its node; stop whatever holds it first')
children = [(host, launch(host)) for host in reversed(nodes)]


def stop_nodes():
    t1 = time.time()
    for sig, wait in (('TERM', 60), ('KILL', 30)):
        running = [(h, c) for h, c in children if c.poll() is None]
        if not running:
            break
        for p in [subprocess.Popen(['ssh', h, f'pkill -{sig} -f "^{ROOT}/{a.binary} qwen4 node "']) for h, _ in running]:
            p.wait()
        deadline = time.time() + wait
        for h, c in running:
            try:
                c.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                print(f'{h}: node still running {wait} s after SIG{sig}', file=sys.stderr)
    print(f'nodes stopped after {time.time() - t1:.1f} s', file=sys.stderr)


def stopped(signum, _frame):
    # Stopped from outside (timeout, Ctrl-C): stop the driver and the nodes before exiting, so no engine process is
    # left behind on any Mac.
    subprocess.run(['ssh', a.driver_host, f'pkill -TERM -f "^{ROOT}/{a.binary} qwen4 "'])
    stop_nodes()
    sys.exit(128 + signum)


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, stopped)


# Start the driver once every node listens (a node accepts the probe as an empty session and waits for the next).
# A launch that ends before its node listens (e.g. gpuq run refusing while the previous run's wrapper is still
# exiting) is shown and retried once; a second failure stops the run before the driver starts.
waiting, retried = set(nodes), set()
while waiting and time.time() - start < 120:
    for i, (host, child) in enumerate(children):
        if host in waiting and child.poll() is not None:
            print(f'{host}: node launch ended (rc {child.returncode}): {child.stdout.read().strip()[-300:]}', file=sys.stderr)
            if host in retried:
                stop_nodes()
                sys.exit(f'{host}: node did not start twice; stopping')
            retried.add(host)
            time.sleep(1)
            children[i] = (host, launch(host))
    waiting = {h for h in waiting if not listening(h)}
    if waiting:
        time.sleep(0.2)
if waiting:
    print(f'nodes not listening after 120 s: {sorted(waiting)}', file=sys.stderr)
dcmd, *dargs = shlex.split(a.driver)
argv = [dcmd, *dargs] if a.raw else [f'{ROOT}/{a.binary}', 'qwen4', dcmd, a.model, *dargs]
note = shlex.quote(time.strftime("%FT%T") + " " + a.name + " " + a.binary + " " + a.env + " " + a.driver)
cmd = f'echo {note} >> results/commands.log; ' + wrapped(a.name, argv) + f' > results/{a.name}.json 2> results/{a.name}.err'
t0 = time.time()
rc = subprocess.run(leased(a.driver_host, cmd)).returncode
print(f'nodes up after {t0 - start:.1f} s; driver exited {rc} after {time.time() - t0:.0f} s', file=sys.stderr)
stop_nodes()
subprocess.run(['ssh', a.driver_host, f'cd {ROOT}; tail -5 results/{a.name}.err; head -c 4000 results/{a.name}.json'])
sys.exit(rc)
