#!/usr/bin/env python3
"""Run one qwen4exp chain measurement over leased Mac Pros.

Nodes start first, last machine of the chain first, each under its gpuq lease with clocks logged and the 0 us CPU
latency request; then the driver. After the driver exits every node is stopped (SIGTERM, then SIGKILL).

usage: q4run.py --name NAME --tickets HOST=ID,... --driver-host HOST --nodes HOST,HOST,... --driver 'CMD ARGS'
                [--binary B] [--env 'K=V ...'] [--node-args ARGS]

The driver command is `trashcan qwen4 CMD MODEL ARGS`; give it `--nodes` itself (the addresses are resolved on the
previous machine of the chain, e.g. invite.tb1:7071).
"""
import argparse, os, re, shlex, subprocess, sys, time

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


for host in [a.driver_host] + nodes:
    if subprocess.run(['ssh', host, f'pgrep -f "^{ROOT}/{a.binary} qwen4"'], capture_output=True).returncode == 0:
        sys.exit(f'{host}: an engine process from an earlier run is still running; stop it first')
children = []
for host in reversed(nodes):
    log = f'results/node-{a.name}-{host}.log'
    cmd = wrapped(f'node-{a.name}-{host}', [f'{ROOT}/{a.binary}', 'qwen4', 'node', a.model, '--listen', '0.0.0.0:7071',
                                             *shlex.split(a.node_args)]) + f' > {log} 2>&1'
    children.append((host, subprocess.Popen(leased(host, cmd))))
time.sleep(3)
dcmd, *dargs = shlex.split(a.driver)
subprocess.run(['ssh', a.driver_host, f'cd {ROOT}; mkdir -p results; echo {shlex.quote(time.strftime("%FT%T") + " " + a.name + " " + a.binary + " " + a.env + " " + a.driver)} >> results/commands.log'])
argv = [dcmd, *dargs] if a.raw else [f'{ROOT}/{a.binary}', 'qwen4', dcmd, a.model, *dargs]
cmd = wrapped(a.name, argv) + f' > results/{a.name}.json 2> results/{a.name}.err'
t0 = time.time()
rc = subprocess.run(leased(a.driver_host, cmd)).returncode
print(f'driver exited {rc} after {time.time() - t0:.0f} s', file=sys.stderr)
time.sleep(2)
for host, child in children:
    for signal, wait in (('TERM', 60), ('KILL', 30)):
        subprocess.run(['ssh', host, f'pkill -{signal} -f "^{ROOT}/{a.binary} qwen4 node "'])
        try:
            child.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            print(f'{host}: node still running {wait} s after SIG{signal}', file=sys.stderr)
subprocess.run(['ssh', a.driver_host, f'cd {ROOT}; tail -5 results/{a.name}.err; head -c 4000 results/{a.name}.json'])
sys.exit(rc)
