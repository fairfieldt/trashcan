#!/usr/bin/env python3
"""Run one tensor-parallel measurement over leased Mac Pros.

Nodes start first (in the order given, so give the highest machine index first), each under its gpuq lease with
clocks logged and the 0 us CPU latency request; then the driver on incite. After the driver exits, each node is
stopped once it has logged the end of its session (GPU teardown done).

usage: tprun.py --name NAME --tickets incite=ID,invite=ID,... [--binary B] [--env 'K=V ...']
                --node 'HOST:ARGS' ... --driver 'COMMAND ARGS'

Copied from results/qwen/tp8-20261007/tools with the workspace, model and MTP paths taken from TP8_ROOT, TP8_MODEL
and TP8_MTP when set, and --profile to record dispatch timestamps on every machine.
"""
import argparse, os, re, shlex, subprocess, sys, time

ROOT = os.environ.get('TP8_ROOT', '/home/fairfieldt/trashcan-exp/tp8opt')
MODEL = os.environ.get('TP8_MODEL', '/home/fairfieldt/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf')
MTP = os.environ.get('TP8_MTP', '/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf')

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--tickets', required=True)
p.add_argument('--binary', default='trashcan-base')
p.add_argument('--env', default='')
p.add_argument('--node', action='append', default=[])
p.add_argument('--driver', required=True)
p.add_argument('--model', default=MODEL)
p.add_argument('--profile', action='store_true', help='TRASHCAN_TP_PROFILE with dispatch stamps on every machine')
a = p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.name):
    p.error('bad name')
tickets = dict(t.split('=') for t in a.tickets.split(','))


def leased(host, cmd):
    return ['ssh', host, f'gpuq run {tickets[host]} -- bash -c ' + shlex.quote(f'cd {ROOT}; mkdir -p results/clocks; {cmd}')]


def wrapped(clock, args):
    env = shlex.split(a.env)
    if a.profile:
        env += [f'TRASHCAN_TP_PROFILE={ROOT}/results/profile-{clock}.jsonl', 'TRASHCAN_TP_PROFILE_DISPATCH=1']
    return shlex.join(['scripts/with-clocks.sh', f'results/clocks/{clock}.txt', 'sudo', '-n', 'env', *env,
                       'python3', 'scripts/latency-run.py', f'{ROOT}/{a.binary}', *args])


nodes = []
for spec in a.node:
    host, args = spec.split(':', 1)
    log = f'results/node-{a.name}-{host}.log'
    cmd = wrapped(f'node-{a.name}-{host}', ['qwen-tp-node', a.model, '--mtp', MTP, *shlex.split(args)]) + f' > {log} 2>&1'
    nodes.append((host, log, subprocess.Popen(leased(host, cmd))))
    time.sleep(1)
time.sleep(2)
dcmd, *dargs = shlex.split(a.driver)
subprocess.run(['ssh', 'incite', f'cd {ROOT}; echo {shlex.quote(time.strftime("%FT%T") + " " + a.name + " " + a.binary + " " + a.env + " " + a.driver)} >> results/commands.log'])
cmd = wrapped(f'{a.name}', [dcmd, a.model, *dargs]) + f' > results/{a.name}.json 2> results/{a.name}.err'
rc = subprocess.run(leased('incite', cmd)).returncode
for host, log, child in nodes:
    for _ in range(60):
        r = subprocess.run(['ssh', host, f'grep -q "session ended" {ROOT}/{log}'])
        if r.returncode == 0:
            break
        time.sleep(1)
    else:
        print(f'{host}: node did not end its session; left running', file=sys.stderr)
        continue
    subprocess.run(['ssh', host, f'pkill -TERM -f "^{ROOT}/{a.binary} qwen-tp-node "'])
    child.wait(timeout=30)
subprocess.run(['ssh', 'incite', f'cd {ROOT}; tail -3 results/{a.name}.err; cat results/{a.name}.json | head -c 3000'])
sys.exit(rc)
