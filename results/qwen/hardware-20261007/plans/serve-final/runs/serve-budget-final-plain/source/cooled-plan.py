#!/usr/bin/env python3
"""Cool both TP4 hosts, acquire paired leases, run one short case, and release.

Input is a JSON list of objects with name, binary, and optional command, args,
env, mtp, minutes (4..6), trace, profile, timeout, and fan_regime. No remote operation occurs
with --dry-run. An incomplete GPU run retains its tickets and stops the plan.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HOSTS = ('incite', 'invite')
STREAMS = '/dev/tbstream0,/dev/tbstream1,/dev/tbstream2'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
       '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
READ_TEMPERATURES = '''import json, pathlib, re, time
rows = []
for card in sorted(pathlib.Path('/sys/class/drm').glob('card*')):
    if not re.fullmatch(r'card[0-9]+', card.name): continue
    dev = card / 'device'
    if not (dev / 'vendor').exists() or (dev / 'vendor').read_text().strip() != '0x1002': continue
    sensors = sorted(dev.glob('hwmon/hwmon*/temp1_input'))
    if len(sensors) != 1: raise SystemExit('expected one GPU temperature sensor at ' + str(dev))
    rows.append(dict(card=card.name, pci_device=str(dev.resolve()), sensor=str(sensors[0]),
                     celsius=int(sensors[0].read_text().strip()) / 1000))
if len(rows) != 2 or len({row['pci_device'] for row in rows}) != 2:
    raise SystemExit('expected exactly two distinct AMD GPUs; got ' + repr(rows))
if any(not 0 < row['celsius'] < 125 for row in rows):
    raise SystemExit('GPU temperature outside plausible range')
print(json.dumps(dict(unix=time.time(), gpus=rows)))
'''


def validate_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError('plan must be a nonempty JSON list')
    names = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('each plan entry must be an object')
        name = case.get('name', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
            raise ValueError('run names must be unique and contain letters, digits, underscores or hyphens')
        names.add(name)
        if not isinstance(case.get('binary'), str) or not case['binary']:
            raise ValueError(name + ': binary is required')
        if case.get('command', 'qwen-bench') not in ['qwen-bench', 'qwen-check', 'qwen-validate']:
            raise ValueError(name + ': unsupported command')
        if not 4 <= case.get('minutes', 6) <= 6:
            raise ValueError(name + ': lease estimate must be 4..6 minutes')
        if not 0 < case.get('timeout', 180) <= 240:
            raise ValueError(name + ': driver timeout must be 1..240 seconds')
        for key in ('args', 'env'):
            if key in case and not isinstance(case[key], str):
                raise ValueError(name + ': ' + key + ' must be a shell-quoted string')
            shlex.split(case.get(key, ''))
        for assignment in shlex.split(case.get('env', '')):
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', assignment, re.S):
                raise ValueError(name + ': env must contain NAME=value assignments')
        for key in ('mtp', 'trace', 'profile'):
            if key in case and not isinstance(case[key], bool):
                raise ValueError(name + ': ' + key + ' must be boolean')
        if 'fan_regime' in case and (not isinstance(case['fan_regime'], str)
                                     or not case['fan_regime'].strip()):
            raise ValueError(name + ': fan_regime must be a nonempty string')


def ticket_number(output):
    matches = set(re.findall(r'\bTicket\s*#(\d+)\b', output, re.I))
    if len(matches) != 1:
        raise ValueError('expected one unambiguous Ticket #ID in gpuq request output')
    return int(matches.pop())


def release_evidence(run_dir):
    """Only a finished harness plus finished child scope records proves safe release."""
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    if manifest.get('state') != 'completed':
        return False, 'harness did not complete'
    for host in HOSTS:
        scope = json.loads((run_dir / host / 'scope.json').read_text())
        if not scope.get('finished_unix') or not isinstance(scope.get('returncode'), int):
            return False, host + ' command has no recorded exit'
        if (not scope.get('governors_before') or scope.get('restore_errors')
                or scope.get('governors_after') != scope.get('governors_before')):
            return False, host + ' governor restoration not confirmed'
    return True, 'harness and both GPU commands finished; governors restored'


def harness_command(case, tickets, root, out):
    command = [sys.executable, str(Path(__file__).with_name('run-tp4.py')),
               '--incite', str(tickets['incite']), '--invite', str(tickets['invite']),
               '--root', root, '--streams', STREAMS, '--out', str(out / 'runs'),
               '--binary', case['binary'], '--name', case['name'],
               '--command', case.get('command', 'qwen-bench'),
               '--args', case.get('args', '--pp 2048 --tg 64 --reps 1'),
               '--env', case.get('env', ''), '--timeout', str(case.get('timeout', 180))]
    for flag in ('mtp', 'trace', 'profile'):
        if case.get(flag):
            command.append('--' + flag)
    if 'fan_regime' in case:
        command += ['--fan-regime', case['fan_regime']]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('--max-c', type=float, default=72)
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--max-plan-minutes', type=float, default=30,
                   help='stop between cases or cooling polls; never terminate GPU work')
    p.add_argument('--owner', default='Qwen TP4 cooled hardware validation')
    p.add_argument('--root', default='/home/fairfieldt/trashcan-exp/tp4-offline-20261007')
    p.add_argument('--out', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cases = json.loads(a.plan.read_text())
    try:
        validate_cases(cases)
    except ValueError as error:
        p.error(str(error))
    if not 0 < a.poll_seconds <= 10 or not 0 < a.max_c <= 72 or a.max_plan_minutes <= 0:
        p.error('poll must be 0..10 seconds, max temperature 0..72 C, and duration positive')
    if not a.root.startswith('/home/fairfieldt/trashcan-exp/') or '..' in Path(a.root).parts:
        p.error('root must be an absolute workspace below /home/fairfieldt/trashcan-exp/')
    out = (a.out or Path(__file__).resolve().parents[1] / 'plans' / a.plan.stem).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_text(json.dumps(cases, indent=2) + '\n')
    summary = dict(started_unix=time.time(), dry_run=a.dry_run, max_c=a.max_c,
                   poll_seconds=a.poll_seconds, root=a.root, streams=STREAMS,
                   source_plan=str(a.plan.resolve()), cases=[], active_tickets={})
    deadline = time.monotonic() + a.max_plan_minutes * 60
    sequence = 0

    def save():
        temporary = out / 'summary.tmp'
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        temporary.replace(out / 'summary.json')

    def event(kind, **values):
        record = dict(unix=time.time(), event=kind, **values)
        with (out / 'events.jsonl').open('a') as log:
            log.write(json.dumps(record) + '\n')
        save()

    def run(command, label, timeout=None):
        nonlocal sequence
        sequence += 1
        prefix = out / f'{sequence:04d}-{label}'
        record = dict(command=command, exact_shell_command=shlex.join(command),
                      stdout=str(prefix) + '.stdout.log', stderr=str(prefix) + '.stderr.log')
        event('command_start', **record)
        with open(record['stdout'], 'wb') as stdout, open(record['stderr'], 'wb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        event('local_process', pid=child.pid, label=label)
        limit = time.monotonic() + timeout if timeout else None
        try:
            while child.poll() is None:
                if limit and time.monotonic() >= limit:
                    # All bounded commands here are read-only thermal/rules queries.
                    # Lease mutations and GPU harnesses are never killed on a timer.
                    child.terminate()
                    child.wait(timeout=5)
                    raise TimeoutError(label + ' timed out')
                time.sleep(min(a.poll_seconds, 1))
        except BaseException:
            event('command_interrupted', label=label, pid=child.pid,
                  returncode=child.poll(), process_left_untouched=child.poll() is None)
            raise
        record.update(returncode=child.returncode,
                      stdout_text=Path(record['stdout']).read_text(errors='replace'),
                      stderr_text=Path(record['stderr']).read_text(errors='replace'))
        event('command_end', **record)
        return record

    def checked(command, label, timeout=None):
        result = run(command, label, timeout)
        if result['returncode']:
            raise RuntimeError(label + ' failed: ' + result['stdout_text'] + result['stderr_text'])
        return result

    def temperatures(case_name, stage):
        readings = {}
        for host in HOSTS:
            result = checked(SSH + [host, shlex.join(['python3', '-c', READ_TEMPERATURES])],
                             host + '-temperatures', timeout=20)
            reading = json.loads(result['stdout_text'])
            if len(reading.get('gpus', [])) != 2:
                raise ValueError(host + ': missing GPU temperature reading')
            readings[host] = reading
        maximum = max(row['celsius'] for data in readings.values() for row in data['gpus'])
        event('thermal_sample', case=case_name, stage=stage, readings=readings, maximum_c=maximum)
        return maximum

    def release(tickets, reason):
        failures = []
        for host, ticket in list(tickets.items()):
            result = run(['gpuq', '--on', host, 'release', str(ticket), '-r', reason], host + '-release')
            if result['returncode']:
                failures.append(host)
            else:
                del tickets[host]
                summary['active_tickets'] = dict(tickets)
                save()
        if failures:
            raise RuntimeError('lease release failed for ' + ', '.join(failures))

    save()
    if a.dry_run:
        summary['state'] = 'dry-run; no remote operations'
        summary['dry_run_commands'] = [harness_command(case, dict(incite=1, invite=2), a.root, out)
                                       for case in cases]
        save()
        print(out / 'summary.json', flush=True)
        return 0

    tickets = {}
    harness_started = False
    code = 1
    try:
        for host in HOSTS:
            checked(['gpuq', '--on', host, 'rules'], host + '-rules', timeout=20)
        for case in cases:
            case_record = dict(name=case['name'], settings=case, state='cooling')
            summary['cases'].append(case_record)
            while True:
                if time.monotonic() >= deadline:
                    summary['state'] = 'time budget reached between runs; no leases held'
                    code = 0
                    return code
                maximum = temperatures(case['name'], 'before_request')
                if maximum > a.max_c:
                    print(f"{case['name']}: cooling, hottest GPU {maximum:.1f} C (target <= {a.max_c:g} C)", flush=True)
                    time.sleep(a.poll_seconds)
                    continue
                for host in HOSTS:
                    response = checked(['gpuq', '--on', host, 'request', case['name'], '--est',
                                        str(case.get('minutes', 6)), '--owner', a.owner], host + '-request')
                    tickets[host] = ticket_number(response['stdout_text'] + response['stderr_text'])
                    summary['active_tickets'] = dict(tickets)
                    event('ticket_acquired', host=host, ticket=tickets[host], case=case['name'])
                for host in HOSTS:
                    checked(['gpuq', '--on', host, 'wait', str(tickets[host])], host + '-wait')
                if time.monotonic() >= deadline:
                    release(tickets, case['name'] + ': plan time budget reached; no GPU command started')
                    summary['state'] = 'time budget reached between runs; no leases held'
                    code = 0
                    return code
                # Another owner may have used a GPU while we waited in the queue.
                if temperatures(case['name'], 'after_grant') <= a.max_c:
                    break
                release(tickets, case['name'] + ': GPUs warmed while queued; no GPU command started')
                print(case['name'] + ': released unused leases; cooling again', flush=True)
                time.sleep(a.poll_seconds)
            case_record.update(state='running', tickets=dict(tickets))
            print(case['name'] + ': running ' + json.dumps(tickets), flush=True)
            harness_started = True
            result = run(harness_command(case, tickets, a.root, out), case['name'] + '-harness')
            safe, reason = release_evidence(out / 'runs' / case['name'])
            case_record.update(harness_returncode=result['returncode'], release_evidence=reason)
            if not safe:
                raise RuntimeError(case['name'] + ': ' + reason + '; retaining both tickets')
            harness_started = False
            release(tickets, case['name'] + ': finished; outputs and clocks saved')
            case_record['state'] = 'completed' if result['returncode'] == 0 else 'failed safely'
            event('case_finished', case=case['name'], returncode=result['returncode'])
            if result['returncode']:
                raise RuntimeError(case['name'] + ': harness failed; jobs ended and leases released')
        summary['state'] = 'completed'
        code = 0
    except (Exception, KeyboardInterrupt) as error:
        summary.update(state='stopped', error=repr(error))
        if tickets and not harness_started:
            try:
                release(tickets, 'plan stopped before GPU work; releasing unused leases')
            except Exception as release_error:
                summary['release_error'] = repr(release_error)
        if tickets:
            summary['state'] = 'stopped; tickets retained for inspection'
            print('ERROR: ' + str(error) + '; tickets retained: ' + json.dumps(tickets), file=sys.stderr, flush=True)
        else:
            print('ERROR: ' + str(error), file=sys.stderr, flush=True)
    finally:
        summary.update(active_tickets=dict(tickets), finished_unix=time.time())
        save()
        print(out / 'summary.json', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
