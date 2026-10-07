#!/usr/bin/env python3
"""Read saved final TP4 evidence only; never SSH, run models, or manage leases.

Exit 0: all evidence passes; 1: failed audit; 2: incomplete/pending evidence.
JSON goes to stdout, or --out. Plan directories can be overridden explicitly.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

HERE = Path(__file__).resolve().parent
HARDWARE = HERE.parent
HOSTS = ('incite', 'invite')
REGIME = 'dynamic-smc-maximum'
STREAMS = '/dev/tbstream0,/dev/tbstream1,/dev/tbstream2'
ENV = {'TRASHCAN_QWEN_ATTN=tiled', 'TRASHCAN_QWEN_ATTN_SLAB=64',
       'TRASHCAN_QWEN_GDN=prepared', 'TRASHCAN_TP_SUM_DIRECT=1'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fan_evidence(scope, expected_hash):
    result = {}
    for phase in ('fan_before', 'fan_after'):
        snapshot = scope[phase]
        require(snapshot['fancurve_sha256'] == expected_hash, phase + ': fan script hash differs')
        fans = [row for row in snapshot['hwmon'] if row.get('name') == 'applesmc']
        require(len(fans) == 1, phase + ': expected one applesmc fan')
        row = fans[0]
        require(int(row['fan1_manual']) == 1, phase + ': fan is not in manual mode')
        require(all(int(row[key]) > 0 for key in ('fan1_min', 'fan1_max', 'fan1_output', 'fan1_input')),
                phase + ': missing/nonpositive fan readings')
        result[phase] = dict(unix=snapshot['unix'], fan=row,
                            fancurve_sha256=snapshot['fancurve_sha256'],
                            read_errors=snapshot['read_errors'])
    return result


def lease_evidence(events, host, ticket, name, finished):
    acquired = [row for row in events if row['event'] == 'ticket_acquired'
                and row.get('host') == host and row.get('ticket') == ticket and row.get('case') == name]
    waits = [row for row in events if row['event'] == 'command_end'
             and row.get('command') == ['gpuq', '--on', host, 'wait', str(ticket)]
             and row.get('returncode') == 0]
    releases = [row for row in events if row['event'] == 'command_end'
                and row.get('command', [])[:5] == ['gpuq', '--on', host, 'release', str(ticket)]
                and row.get('returncode') == 0 and row['unix'] >= finished
                and f'#{ticket} released.' in row.get('stdout_text', '')]
    require(len(acquired) == 1 and len(waits) == 1 and len(releases) == 1,
            host + ': exact acquired/granted/released ticket proof missing or ambiguous')
    require(acquired[0]['unix'] <= waits[0]['unix'] <= releases[0]['unix'], host + ': ticket ordering invalid')
    return dict(ticket=ticket, acquired_unix=acquired[0]['unix'], granted_unix=waits[0]['unix'],
                release=releases[0])


def check_result(run, case):
    rows = [json.loads(line) for line in (run / 'incite/stdout.log').read_text().splitlines()
            if line.startswith('{')]
    require(len(rows) == 1, 'expected one qwen-check JSON result')
    row = rows[0]
    require(row['pass'] is True and row['restore_exact'] is True and row['column_mismatches'] == [],
            'checkpoint/verification correctness failed')
    require(row['speculative']['identical'] is True and row['speculative']['divergence'] is None,
            'speculative output differs')
    args = shlex.split(case['args'])
    require(row['prompt_tokens'] == int(args[args.index('--tokens') + 1]) == 4300,
            'long-check prompt must cover 4300 tokens')
    require(row['steps'] == int(args[args.index('--steps') + 1]) == 32 and row['verify_rounds'] > 0,
            'long-check generation/verification coverage incomplete')
    return {key: row[key] for key in ('pass', 'prompt_tokens', 'steps', 'restore_exact',
            'verify_rounds', 'column_mismatches', 'max_column_score_error', 'speculative')}


def serving_result(run, case, manifest, archive_hash, client):
    require(manifest['mode'] == case['mode'], 'serving mode differs from plan')
    require(manifest['tested_source_archive']['sha256'] == archive_hash, 'serving archive hash differs')
    for filename, expected in manifest['sources'].items():
        require(digest(run / 'source' / filename) == expected, 'harness source snapshot differs: ' + filename)
    require(manifest['sources']['serve-budget-client.py'] == digest(HERE / 'serve-budget-client.py'),
            'audit client differs from recorded client; review before comparing')
    capture = run / 'incite/client'
    summary = read(capture / 'client-summary.json')
    require(summary['passed'] and summary['safe_to_stop'] and summary['server_identity_verified']
            and summary['last_generation_terminal_verified'], 'serving completion/idle barrier not proven')
    require(manifest['client_returncode'] == 0 and manifest['client_summary'] == summary,
            'client exit or captured summary differs from manifest')
    require(summary['budgets'] == list(range(1, 9)) and summary['completed_requests'] == 37
            and summary['stop_probe'], 'serving budget/EOS coverage incomplete')
    disabled = case.get('no_prefix_cache', False)
    require(summary['prefix_cache_disabled'] == manifest['no_prefix_cache'] == disabled,
            'cache mode differs from plan')
    rows = [json.loads(line) for line in (capture / 'responses.jsonl').read_text().splitlines()]
    expected = []
    for budget in range(1, 9):
        expected += [(budget, stage) for stage in ('initial', 'followup', 'followup-replay')]
        if budget <= 3:
            expected.append((budget, 'edited-prefix'))
        expected.append((budget, 'initial-replay'))
    expected += [(8, 'stop-initial'), (8, 'stop-replay')]
    require([(row['budget'], row['stage']) for row in rows] == expected, 'request corpus incomplete/out of order')
    for row in rows:
        limit = row['budget'] if row['stage'] in ('initial', 'initial-replay') else 8
        require(row['request']['max_output_tokens'] == limit, 'request output budget differs from corpus')
        client.validate_boundary(row['response'], row['request']['max_output_tokens'],
                                 row['stage'].startswith('stop-'))
    coverage = client.cache_coverage(rows)
    client.validate_cache_coverage(coverage, disabled)
    return dict(requests=len(rows), prefix_cache_disabled=disabled, coverage=coverage,
                idle_barrier_unix=summary['idle_barrier_unix'])


def audit_plan(directory, kind, build, archive_hash, fan_hash, client):
    result = dict(path=str(directory), kind=kind, state='pending', cases=[])
    try:
        cases, summary = read(directory / 'plan.json'), read(directory / 'summary.json')
        result['controller_state'] = summary.get('state', 'in progress')
        result['active_tickets'] = summary['active_tickets']
        require(not summary['dry_run'], 'dry-run records are not hardware evidence')
        require(len(cases) == (1 if kind == 'long-check' else 4), 'unexpected final plan case count')
        if summary.get('state') != 'completed':
            result['reason'] = summary.get('error', 'plan has not completed')
            result['cases'] = [{key: row[key] for key in ('name', 'state', 'harness_returncode') if key in row}
                               for row in summary['cases']]
            return result
        records = {row['name']: row for row in summary['cases']}
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        for case in cases:
            evidence = dict(name=case['name'], state='pending')
            result['cases'].append(evidence)
            record = records.get(case['name'], {})
            if record.get('state') != 'completed':
                evidence['reason'] = record.get('state', 'not started')
                if record.get('harness_returncode') not in (None, 0):
                    evidence['state'] = 'failed'
                continue
            try:
                run = directory / 'runs' / case['name']
                manifest = read(run / 'manifest.json')
                require(record['settings'] == case and record['harness_returncode'] == 0,
                        'controller settings or harness returncode differs')
                require(manifest['state'] == 'completed' and not manifest['dry_run'], 'harness incomplete')
                require(case['binary'] == 'trashcan-final' and Path(manifest['binary']).name == 'trashcan-final',
                        'case did not use final binary')
                require(case['fan_regime'] == manifest['fan_policy']['regime'] == REGIME, 'fan policy differs')
                require(set(shlex.split(case['env'])) == ENV, 'final optimization settings differ')
                if 'expected_binary_sha256' in case:
                    require(case['expected_binary_sha256'] == build['binary_sha256'], 'planned binary hash differs')
                archive = (Path(summary['source_plan']).parent / case['source_archive']).resolve()
                require(digest(archive) == archive_hash, 'planned source archive hash differs')
                evidence['hosts'] = {}
                for host in HOSTS:
                    settings = manifest['hosts'][host]
                    scope = read(run / host / 'scope.json')
                    require(scope['binary_sha256'] == build['binary_sha256'], host + ': binary hash differs')
                    require(scope['finished_unix'] >= scope['started_unix'] and scope['returncode'] in
                            ((0,) if host == 'incite' and kind == 'long-check' else (0, -15, 143)),
                            host + ': process completion failed')
                    require(scope['governors_before'] and scope['governors_before'] == scope['governors_after']
                            and not scope['restore_errors'], host + ': governors not restored')
                    require(scope['governors_during'] and set(scope['governors_during'].values()) == {'performance'}
                            and scope['cpu_latency_us'] == 0, host + ': CPU run policy differs')
                    require(scope['command'] == settings['command'] and ENV <= set(settings['environment']),
                            host + ': command/environment provenance differs')
                    command = scope['command']
                    mtp = case.get('mtp', False) if kind == 'long-check' else case['mode'] != 'plain'
                    require(('--mtp' in command) == mtp, host + ': MTP mode differs')
                    if host == 'incite' and kind == 'serving':
                        require(('--no-prefix-cache' in command) == case.get('no_prefix_cache', False),
                                'server prefix-cache option differs')
                    stream_flag = '--tp-stream' if host == 'incite' else '--stream'
                    require(command[command.index(stream_flag) + 1] == STREAMS, host + ': stream topology differs')
                    require(settings['ticket'] == record['tickets'][host] and settings['retrieval_returncode'] == 0
                            and not settings.get('scope_verification_error'), host + ': host evidence incomplete')
                    evidence['hosts'][host] = dict(binary_sha256=scope['binary_sha256'], command=command,
                        fan=fan_evidence(scope, fan_hash), lease=lease_evidence(events, host, settings['ticket'],
                            case['name'], manifest['finished_unix']))
                evidence['correctness'] = (check_result(run, case) if kind == 'long-check' else
                    serving_result(run, case, manifest, archive_hash, client))
                evidence['state'] = 'passed'
            except FileNotFoundError as error:
                evidence.update(state='pending', reason=str(error))
            except (AssertionError, KeyError, ValueError, TypeError, OSError) as error:
                evidence.update(state='failed', reason=str(error) or type(error).__name__)
        if any(row['state'] == 'failed' for row in result['cases']):
            result['state'] = 'failed'
        elif summary['state'] == 'completed' and all(row['state'] == 'passed' for row in result['cases']):
            require(not summary['active_tickets'], 'completed plan still records active tickets')
            if kind == 'serving':
                require([case['mode'] for case in cases] == ['plain', 'mtp', 'adaptive', 'plain']
                        and [case.get('no_prefix_cache', False) for case in cases] == [False, False, False, True],
                        'final serving modes differ')
                reference = directory / 'runs' / cases[0]['name'] / 'incite/client'
                result['comparisons'] = {case['name']: client.compare(reference,
                    directory / 'runs' / case['name'] / 'incite/client') for case in cases[1:]}
            result['state'] = 'passed'
    except FileNotFoundError as error:
        result['reason'] = str(error)
    except (AssertionError, KeyError, ValueError, TypeError, OSError) as error:
        result.update(state='failed', reason=str(error) or type(error).__name__)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--long-plan', type=Path, default=HARDWARE / 'plans/long-check-maxfan-plan')
    parser.add_argument('--serve-plan', type=Path, default=HARDWARE / 'plans/serve-final-v2')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = dict(state='pending', notes=[
        'Offline artifact audit only; ticket records do not query current queue state.',
        'Fan snapshots prove manual mode and script identity at endpoints, not continuous RPM/clock behavior.',
        'fan1_max is dynamic; raw max/output/input are retained without requiring exact equality.',
        'Correctness evidence does not establish performance or expose internal server KV/token IDs.'])
    try:
        build = read(HARDWARE / 'builds/final/build.json')
        archive_hash = digest(HARDWARE / 'builds/final/source.tar.gz')
        require(archive_hash == build['source_archive_sha256'], 'final build source archive differs')
        require(digest(Path(build['binary_path'])) == build['binary_sha256'], 'local final binary differs')
        fan_hashes = set()
        for host in HOSTS:
            hashes = set(re.findall(r'^([0-9a-f]{64})  /usr/local/sbin/fancurve.sh$',
                (HARDWARE / 'fans' / (host + '-deploy.log')).read_text(), re.M))
            require(len(hashes) == 1, host + ': original fan deployment hash missing or ambiguous')
            fan_hashes.update(hashes)
        require(len(fan_hashes) == 1, 'original fan deployment hashes differ between hosts')
        fan_hash = fan_hashes.pop()
        spec = importlib.util.spec_from_file_location('final_audit_client', HERE / 'serve-budget-client.py')
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        report['build'] = dict(binary_sha256=build['binary_sha256'], source_archive_sha256=archive_hash,
                               base_commit=build['base_commit'], fan_script_sha256=fan_hash)
        report['plans'] = [audit_plan(args.long_plan, 'long-check', build, archive_hash, fan_hash, client),
                           audit_plan(args.serve_plan, 'serving', build, archive_hash, fan_hash, client)]
        states = [plan['state'] for plan in report['plans']]
        report['state'] = 'failed' if 'failed' in states else 'passed' if states == ['passed', 'passed'] else 'pending'
    except FileNotFoundError as error:
        report['reason'] = str(error)
    except (AssertionError, KeyError, ValueError, TypeError, OSError) as error:
        report.update(state='failed', reason=str(error) or type(error).__name__)
    encoded = json.dumps(report, indent=2) + '\n'
    if args.out:
        args.out.write_text(encoded)
        print(report['state'] + ': ' + str(args.out))
    else:
        print(encoded, end='')
    return {'passed': 0, 'failed': 1, 'pending': 2}[report['state']]


if __name__ == '__main__':
    sys.exit(main())
