#!/usr/bin/env python3
"""Offline cleanup audit. Exit 0=passed, 1=failed, 2=pending; never runs SSH/GPU.

Expects build.json with binary_path, binary_sha256, source_archive_sha256 and
source.tar.gz here, plus completed plans/long-check and plans/serve captures.
Uses the saved hardware audit's common evidence checks and response comparator.
"""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parent
HARDWARE = ROOT.parent / 'hardware-20261007'
SCRIPTS = HARDWARE / 'scripts'
BASE_ENV = {'TRASHCAN_QWEN_ATTN=tiled'}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


A = module('cleanup_evidence_helpers', SCRIPTS / 'audit-final-evidence.py')
CLIENT = module('cleanup_response_comparison', SCRIPTS / 'serve-budget-client.py')


def clocks(path, scope):
    rows = []
    for line in path.read_text().splitlines()[1:]:
        fields = line.split()
        A.require(len(fields) == 7, 'invalid clock record: ' + str(path))
        values = list(map(float, fields))
        A.require(all(math.isfinite(value) for value in values), 'nonfinite clock record')
        if scope['started_unix'] <= values[0] <= scope['finished_unix']:
            rows.append(values)
    A.require(rows, 'no clock samples during scope: ' + str(path))
    A.require(all(a[0] <= b[0] for a, b in zip(rows, rows[1:])), 'nonmonotonic clocks')
    counts = Counter(','.join(str(int(row[i])) for i in (1, 2, 4, 5)) for row in rows)
    return dict(path=str(path), samples=len(rows), clock_counts=dict(counts),
                max_c=max(row[i] for row in rows for i in (3, 6)),
                window='whole command scope, including load/idle; not a throughput qualification')


def audit_plan(kind, build, old_plan):
    directory = ROOT / 'plans' / ('long-check' if kind == 'check' else 'serve')
    output = dict(path=str(directory), state='pending', runs=[])
    try:
        cases = A.read(directory / 'plan.json')
        summary = A.read(directory / 'summary.json')
        output['controller_state'] = summary.get('state', 'in progress')
        output['active_tickets'] = summary['active_tickets']
        if summary.get('state') != 'completed':
            output['reason'] = summary.get('error', 'controller has not completed')
            return output
        A.require(not summary['dry_run'] and not summary['active_tickets'], 'not a completed, released hardware plan')
        expected = A.read(ROOT / ('long-check-plan.json' if kind == 'check' else 'serve-plan.json'))
        A.require(cases == expected, 'saved cases differ from cleanup validation plan')
        records = {row['name']: row for row in summary['cases']}
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        for index, case in enumerate(cases):
            run = directory / 'runs' / case['name']
            record = records[case['name']]
            manifest = A.read(run / 'manifest.json')
            A.require(record['state'] == manifest['state'] == 'completed' and record['harness_returncode'] == 0,
                      case['name'] + ': run incomplete or failed')
            A.require(record['settings'] == case and not manifest['dry_run'], 'case settings/dry-run mismatch')
            A.require(case['binary'] == 'trashcan-clean' and Path(manifest['binary']).name == 'trashcan-clean',
                      'run did not use cleanup binary')
            A.require(set(shlex.split(case['env'])) == BASE_ENV, 'unexpected cleanup source flags')
            A.require(manifest['fan_policy']['regime'] == case['fan_regime'] == 'dynamic-smc-maximum',
                      'fan policy differs')
            source = (Path(summary['source_plan']).parent / case['source_archive']).resolve()
            A.require(A.digest(source) == build['source_archive_sha256'], 'source archive differs')
            evidence = dict(name=case['name'], hosts={})
            output['runs'].append(evidence)
            for host in A.HOSTS:
                settings = manifest['hosts'][host]
                scope = A.read(run / host / 'scope.json')
                A.require(scope['binary_sha256'] == build['binary_sha256'], host + ': wrong binary hash')
                A.require(scope['finished_unix'] >= scope['started_unix'] and scope['returncode'] in
                          ((0,) if kind == 'check' and host == 'incite' else (0, -15, 143)),
                          host + ': command not successfully finished')
                A.require(scope['governors_before'] and scope['governors_after'] == scope['governors_before']
                          and not scope['restore_errors'], host + ': governors not restored')
                A.require(scope['governors_during'] and set(scope['governors_during'].values()) == {'performance'}
                          and scope['cpu_latency_us'] == 0, host + ': scoped CPU settings differ')
                A.require(settings['command'] == scope['command'] and settings['retrieval_returncode'] == 0
                          and not settings.get('scope_verification_error'), host + ': command/log provenance failed')
                env = BASE_ENV if kind == 'check' else BASE_ENV | {
                    'TRASHCAN_QWEN_ADAPTIVE_MTP=' + ('1' if case['mode'] == 'adaptive' else '0')}
                A.require(set(settings['environment']) == env, host + ': unexpected or retired runtime flags')
                command = scope['command']
                flag = '--tp-stream' if host == 'incite' else '--stream'
                A.require(command[command.index(flag) + 1] == A.STREAMS, host + ': stream topology differs')
                mtp = case['mtp'] if kind == 'check' else case['mode'] != 'plain'
                A.require(('--mtp' in command) == mtp, host + ': MTP mode differs')
                if kind == 'serve' and host == 'incite':
                    A.require(('--no-prefix-cache' in command) == case.get('no_prefix_cache', False),
                              'prefix-cache mode differs')
                A.require(settings['ticket'] == record['tickets'][host], host + ': ticket differs')
                fan_hash = scope['fan_before']['fancurve_sha256']
                evidence['hosts'][host] = dict(binary_sha256=scope['binary_sha256'], command=command,
                    environment=settings['environment'], clocks=clocks(run / host / 'clocks.txt', scope),
                    fan=A.fan_evidence(scope, fan_hash),
                    lease=A.lease_evidence(events, host, settings['ticket'], case['name'], manifest['finished_unix']))
            if kind == 'check':
                evidence['correctness'] = A.check_result(run, case)
                old_case = A.read(HARDWARE / 'plans/long-check-maxfan-plan/plan.json')[0]
                old_result = A.check_result(HARDWARE / 'plans/long-check-maxfan-plan/runs' / old_case['name'], old_case)
                evidence['prior_correctness_passed'] = old_result['pass']
                def text_result(path):
                    return next(json.loads(line)['text'] for line in path.read_text().splitlines() if line.startswith('{'))
                A.require(text_result(run / 'incite/stdout.log') == text_result(
                    HARDWARE / 'plans/long-check-maxfan-plan/runs' / old_case['name'] / 'incite/stdout.log'),
                    'long-check generated text differs from prior correct run')
                evidence['prior_generated_text_equal'] = True
            else:
                evidence['correctness'] = A.serving_result(run, case, manifest,
                                                          build['source_archive_sha256'], CLIENT)
                old_case = A.read(old_plan / 'plan.json')[index]
                A.require(old_case['mode'] == case['mode'] and old_case.get('no_prefix_cache', False) ==
                          case.get('no_prefix_cache', False), 'previous serving mode differs')
                evidence['comparison_to_prior'] = CLIENT.compare(
                    old_plan / 'runs' / old_case['name'] / 'incite/client', run / 'incite/client')
                if index:
                    evidence['comparison_to_clean_plain'] = CLIENT.compare(
                        directory / 'runs' / cases[0]['name'] / 'incite/client', run / 'incite/client')
            evidence['state'] = 'passed'
        output['state'] = 'passed'
    except FileNotFoundError as error:
        output['reason'] = str(error)
    except (AssertionError, ValueError, KeyError, TypeError, OSError) as error:
        output.update(state='failed', reason=str(error) or type(error).__name__)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-serve', type=Path, default=HARDWARE / 'plans/serve-final-v2')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = dict(state='pending', notes=[
        'Correctness/provenance audit only; no new performance claim.',
        'Cached modes compare exact output/status/usage including cached counts; uncached captures require zero cached tokens.',
        'Only cached_tokens is omitted when comparing an uncached capture; no other usage fields are ignored.',
        'Fan script identity is reported from snapshots and must stay unchanged within each command; deployment acceptance is separate.'])
    try:
        build = A.read(ROOT / 'build.json')
        A.require(A.digest(ROOT / 'source.tar.gz') == build['source_archive_sha256'], 'cleanup archive hash differs')
        A.require(A.digest(Path(build['binary_path'])) == build['binary_sha256'], 'local cleanup binary hash differs')
        report['build'] = {key: build[key] for key in ('binary_path', 'binary_sha256', 'source_archive_sha256')}
        report['plans'] = [audit_plan('check', build, args.prior_serve), audit_plan('serve', build, args.prior_serve)]
        states = [plan['state'] for plan in report['plans']]
        report['state'] = 'failed' if 'failed' in states else 'passed' if states == ['passed', 'passed'] else 'pending'
    except FileNotFoundError as error:
        report['reason'] = str(error)
    except (AssertionError, ValueError, KeyError, TypeError, OSError) as error:
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
