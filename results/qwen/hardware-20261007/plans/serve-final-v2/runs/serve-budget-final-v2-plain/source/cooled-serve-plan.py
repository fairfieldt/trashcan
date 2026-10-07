#!/usr/bin/env python3
"""Run plain/fixed/adaptive serving cases using the existing cooled lease lifecycle.

Only server-case validation and harness-command construction differ from
cooled-plan.py. That module still owns cooling, rules, request/wait/release,
finished-scope/governor checks, and retaining tickets on uncertain GPU state.
No existing helper is edited. --dry-run does not call SSH or gpuq.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('serving_cool_lifecycle', HERE / 'cooled-plan.py')
COOL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(COOL)
VALIDATE_COMMON = COOL.validate_cases


def validate_cases(cases, plan_dir):
    VALIDATE_COMMON(cases)
    allowed = {'name', 'binary', 'mode', 'env', 'minutes', 'timeout', 'budgets', 'draft_tokens', 'stop_probe', 'source_archive', 'no_prefix_cache', 'fan_regime'}
    cached_cases = cases[:-1] if cases[-1].get('no_prefix_cache') else cases
    if ([case.get('mode') for case in cached_cases] not in (['plain', 'mtp'], ['plain', 'mtp', 'adaptive'])
            or any(case.get('no_prefix_cache') for case in cached_cases)
            or (cases[-1].get('no_prefix_cache') and cases[-1].get('mode') != 'plain')):
        raise ValueError('plan must run plain, mtp, optionally adaptive, then optionally plain with no_prefix_cache')
    reference = cases[0]
    for case in cases:
        if set(case) - allowed:
            raise ValueError(case['name'] + ': unsupported fields ' + repr(sorted(set(case) - allowed)))
        if not isinstance(case.get('stop_probe', True), bool) or not 1 <= case.get('draft_tokens', 2) <= 7:
            raise ValueError(case['name'] + ': invalid stop probe or draft depth')
        if not isinstance(case.get('no_prefix_cache', False), bool):
            raise ValueError(case['name'] + ': no_prefix_cache must be boolean')
        try:
            values = [int(v) for v in case.get('budgets', '1,2,3,4,5,6,7,8').split(',')]
            assert values and len(set(values)) == len(values) and all(1 <= v <= 8 for v in values)
        except (AttributeError, ValueError, AssertionError):
            raise ValueError(case['name'] + ': budgets must be unique integers 1..8')
        if not case.get('source_archive') or not (plan_dir / case['source_archive']).is_file():
            raise ValueError(case['name'] + ': tested source archive is required and must exist')
        if any(v.split('=', 1)[0] == 'TRASHCAN_QWEN_ADAPTIVE_MTP' for v in shlex.split(case.get('env', ''))):
            raise ValueError(case['name'] + ': mode controls adaptive MTP')
        for key, default in [('binary', None), ('env', ''), ('budgets', '1,2,3,4,5,6,7,8'),
                             ('draft_tokens', 2), ('stop_probe', True), ('source_archive', None), ('fan_regime', None)]:
            if case.get(key, default) != reference.get(key, default):
                raise ValueError('all modes must use the same ' + key)


def harness_command(case, tickets, root, out, plan_dir, max_c, reference_name):
    command = [sys.executable, str(HERE / 'run-serve-budget.py'),
               '--incite', str(tickets['incite']), '--invite', str(tickets['invite']),
               '--root', root, '--streams', COOL.STREAMS, '--out', str(out / 'runs'),
               '--name', case['name'], '--binary', case['binary'], '--mode', case['mode'],
               '--env', case.get('env', ''), '--timeout', str(case.get('timeout', 240)),
               '--max-c', str(max_c), '--budgets', case.get('budgets', '1,2,3,4,5,6,7,8'),
               '--draft-tokens', str(case.get('draft_tokens', 2)),
               '--source-archive', str((plan_dir / case['source_archive']).resolve())]
    if case.get('stop_probe', True):
        command.append('--stop-probe')
    if case.get('no_prefix_cache'):
        command.append('--no-prefix-cache')
    if 'fan_regime' in case:
        command += ['--fan-regime', case['fan_regime']]
    if case['name'] != reference_name:
        command += ['--reference', str(out / 'runs' / reference_name / 'incite/client')]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', nargs='?', type=Path, default=HERE / 'serve-budget-plan.json')
    p.add_argument('--max-c', type=float, default=65)
    p.add_argument('--max-plan-minutes', type=float, default=30)
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--root', default='/home/fairfieldt/trashcan-exp/tp4-offline-20261007')
    p.add_argument('--owner', default='Qwen TP4 serving budget/cache regression')
    p.add_argument('--out', type=Path, default=HERE.parent / 'plans/serve-budget-cold')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not 0 < a.max_c <= 65:
        p.error('serving plan requires all four GPUs at or below 65 C before acquiring leases')
    cases = json.loads(a.plan.read_text())
    plan_dir = a.plan.resolve().parent
    try:
        validate_cases(cases, plan_dir)
    except (ValueError, TypeError) as error:
        p.error(str(error))
    # Adapt only these public functions in a private imported module instance.
    # The established lifecycle and release-evidence implementation stay intact.
    COOL.validate_cases = lambda selected: validate_cases(selected, plan_dir)
    COOL.harness_command = lambda case, tickets, root, out: harness_command(
        case, tickets, root, out, plan_dir, a.max_c, cases[0]['name'])
    forwarded = [str(HERE / 'cooled-plan.py'), str(a.plan), '--max-c', str(a.max_c),
                 '--max-plan-minutes', str(a.max_plan_minutes), '--poll-seconds', str(a.poll_seconds),
                 '--root', a.root, '--owner', a.owner, '--out', str(a.out)]
    if a.dry_run:
        forwarded.append('--dry-run')
    previous = sys.argv
    try:
        sys.argv = forwarded
        return COOL.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    sys.exit(main())
