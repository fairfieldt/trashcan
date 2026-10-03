#!/usr/bin/env python3
"""Summarize completed and pending evidence without inferring final acceptance.

Run from anywhere; JSON is written to stdout. Files may be copied while a run is
in progress, so missing/invalid/partial results stay explicit in the output.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo


def read_json(path):
    if not path.exists():
        return {'status': 'missing', 'file': path.name}, None
    try:
        return {'status': 'present', 'file': path.name}, json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {'status': 'unreadable_or_in_progress', 'file': path.name,
                'error': str(error)}, None


def bench(path):
    entry, data = read_json(path)
    if data is None:
        return entry
    fields = ['pp', 'pp_seconds', 'pp_tok_s', 'tg', 'tg_seconds', 'tg_tok_s']
    if not isinstance(data, dict) or any(
            not isinstance(data.get(key), (int, float)) or not math.isfinite(data[key])
            or data[key] <= 0 for key in fields):
        return {**entry, 'status': 'invalid_or_in_progress'}
    entry.update(status='complete_measurement', metrics=data)
    return entry


def pooled(entries):
    values = [entry['metrics'] for entry in entries if entry['status'] == 'complete_measurement']
    result = {'complete_runs': len(values), 'expected_runs': len(entries),
              'all_present': len(values) == len(entries)}
    if values:
        for phase in ['pp', 'tg']:
            shapes = sorted({value[phase] for value in values})
            result[phase] = {'tokens_per_run': shapes,
                             'pooled_tok_s': sum(value[phase] for value in values)
                             / sum(value[phase + '_seconds'] for value in values),
                             'min_tok_s': min(value[phase + '_tok_s'] for value in values),
                             'max_tok_s': max(value[phase + '_tok_s'] for value in values)}
    return result


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items() if key not in ('id', 'call_id')}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def signature(response):
    return normalize({key: response.get(key) for key in ('status', 'incomplete_details', 'output')})


def prefix(path):
    entry, data = read_json(path)
    if not isinstance(data, dict):
        return entry, None
    entry.update(status='completed_protocol_checks' if data.get('passed') is True else 'incomplete_or_failed',
                 protocol_checks_passed=data.get('passed') is True,
                 repeat_output_exact=data.get('repeat_output_exact'),
                 cold_comparison_exact=data.get('cold_comparison_exact'),
                 batch=data.get('batch'), draft=data.get('draft'), cold=data.get('cold'),
                 request_count=len(data.get('requests', [])))
    entry['requests'] = []
    for row in data.get('requests', []):
        usage = row.get('response', {}).get('usage', {})
        entry['requests'].append({'name': row.get('name'), 'seconds': row.get('seconds'),
                                  'first_delta_seconds': row.get('first_delta_seconds'),
                                  'input_tokens': usage.get('input_tokens'),
                                  'cached_tokens': usage.get('input_tokens_details', {}).get('cached_tokens'),
                                  'output_tokens': usage.get('output_tokens')})
    return entry, data


def compare_prefix(cold, cached):
    if cold is None or cached is None:
        return {'status': 'pending_reports'}
    finished = cold.get('passed') is True and cached.get('passed') is True
    matching_config = (cold.get('cold') is True and cached.get('cold') is False
                       and cold.get('batch') == cached.get('batch')
                       and cold.get('draft') == cached.get('draft'))
    old = {row['name']: row for row in cold.get('requests', [])}
    new = {row['name']: row for row in cached.get('requests', [])}
    comparisons = []
    for name in sorted(old.keys() & new.keys()):
        before, after = old[name], new[name]
        same_request = normalize(before['request']) == normalize(after['request'])
        exact_output = signature(before['response']) == signature(after['response'])
        same_count = (before['response']['usage']['output_tokens']
                      == after['response']['usage']['output_tokens'])
        seconds_ok = all(isinstance(row.get('seconds'), (int, float))
                         and row['seconds'] > 0 for row in [before, after])
        eligible = finished and matching_config and same_request and exact_output and same_count and seconds_ok
        comparisons.append({'name': name, 'same_request': same_request, 'exact_output': exact_output,
                            'same_output_tokens': same_count, 'comparable_completed_latency': eligible,
                            'cold_seconds': before.get('seconds'), 'cached_seconds': after.get('seconds'),
                            'speedup': before['seconds'] / after['seconds'] if eligible else None})
    comparable = [row for row in comparisons if row['comparable_completed_latency']]
    return {'status': 'complete_reports' if finished else 'incomplete_or_failed_reports',
            'matching_configuration': matching_config,
            'missing_in_cached': sorted(old.keys() - new.keys()),
            'missing_in_cold': sorted(new.keys() - old.keys()),
            'all_outputs_exact': finished and matching_config and bool(comparisons) and old.keys() == new.keys()
            and all(row['same_request'] and row['exact_output'] and row['same_output_tokens']
                    for row in comparisons),
            'comparisons': comparisons,
            'comparable_request_count': len(comparable),
            'pooled_comparable_latency_speedup':
                sum(row['cold_seconds'] for row in comparable)
                / sum(row['cached_seconds'] for row in comparable) if comparable else None}


def expected_prefix_names(batch):
    names = ['warmup', 'long_history', 'changed_system', 'shorter_prompt', 'tool_continuation']
    for label in [f'boundary_{n}' for n in [batch, batch + 1, batch + 2]] + [
            'stochastic', 'greedy_again', 'ring_wrap', 'tool']:
        names.extend([label + '_first', label + '_repeat'])
    return sorted(names)


def prefix_inventory(reports, raw):
    issues = []
    for label, entry in reports.items():
        data = raw[label]
        if data is None:
            issues.append({'file': entry['file'], 'issue': entry['status']})
            continue
        if data.get('passed') is not True:
            issues.append({'file': entry['file'], 'issue': 'protocol checks incomplete or failed'})
        if data.get('cold') is not label.startswith('cold') or data.get('draft') is not label.endswith('-mtp'):
            issues.append({'file': entry['file'], 'issue': 'report mode does not match filename'})
        if not isinstance(data.get('batch'), int) or data['batch'] < 2:
            issues.append({'file': entry['file'], 'issue': 'invalid or missing batch size'})
            continue
        expected = expected_prefix_names(data['batch'])
        found = [row.get('name') for row in data.get('requests', [])]
        entry['expected_request_count'] = len(expected)
        entry['missing_requests'] = sorted(set(expected) - set(found))
        entry['duplicate_requests'] = sorted(name for name, count in Counter(found).items() if count > 1)
        if sorted(found) != expected:
            issues.append({'file': entry['file'], 'issue': 'expected default request set is incomplete or different',
                           'expected_count': len(expected), 'actual_count': len(found),
                           'missing': entry['missing_requests'], 'duplicates': entry['duplicate_requests']})
    return {'expected_reports': 4, 'expected_requests_per_report': 19,
            'complete_default_report_set': not issues, 'issue_count': len(issues), 'issues': issues,
            'note': 'Completion covers default protocol inventory only; output equality is reported separately.'}


def chat_report(directory):
    prompts = ['transformer', 'photosynthesis', 'csv']
    names = [f'chat-{version}-{prompt}{suffix}' for version in ['baseline', 'final']
             for prompt in prompts for suffix in ['', '-mtp']]
    metrics_pattern = re.compile(r'prefill: (\d+) tokens, ([0-9.eE+-]+) tok/s; decode: (\d+) tokens, ([0-9.eE+-]+) tok/s')
    mtp_pattern = re.compile(r'MTP: accepted (\d+)/(\d+) proposals in (\d+) rounds; draft ([0-9.eE+-]+) ms/proposal, verify ([0-9.eE+-]+) ms/round')
    result, issues = {}, []
    for name in names:
        path = directory / (name + '.log')
        entry = {'file': path.name, 'status': 'missing'}
        if path.exists():
            text = path.read_text()
            matches = list(metrics_pattern.finditer(text))
            mtp = list(mtp_pattern.finditer(text))
            entry['status'] = 'incomplete_log'
            if matches:
                pp, pp_rate, tg, tg_rate = matches[-1].groups()
                entry.update(status='complete_logged_metrics', input_tokens=int(pp), output_tokens=int(tg),
                             prefill_tok_s=float(pp_rate), decode_tok_s=float(tg_rate))
            if name.endswith('-mtp'):
                if not mtp:
                    entry['status'] = 'incomplete_mtp_log'
                else:
                    accepted, proposed, rounds, draft_ms, verify_ms = mtp[-1].groups()
                    entry['mtp'] = {'accepted': int(accepted), 'proposed': int(proposed), 'rounds': int(rounds),
                                    'draft_ms_per_proposal': float(draft_ms), 'verify_ms_per_round': float(verify_ms)}
        output = directory / (name + '.txt')
        prompt = next(prompt for prompt in prompts if '-' + prompt in name)
        reference = directory / f'chat-baseline-{prompt}.txt'
        entry['output_present'] = output.exists()
        entry['matches_baseline_output'] = (output.read_bytes() == reference.read_bytes()
                                            if output.exists() and reference.exists() else None)
        if output.exists():
            entry['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        if entry['status'] != 'complete_logged_metrics' or not entry['output_present']:
            issues.append({'file': path.name, 'issue': entry['status'], 'output_present': entry['output_present']})
        result[name] = entry
    parity_entry, parity = read_json(directory / 'chat-parity.json')
    parity_entry['expected_runs'] = len(names)
    if isinstance(parity, list):
        recorded = [row.get('run') for row in parity]
        parity_entry.update(recorded_runs=len(recorded), missing_runs=sorted(set(names) - set(recorded)),
                            complete_run_set=sorted(recorded) == sorted(names),
                            all_recorded_outputs_match=all(row.get('matches_baseline') is True for row in parity))
    else:
        parity_entry['complete_run_set'] = False
    if not parity_entry['complete_run_set']:
        issues.append({'file': 'chat-parity.json', 'issue': 'missing or incomplete parity inventory'})
    comparisons = []
    for prompt in prompts:
        for suffix in ['', '-mtp']:
            before, after = [result[f'chat-{version}-{prompt}{suffix}'] for version in ['baseline', 'final']]
            comparable = (all(row['status'] == 'complete_logged_metrics' and row['matches_baseline_output'] is True
                              for row in [before, after])
                          and before['input_tokens'] == after['input_tokens']
                          and before['output_tokens'] == after['output_tokens'])
            comparisons.append({'prompt': prompt, 'mtp': bool(suffix), 'comparable': comparable,
                                'decode_speedup': after['decode_tok_s'] / before['decode_tok_s'] if comparable else None,
                                'prefill_speedup': after['prefill_tok_s'] / before['prefill_tok_s'] if comparable else None})
    return {'runs': result, 'parity': parity_entry, 'comparisons': comparisons, 'issues': issues,
            'rate_precision': 'CLI rates rounded to two decimals; CLI final-forward saving is included in decode.'}


def correctness_json(path):
    entry, data = read_json(path)
    if data is not None:
        entry['metrics'] = data
        entry['status'] = 'reported_pass' if isinstance(data, dict) and data.get('pass') is True else 'incomplete_or_failed'
    return entry


def micro(path):
    rows, errors = [], []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            errors.append(line_number)
    if not rows:
        return {'status': 'empty_or_in_progress', 'file': path.name, 'invalid_lines': errors}
    attention = 'partitions' in rows[0]
    fields = ['tokens', 'dim', 'kv_heads', 'partitions'] if attention else ['tensor', 'batch', 'rows', 'cols', 'kernel']
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    values = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        value = dict(zip(fields, key))
        value['samples'] = len(group)
        value['two_gpu_samples_complete'] = (len(group) == 2 and
                                               (not attention or {row.get('gpu') for row in group} == {0, 1}))
        for field in ['milliseconds', 'original_GB_s', 'repacked_GB_s', 'matrix_GFLOP_s']:
            if all(isinstance(row.get(field), (int, float)) for row in group):
                value['mean_' + field] = statistics.mean(row[field] for row in group)
        if attention:
            value['cpu_reference_checks_passed'] = all(row.get('pass') is True for row in group)
        values.append(value)
    complete = not errors and all(value['two_gpu_samples_complete'] for value in values)
    if attention:
        expected = {(dim, parts) for dim in [256, 512] for parts in {row['partitions'] for row in rows}}
        complete = complete and {(row['dim'], row['partitions']) for row in rows} == expected
    return {'file': path.name, 'status': 'complete_samples' if complete else 'partial_samples',
            'rows': len(rows), 'invalid_lines': errors, 'groups': values}


def clocks(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 7:
            continue
        try:
            rows.append([float(field) for field in fields])
        except ValueError:
            continue
    if not rows:
        return {'file': path.name, 'status': 'empty_or_in_progress'}
    result = {'file': path.name, 'status': 'observed_samples', 'samples': len(rows),
              'scope': 'entire log, including idle/uploads; not a utilization measurement',
              'first_sample_local': datetime.fromtimestamp(rows[0][0], ZoneInfo('America/Los_Angeles')).isoformat(),
              'last_sample_local': datetime.fromtimestamp(rows[-1][0], ZoneInfo('America/Los_Angeles')).isoformat(),
              'cards': []}
    for gpu in range(2):
        offset = 1 + 3 * gpu
        pairs = Counter((int(row[offset]), int(row[offset + 1])) for row in rows)
        run_start, longest = None, 0.
        for row in rows:
            if row[offset] == 850 and row[offset + 1] == 1370:
                if run_start is None:
                    run_start = row[0]
                longest = max(longest, row[0] - run_start)
            else:
                run_start = None
        result['cards'].append({'gpu': gpu,
                                'clock_pair_samples': [{'core_mhz': core, 'memory_mhz': memory, 'samples': count}
                                                       for (core, memory), count in sorted(pairs.items())],
                                'max_temperature_c': max(row[offset + 2] for row in rows),
                                'longest_consecutive_observed_850_1370_seconds': longest})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    directory = args.directory
    measurements = {name: bench(directory / f'{name}.json')
                    for name in ['baseline-1', 'baseline-2', 'selected', 'final-1', 'final-2']}
    baseline = pooled([measurements[name] for name in ['baseline-1', 'baseline-2']])
    final = pooled([measurements[name] for name in ['final-1', 'final-2']])
    bench_comparison = {'baseline': baseline, 'final': final}
    if baseline['all_present'] and final['all_present']:
        for phase in ['pp', 'tg']:
            same_shape = baseline[phase]['tokens_per_run'] == final[phase]['tokens_per_run']
            bench_comparison[phase + '_speedup'] = (final[phase]['pooled_tok_s'] / baseline[phase]['pooled_tok_s']
                                                   if same_shape else None)
    prefix_reports, raw_prefix = {}, {}
    for name in ['cold', 'cached', 'cold-mtp', 'cached-mtp']:
        prefix_reports[name], raw_prefix[name] = prefix(directory / f'prefix-{name}.json')
    long_runs = {label: bench(directory / f'long-{label}.json') for label in ['baseline', 'final']}
    long_comparison = {}
    if all(row['status'] == 'complete_measurement' for row in long_runs.values()):
        before, after = [long_runs[label]['metrics'] for label in ['baseline', 'final']]
        for phase in ['pp', 'tg']:
            long_comparison[phase + '_speedup'] = (after[phase + '_tok_s'] / before[phase + '_tok_s']
                                                   if before[phase] == after[phase] else None)
    result = {
        'scope': 'saved evidence only; benchmark presence does not establish correctness or final acceptance',
        'full_model': measurements, 'full_model_comparison': bench_comparison,
        'prefix_reports': prefix_reports,
        'prefix_inventory': prefix_inventory(prefix_reports, raw_prefix),
        'prefix_comparisons': {name: compare_prefix(raw_prefix['cold' + suffix], raw_prefix['cached' + suffix])
                               for name, suffix in [('ordinary', ''), ('mtp', '-mtp')]},
        'chat': chat_report(directory),
        'long_context': {'runs': long_runs, 'comparison': long_comparison},
        'boundary_check': correctness_json(directory / 'boundary-check.json'),
        'microbenchmarks': [micro(path) for path in sorted(directory.glob('*.jsonl'))],
        'clock_logs': [clocks(path) for path in sorted(directory.glob('*clocks*.txt'))],
        'provenance_files': {path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                        'contents': path.read_text().splitlines()}
                             for path in sorted(directory.glob('*.sha256'))},
        'omitted': ['GPU reference/long-ring acceptance: consult complete validation logs separately'],
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
