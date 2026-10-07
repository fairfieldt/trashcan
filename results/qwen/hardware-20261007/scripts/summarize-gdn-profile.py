#!/usr/bin/env python3
"""Compare GDN components on every TP4 rank; profiled wall throughput is not reported.

Usage: summarize-gdn-profile.py [plans/profile-plan]
Reads the saved plan and per-host run artifacts produced by cooled-plan.py.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def segments(labels, timestamps):
    if len(labels) != len(timestamps):
        raise ValueError('profile label/timestamp length mismatch')
    result, current, opened = [], None, {}
    for label, stamp in zip(labels, timestamps):
        if not math.isfinite(stamp):
            raise ValueError('nonfinite GPU timestamp')
        if label == 'begin':
            if current is not None:
                raise ValueError('nested segment begin')
            current = dict(begin=stamp, spans=[])
        elif label == 'end':
            if current is None or opened:
                raise ValueError('segment ended with missing begin or open dispatch')
            current['end'] = stamp
            result.append(current)
            current = None
        elif current is not None:
            if label in opened:
                start = opened.pop(label)
                if stamp < start:
                    raise ValueError('negative dispatch span')
                current['spans'].append((label, start, stamp))
            else:
                opened[label] = stamp
    if current is not None or opened:
        raise ValueError('incomplete segment')
    return result


def profile(path):
    labels, ranks = {}, {}
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as file:
        for line in file:
            row = json.loads(line)
            key = row['rank'], row['graph']
            if 'labels' in row:
                labels[key] = row['labels']
            # These pp2048/tg1 cases have only prefill forwards. Reject accidental
            # decode/draft rows rather than mixing them into component totals.
            if row['n'] <= 1 or len(row['release']) <= 3:
                raise ValueError('unexpected decode/draft row in prefill-only profile')
            blocks = segments(labels[key], row['ns'])
            if len(blocks) != len(row['release']):
                raise ValueError('GPU segments differ from host release count')
            r = ranks.setdefault(row['rank'], dict(tokens=0, graphs=0, gdn_blocks=0,
                gpu_segment_ms=0., delta_segment_ms=0., gdn_combined_ms=0.,
                prepare_ms=0., recurrence_ms=0., between_segment_ms=0.,
                dispatches=defaultdict(lambda: dict(count=0, ms=0.))))
            r['tokens'] += row['n']; r['graphs'] += 1
            r['between_segment_ms'] += sum(b['begin'] - a['end'] for a, b in zip(blocks, blocks[1:])) / 1e6
            for block in blocks:
                r['gpu_segment_ms'] += (block['end'] - block['begin']) / 1e6
                gdn = {}
                for label, start, end in block['spans']:
                    shader = label.split()[0]
                    dispatch = r['dispatches'][shader]
                    dispatch['count'] += 1; dispatch['ms'] += (end - start) / 1e6
                    if shader in ('qwen_gdn', 'qwen_gdn_prepare', 'qwen_gdn_prepared'):
                        if shader in gdn:
                            raise ValueError('duplicate GDN pass inside one segment')
                        gdn[shader] = start, end
                if gdn:
                    prepared = 'qwen_gdn_prepare' in gdn
                    expected = {'qwen_gdn_prepare', 'qwen_gdn_prepared'} if prepared else {'qwen_gdn'}
                    if set(gdn) != expected:
                        raise ValueError('unpaired GDN prepare/recurrence dispatch')
                    start, end = gdn['qwen_gdn_prepared' if prepared else 'qwen_gdn']
                    r['recurrence_ms'] += (end - start) / 1e6
                    if prepared:
                        prep_start, prep_end = gdn['qwen_gdn_prepare']
                        r['prepare_ms'] += (prep_end - prep_start) / 1e6
                        start = prep_start
                    # Envelope includes the prepare->recurrence barrier, without adding
                    # potentially overlapping top/bottom-of-pipe dispatch measurements.
                    r['gdn_combined_ms'] += (end - start) / 1e6
                    r['delta_segment_ms'] += (block['end'] - block['begin']) / 1e6
                    r['gdn_blocks'] += 1
    for rank, row in ranks.items():
        if row['gdn_blocks'] == 0:
            raise ValueError(f'rank {rank}: no GDN dispatch labels; --profile is required')
        row['gdn_ms_per_block'] = row['gdn_combined_ms'] / row['gdn_blocks']
        row['gdn_percent_of_gpu_segments'] = 100 * row['gdn_combined_ms'] / row['gpu_segment_ms']
        row['gdn_percent_of_delta_segments'] = 100 * row['gdn_combined_ms'] / row['delta_segment_ms']
        row['dispatches'] = dict(sorted(row['dispatches'].items(), key=lambda entry: -entry[1]['ms']))
    return ranks


def clocks(path, window):
    counts, temperatures = Counter(), []
    for line in path.read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) != 7:
            raise ValueError('unexpected clock sample')
        if window[0] <= float(fields[0]) <= window[1]:
            counts[','.join(fields[i] for i in [1, 2, 4, 5])] += 1
            temperatures.extend(float(fields[i]) for i in [3, 6])
    return dict(samples=sum(counts.values()), full_clock_samples=counts['850,1370,850,1370'],
                clock_counts=dict(counts), maximum_c=max(temperatures, default=None))


def summarize_run(plan_dir, case):
    root = plan_dir / 'runs' / case['name']
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('state') != 'completed':
        raise ValueError(case['name'] + ': harness did not complete')
    records = [json.loads(line) for line in (root / 'incite/stdout.log').read_text().splitlines() if line.startswith('{')]
    if len(records) != 1 or records[0].get('pp') != 2048 or records[0].get('tg') != 1:
        raise ValueError(case['name'] + ': expected exactly one pp2048/tg1 result')
    result = records[0]
    run = dict(name=case['name'], source_commit=case.get('source_commit'),
               source_archive=case.get('source_archive'), plan_settings=case,
               experimental=result.get('experimental'), host_settings=result.get('host'),
               peer_host_settings=result.get('peer_host'), token_ids=result['token_ids'],
               prefill_top=result['pp_top'], pp=result['pp'], tg=result['tg'], hosts={})
    for host, expected_ranks in [('incite', {0, 1}), ('invite', {2, 3})]:
        scope = json.loads((root / host / 'scope.json').read_text())
        if scope['binary_sha256'] != case['expected_binary_sha256']:
            raise ValueError(case['name'] + ': ' + host + ' binary differs from recorded source')
        path = root / host / 'profile.jsonl'
        if not path.exists():
            path = path.with_suffix('.jsonl.gz')
        ranks = profile(path)
        if set(ranks) != expected_ranks or any(row['tokens'] != result['pp'] for row in ranks.values()):
            raise ValueError(case['name'] + ': incomplete rank/prefill coverage on ' + host)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rb') as file:
            uncompressed_hash = hashlib.file_digest(file, 'sha256').hexdigest()
        run['hosts'][host] = dict(binary_sha256=scope['binary_sha256'], command=scope['command'],
            environment=manifest['hosts'][host]['environment'], profile_path=str(path),
            profile_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), profile_uncompressed_sha256=uncompressed_hash,
            ranks=ranks, clocks=clocks(root / host / 'clocks.txt', result['pp_window_unix']))
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan_directory', nargs='?', type=Path,
                   default=Path(__file__).resolve().parents[1] / 'plans/profile-plan')
    a = p.parse_args()
    cases = json.loads((a.plan_directory / 'plan.json').read_text())
    if len(cases) != 2:
        p.error('expected exactly baseline and prepared cases, in that order')
    runs = [summarize_run(a.plan_directory, case) for case in cases]
    baseline, prepared = runs
    checks = dict(token_ids_equal=baseline['token_ids'] == prepared['token_ids'],
                  prefill_top_ids_equal=[row[1] for row in baseline['prefill_top']] == [row[1] for row in prepared['prefill_top']],
                  prepared_flag_correct=not baseline['experimental']['prepared_gdn'] and prepared['experimental']['prepared_gdn'])
    comparison = []
    for host in ['incite', 'invite']:
        for rank, b in baseline['hosts'][host]['ranks'].items():
            c = prepared['hosts'][host]['ranks'][rank]
            checks[f'{host}_rank{rank}_gdn_block_count_equal'] = b['gdn_blocks'] == c['gdn_blocks']
            checks[f'{host}_rank{rank}_prepared_dispatches_present'] = (
                b['dispatches'].get('qwen_gdn', {}).get('count') == b['gdn_blocks']
                and 'qwen_gdn_prepare' not in b['dispatches']
                and c['dispatches'].get('qwen_gdn_prepare', {}).get('count') == c['gdn_blocks']
                and c['dispatches'].get('qwen_gdn_prepared', {}).get('count') == c['gdn_blocks']
                and 'qwen_gdn' not in c['dispatches'])
            row = dict(host=host, rank=rank, baseline_gdn_ms=b['gdn_combined_ms'], prepared_gdn_ms=c['gdn_combined_ms'],
                gdn_reduction_percent=100 * (1 - c['gdn_combined_ms'] / b['gdn_combined_ms']),
                baseline_gdn_percent_of_gpu_segments=b['gdn_percent_of_gpu_segments'],
                baseline_gpu_segments_ms=b['gpu_segment_ms'], prepared_gpu_segments_ms=c['gpu_segment_ms'],
                prepared_prepare_ms=c['prepare_ms'], prepared_recurrence_ms=c['recurrence_ms'])
            comparison.append(row)
            print(f"{host} rank {rank}: GDN {row['baseline_gdn_ms']:.2f} -> {row['prepared_gdn_ms']:.2f} ms "
                  f"({row['gdn_reduction_percent']:+.2f}% reduction); baseline GDN share {row['baseline_gdn_percent_of_gpu_segments']:.2f}%")
    medians = {key: statistics.median(row[key] for row in comparison)
               for key in comparison[0] if key not in ('host', 'rank')}
    print(f"Four-rank median: GDN {medians['baseline_gdn_ms']:.2f} -> {medians['prepared_gdn_ms']:.2f} ms; "
          f"paired reduction {medians['gdn_reduction_percent']:+.2f}%")
    summary = dict(notes=[
        'Instrumented component comparison only; profiled wall throughput is intentionally omitted.',
        'GDN combined time spans prepare start through recurrence end, including the intervening barrier.',
        'Top/bottom-of-pipe dispatch spans can overlap; shader totals are diagnostics, not an additive wall-time breakdown.',
        'Micro-batches can overlap on two queues; summed GPU segment times are not an end-to-end latency or strict speedup bound.',
        'Clock samples use the driver prefill Unix window on both hosts and assume synchronized host clocks.',
        'One-token and prefill-top ID matches are narrow checks, not a substitute for full model correctness checks.'],
        checks=checks, comparisons=comparison, four_rank_medians=medians, runs=runs)
    dest = a.plan_directory / 'gdn-profile-summary.json'
    dest.write_text(json.dumps(summary, indent=2) + '\n')
    print('Checks:', json.dumps(checks)); print(dest)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
