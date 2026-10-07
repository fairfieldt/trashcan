#!/usr/bin/env python3
"""Join Qwen JSON/JSONL with per-host clocks and compare explicitly paired runs.

Usage: python3 analyze.py manifest.json --output summary.json --markdown summary.md

Manifest paths are relative to the manifest. Example:
{"runs": [
  {"name": "base", "output": "incite/base.jsonl",
   "clocks": {"incite": "incite/clocks/base.txt", "invite": "invite/clocks/node-base.txt"}},
  {"name": "gdn", "output": "incite/gdn.jsonl",
   "clocks": {"incite": "incite/clocks/gdn.txt", "invite": "invite/clocks/node-gdn.txt"}}
 ], "comparisons": [{"baseline": "base", "candidate": "gdn", "atol": 0.002, "rtol": 0}]}

Each run may supply clock_offsets_seconds (added to that host's sample timestamps),
exclude_reps, and expected_sclk/mclk (default 850/1370 MHz). No warmup is silently
discarded. Clocks are time weighted within the driver's exact pp/tg windows; gaps
larger than three nominal sample intervals are not attributed to nearby samples.
Cross-mode sampled token equality is descriptive only, never a correctness gate.
"""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import sys


def read_rows(path):
    text = path.read_text()
    try:
        value = json.loads(text)
        rows = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        rows = []
        for line, raw in enumerate(text.splitlines(), 1):
            if not raw.strip():
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line}: expected a complete JSON record: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: records must be JSON objects")
    return rows


def read_clocks(path, offset=0):
    rows = []
    for line, raw in enumerate(path.read_text().splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0] == "t":
            continue
        if len(fields) != 7:
            raise ValueError(f"{path}:{line}: expected timestamp and two sclk/mclk/temp triples")
        values = [float(value) for value in fields]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line}: nonfinite clock sample")
        values[0] += offset
        rows.append(values)
    rows.sort(key=lambda row: row[0])
    if any(a[0] == b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"{path}: duplicate timestamps")
    return rows


def weighted_quantile(values, quantile):
    values = sorted(values)
    target = sum(weight for _, weight in values) * quantile
    cumulative = 0
    for value, weight in values:
        cumulative += weight
        if cumulative >= target:
            return value
    return values[-1][0]


def clock_window(rows, window, expected_sclk=850, expected_mclk=1370):
    if not window or len(window) != 2 or window[1] <= window[0]:
        return {"valid": False, "reason": "missing or empty timing window"}
    start, end = window
    gaps = [b[0] - a[0] for a, b in zip(rows, rows[1:]) if b[0] > a[0]]
    nominal = statistics.median(gaps) if gaps else 0.25
    # A sample owns the interval halfway to its neighbors, capped across log gaps.
    cells = []
    for i, row in enumerate(rows):
        left = min((row[0] - rows[i - 1][0]) / 2, nominal * 1.5) if i else nominal / 2
        right = min((rows[i + 1][0] - row[0]) / 2, nominal * 1.5) if i + 1 < len(rows) else nominal / 2
        weight = max(0, min(end, row[0] + right) - max(start, row[0] - left))
        if weight:
            cells.append((row, weight))
    covered = sum(weight for _, weight in cells)
    samples = [row for row in rows if start <= row[0] <= end]
    result = {"valid": covered > 0, "samples": len(samples), "weighted_samples": len(cells),
              "window_seconds": end - start, "covered_seconds": covered,
              "coverage_fraction": covered / (end - start), "nominal_interval_seconds": nominal}
    if not cells:
        return result
    cards = []
    for card in range(2):
        at = 1 + card * 3
        core = [(row[at], weight) for row, weight in cells]
        memory = [(row[at + 1], weight) for row, weight in cells]
        full = sum(weight for row, weight in cells if row[at] >= expected_sclk and row[at + 1] >= expected_mclk)
        histogram = collections.Counter((int(row[at]), int(row[at + 1])) for row in samples)
        cards.append({"card": card, "sclk_min": min(value for value, _ in core),
                      "sclk_p10": weighted_quantile(core, 0.1), "sclk_median": weighted_quantile(core, 0.5),
                      "sclk_mean": sum(value * weight for value, weight in core) / covered,
                      "mclk_min": min(value for value, _ in memory),
                      "mclk_mean": sum(value * weight for value, weight in memory) / covered,
                      "temperature_min": min(row[at + 2] for row, _ in cells),
                      "temperature_max": max(row[at + 2] for row, _ in cells),
                      "full_clock_fraction": full / covered,
                      "sample_clock_counts": {f"{s}/{m}": count for (s, m), count in sorted(histogram.items())}})
    result["cards"] = cards
    result["all_cards_full_clock_fraction"] = sum(weight for row, weight in cells
        if all(row[1 + card * 3] >= expected_sclk and row[2 + card * 3] >= expected_mclk for card in range(2))) / covered
    return result


def identity(row, ordinal):
    case = row.get("case", {})
    return (case.get("name", f"pp{row.get('pp', '?')}"), row.get("rep", ordinal))


def analyze_run(spec, root):
    rows = read_rows(root / spec["output"])
    logs = {host: read_clocks(root / path, spec.get("clock_offsets_seconds", {}).get(host, 0))
            for host, path in spec.get("clocks", {}).items()}
    records = []
    for index, row in enumerate(rows):
        case, rep = identity(row, index)
        record = {"case": case, "rep": rep, "excluded": rep in spec.get("exclude_reps", []),
                  "result": row, "clocks": {}}
        for phase in ("pp", "tg"):
            record["clocks"][phase] = {host: clock_window(log, row.get(f"{phase}_window_unix"),
                spec.get("expected_sclk", 850), spec.get("expected_mclk", 1370)) for host, log in logs.items()}
        records.append(record)
    micro = collections.defaultdict(dict)
    for row in rows:
        if "prepared" in row and "samples_ms" in row:
            key = tuple(row[field] for field in ("gpu", "n", "k_heads", "v_heads", "position"))
            if row["prepared"] in micro[key]:
                raise ValueError(f"{spec['name']}: duplicate GDN micro result {key}/{row['prepared']}")
            micro[key][row["prepared"]] = row
    micro_pairs = []
    for key, pair in sorted(micro.items()):
        if False not in pair or True not in pair:
            continue
        baseline, candidate = pair[False], pair[True]
        micro_pairs.append({**dict(zip(("gpu", "n", "k_heads", "v_heads", "position"), key)),
            "baseline_ms": baseline["ms"], "prepared_ms": candidate["ms"],
            "time_reduction_percent": (1 - candidate["ms"] / baseline["ms"]) * 100,
            "pass": baseline.get("pass") is True and candidate.get("pass") is True,
            "bitwise_equal": candidate.get("bitwise_equal"), "max_abs_error": candidate.get("max_abs_error"),
            "max_rel_error": candidate.get("max_rel_error")})
    return {"name": spec["name"], "output": spec["output"], "fan_condition": spec.get("fan_condition"),
            "instrumentation": spec.get("instrumentation"),
            "clock_offsets_seconds": spec.get("clock_offsets_seconds", {}),
            "records": records, "checks": [row for row in rows if "pass" in row], "gdn_micro_pairs": micro_pairs}


def full_clocks(record, phase):
    clocks = record["clocks"][phase]
    # Both hosts and both cards must be observed. A one-sample startup excursion is
    # retained in raw evidence but can fall within the explicitly stated 99% rule.
    return (len(clocks) >= 2 and all(clock.get("samples", 0) >= 2 and clock.get("coverage_fraction", 0) >= 0.9 and
            clock.get("all_cards_full_clock_fraction", 0) >= 0.99 for clock in clocks.values()))


def compare_scores(base, candidate, atol, rtol):
    left, right = base.get("pp_top"), candidate.get("pp_top")
    if left is None or right is None:
        return {"pass": None, "reason": "prefill top scores unavailable"}
    ids_equal = [value[1] for value in left] == [value[1] for value in right]
    baseline = {token: score for score, token in left}
    got = {token: score for score, token in right}
    common = baseline.keys() & got.keys()
    finite = all(math.isfinite(score) for score in [*baseline.values(), *got.values()])
    errors = [abs(baseline[token] - got[token]) for token in common]
    within = bool(common) and finite and baseline.keys() == got.keys() and all(
        abs(baseline[token] - got[token]) <= atol + rtol * abs(baseline[token]) for token in common)
    return {"pass": ids_equal and within, "ids_equal": ids_equal, "finite": finite,
            "exact": left == right, "max_abs_error": max(errors, default=None),
            "atol": atol, "rtol": rtol, "common_tokens": len(common)}


def compare_records(base, candidate, spec):
    left, right = base["result"], candidate["result"]
    greedy = all(row.get("case", {}).get("temperature", 0) == 0 or
                 row.get("case", {}).get("top_k") == 1 for row in (left, right))
    a, b = left.get("token_ids"), right.get("token_ids")
    exact = a == b if a is not None and b is not None else None
    first = next((index for index, (x, y) in enumerate(zip(a or [], b or [])) if x != y), None)
    if first is None and a is not None and b is not None and len(a) != len(b):
        first = min(len(a), len(b))
    scores = compare_scores(left, right, spec.get("atol", 0), spec.get("rtol", 0))
    same_input = left.get("pp") == right.get("pp") and left.get("case") == right.get("case")
    result = {"case": base["case"], "rep": base["rep"], "same_input": same_input,
              "greedy": greedy, "token_ids_equal": exact, "first_token_difference": first,
              "prefill_top": scores, "correctness_pass": same_input and scores["pass"] is True
                  and (exact is True if greedy else True), "phases": {}}
    for phase in ("pp", "tg"):
        bt, ct = left.get(f"{phase}_seconds"), right.get(f"{phase}_seconds")
        bn, cn = left.get(phase), right.get(phase)
        if bt and ct and bn and cn:
            equal_work = bn == cn
            result["phases"][phase] = {"baseline_tok_s": bn / bt, "candidate_tok_s": cn / ct,
                "baseline_elapsed_ms": bt * 1000, "candidate_elapsed_ms": ct * 1000,
                "throughput_gain_percent": (cn / ct / (bn / bt) - 1) * 100,
                "time_reduction_percent": (1 - ct / bt) * 100 if equal_work else None,
                "equal_token_count": equal_work,
                "both_runs_full_clocks": full_clocks(base, phase) and full_clocks(candidate, phase)}
            ap, bp = left.get(f"{phase}_tp_ms"), right.get(f"{phase}_tp_ms")
            if isinstance(ap, dict) and isinstance(bp, dict):
                result["phases"][phase]["driver_tp_ms"] = {key: {"baseline": ap[key], "candidate": bp[key],
                    "delta": bp[key] - ap[key], "reduction_percent": (1 - bp[key] / ap[key]) * 100 if ap[key] else None}
                    for key in ap.keys() & bp.keys()}
    return result


def compare_runs(spec, runs):
    base, candidate = runs[spec["baseline"]], runs[spec["candidate"]]
    def indexed(run):
        result = {}
        for row in run["records"]:
            if row["excluded"] or "pp" not in row["result"]:
                continue
            key = (row["case"], row["rep"])
            if key in result:
                raise ValueError(f"{run['name']}: duplicate case/repetition {key}")
            result[key] = row
        return result
    a, b = indexed(base), indexed(candidate)
    common = sorted(a.keys() & b.keys())
    pairs = [compare_records(a[key], b[key], spec) for key in common]
    aggregate = {}
    for phase in ("pp", "tg"):
        included = [pair for pair in pairs if phase in pair["phases"]]
        clean = [pair for pair in included if pair["phases"][phase]["both_runs_full_clocks"] and
                 pair["phases"][phase]["equal_token_count"] and pair["correctness_pass"]]
        aggregate[phase] = {"pairs": len(included), "full_clock_correct_pairs": len(clean),
            "median_full_clock_gain_percent": statistics.median(pair["phases"][phase]["throughput_gain_percent"]
                for pair in clean) if clean else None,
            "per_case_full_clock_gain_percent": {case: statistics.median(
                pair["phases"][phase]["throughput_gain_percent"] for pair in clean if pair["case"] == case)
                for case in sorted({pair["case"] for pair in clean})}}
    return {"baseline": spec["baseline"], "candidate": spec["candidate"],
            "unmatched_baseline": sorted(a.keys() - b.keys()), "unmatched_candidate": sorted(b.keys() - a.keys()),
            "pairs": pairs, "summary": aggregate}


def markdown(summary):
    lines = ["# Hardware comparison", "", "Full-clock summaries require at least two in-window samples per host, 90% clock coverage and 99% of observed time at 850/1370 MHz (or configured thresholds) on both cards of both hosts. Shorter windows retain measured latency but cannot independently establish sustained clocks. Excluded repetitions are explicit in the manifest. Numerical tolerances are explicit per comparison.", ""]
    for comparison in summary["comparisons"]:
        lines += [f"## {comparison['baseline']} → {comparison['candidate']}", "",
                  "| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |", "| --- | ---: | --- | ---: | ---: | ---: | --- |"]
        for pair in comparison["pairs"]:
            phases = pair["phases"]
            def gain(phase):
                return f"{phases[phase]['throughput_gain_percent']:+.2f}%" if phase in phases else "—"
            error = pair["prefill_top"].get("max_abs_error")
            ids = ("yes" if pair["token_ids_equal"] else "NO") if pair["greedy"] else "sampled"
            clocks = "/".join("yes" if phases.get(phase, {}).get("both_runs_full_clocks") else "no" for phase in ("pp", "tg"))
            lines.append(f"| {pair['case']} | {pair['rep']} | {ids} | {error if error is not None else '—'} | {gain('pp')} | {gain('tg')} | {clocks} |")
        lines.append("")
        for phase, stats in comparison["summary"].items():
            gain = stats["median_full_clock_gain_percent"]
            lines.append(f"- {phase}: {stats['full_clock_correct_pairs']}/{stats['pairs']} equal-work, correct, full-clock pairs; median gain {f'{gain:+.2f}%' if gain is not None else 'unavailable'}.")
        if comparison["unmatched_baseline"] or comparison["unmatched_candidate"]:
            lines.append(f"- Unmatched: baseline {comparison['unmatched_baseline']}; candidate {comparison['unmatched_candidate']}.")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    run_specs = manifest if isinstance(manifest, list) else manifest["runs"]
    runs = [analyze_run(spec, args.manifest.parent) for spec in run_specs]
    indexed = {run["name"]: run for run in runs}
    if len(indexed) != len(runs):
        raise ValueError("run names must be unique")
    summary = {"manifest": str(args.manifest), "clock_rule": {"minimum_in_window_samples_per_host": 2, "minimum_coverage": 0.9, "minimum_full_clock_fraction": 0.99},
               "runs": runs, "comparisons": [compare_runs(spec, indexed) for spec in
                   ([] if isinstance(manifest, list) else manifest.get("comparisons", []))]}
    output = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        sys.stdout.write(output)
    if args.markdown:
        args.markdown.write_text(markdown(summary))


if __name__ == "__main__":
    main()
