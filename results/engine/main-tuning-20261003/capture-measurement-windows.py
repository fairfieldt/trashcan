#!/usr/bin/env python3
"""Preserve local rsync-restored mtimes and approximate bench/clock alignment; no GPU work."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

BENCHES = ["baseline-1", "baseline-2", "selected", "final-1", "final-2",
           "long-baseline", "long-final"]


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def metadata(path):
    if not path.exists():
        return {"file": path.name, "status": "missing"}
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    return {"file": path.name, "status": "captured",
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "mtime_ns": after.st_mtime_ns, "mtime_utc": iso(after.st_mtime_ns / 1e9),
            "mtime_subsecond_ns": after.st_mtime_ns % 1_000_000_000,
            "changed_during_read": (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size)}


def read_clocks(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 7:
            continue
        try:
            row = [float(value) for value in fields]
        except ValueError:
            continue
        if all(math.isfinite(value) for value in row):
            rows.append(row)
    return sorted(rows)


def interval(start, end, clocks):
    selected = [row for row in clocks if start <= row[0] <= end]
    result = {"approximate_start_epoch_seconds": start,
              "approximate_end_epoch_seconds": end,
              "approximate_start_utc": iso(start), "approximate_end_utc": iso(end),
              "duration_seconds": end - start, "clock_samples": len(selected),
              "clock_log_spans_approximate_window": bool(clocks) and clocks[0][0] <= start and clocks[-1][0] >= end}
    if not selected:
        result["status"] = "no_clock_samples_in_approximate_window"
        return result
    result.update(status="sampled_approximate_window", first_sample_epoch_seconds=selected[0][0],
                  last_sample_epoch_seconds=selected[-1][0],
                  start_to_first_sample_seconds=selected[0][0] - start,
                  last_sample_to_end_seconds=end - selected[-1][0],
                  largest_internal_sample_gap_seconds=max((b[0] - a[0] for a, b in zip(selected, selected[1:])), default=None))
    both = sum(row[1:3] == [850, 1370] and row[4:6] == [850, 1370] for row in selected)
    result["both_cards_at_850_1370_samples"] = both
    result["both_cards_at_850_1370_sample_fraction"] = both / len(selected)
    result["cards"] = []
    for gpu in range(2):
        offset = 1 + 3 * gpu
        counts = Counter((int(row[offset]), int(row[offset + 1])) for row in selected)
        full = counts[(850, 1370)]
        result["cards"].append({"gpu": gpu, "at_850_1370_samples": full,
                                "at_850_1370_sample_fraction": full / len(selected),
                                "max_temperature_c": max(row[offset + 2] for row in selected),
                                "clock_pair_samples": [{"core_mhz": core, "memory_mhz": memory, "samples": count}
                                                       for (core, memory), count in sorted(counts.items())]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.directory
    clock_path = directory / "clocks.txt"
    clocks = read_clocks(clock_path) if clock_path.exists() else []
    files = {}
    runs = []
    for name in BENCHES:
        for suffix in [".json", ".log", ".tsv", ".prefill.tsv"]:
            files[name + suffix] = metadata(directory / (name + suffix))
        source = files[name + ".json"]
        entry = {"name": name, "status": source["status"], "timing_source": name + ".json"}
        try:
            data = json.loads((directory / (name + ".json")).read_text())
            pp, tg = data["pp_seconds"], data["tg_seconds"]
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in (pp, tg)):
                raise ValueError("missing, non-finite or non-positive duration")
            if source.get("changed_during_read"):
                raise ValueError("source changed during snapshot")
            end = source["mtime_ns"] / 1e9
            entry.update(status="approximate_windows_from_complete_bench_json",
                         measured_pp_seconds=pp, measured_tg_seconds=tg,
                         pp_tokens=data.get("pp"), tg_tokens=data.get("tg"),
                         combined=interval(end - pp - tg, end, clocks),
                         prefill=interval(end - pp - tg, end - tg, clocks),
                         decode=interval(end - tg, end, clocks))
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry.update(status="missing_or_invalid_timing_source", error=str(error))
        runs.append(entry)
    # Preserve CLI mtimes, but do not infer duration from rounded printed rates.
    for pattern in ["chat-baseline-*.log", "chat-final-*.log", "chat-baseline-*.txt", "chat-final-*.txt", "chat-baseline-*.tsv", "chat-final-*.tsv"]:
        for path in sorted(directory.glob(pattern)):
            files[path.name] = metadata(path)
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Read-only capture of completed local raw files copied with rsync-preserved mtimes. Git does not preserve these timestamps.",
        "method": "Approximate combined end = bench JSON mtime; start = end - pp_seconds - tg_seconds. Decode ends at that mtime; approximate prefill ends where decode starts.",
        "limitations": [
            "These are reconstructed windows, not recorded exact inference start/end timestamps.",
            "Observed local mtime_ns is preserved exactly; a zero subsecond component does not establish nanosecond precision. Current copied files commonly have whole-second mtimes.",
            "Stdout buffering, filesystem timestamp granularity, profile writes between timed phases, final profile writes/cleanup, and copy behavior can shift the anchor or insert unmeasured gaps. Their jitter is not measured or corrected.",
            "Inference setup/model loading is excluded by the reported phase durations; stderr log mtime may precede inference and is not used as its endpoint.",
            "Assumes rsync retained source timestamps and the benchmark files and clock samples use the same incite wall clock; the script does not independently verify that assumption.",
            "Clock coverage fractions count samples, not elapsed-time duty cycle or GPU utilization. Sampling may miss brief changes, and the ongoing remote log can extend beyond this local snapshot.",
            "CLI chat file mtimes are captured, but no chat windows are inferred from rounded rates. GPU reference validation and performance acceptance are separate from this alignment."
        ],
        "clock_snapshot": {**metadata(clock_path), "parsed_samples": len(clocks),
                           "first_sample_epoch_seconds": clocks[0][0] if clocks else None,
                           "last_sample_epoch_seconds": clocks[-1][0] if clocks else None,
                           "median_sample_interval_seconds": statistics.median(b[0] - a[0] for a, b in zip(clocks, clocks[1:])) if len(clocks) > 1 else None},
        "files": files, "benchmarks": runs,
        "cli_chat_windows": {"status": "omitted", "reason": "Only rounded rates are available in CLI logs; timestamp metadata retained above."}
    }
    destination = args.output or directory / "measurement-windows.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Captured {len(files)} file timestamps and {sum('combined' in row for row in runs)} approximate benchmark windows in {destination}")


if __name__ == "__main__":
    main()
