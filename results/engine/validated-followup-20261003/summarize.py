#!/usr/bin/env python3
"""Rebuild the follow-up's compact evidence summary from its saved outputs."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
report = {"source_baseline": "82e559f", "cpu_latency_request_us": 0, "phases": {}}
for phase in ["baseline", "norm", "merge"]:
    runs = [json.loads((root / f"{phase}-{rep}.json").read_text()) for rep in [1, 2]]
    rate = lambda workload: sum(d[workload] for d in runs) / sum(d[workload + "_seconds"] for d in runs)
    profiles = []
    for rep in [1, 2]:
        rows = list(csv.DictReader((root / f"{phase}-{rep}.tsv").open(), delimiter="\t"))
        profiles.append({
            "gpu_work_ms": [sum(float(d[f"gpu{g}_ms"]) for d in rows) for g in [0, 1]],
            "gaps_ms": [sum(float(d[f"gap{g}_ms"]) for d in rows) for g in [0, 1]],
            "note": "last forward only; GPU times run concurrently",
        })
    report["phases"][phase] = {
        "binary_sha256": (root / f"{phase}-binary.sha256").read_text().split()[0],
        "pp512_tok_s_pooled": rate("pp"), "tg256_tok_s_pooled": rate("tg"),
        "runs": runs, "last_forward_profiles": profiles,
    }
base = report["phases"]["baseline"]["tg256_tok_s_pooled"]
report["combined_decode_gain_percent"] = (report["phases"]["merge"]["tg256_tok_s_pooled"] / base - 1) * 100
report["long_context_single_pair"] = {phase: json.loads((root / f"long-{phase}.json").read_text()) for phase in ["baseline", "merge"]}

micro = [json.loads(line) for line in (root / "norm-micro.jsonl").read_text().splitlines()]
assert len(micro) == 64 and all(d["pass"] and d["bit_identical"] for d in micro)
groups = defaultdict(list)
for d in micro:
    groups[(d["columns"], d["shader"])].append(d["median_us"])
report["norm_micro"] = {f"{columns}col_{shader}_mean_of_medians_us": statistics.mean(values) for (columns, shader), values in groups.items()}
report["norm_micro"]["cpu_worst_scaled_error"] = max(d["cpu_worst_scaled_error"] for d in micro)
report["norm_micro"]["records_passed"] = len(micro)

attn = defaultdict(list)
for path in root.glob("attn-*.jsonl"):
    phase = path.name.split("-")[1]
    for line in path.read_text().splitlines():
        d = json.loads(line)
        assert d["pass"]
        if phase == "merge":
            assert d["merge_bit_identical"]
        attn[(phase, d["tokens"], d["dim"], d["partitions"])].append(d["milliseconds"] * 1000)
assert sum(map(len, attn.values())) == 400 and all(len(v) == 4 for v in attn.values())
report["attention_micro"] = [{"phase": phase, "tokens": tokens, "dim": dim, "partitions": parts, "mean_us": statistics.mean(values)} for (phase, tokens, dim, parts), values in sorted(attn.items())]

report["clocks"] = {}
for path in sorted(root.glob("*-clocks.txt")):
    rows = list(csv.DictReader(path.open(), delimiter=" "))
    if not rows:
        continue
    streak = longest = 0
    for d in rows:
        peak = all((d[f"sclk{g}"], d[f"mclk{g}"]) == ("850", "1370") for g in [0, 1])
        streak = streak + 1 if peak else 0
        longest = max(longest, streak)
    report["clocks"][path.name] = {
        "samples": len(rows),
        "per_gpu_clock_samples": [dict(Counter(f'{d[f"sclk{g}"]}/{d[f"mclk{g}"]}' for d in rows)) for g in [0, 1]],
        "max_temperature_c": [max(int(d[f"temp{g}"]) for d in rows) for g in [0, 1]],
        "longest_consecutive_both_850_1370_samples": longest,
        "note": "0.25 s sampling; whole command phases include loading, idle and serial GPU micro cases",
    }

final_hash = report["phases"]["merge"]["binary_sha256"]
assert (root / "acceptance-binary.sha256").read_text().split()[0] == final_hash
assert (root / "final-binary.sha256").read_text().split()[0] == final_hash
suite = (root / "validation-suite.log").read_text()
assert suite.count("PASS: 132 required tensors") == 6
assert "test result: ok." in suite and "10000" in suite
long_check = json.loads(next(line for line in reversed(suite.splitlines()) if line.startswith('{"batch"')))
boundary = json.loads((root / "boundary-250.json").read_text())
assert long_check["pass"] and long_check["tokens"] == 1700 and boundary["pass"]
parity = json.loads((root / "chat-parity.json").read_text())
assert all(d["byte_identical"] for d in parity.values())
assert (root / "baseline-chat.txt").read_bytes() == (root / "merge-chat.txt").read_bytes()
for phase in ["norm", "merge"]:
    assert (root / "baseline-activation.sha256").read_bytes() == (root / f"{phase}-activation.sha256").read_bytes()
    assert (root / "baseline-b1/greedy.txt").read_bytes() == (root / f"{phase}-b1/greedy.txt").read_bytes()
for phase in ["baseline", "norm", "merge"]:
    manifest = (root / f"{phase}-activation.sha256").read_text().splitlines()
    assert len(manifest) == 134  # 133 f32 files plus the 32-token greedy transcript.
    assert len((root / f"{phase}-b1/greedy.txt").read_text().splitlines()) == 32
    for row in manifest:
        digest, name = row.split(maxsplit=1)
        assert hashlib.sha256((root / f"{phase}-b1" / name).read_bytes()).hexdigest() == digest
report["validation"] = {
    "reference_paths_passed": 6,
    "fresh_baseline_activation_and_logit_files_byte_identical": 133,
    "fresh_baseline_greedy_tokens_byte_identical": 32,
    "long_check": long_check, "boundary_check": boundary, "chat_parity": parity,
    "cpu_test_result": next(line for line in suite.splitlines() if line.startswith("test result:")),
}
mtp = json.loads((root / "mtp-summary.json").read_text())
ordinary = {}
for name, pair in mtp.items():
    ordinary[name] = {}
    for phase, measured in pair.items():
        log = root / (f"{phase}-chat.log" if name == "transformer" else f"{name}-{phase}.log")
        tokens, rate = re.search(r"decode: (\d+) tokens, ([\d.]+) tok/s", log.read_text()).groups()
        ordinary[name][phase] = {"tokens": int(tokens), "tok_s": float(rate)}
        assert measured["ordinary_output_byte_identical"] and measured["tokens"] == int(tokens)
report["mtp_single_pairs"] = mtp
report["ordinary_chat_single_pairs"] = ordinary
report["long_prompt_chat_single_pair"] = {}
for phase in ["baseline", "merge"]:
    text = (root / f"long-water-{phase}.log").read_text()
    pp, pp_rate, tg, tg_rate = re.search(r"prefill: (\d+) tokens, ([\d.]+) tok/s; decode: (\d+) tokens, ([\d.]+) tok/s", text).groups()
    report["long_prompt_chat_single_pair"][phase] = {
        "pp_tokens": int(pp), "pp_tok_s": float(pp_rate), "tg_tokens": int(tg), "tg_tok_s": float(tg_rate),
    }
report["mtp_vs_ordinary_pooled_speedup"] = {
    phase: sum(d[phase]["tokens"] / d[phase]["tok_s"] for d in ordinary.values())
    / sum(d[phase]["tokens"] / d[phase]["tok_s"] for d in mtp.values())
    for phase in ["baseline", "merge"]
}
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(f'PASS: complete evidence, tg256 {base:.3f} -> {report["phases"]["merge"]["tg256_tok_s_pooled"]:.3f} tok/s ({report["combined_decode_gain_percent"]:.2f}%)')
