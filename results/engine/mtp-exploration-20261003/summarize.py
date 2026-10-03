#!/usr/bin/env python3
"""Rebuild numerical summaries from the saved measurements (stdlib only)."""
import collections
import csv
import json
import hashlib
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
report = {}
for phase in ["baseline", "candidate"]:
    bench = [json.loads((root / f"{phase}-bench-{rep}.out").read_text()) for rep in [1, 2]]
    result = {"pp_tok_s": sum(x["pp"] for x in bench) / sum(x["pp_seconds"] for x in bench),
              "tg_tok_s": sum(x["tg"] for x in bench) / sum(x["tg_seconds"] for x in bench),
              "bench_runs": bench, "chats": {}}
    for name in ["transformer", "photosynthesis", "csv"]:
        result["chats"][name] = {}
        for depth in [0, 3]:
            runs = []
            for rep in [1, 2]:
                text = (root / f"{phase}-{name}-d{depth}-{rep}.log").read_text()
                tokens, rate = re.search(r"decode: (\d+) tokens, ([\d.]+) tok/s", text).groups()
                row = {"tokens": int(tokens), "tok_s": float(rate)}
                if depth:
                    values = re.search(r"MTP: accepted (\d+)/(\d+) proposals in (\d+) rounds; draft ([\d.]+) ms/proposal, verify ([\d.]+) ms/round", text).groups()
                    row.update(zip(["accepted", "proposed", "rounds", "draft_ms", "verify_ms"],
                                   [int(x) for x in values[:3]] + [float(x) for x in values[3:]]))
                runs.append(row)
                assert (root / f"{phase}-{name}-d{depth}-{rep}.out").read_bytes() == (root / f"baseline-{name}-d0-{rep}.out").read_bytes()
            result["chats"][name][str(depth)] = {"pooled_tok_s": sum(x["tokens"] for x in runs) / sum(x["tokens"] / x["tok_s"] for x in runs), "runs": runs}
    report[phase] = result

    result["mtp_pooled_tok_s"] = 768 / sum(x["tokens"] / x["tok_s"] for c in result["chats"].values() for x in c["3"]["runs"])
    result["ordinary_pooled_tok_s"] = 768 / sum(x["tokens"] / x["tok_s"] for c in result["chats"].values() for x in c["0"]["runs"])
    mtp_runs = [x for c in result["chats"].values() for x in c["3"]["runs"]]
    result["verify_ms_weighted_by_rounds"] = sum(x["rounds"] * x["verify_ms"] for x in mtp_runs) / sum(x["rounds"] for x in mtp_runs)
    profiles = []
    for rep in [1, 2]:
        rows = list(csv.DictReader((root / f"{phase}-transformer-d3-{rep}.tsv").open(), delimiter="\t"))
        assert len(rows) == 97
        profiles.extend({"gpu": g, "rep": rep,
                         "attention_ms": sum(float(x[f"gpu{g}_ms"]) for x in rows[:96] if int(x["segment"]) % 2 == 0),
                         "ffn_ms": sum(float(x[f"gpu{g}_ms"]) for x in rows[:96] if int(x["segment"]) % 2 == 1),
                         "head_ms": float(rows[96][f"gpu{g}_ms"]),
                         "gaps_ms": sum(float(x[f"gap{g}_ms"]) for x in rows)} for g in [0, 1])
    result["last_four_column_transformer_verification_profiles"] = profiles

for name in ["transformer", "photosynthesis", "csv"]:
    for base, cand in zip(report["baseline"]["chats"][name]["3"]["runs"], report["candidate"]["chats"][name]["3"]["runs"]):
        assert all(base[k] == cand[k] for k in ["accepted", "proposed", "rounds"])
report["mtp_gain_percent"] = (report["candidate"]["mtp_pooled_tok_s"] / report["baseline"]["mtp_pooled_tok_s"] - 1) * 100

matrix = collections.defaultdict(list)
for line in (root / "matrix-micro.jsonl").read_text().splitlines():
    row = json.loads(line)
    assert row["reference_bit_identical"] and row["cpu_rows_checked"] == 3
    matrix[(row["tensor"], row["batch"], row["kernel"])].append(row["milliseconds"] * 1000)
report["matrix_micro_us"] = [{"tensor": t, "batch": b, "kernel": k, "mean_us": statistics.mean(v), "measurements": len(v)} for (t, b, k), v in matrix.items()]

report["depth_exploration"] = {}
for name in ["transformer", "photosynthesis", "csv"]:
    report["depth_exploration"][name] = {}
    for depth in [1, 2, 4]:
        runs = []
        for rep in [1, 2]:
            stem = f"explore-{name}-d{depth}-{rep}"
            text = (root / f"{stem}.log").read_text()
            tokens, rate = re.search(r"decode: (\d+) tokens, ([\d.]+) tok/s", text).groups()
            values = re.search(r"MTP: accepted (\d+)/(\d+) proposals in (\d+) rounds; draft ([\d.]+) ms/proposal, verify ([\d.]+) ms/round", text).groups()
            row = {"tokens": int(tokens), "tok_s": float(rate)}
            row.update(zip(["accepted", "proposed", "rounds", "draft_ms", "verify_ms"],
                           [int(x) for x in values[:3]] + [float(x) for x in values[3:]]))
            runs.append(row)
            assert (root / f"{stem}.out").read_bytes() == (root / f"baseline-{name}-d0-{rep}.out").read_bytes()
        report["depth_exploration"][name][str(depth)] = {"pooled_tok_s": sum(x["tokens"] for x in runs) / sum(x["tokens"] / x["tok_s"] for x in runs), "runs": runs}

attention = collections.defaultdict(list)
for tokens in [8192, 80000]:
    for parts in [64, 128]:
        for rep in [1, 2]:
            for line in (root / f"explore-attention-{tokens}-p{parts}-{rep}.out").read_text().splitlines():
                row = json.loads(line)
                assert row["pass"] and row["merge_bit_identical"]
                attention[(tokens, parts, row["dim"])].append(row["milliseconds"] * 1000)
report["attention_micro_us"] = [{"tokens": t, "parts": p, "dim": d, "mean_us": statistics.mean(v), "measurements": len(v)} for (t, p, d), v in attention.items()]

assert (root / "explore-stochastic-baseline.out").read_bytes() == (root / "explore-stochastic-final.out").read_bytes()
report["stochastic_seed17_output_byte_identical"] = True
head = collections.defaultdict(list)
for batch in range(5, 9):
    for kernel in ["multivec", "multivec4"]:
        for rep in [1, 2]:
            for line in (root / f"head-micro-b{batch}-{kernel}-{rep}.out").read_text().splitlines():
                row = json.loads(line)
                assert row["reference_bit_identical"] and row["cpu_rows_checked"] == 3
                head[(batch, kernel)].append(row["milliseconds"] * 1000)
report["head_followup_us"] = [{"batch": b, "kernel": k, "mean_us": statistics.mean(v), "measurements": len(v)} for (b, k), v in head.items()]
report["head_d4_chats"] = {}
for name in ["transformer", "photosynthesis", "csv"]:
    values = []
    for rep in [1, 2]:
        stem = f"head-{name}-d4-{rep}"
        assert (root / f"{stem}.out").read_bytes() == (root / f"explore-{name}-d4-{rep}.out").read_bytes()
        values.append(float(re.search(r"decode: 128 tokens, ([\d.]+) tok/s", (root / f"{stem}.log").read_text())[1]))
    report["head_d4_chats"][name] = 2 / sum(1 / v for v in values)
assert (root / "head-stochastic-d4-final.out").read_bytes() == (root / "head-stochastic-d4-head-tiles.out").read_bytes()
for file in (root / "head-d7-baseline").iterdir():
    if file.suffix == ".f32" or file.name == "greedy.txt":
        assert file.read_bytes() == (root / "head-d7-head-tiles" / file.name).read_bytes()
assert json.loads((root / "head-check-1700-target.out").read_text())["pass"]
assert (root / "head-default3-confirm.out").read_bytes() == (root / "candidate-transformer-d3-1.out").read_bytes()
project = root.parents[2]
for phase in ["candidate", "load-first", "final"]:
    checked = 0
    for line in (root / f"{phase}-source.sha256").read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        snapshot = root / "tested-source" / phase / name
        path = snapshot if snapshot.exists() else project / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (phase, name)
        checked += 1
    assert checked == 43
    (root / f"{phase}-source-check.txt").write_text(f"PASS: {checked} files reconstructed from current source and tested-source/{phase} snapshots.\n")
report["clocks"] = {}
for path in root.glob("*-clocks.txt"):
    rows = [line.split() for line in path.read_text().splitlines()[1:]]
    streak = longest = 0
    for row in rows:
        streak = streak + 1 if row[1:3] == ["850", "1370"] and row[4:6] == ["850", "1370"] else 0
        longest = max(longest, streak)
    report["clocks"][path.name] = {"samples": len(rows), "max_temperature_c": [max(int(row[j]) for row in rows) for j in [3, 6]],
                                 "clock_samples": [dict(collections.Counter(f"{row[j]}/{row[j + 1]}" for row in rows)) for j in [1, 4]],
                                 "longest_both_850_1370_samples": longest}
suite = (root / "final-validation-suite.log").read_text()
assert suite.count("PASS: 132 required tensors") == 6 and "test result: ok. 30 passed" in suite
assert (root / "acceptance-binary.sha256").read_text().split()[0] == (root / "final-binary.sha256").read_text().split()[0]
assert (root / "head-binary.sha256").read_text().split()[0] == (root / "final-binary.sha256").read_text().split()[0]
boundary = json.loads((root / "boundary-250.out").read_text())
assert boundary["pass"]
long_check = json.loads(next(x for x in reversed(suite.splitlines()) if x.startswith('{"batch"')))
assert long_check["pass"] and long_check["tokens"] == 1700
report["acceptance"] = {"reference_paths": 6, "boundary": boundary, "ring_wrap": long_check, "d7_f32_files": len(list((root / "head-d7-baseline").glob("*.f32"))), "d7_greedy_tokens": 64, "stochastic_d4_byte_identical": True}
(root / "parity-check.txt").write_text("PASS: baseline/candidate ordinary and MTP output bytes; accepted/proposed/round counts; every explored greedy depth; baseline/final stochastic seed17; complete reference and rollback suite.\n")
(root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
