#!/usr/bin/env python3
"""Rebuild local hardware summaries from retrieved data; makes no remote calls."""
import collections
import itertools
import json
import os
from pathlib import Path
import statistics
import time

import analyze

ROOT = Path(__file__).resolve().parent
KNOWN_FAILURES = {
    "check-prepared-700": "Thunderbolt stream permission failure before inference",
    "check-prepared-700-retry": "Cold peer model loading exceeded the readiness interval; driver failed",
}
EXPLORATORY = {
    "base-pp2048-a": "Two streams; the third physical link was replugged during model loading/prefill. Retain as exploratory, exclude from performance comparisons.",
}


def write(name, value):
    write_text(name, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_text(name, text):
    path = ROOT / name
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def micro_summary():
    hosts = {}
    for host in ("incite", "invite"):
        shapes = []
        for path in sorted((ROOT / host).glob("gdn-*.jsonl")):
            log = ROOT / host / "clocks" / (path.stem + ".txt")
            run = analyze.analyze_run({"name": path.stem, "output": str(path.relative_to(ROOT))}, ROOT)
            clocks = analyze.read_clocks(log)
            cards = []
            for card in range(2):
                at = 1 + card * 3
                counts = collections.Counter(f"{int(row[at])}/{int(row[at + 1])}" for row in clocks)
                cards.append({"card": card, "clock_sample_counts": dict(sorted(counts.items())),
                    "full_clock_samples": counts.get("850/1370", 0), "total_samples": len(clocks),
                    "min_temperature": min(row[at + 2] for row in clocks),
                    "max_temperature": max(row[at + 2] for row in clocks)})
            all_full = all(card["full_clock_samples"] == card["total_samples"] for card in cards)
            for pair in run["gdn_micro_pairs"]:
                pair.update(output=str(path.relative_to(ROOT)), clocks=str(log.relative_to(ROOT)),
                    whole_process_clock_samples=cards, whole_process_all_samples_full_clock=all_full)
                shapes.append(pair)
        if not shapes:
            continue
        by_n = {}
        for n in sorted({shape["n"] for shape in shapes}):
            selected = [shape for shape in shapes if shape["n"] == n]
            by_n[n] = {"pairs": len(selected), "baseline_ms_range": [min(shape["baseline_ms"] for shape in selected), max(shape["baseline_ms"] for shape in selected)],
                "prepared_ms_range": [min(shape["prepared_ms"] for shape in selected), max(shape["prepared_ms"] for shape in selected)],
                "time_reduction_percent_range": [min(shape["time_reduction_percent"] for shape in selected), max(shape["time_reduction_percent"] for shape in selected)],
                "median_time_reduction_percent": statistics.median(shape["time_reduction_percent"] for shape in selected)}
        hosts[host] = {"pairs": len(shapes), "all_pass": all(shape["pass"] for shape in shapes),
            "all_bitwise_equal": all(shape["bitwise_equal"] for shape in shapes),
            "maximum_absolute_error": max(value for shape in shapes for value in shape["max_abs_error"]),
            "pairs_with_whole_process_full_clock_samples": sum(shape["whole_process_all_samples_full_clock"] for shape in shapes),
            "by_n": by_n, "shapes": shapes}
    result = {"generated_unix": time.time(), "binary_sha256": "0bb31aab9a45eaa602ce6856623d8ca8884810c7f4926da2c6b17e4e271bca5a",
        "heads": {"key": 4, "value": 12}, "positions": [0, 700], "tokens": [9, 64, 129, 188, 256, 257],
        "iterations_per_mode": 12, "warmup_ms_per_mode": 300,
        "timing_scope": "GPU timestamps include preparation plus barrier and recurrence; exclude input/state restoration. Each replay is a separate submission.",
        "clock_caveats": ["Clock logs cover each complete process, with the two GPUs tested sequentially. Kernel modes have no Unix timing windows, so individual timing samples cannot be joined exactly to clocks.",
            "The n=9, position=0 first process on both hosts includes startup/idle clock samples. It remains a correctness result; clock logs do not establish a clean full-clock timing pair.",
            "Subsequent processes have all sampled cards at 850/1370 MHz, but the approximately 0.25-second sampler cannot prove clocks throughout each millisecond-scale kernel sample.",
            "Baseline then prepared mode is a fixed order with separate 300-ms warmups; these are kernel timings, not TP4 model throughput."],
        "total_pairs": sum(host["pairs"] for host in hosts.values()), "hosts": hosts}
    write("micro-summary.json", result)
    return result


def command_key(command):
    # These options change decoding strategy or repetition count, not the workload.
    skipped = {"--mtp", "--draft-tokens", "--reps"}
    values = []
    index = 1
    while index < len(command):
        token = command[index]
        if token in skipped:
            index += 2
        else:
            values.append(token)
            index += 1
    return values


def fan_condition(manifest, path):
    def regime(value):
        if isinstance(value, dict) and "regime" in value:
            return value["regime"]
        return value
    if manifest.get("fan_policy") is not None:
        return regime(manifest["fan_policy"])
    scopes = {}
    for host in ("incite", "invite"):
        scope = path.parent / host / "scope.json"
        if scope.exists():
            policy = json.loads(scope.read_text()).get("fan_policy")
            if policy is not None:
                scopes[host] = regime(policy)
    if len(scopes) == 2:
        return scopes["incite"] if scopes["incite"] == scopes["invite"] else scopes
    if "maxfan" in str(path):
        return "maximum-fan-condition-named-but-policy-unrecorded"
    return "historical-before-dynamic-maximum-fan-policy"


def instrumentation(manifest):
    env = dict(value.split("=", 1) for value in manifest.get("hosts", {}).get("incite", {}).get("environment", []))
    return {"tp_profile": "TRASHCAN_TP_PROFILE" in env, "dispatch_profile": env.get("TRASHCAN_TP_PROFILE_DISPATCH") == "1"}


def macro_summary():
    run_specs, excluded, pending, skipped, metadata = [], [], [], [], {}
    for path in sorted(ROOT.glob("**/runs/*/manifest.json")):
        manifest = json.loads(path.read_text())
        name = manifest["name"]
        driver = manifest.get("hosts", {}).get("incite", {})
        command = driver.get("command", [])
        if len(command) > 1 and command[1] not in ("qwen-bench", "qwen-check"):
            skipped.append({"name": name, "command": command[1], "reason": "Uses separate nonbenchmark/client evidence"})
            continue
        if name in KNOWN_FAILURES:
            excluded.append({"name": name, "reason": KNOWN_FAILURES[name]})
            continue
        if manifest.get("state") != "completed":
            pending.append({"name": name, "state": manifest.get("state", "running"), "error": manifest.get("error")})
            continue
        if driver.get("ssh_returncode") != 0:
            excluded.append({"name": name, "reason": f"driver exit {driver.get('ssh_returncode')}"})
            continue
        output = path.parent / "incite" / "stdout.log"
        clocks = {host: path.parent / host / "clocks.txt" for host in ("incite", "invite")}
        retrieved = all(manifest.get("hosts", {}).get(host, {}).get("retrieval_returncode") == 0 for host in clocks)
        if not retrieved or not output.exists() or not output.stat().st_size or any(not log.exists() for log in clocks.values()):
            pending.append({"name": name, "state": "completed, retrieval not yet complete"})
            continue
        spec = {"name": name, "output": str(output.relative_to(ROOT)),
                "clocks": {host: str(log.relative_to(ROOT)) for host, log in clocks.items()},
                "source_manifest": str(path.relative_to(ROOT)), "fan_condition": fan_condition(manifest, path),
                "instrumentation": instrumentation(manifest)}
        if name in EXPLORATORY:
            spec.update(exclude_from_comparisons=True, exploratory_reason=EXPLORATORY[name])
        run_specs.append(spec)
        metadata[name] = manifest
    comparisons = []
    for base, candidate in itertools.combinations(sorted(run_specs, key=lambda spec: metadata[spec["name"]]["started_unix"]), 2):
        if base.get("exclude_from_comparisons") or candidate.get("exclude_from_comparisons"):
            continue
        a, b = metadata[base["name"]], metadata[candidate["name"]]
        if base["fan_condition"] != candidate["fan_condition"] or base["instrumentation"] != candidate["instrumentation"]:
            continue
        ac, bc = a["hosts"]["incite"]["command"], b["hosts"]["incite"]["command"]
        if ac[1] != "qwen-bench" or bc[1] != "qwen-bench" or command_key(ac) != command_key(bc):
            continue
        def attention(manifest):
            env = dict(value.split("=", 1) for value in manifest["environment"])
            return env.get("TRASHCAN_QWEN_ATTN", "scalar"), env.get("TRASHCAN_QWEN_ATTN_SLAB", "64")
        tolerance = 0.002 if attention(a) != attention(b) else 0
        comparisons.append({"baseline": base["name"], "candidate": candidate["name"], "atol": tolerance, "rtol": 0})
    manifest = {"runs": run_specs, "comparisons": comparisons,
        "time_sync_evidence": ["incite/time-sync.txt", "invite/time-sync.txt"],
        "clock_alignment": "Both hosts report NTPSynchronized=yes. Recorded NTP offsets are -11.979 ms and -3.927 ms; no post-hoc timestamp adjustment is applied. Clock samples are approximately 250 ms apart.",
        "policy": "Every successful, retrieved run is retained. Compare chronological pairs with identical fan regime and command/workload arguments after ignoring binary, --mtp, --draft-tokens and --reps. Same attention settings require exact top scores; changed attention uses absolute tolerance 0.002. No repetition is silently dropped.",
        "excluded": excluded, "pending": pending, "skipped_nonbenchmark": skipped}
    write("analysis-manifest.json", manifest)
    runs = [analyze.analyze_run(spec, ROOT) for spec in run_specs]
    indexed = {run["name"]: run for run in runs}
    result = {"generated_unix": time.time(), "manifest": "analysis-manifest.json", "runs": runs,
        "comparisons": [analyze.compare_runs(spec, indexed) for spec in comparisons], "excluded": excluded, "pending": pending,
        "skipped_nonbenchmark": skipped}
    write("analysis-summary.json", result)
    write_text("analysis-summary.md", analyze.markdown(result))
    plan_summaries(indexed)
    return result


def plan_summaries(runs):
    for path in sorted((ROOT / "plans").glob("*/plan.json")):
        plan = json.loads(path.read_text())
        if not isinstance(plan, list) or not all(isinstance(item, dict) and "name" in item for item in plan):
            continue
        summary_path = path.parent / "summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text()).get("dry_run") is True:
            write(path.parent.name + "-analysis.json", {
                "plan": str(path.relative_to(ROOT)), "summary": str(summary_path.relative_to(ROOT)),
                "dry_run": True, "measured": False, "complete": False,
                "state": "Non-measured dry-run preview; no remote operations or result matching.",
                "planned_runs": [item["name"] for item in plan], "pending": [],
                "rows": [], "matched_comparisons": [], "configuration_summaries": [], "work_totals": [],
            })
            continue
        if any("mode" in item or item.get("command", "qwen-bench") not in ("qwen-bench", "qwen-check") for item in plan):
            continue
        names = [item["name"] for item in plan]
        available = [name for name in names if name in runs]
        rows = []
        for name in available:
            for record in runs[name]["records"]:
                row = record["result"]
                if "pp" not in row:
                    continue
                rows.append({"name": name, "case": record["case"], "rep": record["rep"], "pp": row["pp"], "tg": row["tg"],
                    "pp_seconds": row["pp_seconds"], "tg_seconds": row["tg_seconds"],
                    "pp_tok_s": row["pp_tok_s"], "tg_tok_s": row["tg_tok_s"],
                    "pp_tp_ms": row.get("pp_tp_ms"), "tg_tp_ms": row.get("tg_tp_ms"), "mtp_stats": row.get("mtp"),
                    "clocks": record["clocks"], "full_clocks": {phase: analyze.full_clocks(record, phase) for phase in ("pp", "tg")}})
        result = {"plan": str(path.relative_to(ROOT)), "complete": available == names,
            "pending": [name for name in names if name not in runs], "rows": rows,
            "phase_meaning": "Driver host instrumentation only. pp_tp_ms covers the whole prefill; tg_tp_ms is per emitted token. add=sum+total, so do not add all three. exchange includes peer/transport synchronization; wait is host time waiting for GPU completion. These are not isolated kernel durations.",
            "matched_comparisons": [],
            "matching_policy": "Separate binary, MTP mode, environment, exact command workload (including stream endpoints) and corpus case. Each candidate observation pairs with the closest reference in planned order, preferring matching repetition. Cross-version plus cross-mode pairs are correctness-only. Summaries wait for the entire plan."}
        if result["complete"]:
            groups = collections.defaultdict(list)
            configs = {}
            for order, item in enumerate(plan):
                run = runs[item["name"]]
                meta_path = (ROOT / run["output"]).parent.parent / "manifest.json"
                meta = json.loads(meta_path.read_text())
                command = meta["hosts"]["incite"]["command"]
                for record in run["records"]:
                    if "pp" not in record["result"]:
                        continue
                    config = {"binary": Path(meta["binary"]).name, "mtp": "--mtp" in command,
                        "environment": dict(value.split("=", 1) for value in meta["environment"]),
                        "workload": command_key(command), "case": record["case"], "fan_condition": fan_condition(meta, meta_path),
                        "instrumentation": instrumentation(meta)}
                    key = json.dumps(config, sort_keys=True)
                    configs[key] = config
                    groups[key].append({"order": order, "run": item["name"], "record": record})
            result["configuration_summaries"] = []
            for key, members in groups.items():
                config = configs[key]
                selected = [row for row in rows if row["name"] in [member["run"] for member in members] and row["case"] == config["case"]]
                metrics = {}
                for phase in ("pp", "tg"):
                    clean = [row for row in selected if row["full_clocks"][phase]]
                    metrics[phase] = {"observations": len(selected), "full_clock_observations": len(clean),
                        "full_clock_tok_s": [row[f"{phase}_tok_s"] for row in clean],
                        "all_elapsed_ms": [row[f"{phase}_seconds"] * 1000 for row in selected],
                        "median_full_clock_tok_s": statistics.median(row[f"{phase}_tok_s"] for row in clean) if clean else None,
                        "median_full_clock_driver_tp_ms": {metric: statistics.median(row[f"{phase}_tp_ms"][metric] for row in clean)
                            for metric in ("sum", "total", "add", "exchange", "wait", "submit", "allreduces")}
                            if clean and all(row.get(f"{phase}_tp_ms") for row in clean) else None}
                entry = {"configuration": config, "runs": [member["run"] for member in members], "phases": metrics}
                if len(members) >= 2:
                    entry["repeat_drift"] = labeled_pair(members[0], members[-1])
                result["configuration_summaries"].append(entry)
            for left, right in itertools.combinations(groups, 2):
                a, b = configs[left], configs[right]
                if (a["workload"] != b["workload"] or a["case"] != b["case"] or a["fan_condition"] != b["fan_condition"]
                        or a["instrumentation"] != b["instrumentation"]):
                    continue
                category, swap = comparison_kind(a, b)
                if swap:
                    left, right = right, left
                    a, b = b, a
                pairs = []
                for candidate in groups[right]:
                    reference = min(groups[left], key=lambda member: (
                        member["record"]["rep"] != candidate["record"]["rep"], abs(member["order"] - candidate["order"])))
                    pairs.append(labeled_pair(reference, candidate))
                summary = {}
                for phase in ("pp", "tg"):
                    clean = [pair["comparison"]["phases"][phase] for pair in pairs if pair["comparison"]["correctness_pass"]
                        and pair["comparison"]["phases"].get(phase, {}).get("both_runs_full_clocks")
                        and pair["comparison"]["phases"][phase]["equal_token_count"]]
                    summary[phase] = {"qualifying_pairs": len(clean), "median_gain_percent": statistics.median(
                        pair["throughput_gain_percent"] for pair in clean) if clean and category != "combined_correctness_only" else None}
                result["matched_comparisons"].append({"category": category, "baseline": a, "candidate": b, "pairs": pairs, "summary": summary})
            work = collections.defaultdict(list)
            for key, members in groups.items():
                config = dict(configs[key])
                name = config.pop("case")
                prefix, _, suffix = name.rpartition("-")
                config["case_family"] = prefix if suffix.isdigit() else name
                family_key = json.dumps(config, sort_keys=True)
                for member in members:
                    row = member["record"]["result"]
                    stats = row.get("mtp")
                    if stats is None:
                        maximum = row.get("case", {}).get("max_tokens", row["tg"])
                        columns = max(0, row["tg"] - 1) + int(row["tg"] < maximum)
                        drafts, rounds, plain = 0, 0, columns
                    else:
                        drafts, rounds, plain = stats.get("proposed", 0), stats.get("rounds", 0), stats.get("plain_steps", 0)
                        columns = rounds + drafts + plain
                    work[family_key].append({"run": member["run"], "case": name, "rep": member["record"]["rep"],
                        "output_tokens": row["tg"], "target_columns": columns, "draft_columns": drafts,
                        "speculative_rounds": rounds, "serial_steps": plain, "elapsed_ms": row["tg_seconds"] * 1000})
            result["work_totals"] = [{"configuration": json.loads(key), "observations": len(items),
                "totals": {field: sum(item[field] for item in items) for field in
                    ("output_tokens", "target_columns", "draft_columns", "speculative_rounds", "serial_steps", "elapsed_ms")},
                "cases": items} for key, items in work.items()]
            result["work_count_meaning"] = "Target columns are speculative rounds + proposed draft count + serial steps from runtime counters. Plain-mode columns are inferred from emitted token count and whether EOS precedes the output limit, for these non-context-limited fixtures. Draft columns equal proposed count. Sum elapsed ms is descriptive across these specific requests, not an independent sustained-throughput benchmark."
        write(path.parent.name + "-analysis.json", result)


def comparison_kind(a, b):
    same_env, same_mode = a["environment"] == b["environment"], a["mtp"] == b["mtp"]
    if a["binary"] != b["binary"]:
        return ("software_version" if same_env and same_mode else "combined_correctness_only"), "baseline" in b["binary"]
    if not same_mode:
        return ("decoding_mode" if same_env else "decoding_mode_and_policy"), a["mtp"]
    difference = {key for key in a["environment"].keys() | b["environment"].keys()
                  if a["environment"].get(key) != b["environment"].get(key)}
    if difference == {"TRASHCAN_QWEN_ADAPTIVE_MTP"}:
        return "adaptive_policy", a["environment"].get("TRASHCAN_QWEN_ADAPTIVE_MTP") == "1"
    return "runtime_configuration", len(a["environment"]) > len(b["environment"])


def labeled_pair(reference, candidate):
    return {"baseline_run": reference["run"], "candidate_run": candidate["run"],
            "comparison": analyze.compare_records(reference["record"], candidate["record"], {"atol": 0, "rtol": 0})}


if __name__ == "__main__":
    micro, macro = micro_summary(), macro_summary()
    print(json.dumps({"micro_pairs": micro["total_pairs"], "completed_runs": [run["name"] for run in macro["runs"]],
        "comparisons": len(macro["comparisons"]), "excluded": macro["excluded"], "pending": macro["pending"]}, indent=2))
