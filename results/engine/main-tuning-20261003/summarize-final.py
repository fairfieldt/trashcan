#!/usr/bin/env python3
"""Summarize saved final-run evidence only; write JSON to stdout, never run a workload.

Incomplete/copied-in-progress files remain explicit. Batch trials are selected as
one invocation matching shared-kv/build.json, never assembled across invocations.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


CONTEXTS = [(1, 8), (128, 16), (768, 32), (1536, 64), (2048, 64), (8192, 64)]
PROMPTS = ["transformer", "photosynthesis", "csv"]
CORE_REVISION = "63290a5b862a8f879e70b3d7739abde685c04b35"


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def mean(values):
    return statistics.mean(values) if values else None


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items() if key not in ("id", "call_id")}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def signature(response):
    return normalize({key: response.get(key) for key in ("status", "incomplete_details", "output")})


class Evidence:
    def __init__(self, directory):
        self.directory = directory
        self.files = {}

    def text(self, path):
        path = Path(path)
        key = str(path.relative_to(self.directory))
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            self.files[key] = {"status": "changed_during_read" if changed else "read",
                               "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            return None if changed else raw.decode()
        except (OSError, UnicodeError) as error:
            self.files[key] = {"status": "missing" if not path.exists() else "unreadable",
                               "error": str(error)}
            return None

    def json(self, path):
        text = self.text(path)
        if text is None:
            return None
        try:
            def bad_constant(value):
                raise ValueError("Non-finite JSON number: " + value)
            return json.loads(text, parse_constant=bad_constant)
        except ValueError as error:
            self.files[str(Path(path).relative_to(self.directory))].update(
                status="invalid_or_in_progress", error=str(error))
            return None

    def run(self, directory, name, suffix="json"):
        window = self.json(directory / (name + ".window.json"))
        command = self.json(directory / (name + ".command.json"))
        data = (self.json(directory / (name + "." + suffix)) if suffix == "json"
                else self.text(directory / (name + "." + suffix)))
        completed = isinstance(window, dict) and window.get("returncode") == 0
        return {"name": name, "command": command, "command_window": window,
                "command_completed": completed}, data


def bench(evidence, directory, name):
    result, data = evidence.run(directory, name)
    valid = isinstance(data, dict) and all(positive(data.get(key)) for key in
                ("pp", "pp_seconds", "pp_tok_s", "tg", "tg_seconds", "tg_tok_s"))
    result["status"] = "complete" if valid and result["command_completed"] else "partial_or_failed"
    result["metrics"] = data if valid else None
    return result


def pooled(runs):
    valid = [run["metrics"] for run in runs if run["status"] == "complete"]
    result = {"complete_runs": len(valid), "expected_runs": len(runs), "complete": len(valid) == len(runs)}
    for phase in ("pp", "tg"):
        result[phase] = ({"tokens": sum(row[phase] for row in valid),
                          "seconds": sum(row[phase + "_seconds"] for row in valid),
                          "weighted_tok_s": sum(row[phase] for row in valid) / sum(row[phase + "_seconds"] for row in valid),
                          "mean_tok_s": mean([row[phase + "_tok_s"] for row in valid]),
                          "min_tok_s": min(row[phase + "_tok_s"] for row in valid),
                          "max_tok_s": max(row[phase + "_tok_s"] for row in valid),
                          "tokens_per_run": sorted({row[phase] for row in valid})} if valid else None)
    return result


def bench_comparison(evidence, directory, names):
    runs = {label: [bench(evidence, directory, name) for name in group] for label, group in names.items()}
    groups = {label: pooled(rows) for label, rows in runs.items()}
    ratios = {}
    for phase in ("pp", "tg"):
        a, b = groups["core"], groups["shared"]
        eligible = (a["complete"] and b["complete"] and a[phase] and b[phase]
                    and a[phase]["tokens_per_run"] == b[phase]["tokens_per_run"])
        ratios[phase + "_speedup"] = b[phase]["weighted_tok_s"] / a[phase]["weighted_tok_s"] if eligible else None
    return {"runs": runs, "pooled": groups, "comparison": ratios,
            "note": "Bench emits exact elapsed phase seconds; aggregate rate is total tokens / total seconds."}


def microbenchmarks(evidence, directory):
    runs, comparisons = [], []
    for tokens, parts in CONTEXTS:
        pair = {}
        for kernel in ("scalar", "selected"):
            run, text = evidence.run(directory, f"micro-{kernel}-{tokens}-p{parts}", "jsonl")
            rows, invalid = [], []
            for number, line in enumerate((text or "").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=lambda value: None)
                    if not isinstance(row, dict):
                        raise ValueError("row is not an object")
                    rows.append(row)
                except ValueError:
                    invalid.append(number)
            expected = {(g, dim, kv) for g in (0, 1) for dim, kv in ((256, 4), (512, 1))}
            found = {(r.get("gpu"), r.get("dim"), r.get("kv_heads")) for r in rows}
            complete = (run["command_completed"] and len(rows) == 4 and found == expected and not invalid
                        and all(r.get("tokens") == tokens and r.get("partitions") == parts
                                and r.get("pass") is True and positive(r.get("milliseconds")) for r in rows))
            run.update(status="complete_pass" if complete else "partial_or_failed",
                       two_gpu_shape_inventory_complete=found == expected and len(rows) == 4,
                       invalid_lines=invalid, rows=rows)
            runs.append(run)
            pair[kernel] = run
        for dim, kv in ((256, 4), (512, 1)):
            complete = all(row["status"] == "complete_pass" for row in pair.values())
            entry = {"tokens": tokens, "partitions": parts, "dim": dim, "kv_heads": kv,
                     "eligible": complete}
            if complete:
                times = {k: {r["gpu"]: r["milliseconds"] for r in v["rows"] if r["dim"] == dim}
                         for k, v in pair.items()}
                entry.update(per_gpu=[{"gpu": g, "scalar_ms": times["scalar"][g],
                                       "selected_ms": times["selected"][g],
                                       "speedup": times["scalar"][g] / times["selected"][g]} for g in (0, 1)],
                             mean_ms_speedup=mean(list(times["scalar"].values())) / mean(list(times["selected"].values())),
                             slowest_gpu_speedup=max(times["scalar"].values()) / max(times["selected"].values()))
            comparisons.append(entry)
    return {"complete": all(row["status"] == "complete_pass" for row in runs),
            "runs": runs, "two_gpu_comparisons": comparisons}


def validation(evidence, directory):
    text = evidence.text(directory / "validation.log") or ""
    suites = [dict(zip(("passed", "failed", "ignored", "measured", "filtered_out"), map(int, groups)))
              for groups in re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out", text)]
    full_cpu = any(row["passed"] >= 30 and row["failed"] == row["ignored"] == row["filtered_out"] == 0 for row in suites)
    activation = len(re.findall(r"^PASS: 132 required tensors, logits, and 10 greedy tokens$", text, re.M))
    tokenizer = re.findall(r"^(\d+) tokenizer fixtures passed$", text, re.M)
    chat = re.findall(r"^(\d+) chat fixtures passed$", text, re.M)
    p2p = re.findall(r"^P2P visibility passed: (\d+) changing rounds x (\d+) elements x two directions$", text, re.M)
    errors = [line for line in text.splitlines() if re.search(r"(^FAIL|^Error:|test result: FAILED|panicked at)", line)]
    complete = full_cpu and activation == 6 and tokenizer == ["30"] and chat == ["4"] and ("10000", "3840") in p2p and not errors
    return {"status": "complete_logged_checks" if complete else "partial_or_failed",
            "cpu_test_suites": suites, "full_cpu_suite_at_least_30_tests_passed": full_cpu,
            "tokenizer_fixture_counts": list(map(int, tokenizer)), "chat_fixture_counts": list(map(int, chat)),
            "p2p_records": [{"rounds": int(a), "elements": int(b)} for a, b in p2p],
            "activation_logit_greedy_passes": activation, "expected_activation_passes": 6,
            "errors": errors, "note": "Pass evidence parsed from the full validation log, including unfiltered CPU test counts. No separate validation-process exit/window file is emitted by the driver."}


def chat(evidence, directory):
    runs = {}
    metric = re.compile(r"prefill: (\d+) tokens, ([0-9.eE+-]+) tok/s; decode: (\d+) tokens, ([0-9.eE+-]+) tok/s")
    mtp = re.compile(r"MTP: accepted (\d+)/(\d+) proposals in (\d+) rounds; draft ([0-9.eE+-]+) ms/proposal, verify ([0-9.eE+-]+) ms/round")
    for prompt in PROMPTS:
        reference = evidence.text(directory / f"chat-core-{prompt}.txt")
        for label in ("core", "shared"):
            for draft in (False, True):
                name = f"chat-{label}-{prompt}" + ("-mtp" if draft else "")
                row, output = evidence.run(directory, name, "txt")
                log = evidence.text(directory / (name + ".log")) or ""
                found, proposals = list(metric.finditer(log)), list(mtp.finditer(log))
                row.update(prompt=prompt, label=label, draft=draft,
                           exact_core_ordinary_output=output == reference if output is not None and reference is not None else None)
                if found:
                    pp, pp_rate, tg, tg_rate = found[-1].groups()
                    row.update(input_tokens=int(pp), output_tokens=int(tg), prefill_tok_s=float(pp_rate), decode_tok_s=float(tg_rate))
                if proposals:
                    accepted, proposed, rounds, draft_ms, verify_ms = proposals[-1].groups()
                    row["mtp"] = {"accepted": int(accepted), "proposed": int(proposed), "rounds": int(rounds),
                                  "draft_ms_per_proposal": float(draft_ms), "verify_ms_per_round": float(verify_ms)}
                complete = (row["command_completed"] and found and (not draft or proposals) and output is not None
                            and positive(row.get("input_tokens")) and positive(row.get("output_tokens"))
                            and positive(row.get("prefill_tok_s")) and positive(row.get("decode_tok_s")))
                row["status"] = "complete" if complete else "partial_or_failed"
                runs[name] = row
    parity = evidence.json(directory / "chat-parity.json")
    parity_complete = (isinstance(parity, list) and len(parity) == 12
                       and {r.get("run") for r in parity if isinstance(r, dict)} == set(runs)
                       and all(r.get("matches_core_ordinary") is True for r in parity))
    groups = {}
    for label in ("core", "shared"):
        for draft in (False, True):
            selected = [row for row in runs.values() if row["label"] == label and row["draft"] == draft]
            valid = [r for r in selected if r["status"] == "complete" and positive(r.get("decode_tok_s"))]
            tokens = sum(r["output_tokens"] for r in valid)
            seconds = sum(r["output_tokens"] / r["decode_tok_s"] for r in valid)
            groups[label + ("_mtp" if draft else "_ordinary")] = {
                "complete_runs": len(valid), "expected_runs": 3,
                "all_outputs_exact": len(valid) == 3 and all(r["exact_core_ordinary_output"] for r in valid),
                "tokens": tokens, "seconds_reconstructed_from_rounded_rates": seconds,
                "weighted_decode_tok_s_approx": tokens / seconds if seconds else None,
                "mean_decode_tok_s": mean([r["decode_tok_s"] for r in valid])}
    comparisons = []
    for prompt in PROMPTS:
        for draft in (False, True):
            suffix = "-mtp" if draft else ""
            a, b = [runs[f"chat-{label}-{prompt}{suffix}"] for label in ("core", "shared")]
            eligible = (all(r["status"] == "complete" and r["exact_core_ordinary_output"] is True for r in (a, b))
                        and a.get("input_tokens") == b.get("input_tokens") and a.get("output_tokens") == b.get("output_tokens")
                        and positive(a.get("decode_tok_s")) and positive(b.get("decode_tok_s")))
            comparisons.append({"prompt": prompt, "draft": draft, "eligible": eligible,
                                "decode_speedup": b["decode_tok_s"] / a["decode_tok_s"] if eligible else None})
    return {"complete": parity_complete and all(r["status"] == "complete" and r["exact_core_ordinary_output"] is True for r in runs.values()),
            "parity_inventory_complete_and_exact": parity_complete, "runs": runs, "groups": groups,
            "comparisons": comparisons, "note": "CLI rates are rounded to two decimals. Reconstructed seconds and weighted CLI rates are approximate; saved logs are the rate evidence."}


def memory(snapshots):
    cards, rss = {}, []
    for snap in snapshots:
        phase = snap.get("phase")
        values = [p.get("memory_kib", {}).get("VmRSS") for p in snap.get("processes", [])]
        total = sum(v for v in values if isinstance(v, int)) if any(isinstance(v, int) for v in values) else None
        if total is not None:
            rss.append((phase, total))
        for row in snap.get("cards", []):
            card = cards.setdefault(row["card"], {"device": row.get("device")})
            for metric in ("vram", "vis_vram", "gtt"):
                used = row.get("values", {}).get(f"mem_info_{metric}_used")
                capacity = row.get("values", {}).get(f"mem_info_{metric}_total")
                target = card.setdefault(metric, {})
                if isinstance(capacity, int):
                    target["total_bytes"] = capacity
                if isinstance(used, int):
                    if used > target.get("peak_bytes", -1):
                        target.update(peak_bytes=used, first_peak_phase=phase)
                    if phase in ("ready", "end"):
                        target[phase + "_bytes"] = used
    for card in cards.values():
        for metric in ("vram", "vis_vram", "gtt"):
            row = card[metric]
            if "peak_bytes" in row and "ready_bytes" in row:
                row["growth_from_ready_to_peak_bytes"] = row["peak_bytes"] - row["ready_bytes"]
            if "peak_bytes" in row and positive(row.get("total_bytes")):
                row["peak_fraction_of_capacity"] = row["peak_bytes"] / row["total_bytes"]
    ready = next((v for phase, v in rss if phase == "ready"), None)
    peak = max((v for _, v in rss), default=None)
    return {"cards": cards, "tracked_processes_rss_kib": {"ready": ready, "peak": peak,
             "growth_ready_to_peak": peak - ready if ready is not None and peak is not None else None},
            "snapshot_count": len(snapshots), "note": "Observed snapshots, not continuous maxima. Device counters include other clients; RSS sums only tracked processes. Ready precedes lazy mappings and first command recording."}


def request_summary(row):
    usage = row.get("response", {}).get("usage", {})
    counts = row.get("counts", {})
    output = counts.get("output_tokens", usage.get("output_tokens"))
    seconds = row.get("seconds")
    return {"name": row.get("name"), "seconds": seconds, "first_delta_seconds": row.get("first_delta_seconds"),
            "input_tokens": counts.get("input_tokens", usage.get("input_tokens")),
            "cached_tokens": counts.get("cached_tokens", usage.get("input_tokens_details", {}).get("cached_tokens")),
            "output_tokens": output, "output_tokens_per_wall_second": output / seconds if positive(output) and positive(seconds) else None,
            "http_status": row.get("http_status"), "error": row.get("error")}


def batch_trial(data):
    requests = [request_summary(row) for row in data.get("requests", [])]
    failed = [row for row in data.get("checks", []) if row.get("passed") is not True]
    expected = {"medium", "long_history"} | {f"{name}_{suffix}" for name in
                ("short", "long_default_budget", "stochastic", "greedy_again") for suffix in ("first", "repeat")}
    expected |= {f"varied_{n}" for n in range(1, 9)}
    found = Counter(row["name"] for row in requests)
    inventory = set(found) == expected and all(value == 1 for value in found.values())
    repeats = []
    for row in data.get("repeat_checks", []):
        exact = row.get("exact_output") is True and row.get("same_token_counts") is True
        eligible = (exact and inventory and data.get("protocol_and_cache_passed") is True
                    and positive(row.get("first_seconds")) and positive(row.get("repeat_seconds")))
        repeats.append({key: value for key, value in row.items() if key != "eligible_speedup"})
        repeats[-1].update(eligible=eligible, speedup=row["first_seconds"] / row["repeat_seconds"] if eligible else None)
    relevant = [r for r in requests if r["name"] in ("stochastic_first", "stochastic_repeat", "greedy_again_first", "greedy_again_repeat")]
    all_valid = [r for r in requests if positive(r["seconds"]) and positive(r["output_tokens"])]
    token_total = sum(r["output_tokens"] for r in all_valid)
    seconds_total = sum(r["seconds"] for r in all_valid)
    return {"label": data.get("label"), "batch": data.get("batch"), "draft": data.get("draft"),
            "batch_argument_omitted": data.get("batch_argument_omitted"), "command": data.get("command"),
            "completed": data.get("completed") is True, "protocol_and_cache_passed": data.get("protocol_and_cache_passed") is True,
            "repeat_output_exact": data.get("repeat_output_exact"), "failed_checks": failed,
            "request_count": len(requests), "expected_requests": 18, "requests": requests,
            "request_inventory_complete": inventory, "missing_requests": sorted(expected - set(found)),
            "duplicate_requests": sorted(name for name, count in found.items() if count > 1),
            "repeat_comparisons": repeats, "eligible_repeat_comparisons": sum(r["eligible"] for r in repeats),
            "output_drift_repeat_names": [r.get("name") for r in repeats if r.get("exact_output") is not True or r.get("same_token_counts") is not True],
            "protocol_test": data.get("protocol_test"), "cleanup_remaining": data.get("cleanup", {}).get("remaining"),
            "error": data.get("error"), "memory": memory(data.get("snapshots", [])),
            "short_after_long_history": relevant,
            "aggregate_observed_requests": {"output_tokens": token_total, "wall_seconds": seconds_total,
                "output_tokens_per_wall_second": token_total / seconds_total if seconds_total else None},
            "rate_note": "HTTP output tokens / request wall time includes prefill, protocol, and cache effects; it is not sustained GPU decode throughput."}


def batch_results(evidence, directory, build, requested_run=None):
    shared_hash = (build or {}).get("binaries", {}).get("shared", {}).get("sha256")
    expected_sources = (build or {}).get("source_sha256", {})
    candidates = []
    for path in sorted(directory.glob("*-summary.json")):
        data = evidence.json(path)
        if not isinstance(data, dict):
            continue
        source = data.get("source_sha256", {})
        common = set(source) & set(expected_sources)
        mismatches = sorted(key for key in common if source[key] != expected_sources[key])
        matches = bool(shared_hash) and data.get("binary_sha256") == shared_hash and bool(common) and not mismatches
        candidates.append({"path": path, "data": data, "matching": matches,
                           "mismatches": mismatches, "common_source_files": len(common)})
    matching = [r for r in candidates if r["matching"] and (requested_run is None or r["data"].get("run_id") == requested_run)]
    order = lambda r: (r["data"].get("epoch_started", 0), r["data"].get("run_id", ""))
    complete = [r for r in matching if r["data"].get("completed") is True]
    selected = max(complete or matching, key=order) if complete or matching else None
    inventory = [{"file": r["path"].name, "run_id": r["data"].get("run_id"),
                  "completed": r["data"].get("completed") is True, "matches_selected_build": r["matching"],
                  "source_mismatches": r["mismatches"], "compared_source_files": r["common_source_files"]} for r in candidates]
    result = {"inventory": inventory, "selection": "latest_complete_matching_invocation" if selected in complete and selected else
              "latest_matching_in_progress" if selected else "no_matching_invocation",
              "identity_match_note": "Exact selected binary hash and all overlapping declared source hashes must match. Entire declared identities are retained; files absent from one manifest are listed, not silently verified."}
    if selected is None:
        return result
    data = selected["data"]
    trials, raw = {}, {}
    for label in data.get("labels", []):
        report = evidence.json(directory / f"{data['run_id']}-{label}.json")
        if isinstance(report, dict):
            raw[label] = report
            trials[label] = batch_trial(report)
        else:
            trials[label] = {"completed": False, "status": "missing_or_partial"}
    comparisons = []
    labels = list(raw)
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            indexed = {r.get("name"): r for r in raw[right].get("requests", [])}
            for a in raw[left].get("requests", []):
                b = indexed.get(a.get("name"))
                if not b or "response" not in a or "response" not in b:
                    continue
                same_request = normalize(a.get("request")) == normalize(b.get("request"))
                exact = signature(a["response"]) == signature(b["response"])
                ac, bc = request_summary(a), request_summary(b)
                counts = all(ac[k] == bc[k] and ac[k] is not None for k in ("input_tokens", "output_tokens"))
                eligible = (same_request and exact and counts and positive(a.get("seconds")) and positive(b.get("seconds"))
                            and all(raw[label].get("protocol_and_cache_passed") is True
                                    and trials[label].get("completed")
                                    and trials[label].get("request_inventory_complete") for label in (left, right)))
                comparisons.append({"reference": left, "candidate": right, "name": a.get("name"),
                    "same_request": same_request, "exact_output": exact, "same_token_counts": counts,
                    "eligible": eligible, "reference_seconds": a.get("seconds"), "candidate_seconds": b.get("seconds"),
                    "speedup": a["seconds"] / b["seconds"] if eligible else None})
    result.update(run_id=data.get("run_id"), declared_identity={key: data.get(key) for key in
        ("git_head", "git_status", "binary", "binary_sha256", "source_sha256", "invocation", "model", "assistant")},
        source_files_only_in_shared_manifest=sorted(set(expected_sources) - set(data.get("source_sha256", {}))),
        source_files_only_in_batch_manifest=sorted(set(data.get("source_sha256", {})) - set(expected_sources)),
        completed=data.get("completed") is True, labels=data.get("labels"),
        local_report_inventory_complete=bool(trials) and all(r.get("completed") and r.get("request_inventory_complete") for r in trials.values()),
        protocol_and_cache_passed=data.get("protocol_and_cache_passed"), repeat_output_exact=data.get("repeat_output_exact"),
        trials=trials, cross_configuration_comparisons=comparisons,
        ineligible_comparison_count=sum(not r["eligible"] for r in comparisons))
    return result


def summarize(directory, requested_run=None):
    evidence = Evidence(directory)
    shared = directory / "shared-kv"
    build = evidence.json(shared / "build.json")
    if not isinstance(build, dict):
        build = None
    core_file = evidence.text(directory / "final-binary.sha256")
    core_expected = core_file.split()[0] if core_file and core_file.split() else None
    identity = {"declared_build": build, "core_checkpoint_revision": CORE_REVISION,
                "core_checkpoint_expected_binary_sha256": core_expected,
                "core_binary_matches_checkpoint_artifact": bool(core_expected and build)
                and build.get("binaries", {}).get("core", {}).get("sha256") == core_expected,
                "note": "Checkpoint mapping is declared by bench-shared-kv.py and verified against the retained core binary hash. The selected build includes uncommitted source; its full declared source hash map is the identity, not git HEAD alone."}
    micro = microbenchmarks(evidence, shared)
    valid = validation(evidence, shared)
    decode = bench_comparison(evidence, shared, {label: [f"decode-{label}-{n}" for n in (1, 2)] for label in ("core", "shared")})
    chats = chat(evidence, shared)
    long = bench_comparison(evidence, shared, {label: [f"long-{label}"] for label in ("core", "shared")})
    checks = []
    for tokens in (250, 2300):
        row, data = evidence.run(shared, f"check-{tokens}")
        row.update(metrics=data, status="complete_pass" if row["command_completed"] and isinstance(data, dict)
                   and data.get("pass") is True else "partial_or_failed")
        checks.append(row)
    batches = batch_results(evidence, directory / "batch-defaults", build, requested_run)
    complete_shared = (micro["complete"] and valid["status"] == "complete_logged_checks" and chats["complete"]
                       and all(v["complete"] for group in (decode, long) for v in group["pooled"].values())
                       and all(r["status"] == "complete_pass" for r in checks))
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Saved evidence only. Missing or in-progress files never count as passed; latency gains require matching outputs/counts. No workload or device query is performed.",
            "shared_inventory_complete": complete_shared, "identity": identity, "microbenchmarks": micro,
            "validation": valid, "decode": decode, "chat": chats, "long_context": long, "boundary_checks": checks,
            "batch_defaults": batches, "evidence_files": evidence.files,
            "timing_note": "Recorded .window.json values are exact whole-command wall windows including model loading and cleanup, not exact inference windows. Clock coverage is deliberately not inferred from whole-command windows."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-run-id", help="Inspect one matching invocation rather than auto-selecting the latest complete one")
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory.resolve(), args.batch_run_id), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
