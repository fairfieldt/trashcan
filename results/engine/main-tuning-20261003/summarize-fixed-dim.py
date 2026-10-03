#!/usr/bin/env python3
"""Summarize fixed-dimension evidence only; no workloads, device queries or writes.

The earlier shared-KV report is a separate phase. Missing, truncated, malformed or
copied-in-progress evidence cannot establish completion or an eligible speedup.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HELPER = Path(__file__).with_name("summarize-final.py")
SPEC = importlib.util.spec_from_file_location("saved_shared_evidence", HELPER)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
LABELS = ("shared", "fixed")
CONTEXTS = ((128, 16), (768, 32), (1536, 64), (8192, 64))
SHADERS = ("attention_shared_body.glsl", "attention_partial_heads2.comp", "attention_partial_heads8.comp")
PROMPT_TEXT = dict(zip(shared.PROMPTS, (
    "Explain how a transformer works, step by step, in English.",
    "Explain photosynthesis to a ten-year-old in three short paragraphs.",
    "Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.",
)))


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def reject_constant(text):
    raise ValueError("non-finite JSON constant: " + text)


def finite_int(text):
    value = int(text)
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("JSON integer exceeds finite numeric range")
    return value


class Evidence(shared.Evidence):
    def json(self, path):
        text = self.text(path)
        if text is None:
            return None
        try:
            return json.loads(text, parse_float=finite_float, parse_int=finite_int, parse_constant=reject_constant)
        except ValueError as error:
            self.files[str(Path(path).relative_to(self.directory))].update(status="invalid_or_in_progress", error=str(error))
            return None

    def run(self, directory, name, suffix="json"):
        row, data = super().run(directory, name, suffix)
        window = mapping(row.get("command_window"))
        command = row.get("command")
        valid_window = (shared.positive(window.get("command_epoch_started"))
                        and shared.positive(window.get("command_epoch_finished"))
                        and window["command_epoch_finished"] >= window["command_epoch_started"])
        row["command_completed"] = bool(row["command_completed"] and valid_window
                                         and isinstance(command, list) and command
                                         and all(isinstance(arg, str) for arg in command))
        return row, data


def command_matches(row, build, label, operation, flags, draft=False, prompt=None):
    command = row.get("command")
    path = binary(build, label).get("path")
    if not path or not isinstance(command, list) or not all(isinstance(arg, str) for arg in command) or command.count(path) != 1:
        return False
    start = command.index(path)
    if len(command) <= start + 2 or command[start + 1] != operation or not command[start + 2].endswith(".gguf"):
        return False
    if prompt is not None and (len(command) <= start + 3 or command[start + 3] != prompt):
        return False
    for flag, value in flags.items():
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command) or command[command.index(flag) + 1] != str(value):
            return False
    return (command.count("--draft") == 1 and command.index("--draft") + 1 < len(command)
            and bool(command[command.index("--draft") + 1])) if draft else "--draft" not in command


def invalidate_comparisons(value):
    if isinstance(value, dict):
        if "eligible" in value:
            value["eligible"] = False
        for key in value:
            if "speedup" in key or key == "time_reduction_percent":
                value[key] = None
            else:
                invalidate_comparisons(value[key])
    elif isinstance(value, list):
        for row in value:
            invalidate_comparisons(row)


def mapping(value):
    return value if isinstance(value, dict) else {}


def records(value):
    return value if isinstance(value, list) else []


def binary(build, label):
    return mapping(mapping(mapping(build).get("binaries")).get(label))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_value(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def window_complete(row):
    return (isinstance(row, dict) and row.get("returncode") == 0
            and shared.positive(row.get("epoch_started")) and shared.positive(row.get("epoch_finished"))
            and row["epoch_finished"] >= row["epoch_started"])


def probe(evidence, directory, prior):
    folder = directory / "fixed-dim-probe"
    declared = mapping(evidence.json(folder / "probe.json"))
    candidate_path = directory / "shader-candidates/fixed-dim/source.json"
    candidate = mapping(evidence.json(candidate_path))
    cpu = mapping(evidence.json(folder / "cpu-compile.json"))
    expected_names = {f"{label}-{rep}-{tokens}-p{parts}" for label in ("baseline", "fixed-dim")
                      for rep in (1, 2) for tokens, parts in CONTEXTS}
    metadata = [r for r in records(declared.get("runs")) if isinstance(r, dict)]
    counts = Counter(r.get("name") for r in metadata)
    indexed = {r.get("name"): r for r in metadata}
    runs = {}
    for name in sorted(expected_names):
        match = re.fullmatch(r"(baseline|fixed-dim)-([12])-(\d+)-p(\d+)", name)
        label, rep, tokens, parts = match.groups()
        tokens, parts = int(tokens), int(parts)
        text = evidence.text(folder / (name + ".jsonl"))
        rows, invalid = [], []
        for number, line in enumerate((text or "").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_float=finite_float, parse_int=finite_int, parse_constant=reject_constant)
                if not isinstance(value, dict):
                    raise ValueError("not an object")
                rows.append(value)
            except ValueError:
                invalid.append(number)
        expected = {(gpu, dim, kv) for gpu in (0, 1) for dim, kv in ((256, 4), (512, 1))}
        inventory = len(rows) == 4 and {(r.get("gpu"), r.get("dim"), r.get("kv_heads")) for r in rows} == expected
        valid_rows = inventory and not invalid and all(
            r.get("pass") is True and shared.positive(r.get("milliseconds"))
            and r.get("tokens") == tokens and r.get("partitions") == parts
            and r.get("iterations") == 100 and r.get("kernel") == "selected"
            and r.get("heads_per_group") == (1 if r.get("dim") == 256 and parts == 64 else 2 if r.get("dim") == 256 else 8)
            and r.get("partial_shader") == ("attention_partial64" if r.get("dim") == 256 and parts == 64
                else "attention_partial_heads" + ("2" if r.get("dim") == 256 else "8") + ("" if parts == 32 else f"_{parts}"))
            for r in rows)
        meta = indexed.get(name)
        command = mapping(meta).get("command")
        expected_binary = "target/release/trashcan-shared-accepted" if label == "baseline" else "target/release/trashcan-fixed-dim"
        command_valid = isinstance(command, list) and all(isinstance(x, str) for x in command)
        command_valid = command_valid and command.count(expected_binary) == 1 and command.index(expected_binary) + 1 < len(command) and command[command.index(expected_binary) + 1] == "attention-bench"
        for flag, value in (("--tokens", tokens), ("--partitions", parts), ("--iterations", 100), ("--kernel", "selected")):
            command_valid = command_valid and command.count(flag) == 1 and command.index(flag) + 1 < len(command) and command[command.index(flag) + 1] == str(value)
        complete = valid_rows and counts[name] == 1 and window_complete(meta) and command_valid
        runs[name] = {"label": label, "repetition": int(rep), "tokens": tokens, "partitions": parts,
                      "command_record": meta, "status": "complete_pass" if complete else "partial_or_failed",
                      "row_inventory_complete": inventory, "invalid_lines": invalid, "rows": rows}
    comparisons = []
    for tokens, parts in CONTEXTS:
        for dim, kv in ((256, 4), (512, 1)):
            repetitions = []
            for rep in (1, 2):
                a, b = [runs[f"{label}-{rep}-{tokens}-p{parts}"] for label in ("baseline", "fixed-dim")]
                eligible = a["status"] == b["status"] == "complete_pass"
                row = {"repetition": rep, "eligible": eligible}
                if eligible:
                    times = {label: {r["gpu"]: r["milliseconds"] for r in run["rows"] if r["dim"] == dim}
                             for label, run in (("baseline", a), ("fixed", b))}
                    means = {label: shared.mean(list(values.values())) for label, values in times.items()}
                    row.update(baseline_mean_ms=means["baseline"], fixed_mean_ms=means["fixed"],
                               speedup=means["baseline"] / means["fixed"],
                               time_reduction_percent=(1 - means["fixed"] / means["baseline"]) * 100,
                               per_gpu=[{"gpu": g, "baseline_ms": times["baseline"][g], "fixed_ms": times["fixed"][g],
                                         "speedup": times["baseline"][g] / times["fixed"][g]} for g in (0, 1)])
                repetitions.append(row)
            eligible = all(r["eligible"] for r in repetitions)
            a = shared.mean([r["baseline_mean_ms"] for r in repetitions]) if eligible else None
            b = shared.mean([r["fixed_mean_ms"] for r in repetitions]) if eligible else None
            comparisons.append({"tokens": tokens, "partitions": parts, "dim": dim, "kv_heads": kv,
                                "role": "scalar_control" if dim == 256 and parts == 64 else "selected_shared_kernel",
                                "eligible": eligible, "repetitions": repetitions, "baseline_mean_ms": a,
                                "fixed_mean_ms": b, "speedup": a / b if eligible else None,
                                "time_reduction_percent": (1 - b / a) * 100 if eligible else None})
    candidate_hashes = mapping(candidate.get("candidate_sha256"))
    source_checks = {}
    for name, expected_hash in candidate_hashes.items():
        path = candidate_path.parent / Path(name).name
        evidence.text(path)
        actual = evidence.files.get(str(path.relative_to(directory)), {}).get("sha256")
        source_checks[name] = bool(expected_hash) and actual == expected_hash
    shader_match = all(declared.get("candidate_shader_sha256", {}).get(name) == candidate_hashes.get(name)
                       and bool(candidate_hashes.get(name)) for name in SHADERS)
    prior_hash = binary(prior, "shared").get("sha256")
    baseline_match = bool(prior_hash) and declared.get("baseline_binary_sha256") == prior_hash
    restored = (declared.get("restored_source_exact") is True and declared.get("restored_binary_identical") is True
                and bool(prior_hash) and declared.get("restored_binary_sha256") == prior_hash)
    cpu_rows = [r for r in records(cpu.get("variants")) if isinstance(r, dict)]
    cpu_inventory = len(cpu_rows) == 4 and {(r.get("variant"), r.get("heads_per_group")) for r in cpu_rows} == {
        (label, heads) for label in ("baseline", "fixed-dim") for heads in (2, 8)}
    cpu_source_match = cpu.get("source_manifest_sha256") == evidence.files.get(
        str(candidate_path.relative_to(directory)), {}).get("sha256") and bool(candidate)
    spv_checks = []
    for row in cpu_rows:
        path = folder / "spv" / f"{row.get('variant')}-attention_partial_heads{row.get('heads_per_group')}.spv"
        try:
            before = path.stat()
            actual = digest(path)
            after = path.stat()
            unchanged = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            spv_checks.append({"file": str(path.relative_to(directory)), "sha256": actual,
                               "bytes": after.st_size, "matches": unchanged and actual == row.get("spirv_sha256")
                               and after.st_size == row.get("spirv_bytes")})
        except OSError:
            spv_checks.append({"file": str(path.relative_to(directory)), "matches": False})
    cpu_complete = (cpu_inventory and cpu_source_match and all(r.get("compile_passed") is True for r in cpu_rows)
                    and all(r["matches"] for r in spv_checks))
    inventory = set(counts) == expected_names and all(v == 1 for v in counts.values())
    row_count = sum(len(r["rows"]) for r in runs.values())
    checks_complete = inventory and all(r["status"] == "complete_pass" for r in runs.values())
    identity_complete = baseline_match and restored and shader_match and bool(source_checks) and all(source_checks.values())
    complete = checks_complete and identity_complete and cpu_complete and declared.get("completed") is True
    if not complete:
        invalidate_comparisons(comparisons)
    return {"complete": complete,
            "checks_complete": checks_complete, "expected_runs": 16, "expected_checks": 64, "recorded_checks": row_count,
            "passing_rows": sum(r.get("pass") is True for run in runs.values() for r in run["rows"]),
            "missing_runs": sorted(expected_names - set(counts)), "extra_runs": sorted(set(counts) - expected_names),
            "duplicate_runs": sorted(k for k, v in counts.items() if v > 1), "declared_identity": declared,
            "baseline_matches_prior_shared_build": baseline_match, "restoration_exact": restored,
            "candidate_shader_manifest_matches": shader_match, "candidate_source_hash_checks": source_checks,
            "cpu_compilation": {"complete": cpu_complete, "declared": cpu, "source_manifest_matches": cpu_source_match,
                                "spirv_file_checks": spv_checks,
                                "separate_spirv_validator_available": bool(cpu.get("validator")),
                                "separate_spirv_validation_complete": cpu_inventory and all(r.get("spirv_validation_passed") is True for r in cpu_rows)},
            "runs": runs, "comparisons": comparisons,
            "note": "Means average two GPUs and two equal-iteration repetitions. Payload/cache micro results do not establish application gains. Scalar controls are separate."}


def benchmarks(evidence, folder, names, pp, tg, build):
    runs = {label: [shared.bench(evidence, folder, name) for name in group] for label, group in names.items()}
    for label, group in runs.items():
        for row in group:
            metrics = mapping(row.get("metrics"))
            row["expected_workload_matches"] = metrics.get("pp") == pp and metrics.get("tg") == tg
            row["command_matches_build_and_workload"] = command_matches(row, build, label, "bench", {"--pp": pp, "--tg": tg, "--batch": 512})
            if row["command_matches_build_and_workload"]:
                row["model_path"] = row["command"][row["command"].index(binary(build, label)["path"]) + 2]
            if row["status"] == "complete" and not (row["expected_workload_matches"] and row["command_matches_build_and_workload"]):
                row["status"] = "unexpected_workload"
    groups = {label: shared.pooled(group) for label, group in runs.items()}
    matching_model = len({r.get("model_path") for group in runs.values() for r in group}) == 1 and all(
        r.get("model_path") for group in runs.values() for r in group)
    valid = all(g["complete"] for g in groups.values()) and bool(matching_model)
    return {"complete": valid, "runs": runs, "pooled": groups,
            "matching_model_paths": bool(matching_model),
            "comparison": {phase + "_speedup": groups["fixed"][phase]["weighted_tok_s"] / groups["shared"][phase]["weighted_tok_s"]
                           if valid else None for phase in ("pp", "tg")},
            "note": "Exact phase seconds; pooled throughput is total tokens / total seconds. Two labels belong to this fresh phase only."}


def chat(evidence, folder, build):
    metric = re.compile(r"prefill: (\d+) tokens, ([0-9.eE+-]+) tok/s; decode: (\d+) tokens, ([0-9.eE+-]+) tok/s")
    mtp = re.compile(r"MTP: accepted (\d+)/(\d+) proposals in (\d+) rounds; draft ([0-9.eE+-]+) ms/proposal, verify ([0-9.eE+-]+) ms/round")
    runs = {}
    for prompt in shared.PROMPTS:
        reference = evidence.text(folder / f"chat-shared-{prompt}.txt")
        for label in LABELS:
            for draft in (False, True):
                name = f"chat-{label}-{prompt}" + ("-mtp" if draft else "")
                row, output = evidence.run(folder, name, "txt")
                log = evidence.text(folder / (name + ".log")) or ""
                found, proposals = list(metric.finditer(log)), list(mtp.finditer(log))
                row.update(prompt=prompt, label=label, draft=draft,
                           exact_shared_ordinary_output=output == reference if output is not None and reference is not None else None)
                if found:
                    pp, pp_rate, tg, tg_rate = found[-1].groups()
                    row.update(input_tokens=int(pp), output_tokens=int(tg), prefill_tok_s=float(pp_rate), decode_tok_s=float(tg_rate))
                if proposals:
                    a, p, r, d, v = proposals[-1].groups()
                    row["mtp"] = {"accepted": int(a), "proposed": int(p), "rounds": int(r),
                                  "draft_ms_per_proposal": float(d), "verify_ms_per_round": float(v)}
                row["command_matches_build_and_workload"] = command_matches(row, build, label, "chat",
                    {"--batch": 128, "--temperature": 0, "--max-tokens": 128}, draft, PROMPT_TEXT[prompt])
                if row["command_matches_build_and_workload"]:
                    row["model_path"] = row["command"][row["command"].index(binary(build, label)["path"]) + 2]
                    row["assistant_path"] = row["command"][row["command"].index("--draft") + 1] if draft else None
                complete = (row["command_completed"] and row["command_matches_build_and_workload"] and found and (not draft or proposals) and output is not None
                            and row.get("output_tokens") == 128 and all(shared.positive(row.get(k)) for k in
                                ("input_tokens", "output_tokens", "prefill_tok_s", "decode_tok_s")))
                row["status"] = "complete" if complete else "partial_or_failed"
                runs[name] = row
    parity = evidence.json(folder / "chat-parity.json")
    parity_complete = (isinstance(parity, list) and len(parity) == 12 and all(isinstance(r, dict) for r in parity)
                       and {r.get("run") for r in parity} == set(runs)
                       and all(r.get("matches_shared_ordinary") is True for r in parity))
    groups, comparisons = {}, []
    for label in LABELS:
        for draft in (False, True):
            rows = [r for r in runs.values() if r["label"] == label and r["draft"] == draft and r["status"] == "complete"]
            tokens = sum(r["output_tokens"] for r in rows)
            seconds = sum(r["output_tokens"] / r["decode_tok_s"] for r in rows)
            groups[label + ("_mtp" if draft else "_ordinary")] = {
                "complete_runs": len(rows), "expected_runs": 3, "all_outputs_exact": len(rows) == 3 and all(r["exact_shared_ordinary_output"] is True for r in rows),
                "tokens": tokens, "seconds_reconstructed_from_rounded_rates": seconds,
                "weighted_decode_tok_s_approx": tokens / seconds if seconds else None}
    for prompt in shared.PROMPTS:
        for draft in (False, True):
            a, b = [runs[f"chat-{label}-{prompt}" + ("-mtp" if draft else "")] for label in LABELS]
            eligible = all(r["status"] == "complete" and r["exact_shared_ordinary_output"] is True for r in (a, b))
            eligible = eligible and a.get("input_tokens") == b.get("input_tokens") and a.get("output_tokens") == b.get("output_tokens")
            eligible = eligible and bool(a.get("model_path")) and a.get("model_path") == b.get("model_path") and a.get("assistant_path") == b.get("assistant_path")
            comparisons.append({"prompt": prompt, "draft": draft, "eligible": eligible,
                                "decode_speedup": b["decode_tok_s"] / a["decode_tok_s"] if eligible else None})
    return {"complete": parity_complete and all(r["status"] == "complete" and r["exact_shared_ordinary_output"] is True for r in runs.values()) and all(r["eligible"] for r in comparisons),
            "expected_runs": 12, "parity_inventory_complete_and_exact": parity_complete, "runs": runs,
            "groups": groups, "comparisons": comparisons,
            "note": "CLI rates are rounded to two decimals; reconstructed seconds and pooled rates are approximate. Output equality is checked independently from saved text."}


def servers(evidence, folder, build, run_id):
    # Reuse the prior parser with the selected fixed binary as its comparison identity.
    adapter = dict(mapping(build))
    adapter["binaries"] = {"shared": binary(build, "fixed")}
    result = shared.batch_results(evidence, folder / "batch-defaults", adapter, run_id)
    if "source_files_only_in_shared_manifest" in result:
        result["source_files_only_in_fixed_manifest"] = result.pop("source_files_only_in_shared_manifest")
    trials = mapping(result.get("trials"))
    required = ("512", "512-mtp")
    acceptance = {}
    independent = {}
    for label in required:
        row = mapping(trials.get(label))
        raw = mapping(evidence.json(folder / "batch-defaults" / f"{result.get('run_id')}-{label}.json")) if result.get("run_id") else {}
        requests = [r for r in records(raw.get("requests")) if isinstance(r, dict)]
        indexed = {r.get("name"): r for r in requests}
        command = raw.get("command")
        draft = label == "512-mtp"
        declared_binary = mapping(result.get("declared_identity")).get("binary")
        expected_path = binary(build, "fixed").get("path")
        path_matches = isinstance(declared_binary, str) and isinstance(expected_path, str) and (declared_binary == expected_path or declared_binary.endswith("/" + expected_path))
        command_ok = (isinstance(command, list) and "--batch" not in command and path_matches and command.count(declared_binary) == 1
                      and (("--draft" in command) == draft))
        request_valid = []
        for request in requests:
            response = mapping(request.get("response"))
            usage, counts = mapping(response.get("usage")), mapping(request.get("counts"))
            actual = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                      "cached_tokens": mapping(usage.get("input_tokens_details")).get("cached_tokens")}
            valid = (request.get("http_status") == 200 and not request.get("error") and response.get("error") is None
                     and isinstance(request.get("request"), dict) and bool(request["request"])
                     and response.get("status") in ("completed", "incomplete") and isinstance(response.get("output"), list)
                     and shared.positive(request.get("seconds")) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in actual.values())
                     and actual["input_tokens"] > 0 and actual["output_tokens"] > 0
                     and actual["cached_tokens"] <= actual["input_tokens"]
                     and all(counts.get(key) == value for key, value in actual.items()))
            request_valid.append(valid)
        pairs = []
        for name in ("short", "long_default_budget", "stochastic", "greedy_again"):
            a, b = [mapping(indexed.get(name + suffix)) for suffix in ("_first", "_repeat")]
            ar, br = mapping(a.get("response")), mapping(b.get("response"))
            same_request = {k: v for k, v in mapping(a.get("request")).items() if k != "stream"} == {
                k: v for k, v in mapping(b.get("request")).items() if k != "stream"}
            exact = bool(ar and br) and shared.signature(ar) == shared.signature(br)
            counts = all(mapping(ar.get("usage")).get(k) == mapping(br.get("usage")).get(k)
                         and mapping(ar.get("usage")).get(k) is not None for k in ("input_tokens", "output_tokens"))
            pairs.append({"name": name, "same_request_except_stream": bool(a and b) and same_request,
                          "exact_output": exact, "same_token_counts": counts})
        long_first, long_repeat = [mapping(indexed.get("long_default_budget_" + side)) for side in ("first", "repeat")]
        lr = mapping(mapping(long_repeat.get("response")).get("usage"))
        long_valid = (all(mapping(mapping(r.get("response")).get("usage")).get("output_tokens") == 256
                          and "max_output_tokens" not in mapping(r.get("request")) for r in (long_first, long_repeat))
                      and isinstance(lr.get("input_tokens"), int) and mapping(lr.get("input_tokens_details")).get("cached_tokens") == lr["input_tokens"] - int(draft))
        snapshots = [r for r in records(raw.get("snapshots")) if isinstance(r, dict)]
        expected_phases = {"ready", "end"} | {"after_" + str(r.get("name")) for r in requests}
        phase_counts = Counter(r.get("phase") for r in snapshots)
        missing_memory = []
        for snapshot in snapshots:
            cards = records(snapshot.get("cards"))
            if len({r.get("card") for r in cards if isinstance(r, dict)}) < 2:
                missing_memory.append({"phase": snapshot.get("phase"), "missing": "two cards"})
            for card in cards:
                for kind in ("vram", "vis_vram", "gtt"):
                    for metric in ("used", "total"):
                        key = f"mem_info_{kind}_{metric}"
                        if not isinstance(mapping(card.get("values")).get(key), int):
                            missing_memory.append({"phase": snapshot.get("phase"), "card": card.get("card"), "missing": key})
            if not any(isinstance(mapping(p.get("memory_kib")).get("VmRSS"), int) for p in records(snapshot.get("processes"))):
                missing_memory.append({"phase": snapshot.get("phase"), "missing": "tracked process VmRSS"})
        memory_complete = set(phase_counts) == expected_phases and all(n == 1 for n in phase_counts.values()) and not missing_memory
        checks = records(raw.get("checks"))
        repeat_names = Counter(r.get("name") for r in records(raw.get("repeat_checks")) if isinstance(r, dict))
        independent[label] = {"repeat_comparisons": pairs, "long_default_budget_and_cache_valid": long_valid,
                              "memory_inventory_complete": memory_complete, "missing_memory": missing_memory,
                              "invalid_response_names": [r.get("name") for r, valid in zip(requests, request_valid) if not valid]}
        acceptance[label] = {
            "completed": row.get("completed") is True,
            "request_inventory_complete": row.get("request_inventory_complete") is True,
            "default_batch_exercised": row.get("batch_argument_omitted") is True,
            "protocol_and_cache_passed": row.get("protocol_and_cache_passed") is True and row.get("failed_checks") == [],
            "repeat_output_exact": row.get("repeat_output_exact") is True and row.get("output_drift_repeat_names") == [],
            "protocol_suite_passed": mapping(row.get("protocol_test")).get("returncode") == 0,
            "scoped_cleanup_complete": row.get("cleanup_remaining") == [],
            "report_identity_and_command_valid": raw.get("label") == label and raw.get("batch") == 512 and raw.get("draft") is draft and command_ok,
            "responses_independently_valid": len(requests) == 18 and all(request_valid),
            "checks_present_and_passed": bool(checks) and all(isinstance(c, dict) and c.get("passed") is True for c in checks),
            "repeat_inventory_and_equality_valid": set(repeat_names) == {p["name"] for p in pairs} and all(v == 1 for v in repeat_names.values()) and all(
                p["same_request_except_stream"] and p["exact_output"] and p["same_token_counts"] for p in pairs),
            "long_default_budget_and_cache_valid": long_valid,
            "memory_inventory_complete": memory_complete,
            "no_report_error": not raw.get("error"),
        }
    label_inventory = result.get("labels") is not None and Counter(result["labels"]) == Counter(required)
    complete = (result.get("completed") is True and label_inventory
                and result.get("protocol_and_cache_passed") is True and result.get("repeat_output_exact") is True
                and all(all(flags.values()) for flags in acceptance.values()))
    result.update(required_labels=list(required), required_label_inventory_complete=label_inventory, acceptance=acceptance,
                  independent_checks=independent,
                  complete=complete)
    if not complete:
        invalidate_comparisons(result)
    return result


def summarize(directory, run_id=None):
    evidence = Evidence(directory)
    errors = []

    def safe(name, function):
        try:
            return function()
        except (KeyError, TypeError, ValueError, AttributeError, ArithmeticError, OSError) as error:
            errors.append({"section": name, "error": type(error).__name__ + ": " + str(error)})
            return {"complete": False, "status": "malformed_or_in_progress", "error": errors[-1]["error"]}

    folder = directory / "fixed-dim"
    build = mapping(evidence.json(folder / "build.json"))
    prior = mapping(evidence.json(directory / "shared-kv/build.json"))
    micro = safe("probe", lambda: probe(evidence, directory, prior))
    sources, previous = mapping(build.get("source_sha256")), mapping(prior.get("source_sha256"))
    probe_identity = mapping(micro.get("declared_identity"))
    fixed_hash, baseline_hash = binary(build, "fixed").get("sha256"), binary(build, "shared").get("sha256")
    selected_shader_match = bool(sources) and all(sources.get("engine/shaders/" + name) ==
        mapping(probe_identity.get("candidate_shader_sha256")).get(name) and sources.get("engine/shaders/" + name) for name in SHADERS)
    baseline_match = bool(baseline_hash) and baseline_hash == binary(prior, "shared").get("sha256")
    identity = {"declared_build": build or None, "prior_shared_build": prior or None,
                "shared_baseline_matches_prior_build": baseline_match,
                "fixed_binary_matches_probe": bool(fixed_hash) and fixed_hash == probe_identity.get("candidate_binary_sha256"),
                "fixed_shader_sources_match_probe": selected_shader_match,
                "changed_source_files_from_shared": [{"path": key, "shared_sha256": previous[key], "fixed_sha256": sources[key]}
                    for key in sorted(set(sources) & set(previous)) if sources[key] != previous[key]],
                "source_files_only_in_fixed": sorted(set(sources) - set(previous)),
                "source_files_only_in_shared": sorted(set(previous) - set(sources)),
                "complete": sha256_value(fixed_hash) and baseline_match and bool(sources) and all(sha256_value(v) for v in sources.values()) and selected_shader_match,
                "note": "The probe and application build are identified separately. Shader hashes must match; a different final binary is not silently attributed probe-wide identity. Fresh application validation establishes the final binary's evidence."}
    valid = safe("validation", lambda: shared.validation(evidence, folder))
    decode = safe("decode", lambda: benchmarks(evidence, folder, {l: [f"decode-{l}-{r}" for r in (1, 2)] for l in LABELS}, 512, 256, build))
    chats = safe("chat", lambda: chat(evidence, folder, build))
    long = safe("long", lambda: benchmarks(evidence, folder, {l: [f"long-{l}"] for l in LABELS}, 4096, 128, build))
    for section in (decode, chats, long):
        section["identity_verified"] = identity["complete"]
        if not identity["complete"]:
            invalidate_comparisons(section)
    checks = []
    for tokens, batch in ((250, 128), (2300, 512)):
        row, raw = evidence.run(folder, f"check-{tokens}")
        data = mapping(raw)
        command_ok = command_matches(row, build, "fixed", "check-batches", {"--tokens": tokens, "--batch": batch}, True)
        passed = (row["command_completed"] and command_ok and data.get("pass") is True and data.get("tokens") == tokens
                  and data.get("batch") == batch and data.get("greedy_tokens") == 64
                  and data.get("serial_chunked_greedy_common_prefix") == 64 and data.get("top64_overlap") == 64
                  and isinstance(data.get("relative_logit_error"), (int, float))
                  and 0 <= data["relative_logit_error"] < 0.05)
        row.update(status="complete_pass" if passed else "partial_or_failed", metrics=raw, command_matches_build_and_workload=command_ok)
        checks.append(row)
    batches = safe("batch_defaults", lambda: servers(evidence, folder, build, run_id))
    application_complete = (identity["complete"] and valid.get("status") == "complete_logged_checks"
                            and all(x.get("complete") is True for x in (decode, chats, long))
                            and all(r["status"] == "complete_pass" for r in checks))
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "summarizer_sha256": digest(Path(__file__)), "parsing_helper_sha256": digest(HELPER),
            "scope": "Local saved evidence only; no GPU/device queries or workloads. Earlier 325b/shared-KV comparisons remain a separate phase.",
            "probe_inventory_complete": micro.get("complete") is True,
            "application_inventory_complete": application_complete,
            "server_acceptance_complete": batches.get("complete") is True,
            "all_recorded_suites_complete": application_complete and micro.get("complete") is True and batches.get("complete") is True and not errors,
            "identity": identity, "probe": micro, "validation": valid, "decode": decode, "chat": chats,
            "long_context": long, "boundary_checks": checks, "batch_defaults": batches,
            "parse_errors": errors, "evidence_files": evidence.files,
            "timing_note": "Whole-command windows include loading and cleanup. They do not delimit inference or establish sustained clocks. Missing/copied-in-progress files never count as passed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--batch-run-id", help="Inspect a specific matching latest-binary server invocation")
    args = parser.parse_args()
    result = summarize(args.directory.resolve(), args.batch_run_id)
    nonfinite = []

    def clean(value, path=""):
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite.append(path)
            return None
        if isinstance(value, dict):
            return {key: clean(item, path + "." + key) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item, path + f"[{n}]") for n, item in enumerate(value)]
        return value

    result = clean(result)
    if nonfinite:
        result["parse_errors"].append({"section": "numeric_output", "nonfinite_paths": nonfinite})
        invalidate_comparisons(result)
        for key in ("probe_inventory_complete", "application_inventory_complete", "server_acceptance_complete", "all_recorded_suites_complete"):
            result[key] = False
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
