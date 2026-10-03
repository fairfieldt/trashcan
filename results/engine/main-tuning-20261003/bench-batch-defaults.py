#!/usr/bin/env python3
"""Run sequential batch-default trials on incite; no third-party Python packages.

Run from the repository after GPU work is available:
  python3 results/engine/main-tuning-20261003/bench-batch-defaults.py
  python3 results/engine/main-tuning-20261003/bench-batch-defaults.py \
      --use-default-512 --labels 512 512-mtp

Each invocation gets new timestamped files, so subset reruns preserve old evidence.
Exit 1 means incomplete/protocol/cache failure; exit 2 means repeat-output drift.
Cross-configuration differences are recorded and excluded from speedup claims.
"""
import argparse
import hashlib
import http.client
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABELS = ("32", "128", "512", "512-mtp")
PROMPT = ("List all integers from 1 to 1000 in order, separated by spaces. "
          "Print only the full list, with no introduction, summary, or abbreviations.")
SHORT = "Explain how a transformer works, step by step, in English."


def notes_prompt(count):
    # Exactly the bench-prefix-cache.py builder, with a configurable repetition count.
    return ("Reference notes:\n" + "The workshop has red doors and blue windows. " * count
            + "\n" + PROMPT)


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items() if key not in ("id", "call_id")}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def signature(response):
    return normalize({key: response[key] for key in ("status", "incomplete_details", "output")})


def output_text(response):
    return "".join(part["text"] for item in response["output"] if item["type"] == "message"
                   for part in item["content"])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    # incite is a synchronized deployment, not necessarily a Git checkout.
    # Exact source and binary hashes below identify the tested build either way.
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def proc_table():
    """Read Linux identities, including start ticks to guard against PID reuse."""
    result = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            result[int(entry.name)] = {"pid": int(entry.name), "ppid": int(fields[1]),
                                       "pgrp": int(fields[2]), "session": int(fields[3]),
                                       "start_ticks": int(fields[19]), "state": fields[0]}
        except (OSError, ValueError, IndexError):
            continue
    return result


def descendants(table, root_pid):
    owned = {root_pid} if root_pid in table else set()
    while True:
        added = {pid for pid, row in table.items() if row["ppid"] in owned}
        if added <= owned:
            return owned
        owned |= added


def alive(identity):
    current = proc_table().get(identity["pid"])
    return (current is not None and current["start_ticks"] == identity["start_ticks"]
            and current["state"] != "Z")


class Resident:
    def __init__(self, command, logfile, port):
        self.command, self.logfile, self.port = command, logfile, port
        self.process = None
        self.log = None
        self.identities = {}

    def snapshot(self, phase):
        table = proc_table()
        pids = descendants(table, self.process.pid) if self.process else set()
        pids |= {pid for pid, old in self.identities.items()
                 if pid in table and table[pid]["start_ticks"] == old["start_ticks"]}
        processes = []
        for pid in sorted(pids):
            identity = table[pid]
            self.identities[pid] = identity
            row = dict(identity)
            try:
                row["command"] = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                status = (Path("/proc") / str(pid) / "status").read_text()
                row["memory_kib"] = {key: int(value) for key, value in
                                     re.findall(r"^(VmRSS|VmSize|VmPeak|VmHWM):\s+(\d+)\s+kB", status, re.M)}
            except OSError as error:
                row["read_error"] = str(error)
            processes.append(row)
        cards = []
        for card in sorted(Path("/sys/class/drm").glob("card*")):
            if not re.fullmatch(r"card\d+", card.name):
                continue
            row = {"card": card.name, "device": str((card / "device").resolve()), "values": {}}
            files = list((card / "device").glob("mem_info_*"))
            files += [card / "device" / name for name in ("pp_dpm_sclk", "pp_dpm_mclk")]
            for path in sorted(files):
                try:
                    value = path.read_text().strip()
                    row["values"][path.name] = int(value) if value.isdigit() else value
                except OSError:
                    pass
            cards.append(row)
        return {"phase": phase, "epoch": time.time(), "cards": cards, "processes": processes}

    def start(self, timeout):
        # Refuse an occupied port rather than accidentally benchmarking another server.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", self.port))
        self.log = self.logfile.open("w")
        self.process = subprocess.Popen(self.command, cwd=ROOT, stdout=self.log, stderr=self.log,
                                        start_new_session=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.snapshot("starting")  # Track the wrapper and its exact descendants for cleanup.
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited {self.process.returncode}; see {self.logfile}")
            if "Responses server listening" in self.logfile.read_text():
                connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                try:
                    connection.request("GET", "/health")
                    response = connection.getresponse()
                    value = json.loads(response.read())
                    if response.status == 200 and value.get("status") == "ok":
                        return
                except (OSError, ValueError, http.client.HTTPException):
                    pass
                finally:
                    connection.close()
            time.sleep(0.25)
        raise TimeoutError(f"server startup exceeded {timeout}s; see {self.logfile}")

    def stop(self):
        result = {"signals": [], "remaining": []}
        if self.process is None:
            if self.log:
                self.log.close()
            return result
        self.snapshot("before_cleanup")

        def send(identity, sig):
            if not alive(identity):
                return
            try:
                os.kill(identity["pid"], sig)
            except ProcessLookupError:
                return
            except PermissionError:
                # Only exact tracked descendants; never broad pkill or service management.
                subprocess.run(["sudo", "-n", "kill", "-" + sig.name.removeprefix("SIG"),
                                str(identity["pid"])], check=True, timeout=10)
            result["signals"].append({"pid": identity["pid"], "signal": sig.name})

        try:
            # Children first lets latency-run.py release QoS and sudo reap its child normally.
            for identity in reversed(list(self.identities.values())):
                send(identity, signal.SIGTERM)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and any(alive(row) for row in self.identities.values()):
                time.sleep(0.1)
            for identity in reversed(list(self.identities.values())):
                send(identity, signal.SIGKILL)
            self.process.wait(timeout=10)
            result["wrapper_returncode"] = self.process.returncode
            result["remaining"] = [row for row in self.identities.values() if alive(row)]
        finally:
            if self.log:
                self.log.close()
        return result


class Trial:
    def __init__(self, args, label, command, prefix):
        self.args, self.label, self.prefix = args, label, prefix
        self.port = args.port + LABELS.index(label)
        self.resident = Resident(command, prefix.with_suffix(".server.log"), self.port)
        self.report = {"label": label, "batch": int(label.split("-")[0]), "draft": label.endswith("-mtp"),
                       "url": f"http://127.0.0.1:{self.port}",
                       "batch_argument_omitted": args.use_default_512 and label.startswith("512"),
                       "command": command, "command_shell": shlex.join(command), "requests": [],
                       "repeat_checks": [], "checks": [], "snapshots": [], "completed": False,
                       "protocol_and_cache_passed": False, "repeat_output_exact": None}

    def save(self):
        write_json(self.prefix.with_suffix(".json"), self.report)

    def check(self, condition, description):
        self.report["checks"].append({"description": description, "passed": bool(condition)})
        if not condition:
            print(f"{self.label}: CHECK FAILED: {description}", flush=True)
        self.save()

    def run(self, name, body, stream=False):
        body = {"temperature": 0, "store": False, **body, "stream": stream}
        item = {"name": name, "request": json.loads(json.dumps(body)), "epoch_started": time.time(),
                "events": [], "first_delta_seconds": None}
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.args.request_timeout)
        started = time.perf_counter()
        try:
            connection.request("POST", "/v1/responses", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            item["http_status"] = response.status
            item["headers_seconds"] = time.perf_counter() - started
            item["headers"] = dict(response.getheaders())
            if response.getheader("Content-Type", "").startswith("text/event-stream"):
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    event = json.loads(line[6:])
                    elapsed = time.perf_counter() - started
                    item["events"].append({"seconds": elapsed, "event": event})
                    if item["first_delta_seconds"] is None and event["type"].endswith(".delta"):
                        item["first_delta_seconds"] = elapsed
                item["seconds"] = time.perf_counter() - started
                events = [record["event"] for record in item["events"]]
                if not events or "response" not in events[-1]:
                    raise AssertionError(f"{name}: missing terminal SSE response")
                item["response"] = events[-1]["response"]
                self.check(all(event.get("sequence_number") == index for index, event in enumerate(events)),
                           f"{name}: SSE sequence numbers")
                self.check(events[-1]["type"] in ("response.completed", "response.incomplete"),
                           f"{name}: SSE terminal event")
                done = [event["item"] for event in events if event["type"] == "response.output_item.done"]
                self.check(done == item["response"]["output"], f"{name}: SSE completed items")
                delta = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
                self.check(delta == output_text(item["response"]), f"{name}: SSE text deltas")
            else:
                item["response"] = json.loads(response.read())
                item["seconds"] = time.perf_counter() - started
        except Exception as error:
            item["error"] = repr(error)
            raise
        finally:
            item.setdefault("seconds", time.perf_counter() - started)
            connection.close()
            self.report["requests"].append(item)
            self.report["snapshots"].append(self.resident.snapshot("after_" + name))
            self.save()
        value = item["response"]
        self.check(item["http_status"] == 200, f"{name}: HTTP 200")
        usage = value["usage"]
        cached = usage["input_tokens_details"]["cached_tokens"]
        item["counts"] = {"input_tokens": usage["input_tokens"], "cached_tokens": cached,
                          "output_tokens": usage["output_tokens"]}
        self.check(0 <= cached <= usage["input_tokens"], f"{name}: valid cached-token count")
        self.check(usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"],
                   f"{name}: total usage")
        print(f"{self.label} {name}: {item['seconds']:.3f}s, cached {cached}/{usage['input_tokens']}, "
              f"output {usage['output_tokens']}, first delta {item['first_delta_seconds']}", flush=True)
        return item

    def repeat(self, name, body):
        first = self.run(name + "_first", body)
        again = self.run(name + "_repeat", body, stream=True)
        exact = signature(first["response"]) == signature(again["response"])
        same_count = all(first["counts"][key] == again["counts"][key]
                         for key in ("input_tokens", "output_tokens"))
        self.report["repeat_checks"].append({"name": name, "exact_output": exact,
                                             "same_token_counts": same_count,
                                             "first_seconds": first["seconds"], "repeat_seconds": again["seconds"],
                                             "eligible_speedup": first["seconds"] / again["seconds"]
                                             if exact and same_count else None})
        if not exact or not same_count:
            print(f"{self.label} {name}: OUTPUT DIFFERENCE; both responses retained, speedup excluded", flush=True)
        self.save()
        return first, again

    def execute(self):
        try:
            self.save()
            self.resident.start(self.args.startup_timeout)
            self.report["snapshots"].append(self.resident.snapshot("ready"))
            self.save()
            self.repeat("short", {"input": SHORT, "max_output_tokens": 32})
            medium = self.run("medium", {"input": notes_prompt(51), "max_output_tokens": 1})
            self.report["medium_input_tokens"] = medium["counts"]["input_tokens"]
            self.check(medium["counts"]["output_tokens"] == 1, "medium: one output token")

            # Omitting max_output_tokens exercises the production default of 256.
            long_prompt = notes_prompt(190)
            first, again = self.repeat("long_default_budget", {"input": long_prompt})
            for item in (first, again):
                self.check(item["counts"]["input_tokens"] > 1536, item["name"] + ": prompt exceeds largest ring")
                self.check(item["counts"]["output_tokens"] == 256, item["name"] + ": default output budget is 256")
                self.check(item["response"]["status"] == "incomplete"
                           and (item["response"].get("incomplete_details") or {}).get("reason") == "max_output_tokens",
                           item["name"] + ": stopped at default token budget")
            expected = (again["counts"]["input_tokens"] - int(self.report["draft"])) if self.report["batch"] == 512 else 0
            self.check(again["counts"]["cached_tokens"] == expected,
                       f"long_default_budget_repeat: expected {expected} cached tokens")
            history = [{"role": "user", "content": long_prompt},
                       {"role": "assistant", "content": output_text(again["response"])},
                       {"role": "user", "content": "What color are the workshop doors? Reply in one word."}]
            follow = self.run("long_history", {"input": history, "max_output_tokens": 32}, stream=True)
            self.check("red" in output_text(follow["response"]).lower(), "long_history: correctly recalls red doors")

            self.repeat("stochastic", {"input": PROMPT, "temperature": 1, "seed": 7,
                                       "top_k": 32, "max_output_tokens": 32})
            self.repeat("greedy_again", {"input": PROMPT, "max_output_tokens": 32})
            for index, count in enumerate((1, 2, 4, 8, 16, 24, 40, 64), 1):
                self.run(f"varied_{index}", {"input": "Reply with one word: red. Notes: " + "blue " * count,
                                             "max_output_tokens": 1})

            if self.report["batch"] == 512:
                command = [sys.executable, str(ROOT / "scripts/test-server.py"),
                           "--url", f"http://127.0.0.1:{self.port}"]
                self.report["protocol_test"] = {"command": command, "command_shell": shlex.join(command)}
                self.save()
                with self.prefix.with_suffix(".protocol.log").open("w") as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log,
                                            timeout=self.args.protocol_timeout, check=False)
                self.report["protocol_test"]["returncode"] = result.returncode
                self.check(result.returncode == 0, "existing test-server.py suite")
            self.report["completed"] = True
        except Exception as error:
            self.report["error"] = repr(error)
            self.report["traceback"] = traceback.format_exc()
            print(f"{self.label}: FAILED: {error}", flush=True)
        finally:
            self.report["snapshots"].append(self.resident.snapshot("end"))
            try:
                self.report["cleanup"] = self.resident.stop()
                self.check(not self.report["cleanup"]["remaining"], "scoped server cleanup")
            except Exception as error:
                self.report["cleanup_error"] = repr(error)
                self.report["checks"].append({"description": "scoped server cleanup", "passed": False})
            self.report["protocol_and_cache_passed"] = (self.report["completed"]
                                                         and all(row["passed"] for row in self.report["checks"]))
            self.report["repeat_output_exact"] = (len(self.report["repeat_checks"]) == 4
                                                   and all(row["exact_output"] and row["same_token_counts"]
                                                           for row in self.report["repeat_checks"]))
            self.report["memory_summary"] = memory_summary(self.report["snapshots"])
            self.save()
        return self.report


def memory_summary(snapshots):
    cards = {}
    rss = []
    for snapshot in snapshots:
        values = [row.get("memory_kib", {}).get("VmRSS") for row in snapshot["processes"]]
        if any(value is not None for value in values):
            rss.append(sum(value or 0 for value in values))
        for row in snapshot["cards"]:
            card = cards.setdefault(row["card"], {"device": row["device"], "peak_vram_used_bytes": None})
            used = row["values"].get("mem_info_vram_used")
            if isinstance(used, int):
                card["peak_vram_used_bytes"] = max(card["peak_vram_used_bytes"] or 0, used)
                if snapshot["phase"] in ("ready", "end"):
                    card[snapshot["phase"] + "_vram_used_bytes"] = used
            total = row["values"].get("mem_info_vram_total")
            if isinstance(total, int):
                card["vram_total_bytes"] = total
            for metric, sysfs in (("visible_vram", "vis_vram"), ("gtt", "gtt")):
                used = row["values"].get(f"mem_info_{sysfs}_used")
                if isinstance(used, int):
                    key = f"peak_{metric}_used_bytes"
                    card[key] = max(card.get(key, 0), used)
                    if snapshot["phase"] in ("ready", "end"):
                        card[snapshot["phase"] + f"_{metric}_used_bytes"] = used
                total = row["values"].get(f"mem_info_{sysfs}_total")
                if isinstance(total, int):
                    card[f"{metric}_total_bytes"] = total
    return {"cards": cards, "peak_tracked_processes_rss_kib": max(rss) if rss else None,
            "note": "VRAM is whole-device sysfs usage, including other clients; RSS sums this trial's tracked processes."}


def comparisons(reports):
    rows = []
    for left, right in itertools.combinations(reports, 2):
        indexed = {row["name"]: row for row in right["requests"]}
        for a in left["requests"]:
            b = indexed.get(a["name"])
            if b is None or any("counts" not in row for row in (a, b)):
                continue
            same_request = normalize(a["request"]) == normalize(b["request"])
            exact = signature(a["response"]) == signature(b["response"])
            same_count = all(a["counts"][key] == b["counts"][key] for key in ("input_tokens", "output_tokens"))
            eligible = same_request and exact and same_count
            rows.append({"reference": left["label"], "candidate": right["label"], "name": a["name"],
                         "same_request": same_request, "exact_output": exact, "same_token_counts": same_count,
                         "eligible": eligible, "reference_seconds": a["seconds"], "candidate_seconds": b["seconds"],
                         "speedup": a["seconds"] / b["seconds"] if eligible else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--use-default-512", action="store_true", help="Omit --batch for 512 and 512-mtp")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/trashcan")
    parser.add_argument("--model", type=Path, default=Path.home() / "models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf")
    parser.add_argument("--assistant", type=Path, default=Path.home() / "models/gemma-4-12b-assistant/model.safetensors")
    parser.add_argument("--output-dir", type=Path, default=HERE / "batch-defaults")
    parser.add_argument("--port", type=int, default=18170, help="Base port; labels use 18170/18171/18172/18173 by default")
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--protocol-timeout", type=float, default=1200)
    args = parser.parse_args()
    if len(set(args.labels)) != len(args.labels):
        parser.error("--labels must not contain duplicates")
    for name in ("binary", "model", "assistant", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    required = [args.binary, args.model, ROOT / "scripts/latency-run.py", ROOT / "scripts/test-server.py"]
    if "512-mtp" in args.labels:
        required.append(args.assistant)
    for path in required:
        if not path.is_file():
            parser.error(f"required file missing: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    source_paths = sorted((ROOT / "engine/src").glob("*.rs")) + sorted((ROOT / "engine/shaders").glob("*.comp"))
    source_paths += [ROOT / "engine/build.rs", ROOT / "scripts/latency-run.py", ROOT / "scripts/test-server.py", Path(__file__).resolve()]
    summary = {"run_id": run_id, "epoch_started": time.time(), "invocation": sys.argv,
               "labels": args.labels, "use_default_512": args.use_default_512,
               "binary": str(args.binary), "binary_sha256": sha256(args.binary),
               "model": {"path": str(args.model), "bytes": args.model.stat().st_size},
               "assistant": str(args.assistant),
               "git_head": git_output("rev-parse", "HEAD"),
               "git_status": git_output("status", "--short"),
               "git_note": "Null Git metadata means this deployment has no Git checkout; use exact source/binary hashes.",
               "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
               "reports": [], "comparisons": [], "completed": False}
    summary_path = args.output_dir / (run_id + "-summary.json")
    reports = []
    write_json(summary_path, summary)
    try:
        for label in args.labels:
            port = args.port + LABELS.index(label)
            command = ["sudo", "-n", sys.executable, str(ROOT / "scripts/latency-run.py"), str(args.binary),
                       "serve", str(args.model), "--host", "127.0.0.1", "--port", str(port), "--context", "8192"]
            if not (args.use_default_512 and label.startswith("512")):
                command += ["--batch", label.split("-")[0]]
            if label.endswith("-mtp"):
                command += ["--draft", str(args.assistant)]
            prefix = args.output_dir / (run_id + "-" + label)
            trial = Trial(args, label, command, prefix)
            report = trial.execute()
            reports.append(report)
            summary["reports"].append({"label": label, "path": str(prefix.with_suffix(".json")),
                                       "completed": report["completed"],
                                       "protocol_and_cache_passed": report["protocol_and_cache_passed"],
                                       "repeat_output_exact": report["repeat_output_exact"],
                                       "memory_summary": report["memory_summary"]})
            summary["comparisons"] = comparisons(reports)
            write_json(summary_path, summary)
            # Never launch another model if cleanup did not establish that this one stopped.
            if report.get("cleanup_error") or report.get("cleanup", {}).get("remaining"):
                raise RuntimeError(f"{label}: cleanup incomplete; remaining configurations not launched")
        summary["completed"] = True
    finally:
        summary["epoch_finished"] = time.time()
        summary["protocol_and_cache_passed"] = (len(reports) == len(args.labels)
                                                  and all(row["protocol_and_cache_passed"] for row in reports))
        summary["repeat_output_exact"] = len(reports) == len(args.labels) and all(row["repeat_output_exact"] for row in reports)
        summary["cross_config_ineligible_count"] = sum(not row["eligible"] for row in summary["comparisons"])
        write_json(summary_path, summary)
        print(f"\nSummary: {summary_path}", flush=True)
        for row in summary["reports"]:
            print(f"  {row['label']}: complete={row['completed']}, protocol/cache={row['protocol_and_cache_passed']}, "
                  f"repeat output exact={row['repeat_output_exact']}", flush=True)
            for card, memory in row["memory_summary"]["cards"].items():
                used = memory["peak_vram_used_bytes"]
                if used is not None:
                    print(f"    {card}: peak observed VRAM {used / 2**20:.1f} MiB", flush=True)
            rss = row["memory_summary"]["peak_tracked_processes_rss_kib"]
            if rss is not None:
                print(f"    peak tracked process RSS: {rss / 1024:.1f} MiB", flush=True)
        for row in summary["comparisons"]:
            if row["name"] in ("medium", "long_default_budget_first", "long_default_budget_repeat", "long_history"):
                result = f"{row['speedup']:.3f}x" if row["eligible"] else "excluded (request/output/count differs)"
                print(f"  {row['reference']} -> {row['candidate']} {row['name']}: {result}", flush=True)
        print(f"  Cross-configuration comparisons excluded: {summary['cross_config_ineligible_count']}", flush=True)
    if not summary["protocol_and_cache_passed"]:
        return 1
    return 0 if summary["repeat_output_exact"] else 2


def interrupted(_signal, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
