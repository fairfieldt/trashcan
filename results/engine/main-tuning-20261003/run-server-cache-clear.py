#!/usr/bin/env python3
"""Run the isolated cache-clear diagnostic; this command DOES use both GPUs."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[2]
    os.chdir(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path,
                        default=root / "target/server-diagnostic-cache-clear/release/trashcan")
    parser.add_argument("--output-dir", type=Path, default=out / "diagnostic-cache-clear-full")
    parser.add_argument("--port", type=int, default=18164)
    parser.add_argument("--graph-policy", choices=["clear-switch", "retain"], default="clear-switch")
    parser.add_argument("--skip-short-followup", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    run = args.output_dir.resolve()
    assert binary.is_file(), "Build the isolated binary first"
    run.mkdir(exist_ok=False)
    model = Path.home() / "models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf"
    command = ["sudo", "-n", "python3", "scripts/latency-run.py", "env",
               "TRASHCAN_REQUEST_PROFILE=" + str(run),
               "TRASHCAN_CLEAR_GRAPHS_ON_TOP_K_SWITCH=" + ("1" if args.graph_policy == "clear-switch" else "0"),
               str(binary), "serve", str(model), "--batch", "128", "--port", str(args.port),
               "--no-prefix-cache"]
    (run / "run.json").write_text(json.dumps({
        "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "command": command, "graph_policy": args.graph_policy,
        "short_followup_same_process": not args.skip_short_followup,
        "source_manifest": "server-cache-clear-base.json",
        "note": "Full19 profile indices are 1..19; optional short followup profile indices are 20..24. Join by response_id."
    }, indent=2) + "\n")
    logfile = run / "server.log"
    with logfile.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 240
            while "Responses server listening" not in logfile.read_text():
                if process.poll() is not None:
                    raise RuntimeError(logfile.read_text())
                if time.monotonic() >= deadline:
                    raise RuntimeError("Server startup timed out")
                time.sleep(0.5)
            common = ["python3", str(out / "server-memory-diagnostic.py"),
                      "--url", f"http://127.0.0.1:{args.port}", "--server-pid", str(process.pid)]
            with (run / "client.log").open("w") as client_log:
                subprocess.run(common + ["--replay-report", str(out / "prefix-cold.json"),
                                         "--output", str(run / "client.json")],
                               stdout=client_log, stderr=client_log, check=True)
            if not args.skip_short_followup:
                with (run / "short-client.log").open("w") as client_log:
                    subprocess.run(common + ["--output", str(run / "short-client.json")],
                                   stdout=client_log, stderr=client_log, check=True)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
    print(f"Diagnostic complete: {run}", flush=True)


if __name__ == "__main__":
    main()
