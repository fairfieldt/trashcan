#!/usr/bin/env python3
"""Measure the unchanged three-draft policy with each pinned binary on incite."""
import json
from pathlib import Path
import re
import subprocess

root = Path("results/engine/validated-followup-20261003")
model = Path.home() / "models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf"
draft = Path.home() / "models/gemma-4-12b-assistant/model.safetensors"
prompts = {
    "transformer": "Explain how a transformer works, step by step, in English.",
    "photosynthesis": "Explain photosynthesis to a ten-year-old in three short paragraphs.",
    "csv": "Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.",
}
report = {}
for name, prompt in prompts.items():
    pair = {}
    for phase in ["baseline", "merge"]:
        out = root / f"mtp-{name}-{phase}.txt"
        log = root / f"mtp-{name}-{phase}.log"
        args = ["sudo", "-n", "python3", "scripts/latency-run.py", str(root / "bin" / phase),
                "chat", str(model), prompt, "--batch", "512", "--temperature", "0",
                "--max-tokens", "128", "--draft", str(draft), "--draft-tokens", "3"]
        with out.open("w") as stdout, log.open("w") as stderr:
            subprocess.run(args, stdout=stdout, stderr=stderr, check=True)
        ordinary = root / (f"{phase}-chat.txt" if name == "transformer" else f"{name}-{phase}.txt")
        assert out.read_bytes() == ordinary.read_bytes(), (name, phase)
        text = log.read_text()
        tokens, rate = re.search(r"decode: (\d+) tokens, ([\d.]+) tok/s", text).groups()
        accepted, proposed, rounds, draft_ms, verify_ms = re.search(
            r"MTP: accepted (\d+)/(\d+) proposals in (\d+) rounds; draft ([\d.]+) ms/proposal, verify ([\d.]+) ms/round", text
        ).groups()
        pair[phase] = {"tokens": int(tokens), "tok_s": float(rate), "accepted": int(accepted),
                       "proposed": int(proposed), "rounds": int(rounds), "draft_ms": float(draft_ms),
                       "verify_ms": float(verify_ms), "ordinary_output_byte_identical": True}
        print(f"PASS: {name} {phase} MTP output matches ordinary; {rate} tok/s", flush=True)
    assert (root / f"mtp-{name}-baseline.txt").read_bytes() == (root / f"mtp-{name}-merge.txt").read_bytes()
    report[name] = pair
(root / "mtp-summary.json").write_text(json.dumps(report, indent=2) + "\n")
