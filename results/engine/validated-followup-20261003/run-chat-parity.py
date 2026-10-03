#!/usr/bin/env python3
"""Run on incite from ~/trashcan, while holding ~/trashcan-exp/.lock."""
import json
from pathlib import Path
import subprocess

root = Path("results/engine/validated-followup-20261003")
model = Path.home() / "models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf"
prompts = {
    "photosynthesis": "Explain photosynthesis to a ten-year-old in three short paragraphs.",
    "csv": "Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.",
    "long-water": (
        "The sun heats the surface of the Earth. Water evaporates, condenses into clouds, and falls as rain. " * 80
        + "\nExplain the water cycle in detail, with examples, in English."
    ),
}
report = {}
for name, prompt in prompts.items():
    (root / f"{name}-prompt.txt").write_text(prompt)
    for mode in ["baseline", "merge"]:
        command = [
            "sudo", "-n", "python3", "scripts/latency-run.py",
            str(root / "bin" / mode), "chat", str(model), prompt,
            "--batch", "512", "--temperature", "0", "--max-tokens", "128",
        ]
        with (root / f"{name}-{mode}.txt").open("w") as out, (root / f"{name}-{mode}.log").open("w") as err:
            subprocess.run(command, stdout=out, stderr=err, check=True)
    assert (root / f"{name}-baseline.txt").read_bytes() == (root / f"{name}-merge.txt").read_bytes(), name
    report[name] = {"byte_identical": True, "max_tokens": 128}
    print(f"PASS: {name} baseline/merge greedy output is byte-identical", flush=True)
(root / "chat-parity.json").write_text(json.dumps(report, indent=2) + "\n")
