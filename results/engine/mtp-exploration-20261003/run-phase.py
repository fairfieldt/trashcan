#!/usr/bin/env python3
"""Run on incite while holding ~/trashcan-exp/.lock; pin each phase's binary."""
from pathlib import Path
import subprocess
import sys

root = Path("results/engine/mtp-exploration-20261003")
phase = sys.argv[1]
model = Path.home() / "models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf"
assistant = Path.home() / "models/gemma-4-12b-assistant/model.safetensors"
prompts = {
    "transformer": "Explain how a transformer works, step by step, in English.",
    "photosynthesis": "Explain photosynthesis to a ten-year-old in three short paragraphs.",
    "csv": "Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.",
}

def run(label, args):
    command = ["sudo", "-n", "python3", "scripts/latency-run.py", str(root / "bin" / phase), *map(str, args)]
    with (root / f"{phase}-{label}.out").open("w") as out, (root / f"{phase}-{label}.log").open("w") as err:
        subprocess.run(command, stdout=out, stderr=err, check=True)
    print(f"DONE {phase} {label}", flush=True)

for rep in [1, 2]:
    run(f"bench-{rep}", ["bench", model, "--pp", 512, "--tg", 256, "--batch", 512,
                         "--profile", root / f"{phase}-bench-{rep}.tsv"])
    for name, prompt in prompts.items():
        for depth in [0, 3]:
            args = ["chat", model, prompt, "--batch", 512, "--temperature", 0, "--max-tokens", 128]
            if depth:
                args += ["--draft", assistant, "--draft-tokens", depth,
                         "--profile", root / f"{phase}-{name}-d{depth}-{rep}.tsv"]
            run(f"{name}-d{depth}-{rep}", args)
        assert (root / f"{phase}-{name}-d0-{rep}.out").read_bytes() == (root / f"{phase}-{name}-d3-{rep}.out").read_bytes()
        if phase != "baseline":
            for depth in [0, 3]:
                assert (root / f"{phase}-{name}-d{depth}-{rep}.out").read_bytes() == (root / f"baseline-{name}-d{depth}-{rep}.out").read_bytes()

for batch in [2, 3, 4]:
    run(f"b{batch}", ["validate", model, root / f"{phase}-b{batch}", "--batch", batch, "--greedy", 16])
    if phase != "baseline":
        baseline = root / f"baseline-b{batch}"
        candidate = root / f"{phase}-b{batch}"
        files = list(baseline.glob("*.f32")) + [baseline / "greedy.txt"]
        for file in files:
            assert file.read_bytes() == (candidate / file.name).read_bytes(), (batch, file.name)
        print(f"PASS b{batch}: {len(files)} files byte-identical", flush=True)
