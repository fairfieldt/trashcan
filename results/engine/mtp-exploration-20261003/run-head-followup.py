#!/usr/bin/env python3
"""Measure and validate the head selector after the original depth exploration."""
from pathlib import Path
import subprocess

root = Path("results/engine/mtp-exploration-20261003")
model = Path.home() / "models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf"
assistant = Path.home() / "models/gemma-4-12b-assistant/model.safetensors"
prompts = {
    "transformer": "Explain how a transformer works, step by step, in English.",
    "photosynthesis": "Explain photosynthesis to a ten-year-old in three short paragraphs.",
    "csv": "Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.",
}

def run(label, args, binary="head-tiles"):
    command = ["sudo", "-n", "python3", "scripts/latency-run.py", str(root / "bin" / binary), *map(str, args)]
    with (root / f"head-{label}.out").open("w") as out, (root / f"head-{label}.log").open("w") as err:
        subprocess.run(command, stdout=out, stderr=err, check=True)
    print(f"PASS {label}", flush=True)

for rep in [1, 2]:
    for batch in [5, 6, 7, 8]:
        kernels = ["multivec", "multivec4"] if rep == 1 else ["multivec4", "multivec"]
        for kernel in kernels:
            run(f"micro-b{batch}-{kernel}-{rep}", ["matvec-bench", model, "--tensor", "token_embd.weight",
                                                  "--batch", batch, "--kernel", kernel, "--iterations", 100])
    for name, prompt in prompts.items():
        label = f"{name}-d4-{rep}"
        run(label, ["chat", model, prompt, "--batch", 512, "--temperature", 0,
                    "--max-tokens", 128, "--draft", assistant, "--draft-tokens", 4])
        assert (root / f"head-{label}.out").read_bytes() == (root / f"explore-{label}.out").read_bytes()

# The same seeded stochastic depth exercises the new five-column head path.
for binary in ["final", "head-tiles"]:
    run(f"stochastic-d4-{binary}", ["chat", model, prompts["transformer"], "--batch", 512,
                                   "--temperature", 0.8, "--seed", 17, "--max-tokens", 128,
                                   "--draft", assistant, "--draft-tokens", 4], binary)
assert (root / "head-stochastic-d4-final.out").read_bytes() == (root / "head-stochastic-d4-head-tiles.out").read_bytes()

# Seven proposals exercise the eight-column target head on real activations.
for binary in ["baseline", "head-tiles"]:
    run(f"validate-d7-{binary}", ["validate", model, root / f"head-d7-{binary}",
                                 "--batch", 8, "--draft", assistant, "--draft-tokens", 7, "--greedy", 64], binary)
assert (root / "head-d7-baseline/greedy.txt").read_bytes() == (root / "head-d7-head-tiles/greedy.txt").read_bytes()
for file in (root / "head-d7-baseline").glob("*.f32"):
    assert file.read_bytes() == (root / "head-d7-head-tiles" / file.name).read_bytes(), file.name
run("check-1700-target", ["check-batches", model, "--tokens", 1700, "--batch", 512])
run("default3-confirm", ["chat", model, prompts["transformer"], "--batch", 512,
                         "--temperature", 0, "--max-tokens", 128, "--draft", assistant, "--draft-tokens", 3])
assert (root / "head-default3-confirm.out").read_bytes() == (root / "candidate-transformer-d3-1.out").read_bytes()
print("PASS head selector: stochastic d4 parity, greedy d7 parity, target-only ring wrap and eight-column verification", flush=True)
