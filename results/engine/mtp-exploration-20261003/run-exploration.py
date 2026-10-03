#!/usr/bin/env python3
"""Run extra live draft depths and long-context micro checks under incite flock."""
from pathlib import Path
import shutil
import subprocess

root = Path("results/engine/mtp-exploration-20261003")
model = Path.home() / "models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf"
assistant = Path.home() / "models/gemma-4-12b-assistant/model.safetensors"
prompts = {
    "transformer": "Explain how a transformer works, step by step, in English.",
    "photosynthesis": "Explain photosynthesis to a ten-year-old in three short paragraphs.",
    "csv": "Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.",
}

def run(label, args, binary="final"):
    command = ["sudo", "-n", "python3", "scripts/latency-run.py", str(root / "bin" / binary), *map(str, args)]
    with (root / f"explore-{label}.out").open("w") as out, (root / f"explore-{label}.log").open("w") as err:
        subprocess.run(command, stdout=out, stderr=err, check=True)
    print(f"PASS {label}", flush=True)

shutil.copy2("target/release/trashcan", root / "bin/final")
for batch, tensor, kernel in [(2, "blk.0.ffn_gate.weight", None),
                              (3, "blk.0.ffn_gate.weight", None),
                              (8, "token_embd.weight", None),
                              (4, "blk.0.ffn_gate.weight", "gemm")]:
    args = ["matvec-bench", model, "--batch", batch, "--tensor", tensor, "--iterations", 20]
    if kernel:
        args += ["--kernel", kernel]
    run(f"final-micro-b{batch}-{kernel or 'selected'}", args, "final")

for rep in [1, 2]:
    depths = [1, 2, 4] if rep == 1 else [4, 2, 1]
    for name, prompt in prompts.items():
        for depth in depths:
            label = f"{name}-d{depth}-{rep}"
            run(label, ["chat", model, prompt, "--batch", 512, "--temperature", 0,
                        "--max-tokens", 128, "--draft", assistant, "--draft-tokens", depth])
            assert (root / f"explore-{label}.out").read_bytes() == (root / f"candidate-{name}-d0-{rep}.out").read_bytes()
    for tokens in [8192, 80000]:
        for parts in [64, 128]:
            run(f"attention-{tokens}-p{parts}-{rep}", ["attention-bench", "--tokens", tokens,
                                                       "--partitions", parts, "--iterations", 20])

# Same RNG seed and depth before/after also checks the stochastic verifier's live outputs.
for binary in ["baseline", "final"]:
    run(f"stochastic-{binary}", ["chat", model, prompts["transformer"], "--batch", 512,
                                 "--temperature", 0.8, "--seed", 17, "--max-tokens", 128,
                                 "--draft", assistant, "--draft-tokens", 3], binary)
assert (root / "explore-stochastic-baseline.out").read_bytes() == (root / "explore-stochastic-final.out").read_bytes()
print("PASS stochastic baseline/final output byte equality", flush=True)
