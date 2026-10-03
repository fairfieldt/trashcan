#!/usr/bin/env python3
"""Paired same-binary matrix sweep. Call under the incite flock."""
from pathlib import Path
import subprocess

root = Path("results/engine/mtp-exploration-20261003")
model = Path.home() / "models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf"
tensors = ["blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.attn_q.weight",
           "blk.5.attn_q.weight", "blk.5.attn_k.weight", "blk.0.attn_output.weight", "token_embd.weight"]
with (root / "matrix-micro.jsonl").open("w") as out, (root / "matrix-micro.log").open("w") as err:
    for rep in [1, 2]:
        for tensor in tensors:
            for batch in [2, 3, 4, 5, 6, 7, 8]:
                selected = {2: "multivec4x2", 3: "multivec4x3"}.get(batch, "multivec4")
                kernels = ["multivec4_reference", selected]
                if rep == 2:
                    kernels.reverse()
                for kernel in kernels:
                    args = ["sudo", "-n", "python3", "scripts/latency-run.py", str(root / "bin/candidate"),
                            "matvec-bench", str(model), "--tensor", tensor, "--batch", str(batch),
                            "--kernel", kernel, "--iterations", "100", "--warmup-ms", "500"]
                    subprocess.run(args, stdout=out, stderr=err, check=True)
                    out.flush()
                print(f"PASS rep={rep} tensor={tensor} batch={batch}", flush=True)
