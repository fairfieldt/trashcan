# Fixed-dimension shared-KV candidate

Prepared only; production shader sources are unchanged. `source.json` pins the exact starting
production shaders and candidate files. `baseline/` is a complete standalone copy of the starting
modules, including their common include and license. Compilation and any later numerical/timing
evidence must be recorded separately; preparing this candidate is not an accepted improvement.

The candidate specializes the two modules already selected by the host:

| Module | Query heads | KV heads | Dimension | Heads/group | Partitions | K/V tile floats each |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `attention_partial_heads2.comp` | 8 | 4 | 256 | 2 | 16, 32 | 256 |
| `attention_partial_heads8.comp` | 8 | 1 | 512 | 8 | 16, 32, 64 | 512 |

All dimension arithmetic uses compile-time `DIM`; private q/acc arrays and component loops have
`DIM / 64` entries. The dot-product order, online softmax updates, f16 loads, barriers, partition
mapping and scratch layout are unchanged for these shapes. K and V each use one KV-head plane per
workgroup, so tiles shrink from 512/1024 floats to 256/512. The local workgroup sizes remain 128 and
512 threads. The scalar eight-partition path, merge shader and large-batch prefill are unchanged.

**Shape precondition:** these modules require exactly the dimensions and eight-query-head mappings
above. `selected_attention_shaders` restricts dimension/KV-head combinations, while target,
assistant and attention micro callers currently supply eight query heads. A future caller with a
different head count must not reuse these smaller tiles without rechecking KV-group boundaries.
The candidate has no new runtime guard or host selector changes.

## CPU compilation

Run on a machine with `glslc`; this creates SPIR-V only and never initializes Vulkan or runs GPU
work. Local macOS preparation did not have `glslc` installed. `compile.py` checks every pinned
source hash, compiles both baseline and candidate modules with the same flags as `engine/build.rs`,
and runs `spirv-val` when installed. A missing validator remains explicitly unvalidated in its
JSON; compiler or validator errors exit nonzero. The temporary output directory is printed in the
recorded commands and may be removed after collecting the report.

```sh
python3 results/engine/main-tuning-20261003/shader-candidates/fixed-dim/compile.py > fixed-dim-cpu-compile.json
```

For an isolated incite compile, copy this directory to a new temporary directory and run the same
script there. Do not overwrite `engine/shaders` or the running engine build for this CPU check.
Compare the baseline/candidate SPIR-V sizes and hashes as compiler evidence only; they do not
measure register allocation, GPU cycles, correctness on activations or throughput.

## Later GPU comparison, owned by the root coordinator

First preserve the production three shader files and record the current binary/source hashes.
Compile a baseline binary, then replace only `attention_shared_body.glsl` and the heads2/heads8
wrappers with these candidate files and compile a separate candidate binary. The candidate's
`common.glsl` is a pinned copy for standalone compilation, not an intended production change.
Restore all three production files in a `finally`/trap cleanup and rebuild the restored source;
do not let a failed trial leave the candidate selected accidentally.

Use **`--kernel selected`** for both baseline and candidate. `--kernel scalar` bypasses the modules
under test. Keep the existing selector and trimmed dispatch count. For both binaries, run the
following through `scripts/latency-run.py`, saving command, exit status, source/binary hashes, raw
JSONL and clocks for each invocation:

```sh
sudo -n python3 scripts/latency-run.py BINARY attention-bench --tokens 128 --partitions 16 --iterations 100 --kernel selected
sudo -n python3 scripts/latency-run.py BINARY attention-bench --tokens 768 --partitions 32 --iterations 100 --kernel selected
sudo -n python3 scripts/latency-run.py BINARY attention-bench --tokens 1536 --partitions 64 --iterations 100 --kernel selected
sudo -n python3 scripts/latency-run.py BINARY attention-bench --tokens 8192 --partitions 64 --iterations 100 --kernel selected
```

Each run must contain both GPUs and both dimension checks with `pass: true`. Dimension256 at
64 partitions intentionally falls back to scalar and is a control, not a selected sliding-layer
gain. Pair repeated baseline/candidate runs to detect noise. Reject absent, inconsistent or
negligible gains; a smaller SPIR-V file alone is not a performance result. Retain only if the
measured gains justify full activation/logit/greedy validation and a fresh paired application
benchmark. Existing final-build evidence does not validate this candidate.
