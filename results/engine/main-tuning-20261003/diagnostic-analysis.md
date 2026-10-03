# Resident-server slowdown diagnostic

The full 19-request replay reproduces the slowdown. The measured delay is between GPU segments, not in the shader work. A fresh server running the five short requests does not reproduce it.

| Sequence / request | HTTP s | Prefill ms | Mean forward ms | GPU0 work ms | GPU0 gaps ms | GPU1 gaps ms |
|---|---:|---:|---:|---:|---:|---:|
| cold / stochastic_repeat | 1.713 | 450.2 | 40.51 | 36.00 | 4.10 | 4.00 |
| cold / greedy_repeat | 1.660 | 450.5 | 38.83 | 35.50 | 3.04 | 2.96 |
| cached / stochastic_repeat | 1.210 | 0.0 | 38.87 | 36.00 | 2.64 | 2.55 |
| cached / greedy_repeat | 1.202 | 0.0 | 38.62 | 35.52 | 2.85 | 2.75 |
| cold-full / stochastic_repeat | 1.678 | 452.7 | 39.27 | 35.99 | 2.99 | 2.89 |
| cold-full / greedy_again_first | 4.915 | 595.7 | 139.13 | 35.60 | 97.84 | 96.45 |
| cold-full / greedy_again_repeat | 5.979 | 453.2 | 177.97 | 35.66 | 141.98 | 140.03 |
| cold-full / ring_wrap_first | 14.562 | 13031.5 | 48.61 | 45.45 | 2.91 | 2.81 |
| cold-full / shorter_prompt | 2.144 | 401.8 | 248.13 | 34.46 | 174.82 | 174.67 |
| cold-full / tool_repeat | 2.397 | 791.8 | 133.56 | 35.94 | 97.37 | 95.93 |
| cold-full / tool_continuation | 5.502 | 1348.6 | 197.17 | 36.65 | 160.15 | 158.05 |

GPU values are per card, averaged over ordinary decode forwards; adding the two cards would not represent elapsed time. HTTP includes prefill and protocol work. Exact per-request data, source hashes and clock samples are in `diagnostic-analysis.json`.

All 32 emitted token IDs are identical across the 12 stochastic/greedy responses in the cold, cached and full sequences. All relevant full-sequence clock samples stayed at 850/1370 MHz on both cards; peak sampled temperatures were 82/80°C. The original report showed the same symptom (cold stochastic repeat 1.652 s, greedy repeat 4.523 s).

The final-forward segment profile makes the delay explicit: full request 9 has median inter-segment gaps of 0.0265/0.0267 ms; request 11 has 0.8572/0.8553 ms. Its 96 gaps sum to 82.247/80.979 ms on that final forward. Attention, FFN and head work stay approximately 14.58, 18.16 and 2.99 ms on GPU0. Delay is distributed across the boundaries rather than concentrated in one shader. Host submission remains sub-millisecond; fence waiting accounts for the added time.

The next long-context greedy request immediately recovers to normal gaps. Later short prompts and tools become slow again. This supports a graph/history-dependent submission or synchronization issue. The exact driver mechanism is not established. CPU PM QoS was active throughout; CPU-utilization and memory-pressure traces were not captured.

## Graph history and next diagnostic

The inferred cache contains four graph keys after the long requests, six after stochastic, and eight when the slow short greedy graphs are created (1,552 command buffers across both GPUs). The short greedy decode graph is new at onset, so reuse of an old short greedy graph alone cannot explain it. All ordinary runs use `all_logits=false`; the relevant decode keys differ in top-k (64 versus 1), while both short paths select attention partitions (16,16). The complete inferred graph-key timeline is in the JSON.

1. Diagnostic-only explicit free/drain of all target cached command buffers at each sampler top-k transition, before prefill. Preserve target GPU buffers, descriptors, query pools, semaphores, counter and KV. Wait for both devices idle before freeing, and record old/new key counts. Retain current per-forward timing. Keeps exact full19 request order and long workload, while removing the prior graph/command-pool residency at transitions into stochastic and greedy. Merely HashMap::clear leaks commands because Graph has no Drop; use the explicit Engine::drop-style free_command_buffers loop. Recovery of subsequent steady forwards supports graph/command-pool residency or stale command references; persistence points toward queue/timeline/buffer state or another process-history effect. Extra first-forward recording time is expected; compare GPU gaps and forwards after the first, not just total HTTP wall.
2. If transition clearing recovers performance, repeat with device_wait_idle only (retain graphs) to separate queue drain from command freeing. If needed, clear only unneeded graphs rather than all graphs in a follow-up diagnostic. The initial cache-clear diagnostic otherwise changes both cache residency and an explicit device idle operation.
3. If clearing does not recover, force target top-k=64 for greedy sampling in the diagnostic build and replay the same full19 bodies, still using argmax CPU sampling. A separate prewarm-short-greedy-before-long sequence can distinguish graph allocation/creation order from later execution history. Fast top-64 and slow top-1 graphs process exactly the same tokens at the same positions. This changes graph identity and only the output-selector specialization, isolating the association without claiming top-k compute is the bottleneck.

No production code was edited and no GPU commands were run for this analysis. All timings are observations from the completed diagnostic runs, not extrapolated peak claims.


## Cache-clear experiment

The isolated setter-only variant completed the full19 cold replay and a five-request same-process followup. All 19 normalized responses, usage counts, and complete ordinary-forward `(position, input_token, next_token)` sequences exactly match the original full replay. All eight 32-token list outputs also match the earlier fast/slow sequence.

| Request | Before clear experiment s | Clear experiment s | Clear GPU0 gap ms | Visible VRAM after, MiB (GPU0 / GPU1) |
|---|---:|---:|---:|---:|
| 8: stochastic_first | 1.821 | 1.666 | 2.659 | 222.250 / 229.574 |
| 9: stochastic_repeat | 1.678 | 1.714 | 3.999 | 222.250 / 229.574 |
| 10: greedy_again_first | 4.915 | 1.666 | 3.132 | 222.250 / 229.574 |
| 11: greedy_again_repeat | 5.979 | 1.701 | 4.084 | 222.250 / 229.574 |
| 14: long_history | 14.610 | 14.318 | 2.552 | 240.438 / 247.762 |
| 15: changed_system | 15.204 | 15.168 | 3.575 | 249.531 / 255.996 |
| 16: shorter_prompt | 2.144 | 1.961 | 85.213 | 256.000 / 255.996 |
| 21: stochastic_first | — | 5.129 | 112.718 | 256.000 / 255.996 |
| 22: stochastic_repeat | — | 5.331 | 119.386 | 256.000 / 255.996 |
| 23: greedy_first | — | 1.707 | 4.175 | 256.000 / 255.996 |
| 24: greedy_repeat | — | 1.714 | 4.449 | 256.000 / 255.996 |

Clearing on sampler switches removes the original short-greedy slowdown. Visible VRAM does not decrease when commands are freed, but subsequent command recording reuses the pooled storage instead of growing it immediately. Each additional graph initially adds 9.09375 MiB of visible VRAM per card. Later prompt shapes regrow the pool until the 256 MiB visible-VRAM region fills, and slow gaps recur.

The later short stochastic followup is slow even after freeing 11 keys; following greedy is fast. The delay therefore follows retained/pooled command-buffer state rather than greedy arithmetic. This is strong evidence consistent with a visible-VRAM residency boundary, not a direct driver allocation trace or proof that an idle-only control could have no effect. A graph bound enforced before new recording is the next supported intervention. Wider batch buffers need their own memory snapshots and correctness/performance validation.

All per-request memory snapshots, source hashes, sparse clock alignment and exact comparison results are preserved in `diagnostic-analysis.json`. Compilation used an isolated source copy and a separate binary; no production source was changed.
