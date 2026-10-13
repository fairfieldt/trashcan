# Flash-Next 2-4-token decode batches (MTP verification), 2026-10-12: unfinished

**State.** Stopped after one single-Mac session (the Macs were shut down for the night). Nothing here is ready to
merge: no six-Mac measurement was made, and the one candidate that helps (the lane-per-row second mixer half) is still
behind a temporary switch (`TRASHCAN_VFY_MIX=rows`) together with a first half that does not help. Measured so far:
where each extra token's time goes on one Mac, and that lane-per-row kernels (one lane per matrix row, inputs through
scalar loads, whole 64-byte lines) are much slower than main's kernels for the dense projections and the experts at
these batch sizes, but faster for the mixer's 10240 x 320 up projection.

Tested source: main 26b8d1b (MTP merged) as baseline (`trashcan-26b8d1b`, sha256 fcd79026...), against this branch's
experiment tree (`trashcan-vfy`, sha256 ee023c60...: the WIP commit plus the mixer kernels; the failed kernels are
kept in [removed/](removed/)). invite, layers 0-7 on both D700s, 850/1370 MHz while computing on both GPUs
([invite/clocks/](invite/clocks/)).

```sh
# under a gpuq lease on invite (tools/s1.sh): per batch B = 2, 3, 4, profile on
TRASHCAN_Q4_PROFILE=1 [TRASHCAN_VFY_...] trashcan qwen4 stage-bench G-iq4-00001-of-00006.gguf --ranges 0:4,4:8 \
    --gpus 0,1 --context 32768 --batch B --head-rows 0:21504,21504:43008 --decode 8 --start 64 --prefill 8
python3 tools/prof.py --layers 5,7 invite/vfy-s1-*.json      # profile-s1.txt
```

The B-token chunk at position 0 is the verification-shaped batch (as in results/qwen4/mtp-20261012's
`mtp-stagebench.sh`); decode steps are single tokens at 64.

## Where the extra tokens go (main 26b8d1b, one Mac, 8 layers)

| batch | chunk GPU ms (8 layers) | / one token (8.36 ms) | mixers | DeltaNet | router | experts |
|---|---|---|---|---|---|---|
| 1 | 8.36 | 1 | 1.70 | 2.59 | 0.61 | 2.15 |
| 2 | 13.49 | 1.61 | 3.07 | 3.90 | 0.82 | 3.77 |
| 3 | 16.27 | 1.95 | 4.01 | 4.11 | 0.83 | 5.32 |
| 4 | 19.23 | 2.30 | 4.88 | 4.48 | 0.84 | 6.89 |

Per dispatch, DeltaNet layer 5, microseconds (one token / 2 / 3 / 4 tokens; [profile-s1.txt](profile-s1.txt)):

| dispatch | 1 | 2 | 3 | 4 | why it grows |
|---|---|---|---|---|---|
| mixer first half (`_sk`, 2-8 tokens `_sk4`), each of two | 50 | 96-99 | 118-128 | 145-154 | reads the matrix once, but per token re-reads and recombines the residual in every 4-row wave (activation bytes 8x the weight bytes at 4 tokens) and its finisher walks the tokens in turn |
| mixer second half `qwen4_mix_up`, each of two | 51 | 91-92 | 122-123 | 148-150 | one workgroup row per token: the 3.5 MB matrix is read once per token |
| qkv + z (+ alpha/beta) group | 242 | 371 | 369 | 378 | `_c4` kernels always compute 4 columns (2 and 3 tokens cost as much as 4); 84 VGPRs, x loads 4x the weight bytes |
| out projection 2560 x 6144 | 118 | 205 | 218 | 249 | same, 2 rows per wave: x loads 8x the weight bytes |
| `qwen4_gdn` | 47 | 57 | 78 | 98 | the recurrence walks the tokens (+20 per token) |
| router f32 513 x 2560 | 40 | 67 | 68 | 70 | |
| experts gate/up `qwen4_moe_gu_iq4xs` | 175 | 315 | 451 | 582 | per (token, slot): every token reads its own 11 experts; ~6 VALU per weight (ISA), VALU-bound |
| experts down `qwen4_moe_down_w_iq4nl` | 88 | 157 | 216-225 | 275 | per token |

The QSA layer's q projection (12288 rows) goes 196 -> 345 us at 3 tokens, its out projection 117 -> 215; attention
itself does not grow. At 3 tokens the extra 7.9 ms of the 8 layers are mixers 2.3, experts 3.2, DeltaNet 1.5 (dense
projections 1.3, recurrence 0.2), QSA 0.4, router 0.2.

## Tried (temporary switches on stage-bench; per dispatch in profile-s1.txt)

| candidate | result |
|---|---|
| lane-per-row Q8_0 matrix kernels for 1-4 tokens (`qwen4_q8_rows{4,8,16}`: a lane per row walking 64-byte lines over a K segment, the K segments over the waves of a workgroup summed in LDS, x through scalar loads; 2 VALU per weight plus one FMA per token in the ISA against ~8.6 in `_c4`) | much slower: qkv 10240 x 2560 at 3 tokens 411-419 us against 369, z 255-265 against 168, out 2560 x 6144 440 against 218; one token 400 us against 242 (70 GB/s). K splits of 4, 8 and 16 waves all slower |
| lane-per-row experts (`qwen4_moe_gu_rows`: 32 gate and 32 up rows per workgroup, one 256-weight super-block per wave; `qwen4_moe_down_rows`: a lane per output row, a wave per slot) | slower: gate/up at 3 tokens 690 us against 451, down 347-359 against 216; one token 285 / 144 against 175 / 88 |
| `qwen4_gdn` loading the next token's inputs (read-only views) while computing one | no change (78 / 98 us at 3 / 4 tokens, 47 for one); reverted |
| lane-per-row mixer: `qwen4_mix_prep` (pending combine in place, gamma * x, sums of squares; 7.5 us) + `qwen4_mix_rows_down` (first half, lanes per row, split-K finisher as `_sk`) + `qwen4_mix_rows_up` (second half, 5 waves = the 5 lines of 64 rows) | **second half 39-51 us at 2-4 tokens against 91-150** (and 38-40 against 51 for one token); first half 105 / 137 / 159 us at 2 / 3 / 4 tokens against 96 / 123 / 150 (73 against 50 for one). Mixers of 8 layers 3.07 / 4.01 / 4.88 -> 2.57 / 3.11 / 3.65 ms at 2 / 3 / 4 tokens. **Kept behind `TRASHCAN_VFY_MIX=rows`, unfinished** |

Why lane-per-row lost: decoding work is not the limit at 1-4 tokens; the access pattern is. Lanes reading 64 rows
2.5 KB apart (dense) or 1.3 KB apart (experts), each a different 64-byte line, stream at 65-70 GB/s on the D700 even
with whole, 64-byte-aligned lines (this tree aligns repacked blocks and arena buffers to 64 bytes for them); the
lanes-over-K kernels read 256 contiguous bytes per row and instruction. The prefill tile kernels can use the
lane-per-row layout because each line feeds up to 8 slots there. The mixer's up projection is the exception: its
rows are 320 bytes (5 lines), and a workgroup's 5 waves together read 64 whole consecutive rows (20 KB contiguous).

Correctness of the experiment tree (all lane-per-row kernels on) against main, layers 0-7, the 14-token reference
prompt (`stage-bench --tokens-file dec-ref14.ids --out`), last chunk of 2, 4 and 3 tokens (chunks of 3, 5, 11) and
single tokens: every tensor within rel 1.25e-5 (worst: the QSA output; [compare-s1-*.txt](.)). Not run: the six-Mac
`qwen4 validate`, `mtp spec`, `bench`.

Offline shader statistics: the kbuild VM now has Mesa 25.0.7 (`mesa-vulkan-drivers`, the Macs' version), and
`RADV_FORCE_FAMILY=tahiti RADV_DEBUG=shaderstats` (or `shaders` for the ACO program) on a native aarch64 build prints
the D700's VGPRs, waves per SIMD and instruction mix for every pipeline without a GPU ([tools/vmstats.sh](tools/vmstats.sh),
`isa.py`, `loopmix.py`).

## Next steps

1. Mixer: keep `qwen4_mix_rows_up` (move the pending combine into its epilogue so it can follow main's
   `_sk`/`_sk4` first half without `qwen4_mix_prep`), drop `qwen4_mix_rows_down` and the prep pass, and measure
   whether it also wins for one token (38-40 against 51 us measured with the prep pass in front). Expected: about
   -0.8 ms per Mac at 2 tokens, -1.3 ms at 3, -1.6 at 4 (16 second halves), about -0.2 ms for one token.
2. First half for 2-4 tokens: `_sk4` re-reads and recombines the residual per 4-row wave; options: more rows per wave
   (8, activation:weight bytes 2:1), a prep pass feeding plain gamma * x (measured 7.5 us), or the finisher summing
   the tokens in parallel instead of in turn.
3. Dense Q8_0 for 2-3 tokens: specialize `qwen4_q8_mvw4_*_c4` to 2 and 3 columns (today 2 and 3 tokens cost as much
   as 4: qkv group 371 us at 2 tokens) and try 8 rows per wave for 3-4 columns (x traffic 4C/R of the weight bytes).
4. Experts (the largest share: +135 us gate/up and +60 down per token and layer at ~140 GB/s): the per-slot kernel is
   VALU-bound (~6 VALU per weight: per-lane IQ4_XS scale decode, LDS table addressing); cut its instruction count while
   keeping the lanes-over-K access pattern (e.g. scales decoded once per 8 blocks, the shared expert's rows read once
   for all tokens of the batch).
5. Then: six-Mac `qwen4 validate` (reference prompt and p100), `mtp spec` on the chat prompts alternating with
   `trashcan-26b8d1b` (verification ms by batch size from its JSON), and `qwen4 bench --pp 512 --tg 128` for plain
   decode, per AGENTS.md.

## Files

- `invite/`: stage-bench JSON and stderr of session 1 (`vfy-s1-base-*`: main 26b8d1b; `vfy-s1-all-*`: every
  lane-per-row kernel on; `alln1`: also for single tokens; `ks4/8/16`: forced K splits), `clocks/`.
- `profile-s1.txt`: walls, phases and per-dispatch times (layers 5 and 7) of every run; `compare-s1-*.txt`: dumps
  against main.
- `removed/`: the lane-per-row dense and expert kernels (removed from the engine).
- `tools/`: `s1.sh` (the session), `sb.sh` (stage-bench runs), `dump.sh` (dumps), `prof.py`, `vmstats.sh`, `stats.py`,
  `isa.py`, `loopmix.py` (offline RADV shader statistics).
