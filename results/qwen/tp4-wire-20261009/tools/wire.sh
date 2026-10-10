#!/bin/bash
# TP4 on incite + invite: the f16 all-reduce payload (the default below prefill size) against TRASHCAN_TP_WIRE=f32,
# same binary, alternating in ABBA order, then the correctness gates.
# usage: TICKETS=incite=ID,invite=ID wire.sh PHASE      PHASE: warm | decode | mtp | gates | mtpcool
set -uo pipefail
here=$(dirname "$0")
MTP=/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
DECODE="qwen-bench --pp 512 --tg 128 --reps 2"
F32=TRASHCAN_TP_WIRE=f32
run() { "$here/tp4.sh" "$@"; }
# abba PREFIX 'DRIVER COMMAND': f16, f32, f32, f16, f16, f32. With COOL=LIMIT, every card of both Macs is first
# cooled to at most LIMIT C.
abba() {
  local n16=0 n32=0 side
  for side in f16 f32 f32 f16 f16 f32; do
    [ -n "${COOL:-}" ] && "$here/cool.sh" "$COOL" incite invite
    if [ $side = f16 ]; then n16=$((n16 + 1)); run $1-f16-$n16 "$2" ""
    else n32=$((n32 + 1)); run $1-f32-$n32 "$2" "$F32"; fi
  done
}
case ${1:?phase} in
warm) run w-warm "qwen-bench --pp 512 --tg 16 --reps 1" "" ;;  # loads the model into both page caches; not counted
decode) abba w-dec "$DECODE" ;;
mtp) abba w-mtp "$DECODE --mtp $MTP" ;;
gates)
  run w-val-f16 "qwen-validate results/w-val-f16" ""
  run w-check-mtp-f16 "qwen-check --mtp $MTP" ""
  run w-check-v3-f16 "qwen-check --verify 3" ""
  run w-check-mtp-f32 "qwen-check --mtp $MTP" "$F32"
  run w-check-v3-f32 "qwen-check --verify 3" "$F32" ;;
# The mtp phase heated invite's first card into throttling (README): the same runs, each from at most 55 C.
mtpcool) COOL=55 abba w-mtpc "$DECODE --mtp $MTP" ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
