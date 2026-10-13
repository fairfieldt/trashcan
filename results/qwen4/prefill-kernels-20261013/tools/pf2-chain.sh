#!/bin/bash
# Six-Mac session (run under chain6.sh, TICKETS set): validation of trashcan-pf2 (this branch) against the CPU
# reference and trashcan-6f6635a (main), then prefill A/B on real text, alternating. The A/B runs only if the 100- and
# 300-token prompts give main's greedy tokens and logits within rel 1e-2 of main's.
# usage: pf2-chain.sh TAG [val] [ab]
# (Session c1 also had a single-Mac tail, neutralized before it ran: single-Mac work runs under one-Mac leases.)
tag=$1; shift
# Queued ahead of time: run only when the binary on the Macs has passed the single-Mac checks.
[ -e /Users/fairfieldt/dev/claude-work/flashnext-20261012/pf2/chain-ready ] || { echo "pf2 chain $tag: not ready, released"; exit 3; }
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a2c014fa1f29b2375/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
S=/Users/fairfieldt/dev/claude-work/flashnext-20261012/pf2/chain
B=trashcan-6f6635a X=trashcan-pf2
mkdir -p $S
run() { python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
    --nodes indite,inflight,alight,invite,enlight --driver "$3" > /dev/null 2>&1; echo "$1 rc=$?"; }
check() {  # NAME: greedy tokens equal to main's and logits within rel 1e-2
    rsync -a "incite:trashcan-exp/flashnext/results/pf2-$tag-$1" "incite:trashcan-exp/flashnext/results/pf2-$tag-$1-base" \
        "incite:trashcan-exp/flashnext/results/pf2-$tag-$1.json" "incite:trashcan-exp/flashnext/results/pf2-$tag-$1-base.json" $S/
    a=$(grep greedy $S/pf2-$tag-$1.json) b=$(grep greedy $S/pf2-$tag-$1-base.json)
    rel=$(python3 /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a2c014fa1f29b2375/tools/compare_activations.py \
        $S/pf2-$tag-$1 $S/pf2-$tag-$1-base --all | awk '$1 ~ /^result_output/ {print $3; exit}')
    echo "check $1: mine [$a] main [$b] logits rel $rel"
    [ -n "$a" ] && [ "$a" = "$b" ] && python3 -c "import sys; sys.exit(0 if float('$rel') < 1e-2 else 1)"
}
for what in "$@"; do
  if [ "$what" = val ]; then
    run pf2-$tag-val14 $X "validate results/pf2-$tag-val14 $C --greedy 10"
    run pf2-$tag-val100 $X "validate results/pf2-$tag-val100 $C --greedy 10 --tokens-file $W/p100.ids"
    run pf2-$tag-val100-base $B "validate results/pf2-$tag-val100-base $C --greedy 10 --tokens-file $W/p100.ids"
    run pf2-$tag-val300 $X "validate results/pf2-$tag-val300 $C --greedy 10 --tokens-file $W/pf2-p300.ids"
    run pf2-$tag-val300-base $B "validate results/pf2-$tag-val300-base $C --greedy 10 --tokens-file $W/pf2-p300.ids"
    check val100 && check val300 || { echo "pf2 chain $tag: validation differs from main, no A/B"; exit 4; }
  fi
  if [ "$what" = ab ]; then
    for r in 1 2; do for b in $B $X; do
      run pf2-$tag-pp512-$b-$r $b "bench $C --pp 512 --tg 16 --tokens-file $W/long.ids"
      run pf2-$tag-pp2048-$b-$r $b "bench $C --pp 2048 --tg 16 --tokens-file $W/long.ids"
      run pf2-$tag-pp8192-$b-$r $b "bench $C --pp 8192 --tg 16 --tokens-file $W/long8k.ids"
    done; done
    for b in $B $X; do
      run pf2-$tag-pp1024-$b $b "bench $C --pp 1024 --tg 16 --tokens-file $W/long.ids"
      run pf2-$tag-pp4096-$b $b "bench $C --pp 4096 --tg 16 --tokens-file $W/long4k.ids"
    done
  fi
done
echo "pf2 chain $tag done"
