#!/bin/bash
# The policy replays with the round costs of main eb4dbeb (1-4-token verification batches: 49.4 / 67 / 83 / 106 ms for
# 1-4 tokens) against those of session 3 (main 2293550): which unsure threshold pays with cheaper verification
# (gate/policysim-eb4dbeb.txt), cost-aware gates against the fixed one (gate/policysim2-eb4dbeb.txt), and main's
# drafting with its own draft step and k = 0 round against the engine's (the expected gain on eb4dbeb).
# usage: policysim-eb4dbeb.sh TRACE_DIR   (the sampled and greedy runs and replays of sampling/study.sh)
G=$(cd "$(dirname "$0")/../gate" && pwd)
cd "${1:?trace directory}" || exit 1
for t in 0 1.0; do
    echo "== 2293550 T $t unsure 0.6"
    python3 "$G/policysim.py" . --t $t --draft 5.1 --verify 52.6,84.1,100.2,118.1 --plain 51.3 | grep -v '^T '
    for u in 0.3 0.4 0.5 0.6 0.7; do
        echo "== eb4dbeb T $t unsure $u"
        python3 "$G/policysim.py" . --t $t --draft 5.1 --verify 49.4,67,83,106 --plain 49.2 --unsure $u | grep -v '^T '
    done
done > "$G/policysim-eb4dbeb.txt"
{
    for t in 0 0.6 1.0; do
        python3 "$G/policysim2.py" . --t $t
        python3 "$G/policysim2.py" . --t $t --calib raw | grep -v fixed
    done
    echo
    echo "== old costs (2293550)"
    for t in 0 1.0; do
        python3 "$G/policysim2.py" . --t $t --verify 52.6,84.1,100.2,118.1 --plain 51.3
    done
    echo
    echo "== main's drafting (5.85 ms draft steps, k = 0 rounds 2 ms over a plain step) against the engine's (5.1, 1 ms)"
    for t in 0 1.0; do
        python3 "$G/policysim.py" . --t $t --draft 5.85 --verify 51.2,67,83,106 --plain 49.2 | grep '^main\|^T'
        python3 "$G/policysim.py" . --t $t --draft 5.1 --verify 50.2,67,83,106 --plain 49.2 | grep '^two-rate'
    done
} > "$G/policysim2-eb4dbeb.txt"
