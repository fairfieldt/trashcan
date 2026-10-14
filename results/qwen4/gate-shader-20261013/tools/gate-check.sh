#!/bin/bash
# The bounded gate's single-GPU check (`qwen4 gate-check`, branch host-gate-shader-20261013, binary trashcan-host-gate1)
# on one Mac under a gpuq lease: both GPUs at amdgpu's own clock levels, then GPU 0 with the clocks held high.
# Every submission ends by its own bounds (see bench.rs gate_check). DRY=1 prints the commands.
# usage: gate-check.sh HOST   (not invite)
set -uo pipefail
h=${1:?host}
[ "$h" = invite ] && { echo "not on invite" >&2; exit 1; }
W=trashcan-exp/flashnext
run=host-gate-check-run.sh
cat > "/Users/fairfieldt/dev/claude-work/flashnext-20261012/host/$run" <<EOF
#!/bin/bash
# Run on $h under a gpuq lease (written by gate-check.sh).
cd ~/$W || exit 1
for g in 0 1; do
    scripts/with-clocks.sh results/clocks/host-gate1-$h-\$g.txt ./trashcan-host-gate1 qwen4 gate-check --gpu \$g --reps 200 \
        > results/host-gate1-$h-\$g.json 2> results/host-gate1-$h-\$g.err
    echo "gpu \$g rc \$?"
done
scripts/with-clocks.sh results/clocks/host-gate1-$h-0-high.txt ./trashcan-host-gate1 qwen4 gate-check --gpu 0 --reps 200 --high \
    > results/host-gate1-$h-0-high.json 2> results/host-gate1-$h-0-high.err
echo "gpu 0 high rc \$?"
echo "amdgpu timeouts/resets in dmesg: \$(sudo -n dmesg | grep -c -i -E 'amdgpu.*(timeout|reset)|gpu reset')"
EOF
rsync -a "/Users/fairfieldt/dev/claude-work/flashnext-20261012/host/$run" "$h:$W/$run"
if [ -n "${DRY:-}" ]; then
    ssh -- "$h" "cd $W && ls -la trashcan-host-gate1 scripts/with-clocks.sh $run && md5sum trashcan-host-gate1 && cat $run && echo dmesg errors: \$(sudo -n dmesg | grep -c -i -E 'amdgpu.*(timeout|reset)|gpu reset')"
    exit 0
fi
id=$(gpuq --on "$h" request "host gate-check" --est 3 --owner "Flash-Next decode host agent" | sed -n 's/^Ticket #\([0-9]*\) created.*/\1/p')
[ -n "$id" ] || exit 1
echo "ticket $h #$id"
gpuq --on "$h" wait "$id" || exit 1
ssh -- "$h" "gpuq run $id -- bash $W/$run"
rc=$?
gpuq --on "$h" release "$id" -r "host gate-check: rc $rc" > /dev/null
ssh -- "$h" "cd $W && cat results/host-gate1-$h-*.json; tail -2 results/host-gate1-$h-*.err"
exit $rc
