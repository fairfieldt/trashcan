#!/bin/bash
# One configuration of the Thunderbolt loopback benchmark (MTU 1500).
# thunderbolt0 (domain0, NHI 16:00.0) = 10.77.0.1 in the root namespace (client side),
# thunderbolt1 (domain1, NHI 5d:00.0) = 10.77.0.2 in namespace tbB (server side).
# Usage: bench.sh OUTDIR
# Env: PIN=1 pins NHI IRQs and iperf3/sockperf processes (layout below).
#      REPS=3 repetitions of each one-way iperf3 run.
set -u
out=${1:?outdir}
PIN=${PIN:-0} REPS=${REPS:-3}
mkdir -p "$out"
NS="sudo ip netns exec tbB"
A=10.77.0.1 B=10.77.0.2
# Pin layout (CPU 2k and 2k+1 are HT siblings of core k):
IRQ_A=4 APP_A=6 IRQ_B=8 APP_B=10

wait_link() {
  for _ in $(seq 60); do
    ip link show thunderbolt0 >/dev/null 2>&1 && { ip link show thunderbolt1 >/dev/null 2>&1 || $NS ip link show thunderbolt1 >/dev/null 2>&1; } && return 0
    sleep 1
  done
  echo "link did not come up" >&2; return 1
}

setup() {
  wait_link || exit 1
  sudo ip netns add tbB 2>/dev/null || true
  if ip link show thunderbolt1 >/dev/null 2>&1; then sudo ip link set thunderbolt1 netns tbB; fi
  sudo ip addr flush dev thunderbolt0
  sudo ip addr add $A/24 dev thunderbolt0
  sudo ip link set thunderbolt0 mtu 1500 up
  $NS ip addr flush dev thunderbolt1
  $NS ip addr add $B/24 dev thunderbolt1
  $NS ip link set thunderbolt1 mtu 1500 up
  $NS ip link set lo up
  for _ in $(seq 30); do ping -c 1 -W 1 -q $B >/dev/null && break; sleep 1; done
  sleep 2
  ping -c 3 -W 2 -q $B >/dev/null || { echo "no ping" >&2; exit 1; }
}

irqs_of() { grep "PCI-MSIX-0000:$1 " /proc/interrupts | cut -d: -f1 | tr -d ' '; }

pin_irqs() {
  for i in $(irqs_of 16:00.0); do echo $IRQ_A | sudo tee /proc/irq/$i/smp_affinity_list >/dev/null; done
  for i in $(irqs_of 5d:00.0); do echo $IRQ_B | sudo tee /proc/irq/$i/smp_affinity_list >/dev/null; done
}

if [ "$PIN" = 1 ]; then TA="taskset -c $APP_A" TB="taskset -c $APP_B"; else TA="" TB=""; fi

env_dump() {
  {
    date -Is; uname -r
    echo "PIN=$PIN REPS=$REPS"
    for p in /sys/module/thunderbolt/parameters/*; do echo "$p=$(cat $p)"; done
    echo "srcversion=$(cat /sys/module/thunderbolt/srcversion)"
    for d in /sys/bus/thunderbolt/devices/*-[1-9]; do
      echo "$d rx=$(cat $d/rx_speed)x$(cat $d/rx_lanes) tx=$(cat $d/tx_speed)x$(cat $d/tx_lanes)"
    done
    echo "governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
    echo "min=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq) max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
    for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do echo "$(cat $s/name) disable=$(cat $s/disable)"; done
    for i in $(irqs_of 16:00.0) $(irqs_of 5d:00.0); do echo "irq $i eff=$(cat /proc/irq/$i/effective_affinity_list)"; done
  } > "$out/env.txt" 2>&1
}

# Sample clocks of the pinned CPUs and per-CPU busy time across a run.
stat_cpus() { awk '/^cpu[0-9]/{b=$2+$3+$4+$7+$8; t=b+$5+$6; print $1, b, t}' /proc/stat; }

iperf() { # name, client args...
  local name=$1; shift
  $NS $TB iperf3 -s -1 -D -B $B >/dev/null; sleep 0.5
  stat_cpus > "$out/.stat0"
  ( sleep 5; grep MHz /proc/cpuinfo | awk '{print NR-1, $4}' > "$out/$name.mhz" ) &
  $TA iperf3 -c $B -J "$@" > "$out/$name.json" 2>&1
  wait
  stat_cpus > "$out/.stat1"
  join "$out/.stat0" "$out/.stat1" | awk '{bd=$4-$2; td=$5-$3; if (td>0 && bd/td>0.05) printf "%s %.0f%%\n", $1, 100*bd/td}' > "$out/$name.cpu"
  rm -f "$out/.stat0" "$out/.stat1"
  sleep 0.5
}

setup
[ "$PIN" = 1 ] && pin_irqs
env_dump

ping -c 2000 -i 0.002 -q $B > "$out/ping-56B.txt" 2>&1
$NS $TB sockperf server -i $B --daemonize >/dev/null 2>&1; sleep 0.5
$TA sockperf ping-pong -i $B -t 10 -m 64 --full-rtt > "$out/sockperf-64B.txt" 2>&1
sudo pkill -x sockperf; sleep 0.5

for r in $(seq $REPS); do
  iperf tcp-fwd-$r -t 10
  iperf tcp-rev-$r -t 10 -R
done
iperf tcp-bidir -t 10 --bidir
iperf tcp-p4 -t 10 -P 4
echo done
