#!/bin/bash
# Thunderbolt 2 controller-to-controller network test on one host (incite).
# thunderbolt0 (domain0, NHI 16:00.0) stays in the root namespace as 10.77.0.1;
# thunderbolt1 (domain1, NHI 5d:00.0) moves to namespace tbB as 10.77.0.2 so
# traffic crosses the cable instead of loopback.
# Usage: run.sh OUTDIR [MTU...]
set -u
out=${1:?outdir}; shift
mtus=${*:-1500 9000 65522}
mkdir -p "$out"
NS="sudo ip netns exec tbB"
A=10.77.0.1 B=10.77.0.2

setup() {
  sudo ip netns add tbB 2>/dev/null || true
  if ip link show thunderbolt1 >/dev/null 2>&1; then sudo ip link set thunderbolt1 netns tbB; fi
  sudo ip addr flush dev thunderbolt0
  sudo ip addr add $A/24 dev thunderbolt0
  sudo ip link set thunderbolt0 up
  $NS ip addr flush dev thunderbolt1
  $NS ip addr add $B/24 dev thunderbolt1
  $NS ip link set thunderbolt1 up
  $NS ip link set lo up
}

setmtu() {
  sudo ip link set thunderbolt0 mtu "$1" && $NS ip link set thunderbolt1 mtu "$1"
}

env_dump() {
  {
    date -Is; uname -a
    for d in /sys/bus/thunderbolt/devices/*-[1-9]; do
      echo "$d $(cat $d/device_name) uid=$(cat $d/unique_id) rx=$(cat $d/rx_speed)x$(cat $d/rx_lanes) tx=$(cat $d/tx_speed)x$(cat $d/tx_lanes)"
    done
    for s in 0-0 1-0 2-0; do echo "$s $(cat /sys/bus/thunderbolt/devices/$s/unique_id)"; done
    readlink -f /sys/class/net/thunderbolt0/device
    $NS readlink -f /sys/class/net/thunderbolt1/device
    ip -d link show thunderbolt0
    $NS ip -d link show thunderbolt1
    ethtool -k thunderbolt0 2>/dev/null
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    grep MHz /proc/cpuinfo | sort | uniq -c
    for b in 11:02.0 11:09.0 14:00.0 5b:00.0; do echo "$b $(sudo lspci -s $b -vv | grep -E 'LnkSta:' | tr -s ' ')"; done
  } > "$out/env.txt" 2>&1
}

iperf() { # name, client args...
  local name=$1; shift
  $NS iperf3 -s -1 -D -B $B >/dev/null; sleep 0.5
  iperf3 -c $B -J "$@" > "$out/$name.json" 2>&1
  sleep 0.5
}

setup
sleep 2
env_dump
sudo dmesg | grep -i thunderbolt | tail -20 > "$out/dmesg-tb.txt"

for m in $mtus; do
  if ! setmtu "$m"; then echo "mtu $m rejected" >> "$out/notes.txt"; continue; fi
  sleep 1
  ping -c 3 -q $B > /dev/null
  ping -c 2000 -i 0.002 -q $B > "$out/ping-mtu$m-56B.txt" 2>&1
  ping -c 1000 -i 0.002 -q -s 1400 $B > "$out/ping-mtu$m-1400B.txt" 2>&1
  $NS sockperf server -i $B --daemonize >/dev/null 2>&1; sleep 0.5
  for sz in 64 1024; do
    sockperf ping-pong -i $B -t 10 -m $sz --full-rtt > "$out/sockperf-udp-mtu$m-${sz}B.txt" 2>&1
  done
  sudo pkill -f "sockperf server" ; sleep 0.5
  iperf tcp-fwd-mtu$m -t 15            # domain0 -> domain1
  iperf tcp-rev-mtu$m -t 15 -R         # domain1 -> domain0
  iperf tcp-p4-mtu$m -t 15 -P 4
  iperf tcp-bidir-mtu$m -t 15 --bidir
done
setmtu 1500
sudo dmesg | grep -i thunderbolt | tail -20 > "$out/dmesg-tb-after.txt"
echo done
