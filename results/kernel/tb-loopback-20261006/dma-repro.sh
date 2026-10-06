#!/bin/bash
# Bounce every Thunderbolt network interface down/up N times; count DMA path activation failures.
# usage: dma-repro.sh N
n=${1:-10}
NS="sudo ip netns exec tbB"
for x in $($NS ls /sys/class/net 2>/dev/null); do [ "$x" = lo ] || $NS ip link set "$x" netns 1; done
before=$(sudo dmesg | grep -c "DMA tunnel activation failed\|failed to enable DMA paths")
for i in $(seq "$n"); do
  for d in /sys/class/net/thunderbolt*; do sudo ip link set "$(basename $d)" down; done
  sleep 1
  for d in /sys/class/net/thunderbolt*; do sudo ip link set "$(basename $d)" up; done
  sleep 4
  echo "round $i: carriers $(cat /sys/class/net/thunderbolt*/carrier 2>/dev/null | tr -d '\n') failures $(( $(sudo dmesg | grep -c "DMA tunnel activation failed\|failed to enable DMA paths") - before ))"
done
