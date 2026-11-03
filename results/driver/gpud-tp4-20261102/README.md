# Qwen3.8-27B TP4 on two Macs, userspace drivers end to end (2026-11-02)

Source: trashcan main 3643e670 for the engine (with driver/tbd-client, built in the kbuild VM: lab
scripts/cross-build.sh's file list plus the `lights` workspace member) and for gpud and libgpud_drm.so. Machines:
enlight (machine 0, the driver) and invite (machine 1, `qwen-tp-node`), one Thunderbolt 2 cable between them
(enlight domain 0 port 1, NHI 0000:16:00.0, to invite), two lanes. gpuq leases enlight #200, invite #317. Model:
`Qwen3.8-27B-trashcan-Q4_0.gguf` with its MTP layer; each GPU holds 4341.7 MiB.

The cable on enlight runs on tbd, the userspace Thunderbolt driver (task 1745: `vfio.sh take 0000:16:00.0`,
`tbd run 0000:16:00.0 --stream trashcan:15:15`, 10.99.36.2/30 on tbd0; `enlight/tbd-run.log`); invite runs Linux's
thunderbolt driver. Both runs report `stream up (tbd invite.trashcan.ring)`: the engine's link uses tbd's stream,
whose rings run in the engine process. enlight booted 1745's one-time `tbd-iommu` entry, invite a one-time
`gpud-iommu` entry (added to its custom.cfg; both `intel_iommu=on iommu=pt vfio_pci.disable_idle_d3=1`), no GUI.

| | pp512 | tg128 | chat prefill (23) | chat decode with MTP (86) |
|---|---|---|---|---|
| amdgpu on all four D700s | 129.5 / 133.7 | 25.6 / 25.5 | 23.84 | 28.86 |
| gpud on all four D700s | 109.1 / 111.3 | 24.0 / 24.2 | 20.33 | 24.27 |

tok/s; `qwen-bench MODEL --tp-connect 10.99.36.1:7072 --pp 512 --tg 128 --reps 2` (`enlight/bench-*-tp4.txt`) and
`qwen-chat MODEL "Explain in three sentences what a GPU driver does." --mtp MTP --tp-connect ... --context 2048
--max-tokens 160` (`enlight/chat-*-tp4.txt`). The chat text is the same and MTP accepted the same 43 of 86 drafts
at the same positions. gpud reaches 94% of amdgpu's plain decode and 83% of its prefill.

On the GPUs, not the CPU: the engine allocated its ranks on four Vulkan GPUs (`GPUs per machine [2, 2]`), each of
the four gpud servers logged the engine as a client (`*/serve-0[26].log`), and gpud's clock sensors (`drminfo
clockmon`: `enlight/clocks-gpud-tp4.txt`, `invite/clocks-gpud-node.txt`) show all four at 850/1370 MHz through the
measured phases, as amdgpu's hwmon did for the native run (`enlight/clocks-native-tp4.txt`,
`invite/clocks-native-node.txt`).

Against the record in docs/qwen.md (149 / 28.3 tok/s, older engine, Linux's thunderbolt driver on both ends over
USB4STREAM) the native run here is slower: different engine build and transport, not compared further.

Afterwards: every gpud stopped with its teardown, tbd stopped and the NHI given back to the thunderbolt driver, both
Macs rebooted to their default entries (GPUs on amdgpu, cable up on the stock driver), leases released.
