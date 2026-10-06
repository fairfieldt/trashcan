# P2P boot options and GPU writes to system memory — 2026-10-05

incite, kernel `6.12.111-p2p` (0001-0003 applied), fan at 1900 rpm (`fancurve.service`). Each configuration was a
one-shot GRUB entry (`/boot/grub/custom.cfg`, removed afterwards). `collect.sh NAME` records the cmdline, IOMMU and
PCIe state, then runs `xgpu p2p-wall` and `xgpu p2p-verify`. Bandwidth is submit-to-fence wall time minus an empty
submit (see `bench/xgpu`); the GPU-timestamp numbers overstate writes below a few MiB.

## Why: P2P writes were 20% below NVIDIA's Ivy Bridge figure

NVIDIA measured 9.8 GB/s peer writes on an E5-2690 v2 (2014). incite measured 7.9 GB/s. Two suspects were Apple's
128-byte Max Payload Size and IOMMU translation of every peer request. ACS request redirect on the root ports sends
P2P through VT-d.

| Config | cmdline | MPS | IOMMU | ACS redirect | P2P write | P2P read | data check |
|---|---|---|---|---|---:|---:|---|
| baseline | (default) | 128 | translated | on | 7.88 GB/s | 3.72 GB/s | pass |
| mps256 | `pci=pcie_bus_perf` | 256 | translated | on | 8.53 | 3.72 | pass |
| iommuoff | `intel_iommu=off` | 128 | off | off | 8.53 | 3.73 | pass |
| iommupt | `iommu=pt` | 128 | passthrough | on | **4.16** | 3.73 | pass |
| both | `intel_iommu=off pci=pcie_bus_perf` | 256 | off | off | 8.57 | 3.73 | pass |

64 MiB copies; 32 MiB agrees to within 0.05 GB/s. Each data check covers 3 GiB.

- Each change alone gains 8%, but they don't stack. Something else caps P2P writes at about 8.5 GB/s. The
  host-write results below point at how Tahiti emits writes, not at the link or the IOMMU.
- `iommu=pt` halves P2P writes. Requests still go to the root complex (ACS redirect on) but untranslated. We have
  no explanation. Passthrough isn't a configuration we use.
- Reads hold at 3.7 GB/s throughout, matching NVIDIA's 3.4-3.7 GB/s. That's the Ivy Bridge root complex.
- Kernel warnings are identical in every config: the known `apple-gmux` WARN, plus journal recovery after the
  power cycle.
- None of these is adopted. MPS 256 is the candidate: it keeps the IOMMU and ACS isolation. It would need an
  end-to-end engine measurement (TP prefill) to show it's worth keeping.

## GPU writes to system memory (`host-write.txt`, `sdma-benchmark.txt`)

`xgpu host-write` has GPU0 write a 32 MiB region by several methods into five destinations, timed submit-to-fence.

| destination | copy / fill / contiguous stores | one 4-16 B store per 64 B line |
|---|---:|---:|
| local VRAM | 68-114 GB/s, 1060-1790 Mlines/s | 1990-2110 Mlines/s |
| peer VRAM (P2P) | 8.5 GB/s, 133 Mlines/s | 163-166 Mlines/s |
| cached GTT (snooped) | 1.56 GB/s, 24 Mlines/s | 117 Mlines/s |
| imported malloc (snooped) | 1.1-1.3 GB/s, 18-20 Mlines/s | 53 Mlines/s |
| WC GTT (not snooped) | 0.13 GB/s, 2.0 Mlines/s | 4.0 Mlines/s |

The kernel's SDMA benchmark (`amdgpu_benchmark` tests 1, 3, 4: copies between VRAM and GTT) writes VRAM to GTT at
12.7 GB/s at 64 MiB. It reads GTT to VRAM at 9.6 GB/s, the same as Vulkan's host-to-VRAM reads.

What this shows:

- The link, the IOMMU and the root complex can take GPU writes to DRAM at full speed. SDMA does 12.7 GB/s.
- Shader stores to system memory are slow whatever issues them: RADV's copy shader, RADV's fill, or our own
  stores of either width.
- Time follows the number of write pieces, not bytes. Writing a whole line costs more than writing one store into
  it: 2x more on WC GTT, 4.8x on cached GTT. So a fully written line leaves the GPU as several sub-line writes.
  Each piece costs about 250 ns into non-snooped memory and 10-20 ns into snooped memory. Partial writes to DRAM
  need a read-modify-write that a peer BAR doesn't, which is consistent with P2P being fast on the same path.
- The exact TLP sizes would take a PCIe analyzer to confirm.

Impact: the engine allocates host buffers HOST_CACHED (snooped, about 1.5 GB/s) and does top-k on the GPU, so the
default path only reads back a few hundred bytes per token. Only `--cpu-reduce` writes bulk data to the host:
about 5 ms per 7.5 MB prefill allreduce, against about 0.9 ms over P2P. A fast GPU-to-host path would have to go
through SDMA, which RADV doesn't expose on SI.
