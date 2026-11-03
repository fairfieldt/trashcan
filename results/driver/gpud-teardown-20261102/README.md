# gpud teardown hang, the two incite crashes, and recovery after a killed gpud (2026-11-02)

Source: trashcan main 4a28d629 (`driver/`), built with `cargo build --release --target x86_64-unknown-linux-gnu`
in the kbuild VM; final binary sha256 2e8222099c49c87c0a66447325ed724cced811c1ce09a6a0a6fe76540ceb5382. Machine:
incite, 02:00.0 on vfio-pci (one-time `gpud-iommu` boot, `to-vfio.sh` and `allow_unsafe_interrupts` as in
`../vfio-20261102`), 06:00.0 on amdgpu, gpuq leases #328 and #329. Runs with `GPUD_DEBUG=1 GPUD_KMSG=1` and
`sysctl kernel.printk_devkmsg=on`, so every gpud step also went out on netconsole.

## The crashes

incite reset itself twice (15:26 and 15:30, `netconsole-hangs.txt`): a PCIe completion timeout on root port
00:02.0 about 0.35-0.4 s after `gpud init` opened the GPU, a stream of AER errors, `pciehp: Slot(3-3): Power fault`,
then a hardware reset. The second came from a fresh boot, so it was not state left by an earlier gpud.

Rerun with each step in the kernel log and a guard that stops all MMIO at the first dead read (`init-8.txt`):
init passed (5/5 ring and IB tests), and the GPU stopped answering in the teardown, in gfx_v6_0's hw_fini at
register 0xa00f (DB_DEPTH_INFO). `gfx_v6_0_rlc_stop` halts the RLC and then reads that GFX register; the read
never completes if GFX is clock-gated with the RLC halted. With the guard there were two completion timeouts (the
read and the guard's check), the slot power fault, and incite stayed up; PCI config space still answered, MMIO
did not, until a reboot. Without the guard, `rlc_stop` kept polling `RLC_STAT` (up to 100000 reads, each a
completion timeout): that stream is what reset the machine.

Why GFX was gated: gpud's GFX soft reset at init (needed after amdgpu's teardown, or the CP stalls) puts the RLC's
registers back to their reset defaults, with coarse-grain clock gating on. amdgpu never enables that on SI
(`AMD_CG_SUPPORT_GFX_CGCG` is commented out in si.c), gates at late init and ungates in fini_early, before hw_fini;
gpud did neither. Register trace (`init-9.txt`):

| point | RLC_CGCG_CGLS_CTRL | RLC_CGTT_MGCG_OVERRIDE |
|---|---|---|
| after amdgpu's teardown | 0x20003c (CGCG, CGLS off) | 0xffffffc3 (MGCG off) |
| after gpud's GFX soft reset | 0x20003f (CGCG, CGLS on) | 0xfffffffc (MGCG on) |
| after amdgpu-order gating at late init | 0x20003c | 0xffffffc0 |
| before hw_fini, after ungating | 0x20003c | 0xffffffc3 |

Fix (main 4a28d629): gating and ungating in amdgpu's order. Init now takes 460-490 ms instead of 156: amdgpu's
gfx_v6_0 gating code waits on `RLC_STAT`, whose RLC_BUSY bit stays set while the RLC runs, so each wait runs its
full 100 ms timeout (native amdgpu pays the same).

## Tests of the fix

| run | file | result |
|---|---|---|
| init from a fresh boot (after amdgpu's teardown) | `init-9.txt` | 5/5 ring and IB tests, teardown completes |
| init x3, each after gpud's teardown | `init-10..12.txt` | 5/5 each, no completion timeouts |
| serve, vkcompute | `vkcompute-gpud-2.txt` | exact results, 15.7 GB/s copy (as before; boot clocks) |
| serve, engine micro-benchmarks | `bench-gpud2-*.txt` | all pass; error figures identical to the earlier gpud run, times within 0-5% |
| SIGTERM to a serve whose signal mask was set too late | `init-13.txt` | serve died without teardown; init found CP, RLC, DMA, IH running, stopped them, 5/5 |
| SIGTERM with the fix (signals blocked before the first thread) | `init-14.txt` | serve log shows ungating and every hw_fini; init found all stopped, 5/5 |
| SIGKILL idle, then init | `init-15.txt` | 5/5 |
| SIGKILL mid-benchmark, then init | `init-16.txt` | 5/5, but see below |
| final binary on a fresh boot | `init-17/18.txt`, `vk-fresh-1.txt` | 5/5; vkcompute exact |

## Known: a SIGKILL mid-dispatch can wedge the shader core

After a SIGKILL during `prefill-attention-bench`, some restarts find `GRBM_STATUS` 0xa0403028 at quiesce (SPI and IA
busy) and some 0xa0003028. With SPI busy, init and the IB tests still pass (VMID 0, no shaders), but the first client
dispatch never completes (`vk-dbg2.txt`: gfx ring, 3 IBs, fence 7). Kills at 1.5, 2.5 and 3.2 s into the benchmark
left SPI idle and clients ran (`vk-afterkill.txt`); a kill at 4.4 s left SPI busy and clients hung (`vk-k.txt`,
`serve.log`; `serve.prev.log` is the 3.2 s case). A live dump (SIGUSR1 to serve) of a stuck client submission:
GRBM_STATUS 0xa04c3028 (CP, SPI, IA busy), CP_STAT 0x80038647, CP_ME_HEADER_DUMP 0xc0004600 (EVENT_WRITE: the CP
waits for the waves to drain), CP_RB0_RPTR 0x770, WPTR 0x800.

Reading: the killed process's waves were left in the shader core waiting on memory that vfio-pci's close took away
(IOVAs unmapped, bus mastering off). SI's GRBM soft reset has no shader-sequencer bit and SI has no SQ_CMD wave
kill, so only an ASIC reset (PCI config reset) and a VBIOS re-POST clear it: amdgpu's si_asic_reset, which needs
atom.c in gpud. Until then: reboot.

## Notes

`restart.sh` is the serve restart used (SIGTERM, wait for the teardown, start under the lease). The lost-device
line of the caught hang was written at 32.616 s but reached the netconsole receiver four minutes later, at the next
reboot; the step markers before it arrived immediately.
