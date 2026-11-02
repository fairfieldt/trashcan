# gpud plan

Where `gpud` is going and how it gets there: architecture, licensing, the path to command submission on GCN 1, and
what each milestone has to show. Facts about the Linux code are from the audit in `linux-deps.md` (Linux
7.3-rc7); facts about the hardware are measured unless marked otherwise.

## Goal

A purpose-built appliance: a small microkernel plus userspace services running distributed inference on Mac Pro
6,1s (two D700s each, Thunderbolt between machines). Linux is the harness for developing and checking the pieces,
and the reference to measure them against; it is not a required part of the result.

Layers, top to bottom:

1. The trashcan engine (Rust, Vulkan) and Mesa RADV (NIR/ACO shader compilation, command buffers), largely
   unchanged at first.
2. A thin adapter from the DRM ABI subset RADV uses (11 amdgpu ioctls, 16 core render-node ioctls, `mmap` of BOs;
   `linux-deps.md` §4) to `gpud`'s IPC; perhaps later a RADV winsys of our own.
3. `gpud`: Linux's SI hardware code in C (imported, minimally changed) on a C compatibility layer, with Rust for
   orchestration: device setup sequence, memory and VM management, submission, fences, IPC.
4. The platform interface (`gpud-platform`, Rust): PCI config, MMIO, DMA/IOVA, interrupts. Linux VFIO backend now;
   a microkernel backend later.

GPU service, fabric (Thunderbolt/USB4STREAM), storage/model loading and Apple SMC/fan control stay separate
services. Fixed hardware and a single trusted client allow what a general driver can't: static pinned allocations,
no eviction, no display, no video, no suspend/resume, precompiled shaders.

## What is measured (milestone 1)

On a Mac Pro 6,1, 02:00.0 handed from amdgpu to vfio-pci (`vfio.md`):

- Exclusive ownership through VFIO type1: an IOMMU group of its own; it needs `allow_unsafe_interrupts`, as the
  machine has no interrupt remapping.
- Config space, BARs, capabilities, link (gen 3 x16); the VBIOS from ACPI VFCT by exact bus match, byte-identical
  to what amdgpu loads; the ROM BAR is blank.
- Registers through BAR2, identical to amdgpu's debugfs view except for what its teardown switched off.
- VRAM through BAR0 and through MM_INDEX/MM_DATA, including past the 256 MiB window.
- IOMMU mappings (4 KiB to 64 MiB, 2 MiB-aligned IOVAs for large buffers), the other GPU's BAR mappable as a peer
  target, MSI to an eventfd.
- One device-initiated DMA path: the async DMA engine (no microcode) wrote host memory at IOVAs through the
  memory controller's AGP aperture. Unmapped IOVAs didn't land, but the IOMMU reported no fault (open question).

Not shown: anything involving the command processor, interrupts actually firing, clocks above the boot level.

## Licensing

- Rust code (`gpud-platform`, `gpud`, IPC, adapters): MIT.
- Imported Linux files: unchanged license text, under `driver/linux/<upstream path>`, with a `PROVENANCE` file
  (upstream version 7.3-rc7, commit 24a19b953e2a, per-file sha256, and every local change as a patch). The 12 SI
  hardware files and most of amdgpu core carry AMD's MIT-style text; `drivers/dma-buf/` and other kernel
  infrastructure are GPL-2.0, and importing any of it makes the `gpud` binary GPL-2.0 as a whole.
- The compatibility layer (`linuxkpi`-style headers and runtime) is written for gpud (MIT); FreeBSD's LinuxKPI
  (BSD-2-Clause) is the model and a source where its code fits userspace.
- The engine and RADV talk to `gpud` across a process boundary and never link its code.

## Architecture of gpud

```
 clients (engine + RADV, through the DRM-ABI adapter)
        |  IPC: requests, shared-memory BOs, fence notifications
 gpud   +-- server (Rust): sessions, BO table, VM per client, submission queues, fences, syncobjs
        +-- device core (C or Rust, small): IP block sequencer, BO carve-out allocator, GART/VM binding,
        |   ring/IB/fence glue, IH thread -> amdgpu_irq dispatch
        +-- SI hardware modules (C, imported): si.c, gmc_v6_0.c, si_ih.c, gfx_v6_0.c, si_dma.c, atom.c,
        |   amdgpu_atombios.c, si_dpm.c, si_smc.c, legacy_dpm.c, ... + the amdgpu core files that are cheap
        +-- linuxkpi (C): kmalloc, locks, delays, printk, lists/idr, workqueue thread, dma_fence subset,
        |   firmware files, PCI config through the platform
        +-- gpud-platform (Rust): Mmio, DmaSpace, IrqEvent, PciDevice   <- VFIO now, microkernel later
```

What the audit says about where the line falls (`linux-deps.md` §5):

- **Import as is**: the 12 hardware files (22,852 lines, 76 outside symbols, 1,010 register accesses through
  one funnel), and the core files that serve them without TTM or the scheduler: ring, IB, IH, IRQ, write-back
  slots, GART, GMC helpers, ucode, RLC, GFX helpers, suballocator, register access, atombios/atomfirmware helpers.
- **Replace with a small implementation of the same functions**: `amdgpu_device.c`'s IP framework (a sequencer
  over the six `amd_ip_funcs` tables of si_common, gmc_v6_0, si_ih, gfx_v6_0, si_dma, si_smu); kernel BOs
  (`amdgpu_bo_create_kernel/_reserved/_at`, `kmap`, `gpu_offset`) from a static carve-out of VRAM and of pinned
  host memory; firmware loading (six Tahiti files: mc, pfp, me, ce, rlc, smc); IRQ registration (one MSI, an
  eventfd, a thread); PCI helpers; display hooks (`drm_mode_vrefresh` for DPM: no displays).
- **Emulate**: the dma-fence subset (refcount, signal, callbacks, timed wait) that ring and IB tests use; later,
  for user submissions, what RADV's CS ioctl needs from gpu-sched, dma-resv and syncobjs, in a single-client form.
- **Leave out**: TTM, drm_sched, KMS/DC, UVD/VCE, PSP/discovery/KFD/RAS/SR-IOV/XGMI, runtime PM.

Mac Pro specifics that must survive the port: the exact-bus VFCT image (done in `gpud-platform`, as kernel patch
0001 does for amdgpu); the SI DPM AC/DC fix (upstream in 7.3, comes with si_dpm.c); peer DMA between the D700s (in
VFIO, both GPUs in one container and each one's BAR0 mapped as an IOVA: no host-bridge P2PDMA whitelist involved,
only the 256 MiB window of the peer's VRAM is reachable).

## GCN 1 command submission: the plan (milestone 4)

How SI submits work, from `gfx_v6_0.c` and `si_dma.c` (7.3):

- **Rings.** One GFX ring (CP_RB0) and two compute rings (CP_RB1, CP_RB2) on the command processor; two async DMA
  rings (DMA0, DMA1 at +0x200 dwords). No doorbells on SI: the write pointer is an MMIO write (`CP_RBn_WPTR`,
  `DMA_GFX_RB_WPTR`); the read pointer is written back to memory (`*_RPTR_ADDR`).
- **Indirect buffers.** The ring carries `PACKET3_INDIRECT_BUFFER` (address, length, and the VMID in bits 24-27 of
  the control word: `gfx_v6_0_ring_emit_ib`); DMA uses `DMA_IB_PACKET` with the VMID in bits 20-23.
- **Fences.** GFX/compute: `SURFACE_SYNC` (TCL1, TC, K$ and I$ actions) then `EVENT_WRITE_EOP`
  (CACHE_FLUSH_AND_INV_TS_EVENT, 32/64-bit sequence number to a fence address, INT_SEL 2 for an interrupt once the
  write lands: `gfx_v6_0_ring_emit_fence`). DMA: `DMA_PACKET_FENCE` then `DMA_PACKET_TRAP`.
- **Interrupts.** The IH ring in host memory, written by the GPU at a bus address (an IOVA here); sources: CP EOP
  181, privileged register/instruction 184/185, DMA0/DMA1 trap 224/244, VM faults 146/147, thermal 230/231.
- **Addressing.** MC address space: VRAM at 0xF4_0000_0000 (6 GiB). amdgpu puts a 1 GiB GART (VM context 0, page
  table in VRAM) at MC 0 for kernel objects, and gives each client a VM of 256 GB, two levels (VMIDs 1-15). Every
  system page an entry points at is an IOVA from `DmaSpace`; nothing in the GPU's page tables is a host physical
  address.

Steps, each with what it must show:

1. **linuxkpi skeleton and atom.c.** atom.c needs only kmalloc, mutex, delays and printk. Parse the VFCT image
   for 02:00.0 and print the firmware info table (default sclk/mclk); the values must match what amdgpu reports
   for the same card. Then the register callbacks (`card_info`: MMIO, MC, PLL, I/O) on `Mmio`.
2. **Register funnel and IP sequencer.** `amdgpu_device_rreg/wreg` and the SMC/PCIe indirect accessors on BAR2;
   `sw_init`/`hw_init` for si_common (golden registers), gmc_v6_0 (MC program, GART on, VM manager), si_ih.
   Shows: GART entries written, VM_CONTEXT0 enabled, the IH ring's write pointer moving when an interrupt source
   fires; the MSI eventfd counting.
3. **si_dma.** Ring test (`DMA_PACKET_WRITE` to a write-back slot) and IB test (an IB with a WRITE, then FENCE
   and TRAP; the fence value arrives and the trap interrupt 224 is dispatched to `amdgpu_fence_process`). This is
   the first "no-op IB and fence", and it needs no microcode.
4. **gfx_v6_0.** Upload the PFP, ME, CE and RLC microcode through MMIO, start the CP (`cp_start`, clear state),
   then the ring test (SET_CONFIG_REG of SCRATCH_REG0) and IB test (the same in an IB, fence by EVENT_WRITE_EOP,
   interrupt 181) on the GFX ring and both compute rings. Done when all five rings pass `amdgpu_ring_test_helper`
   and the IB tests, with fence latency measured.
5. **si_dpm.** Upload tahiti_smc.bin, parse PowerPlay from the VBIOS, enable DPM. Done when the card reaches
   850 MHz sclk and 1370 MHz mclk under load, read the same way `scripts/with-clocks.sh` reads the native card.
6. **Recovery.** A bus reset through VFIO (`VFIO_DEVICE_PCI_HOT_RESET`, with the audio function also on
   vfio-pci), then ASIC init from the VBIOS through atom.c, then steps 2-5. Done when gpud brings the card up from
   reset without amdgpu ever having touched it.

Throughout, 06:00.0 stays on amdgpu as the oracle: same registers after init (`oracle.sh` in the results), same
clocks, and later the same engine results.

## After command submission

- **Milestone 5: compute.** A precompiled compute shader (GCN 1 ISA from RADV/ACO, or assembled with LLVM's
  AMDGPU backend for `tahiti`) dispatched through a compute IB in a client VM; then the DRM-ABI adapter, so RADV
  runs on gpud: device info (`AMDGPU_INFO` queries, including READ_MMR_REG of the registers in
  `si_allowed_read_registers`), BO create/VA/mmap, CS with IB/fence/BO-list/syncobj chunks, syncobjs. Done when
  `scripts/test-engine.sh` passes on one D700 under gpud and the single-GPU benchmark matches amdgpu's numbers on
  the same card.
- **Milestone 6: both GPUs, then the cluster.** Both D700s in one VFIO container, each one's BAR0 mapped for the
  other (shown possible in milestone 1); `bench/xgpu` P2P checks against the kernel numbers. Thunderbolt/USB4STREAM
  as its own service is a separate effort.

## Open questions

- Is a GPU write to an unmapped IOVA blocked silently, or does it never leave the GPU? No DMAR fault was reported;
  checking needs the IOMMU's fault registers (`CONFIG_INTEL_IOMMU_DEBUGFS` in a test kernel).
- How a client process maps CPU-visible VRAM (the 256 MiB BAR0 window) when the device belongs to gpud: share the
  VFIO device fd with the trusted client, or keep host-visible buffers in host memory.
- The DRM-ABI adapter: an `LD_PRELOAD` shim around libdrm (device discovery reads sysfs and expects a DRM char
  device), a patched libdrm_amdgpu, or a RADV winsys. Decide when milestone 5 starts, by what RADV actually needs.
- Synchronization model for user submissions: one submission thread per ring and sequence-number fences may be
  enough for one client; RADV's timeline syncobjs set the minimum.
