# gpud milestone 1: one D700 owned through VFIO on incite (2026-11-02)

Source: trashcan main 2be71989 (`driver/`: gpud-platform, gpud), built with `cargo build --release --target
x86_64-unknown-linux-gnu` in the kbuild VM; binary sha256 0c416e4d09786356e0ad089c629eef2b16d2c40bd8936ad688d3ff239d6aeb99.
`probe-final.txt` and `dma-smoke-final.txt` are that binary's runs; the earlier files are the development runs the
findings below were first seen in (same checks, before formatting and clippy fixes). incite: Linux 7.3.0-rc7-p2p
(`#1p2p+ps9c9a5449`), Debian 13, gpuq lease #327 for every GPU step. Netconsole to this Mac throughout
(`netconsole-recv.py`; kernel lines in `netconsole-kernel.txt`): no oops, no AER, no DMAR fault message.

## Host changes on incite (user-approved)

- `/boot/grub/custom.cfg`: menu entry `gpud-iommu` = the 7.3.0-rc7-p2p entry plus `intel_iommu=on iommu=pt
  vfio_pci.disable_idle_d3=1`. Booted once with `grub-reboot gpud-iommu` (the default entry is unchanged; a reboot
  goes back to it). The reboot took 30 s.
- At runtime, until the next reboot: 02:00.0 on vfio-pci with reset methods cleared, 02:00.1 unbound
  (`driver_override` none) (`to-vfio.sh 02`), and `vfio_iommu_type1 allow_unsafe_interrupts=1`.
- 06:00.0 stayed on amdgpu throughout (boot VGA, GDM).

## Findings

1. **IOMMU groups** (`/sys/kernel/iommu_groups`): {02:00.0, 02:00.1} = group 51, {06:00.0, 06:00.1} = group 52;
   root ports 00:02.0 and 00:03.0 alone in groups 2 and 3. ACS on the root ports was enabled by the kernel (SV RR
   CR UF). No override needed.
2. **No interrupt remapping**: DMAR flags 0x0 (INTR_REMAP clear), no IOAPIC scope (`DMAR.bin` decoded: DRHD
   INCLUDE_PCI_ALL; RMRR 0x7e7d3000-0x7e7d5fff for 00:1d.0 and 00:1a.0; ATSR for root ports 00:01.0, 00:02.0,
   00:03.0). Kernel: "x2apic: IRQ remapping doesn't support X2APIC mode". VFIO_SET_IOMMU failed with EPERM
   ("No interrupt remapping support") until `allow_unsafe_interrupts=1`.
3. **amdgpu unbind of 02:00.0 is clean**: "finishing device", GDM's Xwayland (holding card0) carried on, no
   warnings.
4. **Reset traps** (from vfio-pci and PCI core source, 7.3): vfio-pci resets a function on every open
   (`pci_try_reset_function`) and does a bus reset on the last close when every function on the bus is in its
   device set; D3hot->D0 resets the D700 (`NoSoftRst-`). Hence the cleared reset methods, the unbound audio
   function and `disable_idle_d3`. With them, every open kept the GPU's state (MEMSIZE, FB location, VRAM
   contents).
5. **VBIOS**: the D700's ROM BAR is blank (vfio-pci: "Invalid PCI ROM header signature: expecting 0xaa55, got
   0x0000"; VFIO reports the ROM region as size 0). VFCT holds three entries: bus 02 (113-C3861XA-028, 64 KiB), bus
   06 (113-C3861XB-028, 64 KiB) and an empty terminator. gpud's exact-bus pick for 02:00.0 is byte-identical to
   what amdgpu loaded (sha256 129d6f3a204fd88b7acda00325b862faf8b56ff1ae32a9560f3a58e9558609ff for both).
   VFCT.bin sha256 4ff79815b4b5bb4a4c36d4aa2216e8efefe72a0ea86c98c129a152390f6f8402 (binaries kept on incite in
   `~/trashcan-exp/gpud/`, not committed).
6. **Registers through VFIO vs amdgpu's debugfs** (`oracle-02-amdgpu/amdgpu-regs.txt` before the unbind,
   `probe-readonly.txt` after): identical except what amdgpu's teardown switched off: VM_L2_CNTL 0x000b8603 ->
   0x000b8600, VM_CONTEXT0_CNTL 0x00049259 -> 0 (GART off), CP_ME_CNTL 0 -> 0x15000000 (ME/PFP/CE halted),
   RLC_CNTL 1 -> 0, IH_RB_CNTL 0x0001011d -> 0x0001011c (IH ring off), HDP_HOST_PATH_CNTL 0x0f200029 ->
   0x0fa00029. CONFIG_MEMSIZE 6144, MC_VM_FB_LOCATION 0xf57ff400 (VRAM at MC 0xF400000000, as amdgpu logs).
7. **VRAM** (`probe-full.txt`): 4 KiB written and read back through BAR0 at window offset 0xff00000; the same
   VRAM offset read through MM_INDEX/MM_DATA; VRAM at 0x17ffff000 (past the 256 MiB window) written and read back
   through MM_INDEX/MM_DATA. Contents restored.
8. **IOMMU mappings**: IOVA ranges 0x100000-0xfedfffff and 0xfef00000-0xffffffffff (MSI window excluded; gpud's
   window is [1 MiB, 2^40)); page sizes 4K/2M/1G; 65535 mappings available. 4 KiB, 2 MiB and 64 MiB buffers mapped
   at aligned IOVAs; freed IOVA reused. BAR0 (2 MiB of it) mapped into the container as a peer target: accepted.
9. **MSI**: routed to an eventfd and removed; no interrupts in 20 ms (the IH is off).
10. **Device-initiated DMA** (`dma-smoke-3.txt`): the async DMA engine (DMA0, no microcode), ring in VRAM, wrote
    0xDEADBEEF to VRAM and then dwords to host buffers at IOVA 0x100000 and 0x9ff000 through the MC's AGP
    aperture (MC [0, 4 GiB) -> bus addresses, L1 TLB as amdgpu left it, 0x18). Writes to unmapped IOVAs 0x101000,
    0xf0000000 and 0x40000000 did not land in the mapped buffers, but **no DMAR fault was reported** (DMAR-MSI
    interrupt count stays 0 since boot). Open: whether the IOMMU blocked them silently or they never left the GPU.
    The boot log has "DMAR: DRHD: handling fault status reg 702" without a fault record printed, so stale firmware
    fault records may keep PPF set; reading FSTS/FRCD needs `CONFIG_INTEL_IOMMU_DEBUGFS`.

## Commands

```bash
# lease, netconsole receiver, one-time boot
gpuq --on incite request gpud-vfio --est 120 --owner claude-gpud; gpuq --on incite wait 327
python3 -I netconsole-recv.py netconsole-incite.log &
ssh incite 'sudo install -m 644 gpud-custom.cfg /boot/grub/custom.cfg && sudo grub-reboot gpud-iommu && sudo systemctl reboot'
# before the handoff: amdgpu's view of 02:00.0
ssh incite 'gpuq run 327 -- sudo ~/trashcan-exp/gpud/oracle.sh 0000:02:00.0 0 ~/trashcan-exp/gpud/oracle-02-amdgpu'
ssh incite 'gpuq run 327 -- sudo ~/trashcan-exp/gpud/to-vfio.sh 02'
ssh incite 'echo 1 | sudo tee /sys/module/vfio_iommu_type1/parameters/allow_unsafe_interrupts'
ssh incite 'cd ~/trashcan-exp/gpud && gpuq run 327 -- sudo ./gpud probe 0000:02:00.0 --read-only'
ssh incite 'cd ~/trashcan-exp/gpud && gpuq run 327 -- sudo ./gpud probe 0000:02:00.0 --dump-vbios vbios-02-vfct.bin'
ssh incite 'cd ~/trashcan-exp/gpud && gpuq run 327 -- sudo ./gpud dma-smoke 0000:02:00.0'
```

Not measured: clocks (no engine load; DPM is off on 02:00.0 since amdgpu's teardown), anything with the CP.

Side effect while 02:00.0 is on vfio-pci: fancurve.service sees one amdgpu GPU and runs the fan at full speed
(2350 rpm from 14:06:26, its rule for a GPU it can't read). Nothing reads 02:00.0's temperature in this state; it
is idle at boot clocks.
