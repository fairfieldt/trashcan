# Owning a D700 through VFIO on a Mac Pro 6,1

What the host needs before `gpud` can own a GPU, and why. Measured on a Mac Pro 6,1 (2x FirePro D700) running the
project's Linux 7.3.0-rc7-p2p kernel.

## Topology

| Function | What | Root port | IOMMU group (with the IOMMU on) |
|---|---|---|---|
| 02:00.0 / 02:00.1 | D700, VBIOS 113-C3861XA-028, subsystem 106b:0128 / its HDMI audio | 00:02.0 (x16 gen 3) | its own: just these two |
| 06:00.0 / 06:00.1 | D700, VBIOS 113-C3861XB-028, subsystem 106b:0127 / its HDMI audio | 00:03.0 (x16 gen 3) | its own: just these two |

06:00.0 is the boot VGA device: EFI's framebuffer (`efifb`, 1600x1200) lives in its BAR0, and the console and the
login screen run on it. 02:00.0 drives no display and is the one to hand to `gpud`.

Each GPU has BAR0 (256 MiB prefetchable, the host's window on VRAM), BAR2 (256 KiB, registers), BAR4 (256 bytes of
I/O ports) and a 128 KiB expansion ROM. The root ports implement ACS (source validation, request and completion
redirect, upstream forwarding), which the kernel turns on with the IOMMU, so no ACS override is needed. Separate
groups are the kernel's finding, not just the separate root ports: check `/sys/kernel/iommu_groups/`.

The GPU has no FLR. Its PM capability says `NoSoftRst-`, so a D3hot to D0 transition resets it. The kernel lists
`bus` as its reset method: a secondary bus reset of the root port, which also resets the audio function.

## The IOMMU

The DMAR table has one DMA remapping unit covering every device (`INCLUDE_PCI_ALL`), an RMRR for the USB
controllers only (00:1d.0, 00:1a.0), and ATS allowed below root ports 00:01.0, 00:02.0 and 00:03.0. Linux doesn't
turn the IOMMU on by default with this kernel config, so boot with

    intel_iommu=on iommu=pt

`iommu=pt` keeps every host-owned device (the other GPU, Thunderbolt, network, storage) identity-mapped as before;
only devices handed to VFIO get a translated domain.

**There is no interrupt remapping.** The DMAR table's flags are 0 (the INTR_REMAP bit is clear) and it lists no
IOAPIC, so the kernel can't enable it ("x2apic: IRQ remapping doesn't support X2APIC mode"), and VFIO's type1
backend refuses to attach a group unless `vfio_iommu_type1.allow_unsafe_interrupts=1`. What that costs: a device
owned by userspace can write to the MSI address range and raise interrupts on the host. DMA remapping is
unaffected. For a single trusted driver it is a robustness risk, not an isolation boundary; a microkernel on this
machine has the same limit and must treat the GPU driver as trusted with respect to interrupts.

Unmapped IOVAs: the GPU's DMA engine wrote to three of them and nothing landed in mapped buffers, but no DMAR
fault was reported and the fault interrupt count stayed 0. Whether the IOMMU dropped the writes silently or they
never left the GPU is open (the boot log's "handling fault status reg 702" with no record printed hints at stale
fault records from firmware). Don't rely on DMAR faults to catch a stray GPU write on this machine yet.

## Handing a GPU to vfio-pci without resetting it

`results/driver/vfio-20261102/to-vfio.sh` does this; the steps and why:

1. `vfio_pci.disable_idle_d3=1` (kernel command line, or before vfio-pci loads): otherwise vfio-pci puts an
   unused device in D3hot, and with `NoSoftRst-` coming back to D0 resets the GPU.
2. Clear the GPU's reset methods (`echo > /sys/bus/pci/devices/0000:02:00.0/reset_method`): otherwise every open
   through VFIO resets the function (`vfio_pci_core_enable` calls `pci_try_reset_function`).
3. Unbind the HDMI audio function and keep it driverless (`driver_override` set to `none`), rather than binding it
   to vfio-pci: when every device on the bus is in vfio-pci's device set, the last close does a bus reset
   (`vfio_pci_dev_set_try_reset`). An unbound device keeps the group viable.
4. Unbind the GPU from amdgpu (`.../drivers/amdgpu/unbind`), set its `driver_override` to `vfio-pci` and probe.

amdgpu's unbind of an idle D700 is clean ("finishing device"); GDM's Xwayland had the card open and carried on.
What the GPU keeps afterwards is what amdgpu's teardown left: the memory controller trained and configured (VRAM
reachable), engines stopped, the GART off. The async DMA engine still runs without microcode in that state, and
its writes reach host memory through the memory controller's AGP aperture (`gpud dma-smoke`). Coming back from a
reset needs the VBIOS's ASIC init (`atom.c`), which `gpud` doesn't have yet.

To give the GPU back without a reboot (not tried yet): unbind it from vfio-pci, clear `driver_override`, `echo
default > reset_method`, and probe it so amdgpu takes it. A reboot always restores everything.
