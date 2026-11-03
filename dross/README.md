# dross

dross is the kernel of Oscar, the operating system of the trashcan appliance: Mac Pro 6,1s running distributed
inference, with the drivers (`driver/`: gpud for the D700s, tbd for Thunderbolt, ethd for Ethernet) as user
programs. It is written in Rust and developed in QEMU; Linux stays the harness for the drivers until dross can
host them.

Status: dross boots from UEFI, brings up all CPUs and the IOMMU, and runs user programs: processes, threads pinned
to CPUs, channels and notifications, memory objects, and handles as capabilities, including device memory, I/O
ports, MSIs and DMA domains. init runs a self-test of all of that at boot, then starts pcid and drivers; a driver
for QEMU's `edu` device checks the whole driver path (BAR, MSI, DMA through the IOMMU, and a blocked DMA), and
usbd drives USB keyboards and mice on an xHCI controller.

| Path | What |
|---|---|
| `boot/` | `dross-boot`, the UEFI loader: loads `\dross.elf` from its own volume, builds the page tables, hands over (`x86_64-unknown-uefi`) |
| `bootinfo/` | The handover between loader and kernel: memory map, RSDP, framebuffer, the page table layout |
| `kernel/` | The kernel (`x86_64-unknown-none`) |
| `acpi/` | Parsers for the static ACPI tables (RSDP, XSDT, MADT, MCFG, HPET, DMAR, FADT, `\_S5`); tested on the Mac Pro 6,1's own tables (`acpi/tests/macpro61`) |
| `pmm/` | The physical frame allocator (a bitmap; contiguous, aligned and below-a-ceiling allocations) |
| `abi/` | The system call interface (numbers, errors, rights, flags) |
| `user/rt/` | `dross-rt`, the runtime of every program: system calls, entry, heap, threads, bootstrap messages |
| `user/init/` | The first process: holds the root resources and starts the rest |
| `user/selftest/` | Checks the kernel from user space at every boot |
| `user/pcid/` | Owns PCI config space; gives drivers their functions (filtered config access, BARs, a DMA domain each, MSIs) |
| `user/rt/src/pci.rs` | The driver side: pcid's protocol and the platform interface (`Device`, `Mmio`, `DmaSpace`, `IrqEvent`), shaped like `gpud-platform` |
| `user/edutest/` | A driver for QEMU's `edu` test device: the driver path end to end |
| `user/usbd/` | xHCI driver for USB keyboards and mice (HID boot protocol) on root ports: logs keys and mouse movement |

## What the kernel does at boot

1. The loader maps all physical memory (RAM and everything below 4 GiB) at `0xffff800000000000` with 2 MiB
   pages, and the kernel image at `0xffffffff80000000`; it allocates the kernel's stack, a trampoline page
   below 1 MiB and the page tables below 4 GiB, then leaves boot services and jumps.
2. The log goes to COM1 (QEMU) and to the firmware framebuffer (the Mac Pro has no serial port).
3. CPU features (x2APIC and NX are required), the frame bitmap from the UEFI memory map, GDT/TSS with separate
   stacks for double fault, NMI and machine check, an IDT covering all 256 vectors.
4. ACPI: the tables above, checked; the VFCT (VBIOS images) is located for gpud.
5. Time: the HPET as reference, the TSC calibrated against it (median of three 10 ms runs).
6. Interrupts: the 8259s masked, every I/O APIC input masked, the local APIC in x2APIC mode with LINT1 as NMI.
   The APIC timer runs in TSC-deadline mode when the CPU has it (the Mac Pro does), periodic otherwise.
7. SMP: each CPU in the MADT gets INIT and STARTUP IPIs; the trampoline goes from real mode straight to long
   mode on the kernel's page tables. Each CPU loads its own GDT/TSS/IDT and x2APIC, then takes a test IPI.

8. The identity map goes; every process shares the kernel half of the page tables.
9. VT-d: translation on with every device blocked until something attaches it to a DMA domain; reserved
   regions (RMRRs) identity-mapped for their devices first.
10. init starts from the `init` module with a bootstrap message: the other modules and the root resources.

Faults in the kernel print the registers and stop the machine (in QEMU, end the run with a failure); faults in
user mode end the process.

## The kernel's interface

System calls take handles: per-process integers naming kernel objects, each with rights (`abi/src/lib.rs`).

- **Processes and threads.** `SPAWN` loads a static ELF from a memory object and passes one handle (a bootstrap
  channel). Threads are pinned: each CPU has its own run queue, a time slice runs only when others wait, and a
  CPU with nothing to do and no deadlines takes no timer interrupts. FPU/SSE/AVX state is saved with XSAVE.
- **Memory objects.** Pinned and zeroed at creation, mapped with `VM_MAP` (read/write/execute) into any number
  of places, and into DMA domains. `MEM_PHYSICAL` makes one from device memory (uncached), only outside RAM.
- **Channels** carry bytes and handles (which move); writes don't block, reads wait with a deadline.
  **Notifications** are 64 bits that signals OR in and a wait takes; a server binds its channels and interrupts
  to one notification and waits there.
- **Resources** are the root capabilities: physical address ranges, I/O ports (loaded into the TSS bitmap when the
  process runs), interrupt vectors (`IRQ_MSI` gives the MSI address/data to program and signals a notification),
  the IOMMU (creating DMA domains, attaching devices) and the system (ending a run). init narrows them with
  `RESOURCE_SUB` before handing them on.

## Building and running

Needs the Rust targets `x86_64-unknown-uefi` and `x86_64-unknown-none` and, for `run-qemu.sh`, QEMU with OVMF
(Homebrew's `qemu` has both).

```bash
cargo test -p acpi-tables -p pmm        # host tests, including the Mac Pro tables
./build.sh                              # target/esp: EFI/BOOT/BOOTX64.EFI, dross.elf, modules/
./run-qemu.sh                           # SMP=24 MEM=8G by default; PROFILE=release for an optimized build
```

`run-qemu.sh` boots a q35 machine shaped like the Mac Pro (Ivy Bridge CPU, 12 cores x 2 threads, an Intel
IOMMU without interrupt remapping) plus QEMU's `edu` device and an xHCI controller with a full-speed keyboard and
a high-speed mouse. Once usbd reports its devices ready, the script types "ab", moves the mouse and clicks
through QEMU's monitor. It exits 0 when init reports that every check passed (usbd saw the input) and the kernel
logged the IOMMU fault for the edu driver's deliberate DMA to an unmapped address. The log is in
`target/qemu.log`. Measured on QEMU 11.1
(TCG on an arm64 Mac): 24 of 24 CPUs online; the self-test's 14 checks pass with 4 and 24 CPUs, debug and release
builds. Its timings under TCG (a 20 ms deadline taking up to 90 ms, 50-460 us channel round trips between CPUs)
reflect the emulator and the host's timer coalescing, not the kernel.

Under TCG on macOS the host coalesces QEMU's timer wakeups: 1 kHz APIC timer ticks arrive 2-18 ms apart and the
missed ones are dropped. The timer check is therefore strict (count within 10%, no early tick) only on hardware;
in a VM it checks that ticks arrive. TCG has no TSC-deadline timer, so QEMU runs the periodic path.

## The Mac Pro 6,1, as dross sees it

From its ACPI tables (`acpi/tests/macpro61`) and Linux on the same machine:

- 24 CPUs, x2APIC IDs 0-11 and 16-27; two I/O APICs (GSIs 0-23 and 24-47); LINT1 is NMI on every CPU.
- One VT-d unit for all devices, 46-bit DMA, **no interrupt remapping**: a device that can DMA can also
  write MSIs, so drivers in userspace are isolated against faults, not against malice. One reserved region
  (RMRR) for the USB controllers at 00:1a.0 and 00:1d.0.
- ECAM at 0xe0000000 for buses 0-255; HPET at 0xfed00000; reset by writing 6 to I/O port 0xcf9; soft-off
  is SLP_TYP 7.
- The TCO watchdog can't be used: the firmware locks its NO_REBOOT bit (Linux: "unable to reset NO_REBOOT
  flag, device disabled by hardware/BIOS").
- The Apple SSD's AHCI controller runs on legacy INTx under Linux (IRQ 16), not MSI.

## Where it is going

The kernel keeps only what needs ring 0 or is global to the machine; everything with a device behind it is a
server.

- **Kernel**: CPUs, physical memory (pinned, contiguous, 2 MiB pages), address spaces, threads and a scheduler
  that pins and can leave cores tickless for busy polling, IPC and shared memory, interrupts delivered as
  notifications, timers, the IOMMU's page tables (DMA safety depends on knowing when memory is freed; one GPU's
  BAR must be mappable for the other), capabilities for MMIO and I/O port ranges, a few MSRs (thermal status,
  P-states), reset and power-off, the log.
- **Servers**: init (holds capabilities, restarts servers), pcid (enumeration, config space with the MSI
  capability and command register filtered, bus resets, AER), gpud, tbd, the Ethernet driver with a TCP/IP
  stack, the SMC (fans: it must set full speed before any GPU work), storage (AHCI, read-only model store), and
  the engine with RADV on a POSIX layer.
- **Data paths** (Thunderbolt streams, GPU submission and fences) run through shared-memory rings mapped into
  the engine, busy-polled on dedicated cores; the servers own setup, teardown and recovery.

usbd's limits: devices must be on a root port (no hubs: Apple keyboards with a built-in hub won't work yet), boot
protocol only, and it has run only on QEMU's xHCI. On the Mac Pro the rear ports are on a Fresco Logic FL1100
(1b73:1100, MSI); low-speed devices and the FL1100's quirks are untested.

Next: a dross backend for `gpud-platform` so the existing drivers can run on it.
