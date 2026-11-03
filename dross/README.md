# dross

See [Native program interfaces](API.md) for handles, memory objects, messages,
object-wait subscriptions and deadlines, and [Memory ownership and allocation](MEMORY.md)
for reclamation and allocation budgets.

dross is the kernel of Oscar, the operating system of the trashcan appliance: Mac Pro 6,1s running distributed
inference, with the drivers (`driver/`: gpud for the D700s, tbd for Thunderbolt, ethd for Ethernet) as user
programs. It is written in Rust. QEMU provides repeatable kernel tests; the Mac Pro
provides the hardware checks for its drivers.

Status: dross boots from UEFI, brings up all CPUs and the IOMMU, and runs user programs: processes, threads pinned
to CPUs, channels and notifications, memory objects, and handles as capabilities, including device memory, I/O
ports, MSIs and DMA domains. init runs a self-test of all of that at boot, then starts pcid and drivers; a driver
for QEMU's `edu` device checks the whole driver path (BAR, MSI, DMA through the IOMMU, and a blocked DMA), and
usbd drives USB keyboards and mice on an xHCI controller.

dross also runs Linux executables on musl, static or dynamically linked: gpud, tbd and ethd run on it unchanged,
built for `x86_64-unknown-linux-musl`, with gpud-platform choosing its dross backend at run time. Their unit tests
pass on dross, and each driver reaches its device through pcid. gpud and ethd also
run on a physical Mac Pro under dross. Vulkan rendering through gpud has passed
exact-pixel checks there. This does not establish that every driver or graphics
path has been tested on hardware.

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
| `user/rt/src/pci.rs` | The driver side: pcid's protocol and a small platform interface (`Device`, `Mmio`, `DmaSpace`, `IrqEvent`) |
| `../driver/platform/src/dross.rs` | The dross backend of `gpud-platform` (feature `dross`, no `std`): the interface gpud, tbd and ethd use on Linux |
| `user/edutest/` | A driver for QEMU's `edu` test device, written against `gpud-platform` as gpud is: the driver path end to end |
| `user/usbd/` | xHCI driver for USB keyboards and mice (HID boot protocol) on root ports: logs keys and mouse movement |
| `kernel/src/linux/` | The Linux system call interface for Linux executables |
| `linux/dyntest/` | A dynamically linked C program that dlopens a library: checks the dynamic linking path |
| `linux/posixtest/` | A Rust `std` program that checks the Linux interface (its own workspace: it needs unwinding) |
| `linux/rootfs/` | Files Linux programs see (packed into the `rootfs.tar` module) |
| `linux/*.conf` | What init runs: `qemu-init.conf` for `run-qemu.sh`, `macpro-init.conf` for a first boot on a Mac Pro |

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
- **Event queues** observe object conditions through explicit subscriptions. Each
  subscription reserves room for one event and must be rearmed after delivery.
  **Timers** signal expiration at an absolute monotonic deadline. The contracts
  and examples are in [Native program interfaces](API.md).
- **Resources** are the root capabilities: physical address ranges, I/O ports (loaded into the TSS bitmap when the
  process runs), interrupt vectors (`IRQ_MSI` gives the MSI address/data to program and signals a notification),
  the IOMMU (creating DMA domains, attaching devices) and the system (ending a run). init narrows them with
  `RESOURCE_SUB` before handing them on.

## Building and running

Needs the Rust targets `x86_64-unknown-uefi` and `x86_64-unknown-none`, Zig for fsd's freestanding lwext4 build,
and, for `run-qemu.sh`, QEMU with OVMF
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
(TCG on an arm64 Mac): the current audit build passes 55 native and 28 POSIX checks
with one CPU, and 60 native and 28 POSIX checks with four CPUs. The same 60 native
and 28 POSIX checks pass on a physical 12-CPU Mac Pro 6,1. Timings under TCG reflect
the emulator and the host's timer coalescing; use hardware runs for latency work.

Under TCG on macOS the host coalesces QEMU's timer wakeups: 1 kHz APIC timer ticks arrive 2-18 ms apart and the
missed ones are dropped. The timer check is therefore strict (count within 10%, no early tick) only on hardware;
in a VM it checks that ticks arrive. TCG has no TSC-deadline timer, so QEMU runs the periodic path.

## Linux programs

A process started with `SPAWN_LINUX` gets Linux's system call numbers; its native dross calls still work with
`NATIVE` set in the number (`dross-rt` sets it, so gpud-platform's dross backend works inside a Linux program).
The interface lives in the kernel for now (`kernel/src/linux/`):

- programs: static, static-PIE, and dynamically linked ones (their PT_INTERP, musl's dynamic linker
  `/lib/ld-musl-x86_64.so.1`, is loaded beside them and started with the auxiliary vector it needs);
- memory: anonymous and memfd mappings, files mapped by copying them (private, or shared read-only), `munmap`
  and `mprotect` of any range (mappings split), reservations, `membarrier`;
- threads: `clone` for threads, futexes (waking the oldest waiters), TLS (the FS base is part of a thread's
  state), clear-tid on exit; every thread is pinned where it was created (`sched_setaffinity` is accepted);
- files: a read-only tree from the `rootfs.tar` module, the ACPI tables in `/sys/firmware/acpi/tables` (gpud
  reads the VFCT there), `/dev/null`, `/dev/zero`, `/dev/urandom`, stdout and stderr into the kernel log;
- Unix sockets (stream, datagram, seqpacket) with `SCM_RIGHTS`, pipes, memfd, eventfd, poll;
- time, sleeping, `getrandom` (RDRAND);
- an fd for a dross handle (`HANDLE_FD`): a notification polls readable while bits are pending (tbd polls its
  interrupts with sockets), a memory object can be mapped (tbd hands its NHI's BAR to clients that way).

Not there: fork and exec, signal delivery (handlers are accepted; an abort ends the process), epoll, file
writes, and networking: ethd's and tbd's TAP interfaces need a network stack inside dross, so ethd can't run its
`run` mode and tbd runs streams only (`--net none`). (The `unix` kernel feature adds these, with a userspace
TCP/IP stack behind sockets: `unix/README.md`.)

```bash
./build-linux.sh        # target/linux: posixtest, gpud, tbd, ethd, test-*, dyntest and rootfs.tar
./run-qemu.sh           # now also runs the unit tests, posixtest, the drivers' probes and dyntest (linux/qemu-init.conf)
```

`build-linux.sh` builds static musl executables with `rust-lld` and Rust's own musl, gpud's Linux C with
`zig cc`, and dyntest and its library with zig against musl's dynamic linker, which it takes from Alpine's
minirootfs (downloaded once into `target/alpine`, checked by SHA-256); `FIRMWARE=DIR` adds `DIR` as
`/lib/firmware` (gpud's `init` and `serve` need Tahiti's microcode there, from linux-firmware).

## On a Mac Pro

dross runs on a Mac Pro 6,1 (E5-1650 v2, 6 cores x 2 threads, 48 GiB). Measured there (release build, booted once
with `grub-reboot` from Linux's GRUB, which chainloads the loader on the EFI system partition):

- "boot ok" 390 ms after the loader: 12 of 12 CPUs, each CPU's 1 ms TSC-deadline timer fires after 1000 us, the
  TSCs agree within 300 cycles, VT-d on with the USB controllers' reserved region identity-mapped;
- the self-test's 15 checks pass; 10 ms deadlines take 10.00 ms on every CPU; preemption shares a CPU evenly;
- the drivers' 42 unit tests pass as Linux programs;
- gpud's read-only probe passes on both D700s through pcid (config space, BARs, each card's VBIOS from the VFCT,
  memory controller registers, DMA buffers in the GPU's IOMMU domain); tbd's probe passes on a Thunderbolt
  controller; usbd brings up the Fresco Logic FL1100;
- `gpud init` brings both D700s up under dross as under VFIO: memory controller and GART through the IOMMU, the
  CP, RLC and SMC microcode (Tahiti's files from linux-firmware in the rootfs, `FIRMWARE=` in build-linux.sh), SI
  power management, and ring and IB tests on all five rings (gfx, two compute, two SDMA; IB tests 22-24 us, fence
  interrupts as dross notifications), 670 ms per card; `gpud dma-smoke` passes on both (the DMA engine writes VRAM
  and host memory, through the AGP aperture and the IOMMU, and its writes to unmapped addresses are stopped);
- `ethd serve` drives the BCM57762 for Oscar's network stack over a channel (OETH, `dross/unix/ethernet`): from
  the stack's side, an ARP request goes out and the gateway's reply comes back through it (0.6 ms);
- ethd sends the kernel log to the LAN (`ethd netlog`, below); at the end `reboot` resets back into Linux;
- with the `unix` kernel feature (`unix/README.md`), Oscar's Unix userland runs on the real network: ethd serve,
  netd (smoltcp) and Dropbear, logged into with OpenSSH from a Mac on the same LAN. A session (key exchange,
  public-key login, one command) takes 0.11 s; 7 MB go up at 5.1 MB/s and down at 10.1 MB/s (hashes checked).
  Uploads first ran at 125 KB/s: netd told smoltcp its device takes one frame at a time, and smoltcp then caps
  the receive window at one segment, so every segment waited for its acknowledgement.

Channel messages between threads on two CPUs: 2.7 us one way to a thread blocked in the kernel on a halted CPU
(p50; p99 2.8 us), 1.4 us to one spinning on non-blocking reads, round trips 5.6 us (p99 6.4 us, max 6.7 us over
1000). An earlier mean of 116 us was another thread busy on the same CPU (ethd bringing its link up) costing whole
time slices; new threads now go to the CPU with the fewest threads.

Without a serial console, the log comes over the network: ethd's `netlog` mode reads `/dev/kmsg` (dross's log
from boot) and sends each line as a UDP datagram to the netconsole port, and `driver/ethd/netlog-listen.py` on
another machine prints them. `init.conf` starts it before anything else: `start ethd netlog 0000:0c:00.0 --ip
ADDR --to RECEIVER --to-mac MAC` (macOS drops broadcasts from 0.0.0.0, so give real addresses). On hardware a
panic, a fatal kernel fault or init.conf's `reboot` resets the machine after 1 s (the FADT reset register), a
Linux program's `reboot(2)` (BusyBox `reboot -f`, e.g. over SSH) after 300 ms, and dross clears the chipset's
"stay off after power loss" bit, so a machine on a switched outlet can always be power-cycled.

What init runs comes from `init.conf`: `start PROG ARGS` (a Linux program) and `native PROG ARGS` (a dross
program) run before the self-test, up to the first `wait`; `channel NAME` makes a channel whose ends go to
programs as `@NAME=HANDLE` (a NAME that isn't a channel passes a duplicate of init's own handle, such as
`module/trash` or `display`); `netservice NAME` registers one end of a channel as the kernel's network service
(`unix` kernels) and hands the other to the stack; `mount NAME PATH` puts the read-only filesystem served at the
other end of channel NAME at PATH (`unix` kernels); `%KEY=VALUE` passes a value, `%KEY=$NAME` one of init's (the
kernel's `display-width` and so on); a `native usbd` line replaces the usbd init starts by itself; then, in order, `wait CHANNEL
[MS]` (init's end of the channel gets a message, e.g. netd's `ready`), the remaining `start` lines, `check PROG
CODE ARGS`, `sleep MS` and `reboot`.

### Oscar's desktop

The desktop (`desktop/`: a window server, toolkit, Trash shell and terminal) runs on the firmware framebuffer.
The kernel hands init the framebuffer as `display` with its geometry; while a program has it mapped, the log
stays off the screen (a panic or a reset still draws). usbd is the seat. With the `unix` kernel and the staged
userland (`unix/stage-hw.py`), the terminal runs BusyBox's shell on a PTY through `unix-terminal-bridge`:

```
channel seat
channel term
native usbd @seat=seat %width=$display-width %height=$display-height
start unix-terminal-bridge @term=terminal %columns=94 %rows=27
native oscar-session @module/oscar-windowd=module/oscar-windowd @module/oscar-terminal=module/oscar-terminal @display=display @seat=input @term=terminal %app=terminal %width=$display-width %height=$display-height %stride=$display-stride %format=$display-format
```

(94 by 27 cells is the terminal's default 760 by 500 window.) On the Mac Pro it draws on the 1600 by 1200
framebuffer, and a keyboard and mouse on the rear ports work. usbd used to switch a port's power off when it
reset it (its mask of the port bits to keep lacked port power; QEMU's controller ignores that bit).

### Disks

usbd drives one USB mass storage device through bulk-only transport and serves READ, WRITE and FLUSH on its
`block` channel, with a shared 1 MiB DMA window. fsd runs lwext4 outside the kernel and selects the first Linux
filesystem GPT partition (or `%partition=N`), else the whole disk. Mounts default to read-only. Add `%writable=1`
to fsd's bootstrap to enable supported journaled ext4 writes. Writable reads use bounded range requests;
read-only model/library access retains shared snapshots with a bounded kernel cache. Filesystem features,
persistence rules, supported operations and current limits are in [fsd/README](user/fsd/README.md).

```
channel disk
mount root /disk
native usbd @disk=block
native fsd @disk=block @root=service
```

`linux/disk-test.sh` checks read-only access in QEMU; `linux/disk-write-test.sh` checks writes and persistence
across two boots of the same disposable USB image. The earlier read-only implementation was measured on the
Mac Pro, from a SanDisk Extreme Pro on the FL1100's SuperSpeed port, formatted on Linux: the listings, a 50 MB file's hash, and a
7 GB GGUF (6652 MiB in 22.6 s, 308 MB/s, its SHA-256 as on Linux).

### Vulkan

`gpud serve` runs under dross on both D700s, and Alpine's musl builds of the Vulkan loader and Mesa's RADV (from
an Alpine root on the ext4 disk) reach them through `libgpud_drm.so`, as on Linux: `vulkaninfo` lists both GPUs
(RADV TAHITI, Mesa 26.1.6). Run with `LD_LIBRARY_PATH` naming the Alpine root's `lib` and `usr/lib`,
`VK_DRIVER_FILES` an ICD file whose `library_path` is relative to it, `LD_PRELOAD` the shim (built for musl with
`zig cc -target x86_64-linux-musl -shared`) and `GPUD_SOCKET` gpud's sockets (`--socket` paths without colons,
which separate the list). What it took: memory objects behind an address for DMA (`VM_MEMORY`, so gpud's
memfd-backed buffers map for the GPU); `open` and `stat` of `/proc/self/fd/N`; one inode per open file in
`fstat`, the same through every descriptor (the shim finds its device's memfd by it); copies between memory
objects page by page and kernel buffers sized by programs allocated fallibly (mapping LLVM's 182 MB library
asked the 64 MiB heap for all of it); and a 4 GiB buffer file per gpud client instead of 32 GiB, since dross
allocates a memfd's memory up front.

### trashcan on Oscar

trashcan itself runs on dross: built for musl in an Alpine container (Alpine's Rust and shaderc), from the ext4
disk, with Vulkan as above. It needs timeline semaphores, which libgpud_drm.so doesn't give the kernel driver
(no timeline syncobjs): Mesa 25 emulates them, Mesa 26 (Alpine 3.24) no longer does, so the Vulkan stack is
Alpine 3.22's (Mesa 25.1.9). Measured on the Mac Pro: Qwen3.5-9B with its MTP draft on both D700s, 42.7 tok/s
decode (46 on Linux), and `trashcan serve` with the Responses API on port 8000, started from init.conf.

`oscar-chat` (also `ocsar-chat`) talks to it from Oscar's shell, and `oscar-chat-gui` is the desktop's chat
window (`desktop/chat`). They reach the server on the same machine through netd's loopback interface:
127.0.0.0/8 and the machine's own address go to a second smoltcp interface on a loopback device. netd used to
end the sending side on `shutdown(SHUT_RD)`, so an HTTP server that shuts reading down before it answers
(trashcan's, for `Connection: close`) sent nothing.

```
channel chatwin
start oscar-chat-gui @chatwin=window
native oscar-session ... @chatwin=client/chat
```

### Network boot

A test machine boots the newest build over the network, so a deploy is a build and a reset:

```bash
./build-linux.sh && INIT_CONF=YOUR.conf PROFILE=release NETBOOT=1 ./build.sh   # target/netboot/dross.efi
./netboot.sh dross     # publish the completed image; linux selects the normal default
# An isolated, already tested image can be published directly:
./netboot.sh dross target/my-test/BOOTX64.EFI
```

`NETBOOT=1` builds the loader with the kernel and modules (stripped, 18 MB) built in, since a network boot has no
volume to read them from; `NETBOOT=1 ./run-qemu.sh` boots it from an otherwise empty volume. Serve
`target/netboot/` over HTTP and let the machine's GRUB (2.12) fetch `boot.cfg` at every boot, from
`/boot/grub/custom.cfg`:

```
if [ -z "${boot_once}" ]; then          # not after grub-reboot
    insmod efinet
    insmod http
    insmod chain
    net_ls_cards
    net_bootp                           # configure the firmware's network cards
    sleep 1
    net_add_dns ROUTER                  # DHCP gave GRUB no DNS server
    set dross_server=SERVER:PORT
    source (http,${dross_server})/boot.cfg
fi
```

`netboot.sh dross` copies the completed loader to a path named by its SHA-256 hash,
then atomically publishes `boot.cfg` pointing to it. An optional image argument
selects a separate tested build. A boot cannot fetch a half-written replacement.
`netboot.sh linux` atomically empties the configuration; without the server the
machine boots its normal default. On the Mac
Pro (Apple's firmware gives GRUB the BCM57762 through SNP), GRUB fetches the image in a second; from `reboot -f`
to dross's "boot ok" takes 59 s, most of it the firmware.

Linux is not part of the normal iteration loop. Keep SSH running in the dross
test configuration and omit its `reboot` line to inspect completed checks. Upload
and run changed applications over SSH. Close GPU clients before restarting their
driver so its fence and buffer retirement can finish. For a kernel change,
publish the new image on the host, then run `reboot -f` from dross. GRUB loads the
published image directly. Select `linux` before a reboot when recovery or an
independent disk check is needed.

Without the server, copy `target/esp/dross.elf` and `target/esp/modules/` to the root of the EFI system partition
and the loader to `EFI/dross/BOOTX64.EFI`, add a GRUB entry that chainloads it and `grub-reboot` into it.

Bugs only the hardware showed, now fixed: an AP's local APIC ignores unmasking LVT entries while it is software
disabled (its timer never fired, so timed waits hung); new threads started with MXCSR 0 (every floating-point
exception unmasked: gpud's tests died of a SIMD exception); pcid protected the largest possible MSI capability and
so refused the BCM57762's MISC_HOST_CTRL right after it; the dross DMA backend never reused device addresses; the
IOMMU's page tables lacked the snoop bit, so the GPU's no-snoop writes (its AGP aperture) went around the CPU caches
and the CPU kept reading stale data. Open: writes the IOMMU blocks log no fault on the real unit (they do in QEMU).

## The Mac Pro 6,1, as dross sees it

From the 12-core model's ACPI tables (`acpi/tests/macpro61`) and Linux on the same machine:

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

`gpud-platform` builds without `std` now (its VFIO backend and mock need the `std` feature, which stays the
default) and has a dross backend. What it doesn't do yet: resets (pcid has none), the expansion ROM, mapping
memory by address into a DMA domain (`map_external`; peer BARs will come from pcid as memory objects), and
counting interrupts (a notification reports one however often it fired).

Next: a network stack (ethd and tbd's ThunderboltIP), and the engine with
RADV, as Alpine's musl builds of Mesa and the Vulkan loader (shared libraries, dlopen'ed).
