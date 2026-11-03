# gpud: a userspace driver server for the Mac Pro 6,1's D700s

`gpud` is the start of a GPU driver that runs as an ordinary process: it owns one FirePro D700 (Tahiti XT, GCN 1)
through an IOMMU and serves the inference engine. Linux is the development harness; the interfaces are drawn so
the same driver can later run on a small microkernel with only the platform backend replaced. The engine in
`engine/` stays the reference implementation, unchanged, running on Linux's amdgpu.

Status: Linux's Southern Islands code (driver/linux, unmodified) runs in `gpud` on a small compatibility layer
(driver/shim) and brings the GPU up: memory controller and GART, interrupts, the GFX, two compute and two DMA rings,
all passing their ring and IB tests. `gpud serve` gives clients their own GPU VM, buffers and submission; with
`libgpud_drm.so` preloaded, an unmodified Mesa RADV runs on it, and so does the engine: Qwen3.5-9B (Q4_0) on one D700
gives activations bit-identical to amdgpu's and the same chat text. Clocks stay at the boot level until SI DPM is
ported, so it decodes at 2.1 tok/s against amdgpu's 24.7. The plan is `docs/plan.md`.

| Path | What |
|---|---|
| `platform/` | `gpud-platform` (Rust): the platform interface (PCI config space, MMIO, DMA/IOVA, interrupts), the VFIO backend, an in-memory mock, VBIOS lookup in ACPI VFCT |
| `gpud/` | `gpud` (Rust): `probe`, `dma-smoke`, `init` (bring-up and ring tests), `serve` (the driver server) |
| `linux/` | Linux 7.3-rc7's SI amdgpu files, unmodified (`PROVENANCE.md`) |
| `shim/` | The kernel API and amdgpu core they run on (C), and the client protocol (`gpud_proto.h`) |
| `client/` | `libgpud_drm.so` (RADV and libdrm onto gpud, loaded with `LD_PRELOAD`) and `drminfo` |
| `vkcompute/` | Vulkan compute checks, run the same way on gpud and on amdgpu |
| `docs/vfio.md` | How a D700 is handed to VFIO on a Mac Pro 6,1, and what the host needs |
| `docs/linux-deps.md` | Measured dependency audit of Linux 7.3's SI amdgpu code |
| `docs/plan.md` | Architecture and the plan to command submission and beyond |

## Building and running

`driver/` is its own Cargo workspace. Tests run anywhere (`cargo test` in `driver/`); the VFIO backend builds on
Linux only (`cargo build --release --target x86_64-unknown-linux-gnu`). On the Mac Pro, with the GPU set up as in
`docs/vfio.md` and nothing else using it:

```bash
sudo ./gpud probe 0000:02:00.0              # --read-only skips the VRAM test and the MSI route
sudo ./gpud dma-smoke 0000:02:00.0
sudo ./gpud init 0000:02:00.0               # bring-up, ring and IB tests, then shut the engines down
sudo ./gpud serve 0000:02:00.0 &            # the server, on /run/gpud/gpud.sock
LD_PRELOAD=$PWD/libgpud_drm.so vulkaninfo --summary
LD_PRELOAD=$PWD/libgpud_drm.so ./vkcompute
```

`gpud serve` stops the engines on SIGINT or SIGTERM, and logs the GPU's state on SIGUSR1 (for a stuck submission).
After a SIGKILL the next `init` or `serve` stops what was left running, with one exception: a kill in the middle of a
dispatch can leave shader waves that never finish (their host memory is gone with the process), and SI can only
clear those with an ASIC reset and a VBIOS re-POST, which `gpud` can't do yet. `init` and the IB tests still pass,
but client dispatches hang; reboot. `GPUD_DEBUG=1` adds amdgpu's debug messages and a marker per bring-up and
teardown step; `GPUD_KMSG=1` copies every log line to the kernel log (with `sysctl kernel.printk_devkmsg=on`), so
netconsole shows the last step before a hang. If the GPU stops answering MMIO, `gpud` stops touching it at the first
dead read and waits to be killed; the GPU then needs a reboot.

`client/build.sh` builds `libgpud_drm.so` and `drminfo` (in the cross-compiling VM).

## Licenses

The Rust code here is MIT, like the rest of the repository. The Linux amdgpu code the driver will import (see
`docs/plan.md`) keeps its own license: the SI hardware files carry AMD's MIT-style text, and anything GPL-2.0 from
the kernel (for example `drivers/dma-buf/`) keeps the GPL, which then covers the `gpud` binary as a whole. Imported
files go under `driver/linux/` unmodified where possible, with their upstream path, version and changes recorded
next to them. The engine never links any of it; it talks to `gpud` across a process boundary.
