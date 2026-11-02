# gpud: a userspace driver server for the Mac Pro 6,1's D700s

`gpud` is the start of a GPU driver that runs as an ordinary process: it owns one FirePro D700 (Tahiti XT, GCN 1)
through an IOMMU and serves the inference engine. Linux is the development harness; the interfaces are drawn so
the same driver can later run on a small microkernel with only the platform backend replaced. The engine in
`engine/` stays the reference implementation, unchanged, running on Linux's amdgpu.

Status (milestone 1): `gpud` owns one GPU through VFIO and has checked config space, the VBIOS, registers, VRAM,
IOMMU mappings, MSI routing, and a write by the GPU's own DMA engine into host memory through the IOMMU. No engine
beyond that one write has run yet; there is no command submission. The plan for it is `docs/plan.md`.

| Path | What |
|---|---|
| `platform/` | `gpud-platform` (Rust): the platform interface (PCI config space, MMIO, DMA/IOVA, interrupts), the VFIO backend, an in-memory mock, VBIOS lookup in ACPI VFCT |
| `gpud/` | `gpud` (Rust): `gpud probe` and `gpud dma-smoke` |
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
```

## Licenses

The Rust code here is MIT, like the rest of the repository. The Linux amdgpu code the driver will import (see
`docs/plan.md`) keeps its own license: the SI hardware files carry AMD's MIT-style text, and anything GPL-2.0 from
the kernel (for example `drivers/dma-buf/`) keeps the GPL, which then covers the `gpud` binary as a whole. Imported
files go under `driver/linux/` unmodified where possible, with their upstream path, version and changes recorded
next to them. The engine never links any of it; it talks to `gpud` across a process boundary.
