# Unix tools and SSH on Oscar

This is an opt-in Unix development image. It builds BusyBox 1.37.0 and Dropbear
2026.94 as static x86-64 musl executables and runs them on dross. The regular
hardware build, init program and init.conf files do not enable it.

## Build and test

Requires Rust's `x86_64-unknown-none`, `x86_64-unknown-uefi` and
`x86_64-unknown-linux-musl` targets, Zig, a host C compiler, make, Python, QEMU/OVMF
and OpenSSH client tools.

```sh
cd dross/unix
./build.sh
./run-qemu.sh
SMP=1 ./run-qemu.sh
```

All build output, downloaded source, guest keys and test logs stay under this
workspace's `target/` or ignored `vendor-cache/`. In particular, this does not
replace `dross/target/esp`, modify a mounted ESP, boot a Mac Pro, configure a real
NIC or change the hardware bring-up configuration. Source archives are pinned by
SHA-256 in `build.py`; Cargo dependencies are locked in both workspaces. Upstream
license texts are included in the guest at `/usr/share/licenses`.

The runner forwards **127.0.0.1:2222** to guest TCP port 22. `SSH_PORT` selects a
free local port when running another guest. It creates dedicated test identities,
uses a known host key with strict verification, ignores the user's SSH config and
agent, and never adds keys to the agent or the user's known_hosts. The test image
contains test private keys and is not an installable production root filesystem.

The suite checks real host OpenSSH connections, remote commands and exit codes,
separate stdout/stderr, binary upload/download with SHA-256 verification, PTY input
and resize, interactive ash with Ctrl-Z, foreground resume and Ctrl-C, rejection of an unauthorized
key, repeated connections, and daemon shutdown. The guest checks process and fd
semantics, writable files, symlinks/scripts, permissions, signals, shared mappings,
quota reclamation and PTYs. The native channel-to-PTY bridge is exercised before
SSH starts. Logs are `target/qemu.log`, `target/ssh-pty.log` and
`target/ssh-interactive.log`.

## Process graph

```text
Unix test init
  ├─ pcid → unix-nic (QEMU e1000, BAR + private VT-d domain)
  │              ↕ OETH frame channel
  ├─ netd (smoltcp IPv4/TCP, no PCI or system resource)
  │              ↕ socket capability channels
  └─ Linux programs: BusyBox ash, utilities, Dropbear, terminal bridge
                         ↕ shared kernel PTYs
                    local Oscar terminal / SSH channels
```

The kernel extension is behind Cargo feature `unix`. It implements the Linux ABI
mechanisms required by these programs; it contains no Ethernet driver or TCP
stack. The default dross build retains its driver-harness personality.

`unix-init` is a separate test supervisor copied as `init` only into this suite's
ESP. It uses the normal native process/channel APIs and registers netd with native
`UNIX_NET_REGISTER` (33): `(system resource, channel)`. Only the supervisor receives
the system resource. The socket RPC contract is in [netd/PROTOCOL.md](netd/PROTOCOL.md).
Every socket gets its own channel capability; its identity
comes from that endpoint, not a client-supplied socket ID. A fork shares an open
socket description, with serialized bounded RPCs. Ordinary signals are deferred
until an internal RPC response is consumed, so replies cannot become misaligned.

Socket `poll`/`select` currently query readiness on a bounded timer. They preserve
the caller's timeout and nonblocking behavior; this is a correctness baseline, not
a performance result. Frames and socket buffers are bounded. No GPU is used.

## NIC interface and hardware boundary

`oscar-ethernet` is a small `no_std` crate with the shared codec and host tests:

```sh
cargo test -p oscar-ethernet
```

The complete wire contract is documented in [ethernet/PROTOCOL.md](ethernet/PROTOCOL.md).
The QEMU driver runs in its own process, owns its PCI function and DMA domain, and
implements that protocol. The BCM57762/BCM57766 driver can provide the same endpoint
without changing netd or the kernel's socket ABI. Physical link state and interrupts
belong to the driver; channel messages wake netd. The existing hardware `ethd
netlog` mode is unaffected.

netd bootstrap handles: `@ethernet` (driver endpoint), `@service` (kernel's socket
factory endpoint), `@ready` (one-shot ready report). Optional values: `ip` (IPv4),
`prefix` (0..32), `gateway` (IPv4). The default QEMU configuration is 10.0.2.15/24
through 10.0.2.2. With an explicit `ip`, no gateway is assumed. The server currently
supports IPv4 TCP, not UDP, IPv6, DHCP, DNS configuration or a loopback interface.

Hardware integration must supply the real NIC endpoint and addressing, enable the
Unix kernel feature deliberately, provision account/key files, and arrange secure
entropy and persistent configuration. It is separate from the QEMU test image.

## Local graphical terminal

`unix-terminal-bridge` is a static musl program. Its bootstrap contains
`@terminal`, a duplex raw-byte channel, and optional `columns`/`rows` (default
80×24; must match the frontend's grid). It converts that explicitly supplied
channel to a Unix fd, allocates a PTY, starts `/bin/sh -i` as a session leader, and
forwards bytes. Closing the frontend hangs up and reaps the shell.

A desktop supervisor creates a channel pair, gives one endpoint to this bridge
and the other to `oscar-terminal` (or `oscar-session` with `app terminal`). This
uses the same PTY implementation as Dropbear. It requires no change to the window
protocol or terminal parser. The existing graphical bootstrap still needs its
separate exclusive framebuffer/input handoff before hardware use.

## Implemented semantics and limits

* Eager private-memory fork with shared open descriptions, shared mappings, child
  status/reaping and close-on-exec; transactional ELF/shebang exec; empty
  arguments preserved. No copy-on-write performance claim.
* Dynamically linked programs: exec loads the PT_INTERP interpreter beside the
  program. The image stages Alpine 3.24's musl dynamic linker
  (`/lib/ld-musl-x86_64.so.1`) and its dynamically linked BusyBox
  (`/opt/alpine/bin/busybox`), plus `/usr/bin/dyntest` (`../linux/dyntest`),
  which dlopens `/usr/lib/libdrosstest.so`. File mappings are copies: musl maps
  a library's whole span and then each segment over it, so a library takes about
  twice its size in memory (and fork copies that), and libraries are files within
  the RAM filesystem's limits below.
* vfork/`posix_spawn`: the parent waits while the child shares backing pages, then
  detaches on exec or exit. Address-space mutation and nested fork in that child
  are rejected. General process-clone sharing flags are unsupported. Fork and exec
  of a multithreaded process return an error.
* Bounded RAM filesystem: directories, file creation/writes/truncation, unlink,
  rename, symlinks, working directories, mode/ownership checks and timestamps.
  Eight MiB per file, 24 MiB total live file contents, 4096 namespace entries.
  File descriptions survive unlink. This is volatile storage, not a disk port.
* Real UID/GID changes and privilege-drop checks replace success stubs for this
  personality. Set-id executable transitions, chroot, a complete `/proc`, full
  Linux filesystem semantics and all BusyBox applet behaviors are not provided.
* Ordinary signal handlers, masked pending signals, handler return, XSAVE
  preservation, process groups and terminal signals. Full Linux siginfo/ucontext,
  alternate signal stacks and general multithreaded signal semantics remain
  unsupported; unsupported handler flags fail explicitly. Synchronous user faults
  still end the process.
* PTYs: canonical/raw input, erase/kill/EOF, echo and CR/LF output, dimensions,
  controlling terminals, foreground groups, hangup and poll. Full termios timing,
  software flow control and every tty ioctl are not implemented.
* Dropbear is built with public-key authentication and without password/PAM auth,
  compression or forwarding. It uses upstream cryptography and static linking;
  there is no custom SSH protocol or crypto implementation. The first guest account
  is root; a general multi-user login policy is a separate task.
* Unix randomness uses ChaCha20 seeded by fresh host entropy in the test image and
  mixes RDRAND when available. Hardware boots require RDRAND or a **fresh,
  one-time** trusted boot seed. There is no time-based fallback in this personality.
  Staged test seeds are regenerated on every run; host keys remain stable.

Native capabilities are not implicitly inherited by Unix fork. Only fd capabilities
are inherited. The only syscall API additions are optional network registration,
secure native random bytes (34), and channel-backed fds through `HANDLE_FD`.

## Validation

Validated on 2026-11-02 using Zig 0.14.1 and QEMU 11.1.0 TCG, in release builds
with both `SMP=4` and `SMP=1`. Each run passed 36 guest checks, the native terminal
bridge, and the complete host OpenSSH suite. The OETH codec's two host tests passed.
The native Unix programs and musl terminal bridge passed Clippy with warnings denied.
With dynamic linking (2026-11-03, `SMP=4` and `SMP=1`) the guest runs 39 checks.

The regular kernel personality also passed its existing native self-test (18 checks),
POSIX test (19 checks), edu DMA and USB checks with:

```sh
cd dross
NO_DRIVERS=1 ./build-linux.sh
SMP=4 PROFILE=release INIT_CONF=unix/regression-init.conf TIMEOUT=120 ./run-qemu.sh
```

The four-CPU and one-CPU acceptance logs are `unix/target/acceptance-smp4.log` and
`unix/target/acceptance-smp1.log`. These are correctness results in QEMU; the Unix
image has not yet been validated on the Mac Pro's physical NIC.
