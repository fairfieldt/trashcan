# Launch a native executable

`native-run IMAGE` starts a native x86-64 Oscar executable from the Unix development
shell and returns its exit status. Copy a new image over SSH, then run it without
rebooting the OS. The launcher uses Linux compatibility; its child uses the native
syscall ABI.
Native statuses outside the shell's range of 0 through 255 are reported as 125.

Build from `dross/`:

```sh
CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS='-C link-self-contained=yes' \
cargo build --manifest-path unix/Cargo.toml -p oscar-native-run \
  --target x86_64-unknown-linux-musl --release
```

The image must use the native entry convention and be an ET_EXEC ELF, with static
relocation and an x86-64 machine type. A Linux executable is not a native image
even if its ELF header has those values.

The child receives an empty BootstrapV2 message, with no device, system, filesystem
or directory capabilities. Its memory is charged under the launcher's default
account. Programs requiring delegated capabilities need a supervisor that passes
them explicitly.

The launcher waits for exit. Stopping the launcher does not terminate the child;
closing a Process handle only releases observation. Use bounded test programs.
This tool does not provide process termination, supervision or crash recovery.

The integration child in `tests/exit.c` validates the empty typed bootstrap and
exits with status 42. On a physical Mac Pro 6,1, uploading the launcher and that
child over SSH returned status 42 without rebooting or changing the running
desktop. Invalid ELF input is rejected before process creation.
