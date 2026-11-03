# Native C interoperability test

This is a freestanding C program that runs as a native Oscar process. It uses the native syscall ABI directly. It has no libc, allocator, Linux syscall layer or host dynamic loader.

Build from the repository root:

```
dross/user/c-api-test/build.sh
```

The output is `dross/target/c-api-test/c-api-test`. Set `OUT` to use another output directory. The script uses Zig's C compiler and the existing native user linker script. It includes `dross/protocol/include/oscar_protocol.h` and its generated schema constants; it does not maintain a separate frame or bootstrap codec.

The native supervisor stages this ELF as module `c-api-test`, spawns it with a typed BootstrapV2 message, and waits for exit 0 with a bounded deadline. Its bootstrap must contain exactly one capability:

| Name | Declared protocol | Authority |
| --- | --- | --- |
| `directory` | DIRECTORY / ODIR, major 1 | A directory view permitting `echo` and denying `forbidden` |

The directory connects `echo` to a provider implementing the ECHO protocol. CONNECT must preserve the requested endpoint rights `READ | WRITE | TRANSFER | WAIT` (`0x93`). No PCI, physical-memory, system or global directory capability is passed to this client.

The client proves that C can:

- Receive and validate the typed bootstrap, then take ownership of its directory endpoint.
- Connect to the permitted provider and exchange a correlated framed binary request/reply, including NUL and non-UTF-8 bytes.
- Receive directory Denied for `forbidden`, with no attached capability, and continue using its permitted endpoint.
- Fail a physical-memory syscall on the directory handle with WrongType, and fail an attempt to duplicate the provider endpoint because DUPLICATE was not granted.
- Close its bootstrap, directory, provider and every received attachment before successful process exit.

Kernel error numbers and service result numbers are separate. The test checks each in its own domain. It uses the shared generated IDs, frame encoder/decoder, bootstrap parser, name validator and little-endian field helpers. The short CONNECT body is constructed from those helpers according to the shared directory contract.

`message.h` owns fixed receive storage for 1,024 bytes and 64 handles. A successful receive owns all returned handles immediately. Taking a handle clears its ownership slot; all unused or rejected attachments are closed even when decoding fails. Control requests attach no handles. The native send wrapper documents the kernel's rule: success moves attachments; failure leaves them owned by the caller. Any unexpected reply or local timeout ends the test and closes its endpoints, so a late reply cannot reach a later call.

Each RPC computes one absolute monotonic reply deadline before encoding and sending. This smoke client does not implement waiting send retries, asynchronous reply dispatch or cancellation. The current directory prototype does not carry an operation deadline in CONNECT; bounding the local wait does not undo a request already sent.

Build validation checks a static x86-64 ELF with entry 0x400000, separate RX/R/RW load segments, no interpreter/dynamic segment and no undefined symbols. Native end-to-end validation is performed by the supervisor's QEMU acceptance run. No live hardware action is needed for this test.
