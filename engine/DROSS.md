# Dross model server

Build with `DROSS_MUSL_LIBC=/path/to/retained/ld-musl-x86_64.so.1 scripts/build-dross.sh`.
The build needs Rust's x86_64 musl target, Zig and glslc. Copy the output from
`target/dross-server/bin/trashcan` to retained results before deployment.
The optional `dross` feature uses the pinned SDK in `vendor/dross`; it does not
fetch another private repository during a build. SDK provenance is recorded
there. Linux/VFIO use without `DROSS_BOOTSTRAP=1` keeps the ordinary server path.

The engine is still a Unix-hosted program. Before loading its model or calling
DRM it reads its bootstrap, publishes its fresh `gpu` handles for the shim, and
requires a private `control` channel. Do not set `GPUD_CHANNEL=bootstrap`: the
engine has already consumed that message. A supervisor retains the actual
process handle and sends the four bytes `stop` to `control`.

Stop wakes the blocked HTTP listener without a periodic poll. Generation stops
between completed forward operations; it does not interrupt submitted GPU work.
After the serving scope drops its model/device owners, the control reply is an
eight-byte little-endian i64: zero for normal completion, minus one for failure.
The supervisor must also verify exit zero, then separately obtain gpud's checked
DMA shutdown. A reply alone is not a driver-drain proof. No HTTP shutdown route,
signal termination or forced owner kill is used by this adapter.

A blocked request-body read, model load or forward may delay completion. Missing
reply, timeout, nonzero exit or failed gpud drain must retain hardware custody;
the controller must not forcibly kill an owner or release uncertain backing.
The existing Linux signal handlers are unchanged.
