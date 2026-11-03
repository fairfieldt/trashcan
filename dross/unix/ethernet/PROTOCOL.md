# OETH v1

A native dross channel transports complete datagrams, with no handles. Each
endpoint is a capability for one Ethernet port. Bootstrap it as `@ethernet`.

All messages have an 8-byte header:

| Offset | Bytes | Value |
| --- | --- | --- |
| 0 | 4 | ASCII `OETH` |
| 4 | 2 | Little-endian version 1 |
| 6 | 2 | Little-endian opcode |

The driver's first message must be **Info** (opcode 1), exactly 24 bytes:

| Offset | Bytes | Value |
| --- | --- | --- |
| 8 | 6 | Station MAC |
| 14 | 2 | Little-endian IP MTU: 1500 |
| 16 | 1 | Link up: 0 or 1 |
| 17 | 3 | Zero |
| 20 | 4 | Little-endian speed in Mbps, 0 if unknown |

**Frame** (opcode 2) goes in either direction. The payload is 14..1514 Ethernet
bytes, including the Ethernet header, without FCS. It is exactly one frame, with
no length prefix or padding required at the channel level. The driver pads short
frames when its hardware requires it. v1 has no VLAN/offload/jumbo metadata.

**Link** (opcode 3) goes from driver to netd when the link changes, exactly 16 bytes:
link boolean at byte 8, zero bytes 9..12, little-endian Mbps at 12..16.

No per-frame ACK is required. Channel writes are nonblocking; keep a bounded retry
queue on `ShouldWait`. Stop draining a source when that queue fills. The reference
implementation limits each software frame queue to 32 entries and processes at
most 32 channel frames per loop. Physical receive-ring overflow can drop packets;
TCP handles loss. Do not block on another process while holding a device ring.

The driver owns IRQ/MSI capabilities and DMA mappings. No interrupt handle crosses
this interface. Netd binds the Ethernet channel to its own notification and wakes
on frame/link messages or a timer. A closed channel ends the session; a supervisor
must establish a fresh capability when restarting a driver. The codec rejects
unknown versions/opcodes, incorrect lengths, non-boolean link values, nonzero
reserved bytes and unsupported MTUs.
