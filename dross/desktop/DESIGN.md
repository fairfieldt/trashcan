# Windowing and application architecture

## Boundaries

Window policy stays in a user server. The kernel needs no window objects, fonts,
input focus, terminal parser or desktop policy. `oscar-windowd` alone receives the
display memory capability. A seat provider alone may deliver keyboard/pointer
events. An application receives a channel endpoint, never the framebuffer,
physical-memory resource, PCI endpoint, IRQ or another application's channel.

One transport connection represents one authority principal. It may own multiple
windows within the global budget; a different application must use a different
connection. A server ID is allocated monotonically and is not reused during that
server's lifetime. Every mutation checks both the window ID and the connection's
identity. Responses and input events are returned only to that connection.
Client-side `Input` packets are rejected: only the trusted seat adapter calls the
input entry point. The host assigns connection 1 internally; dross assigns IDs to
bootstrap endpoints in server-local order. Neither accepts a client-supplied ID.

The server is the sole owner of its mutable scene. It processes bounded batches of
messages, routes input, and renders if dirty. Applications never block the server
on a fence or share writable scene data with it. There are no worker threads,
render locks, callbacks into client code, GPU jobs, or host events in the core.
Clients draw in their own CPU surfaces; the server owns decoration. Client pixels
are opaque, so there is no undefined alpha or cross-client readback.

## Presentation and resource accounting

Each window has a front surface and a staging surface. `Upload` copies a bounded
contiguous pixel span into staging after validating the complete range. `Present`
copies staging to front atomically with respect to server request processing and
queues `FrameDone` with the same serial. A partially uploaded frame cannot reach
the screen. Unmodified staging pixels retain the previous frame. Native messages
carry copied pixels, so there is no shared-memory validation/use race and no
client fence which can wedge the display.

`FrameDone` acknowledges a committed surface; it is **not** a physical vblank or
scanout fence. The initial adapter has no page flipping, tear-free guarantee or
display timing API. The current client allows one outstanding frame and coalesces
further UI changes until the matching acknowledgement. A stalled channel ends the
native client connection instead of accumulating whole frames. The server retries
outbound control events on a bounded 16 ms deadline and disconnects an endpoint
which exceeds its event queue. Seat and each client get bounded batches per loop.

| Resource | Limit |
| --- | --- |
| Packet | 60 KiB, including a 20-byte header (below the dross channel's 64 KiB) |
| Title | 80 UTF-8 bytes, no control characters |
| Windows/connections | 16 each |
| Window dimensions | Up to 2048 per axis; must fit the current display including decoration |
| Window pixels | 512 Ki pixels per window and in total across all windows |
| Window backing memory | At most 4 MiB total, including both front and staging |
| Display | At least 320×240; at most 4096 per axis and 2 Mi pixels total |
| Native outbound events | 32 per connection, in addition to bounded kernel channel queues |
| Terminal | 256×128 cells maximum, 256 history lines, 128-byte CSI buffer |
| Shell | 4096 input bytes, 64 history entries, output capped at 32 KiB per command |

The server's largest framebuffer is 8 MiB; window storage is at most 4 MiB. This
fits the existing 16 MiB dross runtime heap with room for metadata and bounded
packet scratch space. The default 1024×768 display and 760×500 client use less.
This is capacity accounting, not a throughput measurement. No D700 performance,
frame-rate target or hardware upper bound is claimed. CPU software composition is
the baseline until the display/driver path exists and an end-to-end measurement
justifies another renderer or buffer transport.

## Protocol v1

All integers are little endian. A message is exactly one channel datagram. No
messages carry handles. Unexpected transferred handles are closed by the native
adapter, then the connection is rejected. Oversize messages, invalid UTF-8,
unknown enums/flags, trailing bytes and inconsistent lengths are rejected.

| Header offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | ASCII `OSCR` |
| 4 | 2 | Version = 1 |
| 6 | 2 | Opcode |
| 8 | 4 | Request serial (events use 0) |
| 12 | 4 | Window ID (`Create` uses 0) |
| 16 | 4 | Payload length |

| Opcode | Direction | Payload |
| --- | --- | --- |
| 1 Create | Client → server | width:u32, height:u32, remaining UTF-8 title bytes |
| 2 Created | Server → client | width:u32, height:u32; allocated ID in header |
| 3 Upload | Client → server | pixel offset:u32, remaining u32 XRGB8888 pixels |
| 4 Present | Client → server | Empty |
| 5 FrameDone | Server → client | Empty; echoes Present serial |
| 6 Close | Client → server | Empty |
| 7 Closed | Server → client | Empty |
| 8 CloseRequested | Server → client | Empty; application should send Close |
| 9 Focus | Server → client | boolean:u8 |
| 10 Pointer | Server → client, seat → server | x:i32, y:i32, left-button boolean:u8 |
| 11 Key | Server → client, seat → server | key:u32, modifiers:u8 |
| 12 Error | Server → client | fault:u32 (see protocol enum) |

Key codes below U+110000 are printable Unicode scalars. U+110000 through U+11000C
name Enter, Backspace, Delete, Left, Right, Up, Down, Home, End, Tab, Escape, Page Up
and Page Down. Modifier bits 0/1/2 are Shift/Ctrl/Alt. Key events represent text or
repeated key presses, not physical key releases. The seat provider owns keyboard
layout/repeat. Pointer coordinates from a seat are display-relative; delivered
coordinates are content-relative. During pointer capture they may be outside the
content rectangle, allowing widgets to cancel release-outside actions.

Window focus and widget focus are distinct. The server raises/focuses a window on
pointer press, owns title dragging and close-box actions, and consumes Alt-Tab.
The toolkit uses caller-stable IDs, explicit begin/end event transactions,
pointer capture, release activation and tab traversal. All widget and terminal
drawing is clipped through the canvas. Trash switches between terminal input and
widget navigation using Ctrl-Tab. The current desktop client contains a dock,
front panel and terminal rather than a separate privileged desktop process.

## Native startup contract

`oscar-session` is an **opt-in** supervisor, not referenced by the existing init.
It receives a normal `dross-rt` bootstrap message:

| Handle/value | Meaning |
| --- | --- |
| `@module/oscar-windowd` | Native server ELF memory |
| `@module/trash` | Native Trash ELF memory (default application) |
| `@module/oscar-terminal` | Terminal ELF memory if `app terminal` |
| `@display` | Exclusive writable/mappable framebuffer memory object; no root resource |
| `@input` | Trusted seat channel carrying protocol Input packets |
| `@terminal` | Required only in terminal mode: duplex channel of raw terminal bytes |
| `@file/NAME` | Optional read-only file objects exposed to Trash |
| `width`, `height`, `stride` | Display pixels and stride in pixels |
| `format` | `bgr` (XRGB8888 words) or `rgb` (red/blue swapped at scanout) |
| `offset` | Optional byte offset of the framebuffer in the object; 4-byte aligned |
| `client-width`, `client-height` | Optional client dimensions; default 760×500 |
| `app` | Optional `terminal`; otherwise Trash |

The launcher creates the application/server connection and a lifetime channel.
It transfers display/input capabilities to windowd, the other endpoint to the
application, and waits for the application. Closing the lifetime channel shuts
down windowd and returns its display mapping to the kernel. A closed application
channel removes all its windows and recovers focus. A malformed client is
disconnected. A malformed or closed trusted input channel ends the server; clients
observe peer closure and exit. There is no automatic restart policy yet.

For a custom supervisor, windowd accepts `@display`, `@input`, `@control` and up to
16 distinct `@client/NAME` endpoints plus the display values above. Its `control`
channel is a lifetime capability: closing its peer ends the server; sending data
is an error. Trash accepts `@window`, optional `@file/NAME`, and optional
`width`/`height`. `oscar-terminal` additionally requires `@terminal`. Terminal data
is raw bytes in channel messages up to 60 KiB, without handles. Input is VT-style
encoded bytes in the reverse direction. The stream peer is responsible for process
launch, echo, line discipline and matching the terminal dimensions. v1 dimensions
are fixed for the connection's lifetime, so no resize stream message is needed.

## Hardware integration boundary

On dross the kernel hands init the firmware framebuffer as `display` (a memory object) with its geometry as
values (`display-width`, `display-height`, `display-stride`, `display-format`). While any program has it mapped,
the kernel log stays off the screen; a panic or a reset draws on it regardless. usbd is the seat: given a `seat`
channel it sends keyboard and pointer input as protocol Input packets (US layout, key repeat, the pointer
clamped to the display) and stops logging keys. init.conf connects them (`../README.md`, "Oscar's desktop").

This boundary is also where a later modesetting display server can replace the
firmware scanout adapter. Shared surfaces would need explicit ownership and release
semantics; a GPU renderer would need bounded submissions. Neither is assumed by
the UI/toolkit or terminal model.

## Current scope and limits

Implemented: opaque fixed-size movable windows, focus/stacking, capture, close and
disconnect; software decoration/composition; themed buttons, scrollbar and row
layout; terminal widget; Trash's real command interpreter; host PTY adapter; native
channel byte-stream adapter; isolated session launcher.

The terminal handles incremental UTF-8, delayed wrap, CR/LF/backspace/tab, cursor
motion/save/restore, erase, scroll regions, alternate screen, cursor visibility,
16/256/true-color SGR, inverse/bold, and bounded scrollback. OSC/DCS/PM/APC strings
are safely consumed and ignored. It is an ANSI subset, not a full xterm: no wide or
combining glyph layout, selection/clipboard, IME, mouse reporting, terminal replies,
or complete terminfo conformance. The bundled raster font is Basic Latin; other
scalars are retained in cells and drawn as fallback glyphs. `TERM=xterm-256color`
in the host is a pragmatic color hint; full-screen applications may use unsupported
editing sequences. No full-screen application compatibility is claimed.

There is no dynamic connection broker, resizing, multi-display, transparency,
accessibility tree, storage service, native POSIX PTY, fork/exec integration, or
graphical file manager. The toolkit is a small initial widget set with explicit
state; adding menus/text fields can follow the same stable-ID/event contract.
Native Trash does not make Linux's logged stdout into terminal output: that needs
an actual process/PTY service rather than intercepting the kernel log.
