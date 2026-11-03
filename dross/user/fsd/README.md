# ext4 service

fsd runs lwext4 in a userspace process. It owns one explicitly supplied block
channel and serves one mount channel. The kernel contains no ext4 implementation.
Sources and license details are in [PROVENANCE.md](PROVENANCE.md).

A mount is read-only unless its bootstrap includes `%writable=1`:

```
channel disk
mount root /disk
native usbd @disk=block
native fsd @disk=block @root=service %writable=1
```

The first Linux-filesystem GPT partition is selected, or `%partition=N` selects a
partition by number. Without GPT, the whole device is used. The partition must be
within the device and aligned to its physical blocks. Only the given block
capability grants device access; fsd receives no PCI or system resource.

## Filesystem profile

Supported filesystem blocks are 1, 2 and 4 KiB; device blocks are power-of-two
sizes from 512 through 4096 bytes. The recommended image profile is:

```
mkfs.ext4 -b 4096 -O ^orphan_file,^metadata_csum_seed IMAGE
```

Do this only on a newly created disposable image or a volume you intend to format.
The adapter rejects unknown compatible, incompatible and read-only-compatible
feature bits. It also rejects multi-mount protection, external journals, encryption,
inline data, bigalloc and other features outside lwext4's supported masks.
Read-only mounts also accept `metadata_csum_seed` and a clean `orphan_file`
filesystem with no pending orphan recovery. Stored checksum seeds are used for
filesystem metadata; changing the UUID does not change that seed. Writable mounts
still reject these two features and require an internal journal. Read-only mounts reject a journal
that needs recovery; a writable mount recovers the journal before serving requests.
There is no fallback that writes an unsupported volume.

The lwext4 block cache has 32 entries, at most 128 KiB of block data for this
profile. Its allocation failures return errors through the adapter. Other library
metadata and journal allocations use fsd's existing bounded process heap.

## Current operations and limits

Writable mounts support file and directory creation, reads, writes, append,
shrinking and zero-extending truncation, symlinks, hard links, atomic replacement
rename, open-unlinked regular files, empty-directory removal, mode/owner changes,
timestamps, stat and readdir.
The kernel enforces its existing Unix permission checks before requesting changes.
Timestamp updates preserve nanoseconds when the inode has ext4 extra timestamp
fields; unrepresentable timestamps return an error before either time changes.

Read/write operations on writable mounts use bounded range RPCs, so ordinary reads
do not allocate a whole-file snapshot. Read-only mounts preserve whole-file shared
snapshots for model and library access, with the separate bounded kernel cache.
`mmap`/exec can request snapshots; shared writable file mappings are unsupported.

Each writable regular-file open description owns a separate channel capability.
Reads, writes, fstat, truncation and metadata updates use that inode identity after
rename or unlink. No caller-supplied inode number grants access. The final channel
close releases the inode; unnamed inodes remain on the journaled classic orphan
chain until then. A later writable mount reaps orphans left by an interrupted
server. Superblock counts and the orphan head join the same committed transaction
as the allocation bitmaps and group counters. Replacement rename changes both
names in one journal transaction. The
kernel serializes namespace requests with its namespace cache updates.

This is not a complete POSIX filesystem implementation. Directory open descriptions
and cwd still use namespace paths. Shared writable file mappings need a coherent
page-cache/writeback design and remain unsupported; read-only mappings and exec
use snapshots. Writes beyond EOF allocate zero-filled blocks rather than sparse
holes. Existing holes and unwritten extents read as zeroes. External writers must
not share the mounted volume. Large write calls are split into bounded requests,
so append serialization applies to each request rather than an arbitrarily large
user write. At most 256 server open descriptions are active at once.

## Durability and errors

WRITE completion means the USB transport completed, not that data is durable.
FLUSH issues SCSI SYNCHRONIZE CACHE (10), with IMMED clear, and checks command
status and zero residue. An unsupported flush or transport failure returns an
error. The adapter flushes after each lwext4 block-write callback to preserve
journal ordering, and fsync/fdatasync/syncfs/sync flush the library cache and device.
This is a correctness baseline, with no throughput claim.

The adapter refuses device writes independently when mounted read-only. It latches
block I/O failures and detected metadata checksum failures, returns EIO for those
failures and refuses subsequent writes. This also
covers library paths that do not propagate every lower-level write failure.
After an I/O failure, remount and recovery are required. An error may follow a
partially completed write; recovery is not a promise of application-level atomicity.
A device must truthfully implement SYNCHRONIZE CACHE for those guarantees to hold.

## Block channel

One client uses a shared 1 MiB DMA window. Requests are serialized; it must not
change the window while a command is outstanding. Integers are little-endian.

| Request | Layout | Success reply |
|---|---|---|
| INFO (0) | `u8 op` | `i64 status, u32 block_size, u64 block_count`, window handle on first request |
| READ (1) | `u8 op, u64 lba, u32 count` | `i64 status`; blocks start at window offset zero |
| WRITE (2) | same as READ | `i64 status`; blocks come from window offset zero |
| FLUSH (3) | `u8 op` | `i64 status`; preceding writes reached stable storage |

A negative status is an errno. READ/WRITE reject empty, oversized or out-of-range
requests. INFO gives READ, WRITE, MAP and TRANSFER rights on the window.

`usbd %block-start=N %block-count=N` delegates only that physical block range.
Both values are required; count must be nonzero and the range must fit the device.
INFO reports the view size, READ/WRITE validate against it and then translate LBAs;
FLUSH flushes the same physical device. This supports partition and reserved-range
delegation. Do not expose a file's physical extent while its parent filesystem is
mounted or served. `fsd %uuid=UUID` validates the selected filesystem UUID before
mounting, journal replay or any write.

## Filesystem channel

Every request starts with a little-endian `u32 op`. Every reply starts with
`i64 status` (zero or a negative errno). Paths are UTF-8 bytes without a NUL,
absolute below the mount root; `..` components are rejected. The kernel serializes
requests per mount and consumes each response before exposing ordinary signals.

| Operation | Request after op | Success reply after status |
|---|---|---|
| LIST (1) | `u64 cookie, path` | `u64 next_cookie`, entries; zero cookie ends enumeration |
| SNAPSHOT (2) | path | `u64 size`, one transferred read-only memory object |
| CREATE (3) | `u32 mode, uid, gid, path` | none |
| TRUNCATE (4) | `u64 size, path` | none |
| WRITE (5) | `u64 offset, u32 path_len, path, data` | `u64 end_offset, written_count`; max offset means append |
| REMOVE (6) | `u32 directory, path` | none |
| RENAME (7) | `u32 directory, u32 old_len, old, new` | none |
| SYMLINK (8) | `u32 path_len, path, target` | none |
| SYNC (9) | none | none |
| READ (10) | `u64 offset, u32 length, path` | bytes; at most 60 KiB |
| INFO (11) | none | `u8 writable` |
| SETATTR (12) | `u32 mode, uid, gid, path` | none; max fields mean unchanged |
| TIMES (13) | `u32 flags, u64 atime_ns, mtime_ns, path` | none; bits 0/1 select times |
| OPEN (14) | `u32 flags, path`, one transferred channel endpoint | none; endpoint becomes an open-inode service |
| STAT (15) | path | inode information |
| LINK (16) | `u32 old_len, old, new` | none |

LIST entries are `u32 mode, uid, gid, u64 size, mtime_ns, inode,
u16 name_len, target_len`, followed by name and symlink target.
Inode information is `u64 inode, size, atime_ns, mtime_ns, ctime_ns,
u32 mode, uid, gid, links`.
SNAPSHOT relinquishes all fsd mappings/handles before transferring the backing.

Open-inode endpoints accept READ (1: `u64 offset, u32 len`), WRITE (2: `u64 offset, data`), STAT (3), TRUNCATE (4: `u64 size`), SETATTR (5: `u32 mode,
uid, gid`), TIMES (6: `u32 flags, u64 atime_ns, mtime_ns`), SYNC (7) and
SNAPSHOT (8). Their reply layouts match the corresponding mount operations.
The peer closing releases the inode without a blocking kernel destructor.

## Validation

`tests/run-adapter.sh` builds the exact C adapter on the host and uses disposable
images under `dross/target/fsd-adapter`. It checks separate-mount persistence,
write/flush-error propagation, read-only protection, interruption at four block-write
boundaries followed by journal recovery and clean e2fsck, unknown-feature rejection,
a sparse file larger than 5 GiB read across the 32-bit offset boundary,
open-unlinked data, hard links and replacement rename at ten interruption points.
A stored-seed fixture changes its UUID, indexes a 300-entry directory and reads an
external unwritten-extent tree; the image hash must remain unchanged. A deliberately
wrong stored seed with a valid superblock checksum must produce EIO without writes.
Recovery tests also crash after the final close and sync, without clean unmount,
across repeated mounts and ten last-close write cutpoints for each of the
1/2/4 KiB block sizes. Copies go directly to independent
Linux journal replay and a read-only full check before any lwext4 remount.
Correction diagnostics are failures even if e2fsck returns zero; ordinary journal
replay and automatic orphan-chain cleanup are expected recovery work.

With the staged Unix userland, run from `dross`:

```
PROFILE=release SMP=4 MEM=2G ./linux/disk-test.sh
PROFILE=release SMP=4 MEM=2G ./linux/disk-write-test.sh
PROFILE=release SMP=4 MEM=2G ./linux/disk-range-test.sh
PROFILE=release SMP=4 MEM=2G ./linux/disk-hardware-test.sh
```

The writable suite also checks fsync/fdatasync/syncfs, mode persistence, nested
directory rename with an open child, and reads beyond 5 GiB with 2 GiB RAM. It boots the same disposable USB image twice and checks persisted
contents, names, inode identity, hard-link counts and nanosecond timestamps.
Range tests hash guard regions before and after rejected boundary requests and
ordinary writes. Hardware-config tests also prove UUID rejection makes no disk
writes. These are correctness checks; no physical Mac Pro throughput was measured.

On 2026-11-03, the final service passed a physical Mac Pro 6,1 (INDITE) test on
a new 64 MiB image in a verified contiguous USB-file extent. Its parent filesystem
remained unmounted while dross served only the delegated block range. After
rewrite, verification and reboot to Linux, a raw copy went directly to journal-only
replay (without `-f`) and then `e2fsck -fn`, with no repair diagnostics or count
drift. Exact bytes, mode, nanosecond timestamps and concurrent-write chunks
persisted. Adjacent 64 KiB guard hashes were unchanged, and a full read-only check
of the parent USB filesystem was clean.

The tested fsd build is `6d2d2b925d7943c29d94eb0384929e9b035cb494e6d74f8b0b0077e8fe756ca4`.
The deployed stripped module is
`aa2b678928b05d4d43115259f6525a37ee6e5c94652912291de36b3b85a28bab`;
all its allocated ELF sections and its entry point match that build. Exact boot files
are recorded in `results/dross/hardware-20261103/fs-counter-fix-manifest.json`.
The independent recovery, parent check and physical boot logs are
`fs-counter-fix-independent.txt`, `fs-counter-fix-parent-e2fsck.txt` and
`fs-counter-fix-netlog.txt` in the same archive. The physical boot also passed
60 native and 28 POSIX checks. Physical recovery was tested by reboot; host
interruption tests stop between block writes. No throughput measurements were taken.
