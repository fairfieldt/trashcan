# lwext4

Upstream: https://github.com/gkostka/lwext4

Pinned revision: `58bcf89a121b72d4fb66334f1693d3b30e4cb9c5`.
The `vendor/lwext4` source and include directories, README and LICENSE come from
that revision. Local changes are recorded in `patches/`:

- `0001-preserve-journal-generation.patch` preserves the next journal transaction
  generation across clean unmounts, preventing stale records from being accepted
  during recovery in a later mount session.
- `0002-metadata-checksum-seed.patch` uses the stored metadata CRC32C seed when
  `metadata_csum_seed` is set. Superblock and journal checksum rules are unchanged.
- `0003-zero-hole-reads.patch` returns zeroes for sparse and unwritten extents at
  every read alignment, instead of reading the physical-block-zero hole sentinel.
- `0004-latch-metadata-checksum-errors.patch` reports detected metadata checksum
  failures to the adapter independently of debug logging. The adapter latches EIO,
  suppresses subsequent writes and rejects successful service replies.
- `0005-initialize-indexed-directory-leaf.patch` initializes the empty directory
  leaf before checksumming it. Upstream computed the checksum before clearing a
  potentially nonzero inode field in reused buffer memory.
- `0006-journal-final-superblock-state.patch` journals the final summary counters
  and orphan head with every transaction. This prevents last-close inode
  reclamation from committing a bitmap/group update with an older superblock
  count. Transaction-stop failures latch through a local port callback.
- `0007-compatible-mount-and-superblock-replay.patch` marks writable mounts
  unclean without inventing EXT4_ERROR_FS (which stops Linux trusting the orphan
  chain), and updates the in-memory superblock on replay for 1 KiB blocks too.

The adapter, build script and freestanding libc subset are local. The adapter
includes `ext4.c` directly to extend its private namespace/journal helpers with
open-inode capabilities, classic orphan-chain cleanup and atomic replacement
rename. It is not compiled a second time. No runtime source download is needed.
The stored-seed layout follows the Linux ext4 superblock specification:
https://www.kernel.org/doc/html/latest/filesystems/ext4/super.html

The library contains mixed licenses. In this revision `src/ext4_extent.c` and
`src/ext4_xattr.c` state GPL version 2 or any later version. Other source/header
notices include BSD-style terms; retain each file's notice. Upstream's README
also explains that including its GPL files makes the complete library GPLv2.
This build includes the extent and xattr files. The combined fsd executable is
therefore distributed under GPL-2.0-or-later, with the complete corresponding
source in this repository. This does not change other Oscar components' licenses.
The upstream GPLv2 text is also staged at `/usr/share/licenses/lwext4/LICENSE`.
