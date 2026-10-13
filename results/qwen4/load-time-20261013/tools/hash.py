#!/usr/bin/env python3
"""TEMPORARY test builds: read back every weight buffer after upload and print `HASH size fnv64` (stderr), in upload
order. usage: hash.py main|new CHECKOUT   (main: Device::upload of 6f6635a; new: Uploader::finish)"""
import os, sys

which, root = sys.argv[1], sys.argv[2]
HASHFN = '''
/// TEMPORARY (hash test builds): FNV-1a over 64-bit words of a read-back weight buffer.
pub fn load_hash(b: &[u8]) {
    let mut h = 0xcbf29ce484222325u64;
    for w in b.chunks(8) {
        let mut a = [0u8; 8];
        a[..w.len()].copy_from_slice(w);
        h = (h ^ u64::from_le_bytes(a)).wrapping_mul(0x100000001b3);
    }
    eprintln!("HASH {} {h:016x}", b.len());
}
'''


def edit(path, pairs):
    p = os.path.join(root, path)
    s = open(p).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    open(p, 'w').write(s)


if which == 'main':
    edit('engine/src/vulkan.rs', [
        ("        self.copy(cmd, &staging, 0, &dst, 0, staging.size);\n        self.finish(cmd)?;\n        self.run(cmd)?;\n        Ok(dst)\n",
         "        self.copy(cmd, &staging, 0, &dst, 0, staging.size);\n        self.finish(cmd)?;\n        self.run(cmd)?;\n        load_hash(&dst.download()?);\n        Ok(dst)\n"),
        ("pub struct Buffer {\n", HASHFN + "pub struct Buffer {\n"),
    ])
else:
    edit('engine/src/vulkan.rs', [
        ("    jobs: Vec<(Kind, &'a [u8], usize, usize)>,\n    threads: usize,\n}",
         "    jobs: Vec<(Kind, &'a [u8], usize, usize)>,\n    threads: usize,\n    uploaded: Vec<(vk::Buffer, usize)>,\n}"),
        ("            threads: std::thread::available_parallelism().map_or(8, |n| n.get()),\n        };",
         "            threads: std::thread::available_parallelism().map_or(8, |n| n.get()),\n            uploaded: Vec::new(),\n        };"),
        ("            done += n;\n        }\n        Ok(dst)\n    }\n\n    /// A device-local buffer holding a matrix",
         "            done += n;\n        }\n        self.uploaded.push((dst.buf, dst.size));\n        Ok(dst)\n    }\n\n    /// A device-local buffer holding a matrix"),
        ("            done += n;\n        }\n        Ok(dst)\n    }\n\n    /// Submit the last copies",
         "            done += n;\n        }\n        self.uploaded.push((dst.buf, dst.size));\n        Ok(dst)\n    }\n\n    /// Submit the last copies"),
        ("        Ok(())\n    }\n}\n\nimpl Drop for Uploader",
         "\n"
         "        for &(buf, size) in &self.uploaded {\n"
         "            let h = self.gpu.buffer(size, true, false, None)?;\n"
         "            let cmd = self.gpu.begin()?;\n"
         "            unsafe { self.gpu.dev.cmd_copy_buffer(cmd, buf, h.buf, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: size as u64 }]) };\n"
         "            self.gpu.finish(cmd)?;\n"
         "            self.gpu.run(cmd)?;\n"
         "            load_hash(&h.read(size)?);\n"
         "        }\n"
         "        Ok(())\n    }\n}\n\nimpl Drop for Uploader"),
        ("pub struct Buffer {\n", HASHFN + "pub struct Buffer {\n"),
    ])
print('hash build', which, root)
