#!/usr/bin/env python3
"""Build an isolated native worker and prepare a verified, stripped staging copy."""

import hashlib
import json
import os
import pathlib
import shutil
import struct
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / "target/native-build"


def inspect(path):
    data = path.read_bytes()
    if data[:7] != b"\x7fELF\x02\x01\x01":
        raise RuntimeError(f"not a little-endian ELF64 executable: {path}")
    executable_type, machine = struct.unpack_from("<HH", data, 16)
    if (executable_type, machine) != (2, 62):
        raise RuntimeError(f"expected ET_EXEC/x86-64, got {executable_type}/{machine}")
    entry, program_offset = struct.unpack_from("<QQ", data, 24)
    program_size, program_count = struct.unpack_from("<HH", data, 54)
    loads = []
    for index in range(program_count):
        kind, flags, offset, address, _physical, file_size, memory_size, _alignment = struct.unpack_from(
            "<IIQQQQQQ", data, program_offset + index * program_size)
        if kind == 1:
            payload = bytearray(data[offset:offset + file_size])
            if offset == 0:
                # Stripping updates section-table location/count metadata in
                # the mapped ELF header. Program headers and actual code/data
                # must remain byte-identical; the loader doesn't use sections.
                payload[40:48] = b"\0" * 8
                payload[58:64] = b"\0" * 6
            loads.append((address, flags, memory_size, bytes(payload)))
    if not loads:
        raise RuntimeError("executable has no PT_LOAD segments")
    return data, entry, loads


def main():
    environment = dict(os.environ, CARGO_TARGET_DIR=str(BUILD))
    if environment.get("RUSTFLAGS") or environment.get("CARGO_ENCODED_RUSTFLAGS"):
        raise RuntimeError("unset RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS so the reviewed target config applies")
    subprocess.run(["cargo", "build", "-p", "oscar-netstack3-native", "--target",
                    "x86_64-unknown-none", "--locked"], cwd=HERE, env=environment, check=True)
    original = BUILD / "x86_64-unknown-none/debug/oscar-netstack3-native"
    _original_data, original_entry, original_loads = inspect(original)
    strip_tool = shutil.which("llvm-strip")
    if not strip_tool:
        brew = pathlib.Path("/opt/homebrew/opt/llvm/bin/llvm-strip")
        if brew.is_file():
            strip_tool = str(brew)
    if not strip_tool:
        raise RuntimeError("llvm-strip is needed to prepare the staging copy")
    destination = HERE / "target/native-stage/oscar-netstack3-native"
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([strip_tool, "--strip-all", "-o", str(destination), str(original)], check=True)
    data, entry, loads = inspect(destination)
    if entry != original_entry or loads != original_loads:
        raise RuntimeError("stripping changed the entry point or loaded program bytes")
    record = {
        "binary": str(destination.relative_to(HERE)), "sha256": hashlib.sha256(data).hexdigest(),
        "file_bytes": len(data), "elf_type": 2, "machine": 62, "entry": hex(entry),
        "declared_pt_load_memory_bytes": sum(load[2] for load in loads),
        "load_bytes_check": "identical except ELF section-table metadata",
        "execution": "not run by this build script",
    }
    (destination.parent / "artifact.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
