#!/usr/bin/env python3
"""CPU-only standalone shader compilation; never creates a Vulkan device."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    compiler = shutil.which("glslc")
    if compiler is None:
        raise SystemExit("glslc unavailable; no shader compilation or GPU validation performed")
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="fixed-dim-spv-"))
    output.mkdir(parents=True, exist_ok=True)
    validator = shutil.which("spirv-val")
    report = {
        "scope": "CPU-only GLSL/SPIR-V checks; no GPU execution or numerical/performance claim",
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "validator": subprocess.check_output([validator, "--version"], text=True) if validator else None,
        "source_manifest_sha256": hashlib.sha256((source / "source.json").read_bytes()).hexdigest(),
        "variants": [],
    }
    manifest = json.loads((source / "source.json").read_text())
    for label, directory, hashes in [
        ("baseline", source / "baseline", manifest["baseline_sha256"]),
        ("fixed-dim", source, manifest["candidate_sha256"]),
    ]:
        for name, expected in hashes.items():
            path = directory / Path(name).name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise SystemExit(f"Source hash mismatch: {path}")
        for heads in (2, 8):
            name = f"attention_partial_heads{heads}"
            spv = output / f"{label}-{name}.spv"
            command = [compiler, "--target-env=vulkan1.2", "-O", "-I" + str(directory),
                       str(directory / (name + ".comp")), "-o", str(spv)]
            subprocess.run(command, check=True)
            validation_command = [validator, "--target-env", "vulkan1.2", str(spv)] if validator else None
            if validation_command:
                subprocess.run(validation_command, check=True)
            report["variants"].append({
                "variant": label, "heads_per_group": heads,
                "compile_command": command, "compile_passed": True,
                "validation_command": validation_command,
                "spirv_validation_passed": True if validator else None,
                "spirv_bytes": spv.stat().st_size,
                "spirv_sha256": hashlib.sha256(spv.read_bytes()).hexdigest(),
            })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
