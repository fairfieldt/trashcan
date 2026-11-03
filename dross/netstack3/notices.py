#!/usr/bin/env python3
"""Record resolved dependency licenses and copy source notices for review."""

import json
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "target/notices"


def main():
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--locked"], cwd=HERE))
    OUT.mkdir(parents=True, exist_ok=True)
    packages = []
    for package in metadata["packages"]:
        if not package["source"]:
            continue
        root = pathlib.Path(package["manifest_path"]).parent
        folder = OUT / (package["name"] + "-" + package["version"])
        folder.mkdir(exist_ok=True)
        files = set()
        for pattern in ("LICENSE*", "LICENCE*", "COPYING*", "COPYRIGHT*", "NOTICE*"):
            files.update(path for path in root.glob(pattern) if path.is_file())
        if package["license_file"]:
            files.add(root / package["license_file"])
        for path in files:
            shutil.copy2(path, folder / path.name)
        packages.append({
            "name": package["name"], "version": package["version"],
            "source": package["source"], "license": package["license"],
            "notices": sorted(path.name for path in files),
        })
    (OUT / "dependencies.json").write_text(json.dumps(packages, indent=2) + "\n")
    shutil.copy2(HERE / "FUCHSIA-LICENSE", OUT / "FUCHSIA-LICENSE")
    print(f"Recorded {len(packages)} third-party packages in {OUT}")
    missing = [f'{p["name"]}-{p["version"]}' for p in packages if not p["license"] or not p["notices"]]
    if missing:
        print("Manual notice/license review needed:", ", ".join(missing))


if __name__ == "__main__":
    main()
