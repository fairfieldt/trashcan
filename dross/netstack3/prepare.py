#!/usr/bin/env python3
"""Fetch the pinned Fuchsia subset and translate its host Rust targets to Cargo.

Generated manifests and imported sources live under target/, outside the reviewable
glue. This only understands the pinned targets below, not arbitrary Bazel projects.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import pathlib
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import tomllib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "target/upstream"
PIN = "650b15d5ea9cfe67faf22ea68cb40e8bc3293303"
BASE = f"https://fuchsia.googlesource.com/fuchsia/+/{PIN}/"
DIRS = {
    "src/connectivity/network/netstack3/core": "core",
    "src/connectivity/lib/net-types": "src/connectivity/lib/net-types",
    "src/connectivity/lib/packet-formats": "src/connectivity/lib/packet-formats",
    "src/connectivity/lib/internet-checksum": "src/connectivity/lib/internet-checksum",
    "src/connectivity/lib/net-declare": "src/connectivity/lib/net-declare",
    "src/connectivity/lib/ip-test-macro": "src/connectivity/lib/ip-test-macro",
    "src/connectivity/network/lib/diagnostics-traits": "src/connectivity/network/lib/diagnostics-traits",
    "src/connectivity/network/lib/explicit": "src/connectivity/network/lib/explicit",
    "src/lib/network/packet": "src/lib/network/packet",
    "src/lib/replace-with": "src/lib/replace-with",
    "src/lib/fuchsia-sync": "src/lib/fuchsia-sync",
    "src/lib/proptest-support": "src/lib/proptest-support",
    "src/lib/test_util": "src/lib/test_util",
    "src/diagnostics/lib/redactor": "src/diagnostics/lib/redactor",
}
FIDL_ONLY = "//sdk/fidl/fuchsia.net:fuchsia.net_rust_common"


def download(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def fetch(item):
    upstream, local = item
    destination = SOURCE / local
    marker = destination / ".oscar-source-pin"
    if marker.exists() and marker.read_text().strip() == PIN and (destination / ".oscar-source.json").exists():
        return
    url = f"https://fuchsia.googlesource.com/fuchsia/+archive/{PIN}/{upstream}.tar.gz"
    archive = download(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=destination.name + ".fetch-", dir=destination.parent))
    backup = destination.with_name(destination.name + ".previous")
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(stage, filter="data")
        (stage / ".oscar-source-pin").write_text(PIN + "\n")
        (stage / ".oscar-source.json").write_text(json.dumps({
            "revision": PIN, "directory": upstream, "url": url,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
        }, indent=2) + "\n")
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.rename(backup)
        try:
            stage.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print("Fetched", upstream, flush=True)


def source_label(directory):
    relative = directory.relative_to(SOURCE)
    if relative.parts[0] == "core":
        return "//src/connectivity/network/netstack3/" + str(relative)
    return "//" + str(relative)


def absolute_label(label, parent):
    if label.startswith(":"):
        return parent.rsplit(":", 1)[0] + label
    return label if ":" in label else label + ":" + label.rsplit("/", 1)[-1]


def read_targets():
    targets, tests = {}, []
    for build in sorted(SOURCE.rglob("BUILD.bazel")):
        if any(part in ("tests", "bench", "fuzz") for part in build.relative_to(SOURCE).parts):
            continue
        calls = []

        def ignored(*args, **kwargs):
            pass

        def capture(kind):
            def record(**kwargs):
                calls.append(dict(kwargs, kind=kind))
            return record

        # Pinned, trusted Starlark files use a Python-compatible declarative
        # subset. Loads/build rules are inert; no imported code is executed.
        environment = {
            "__builtins__": {}, "load": ignored, "package": ignored,
            "select": lambda choices: choices.get("//conditions:default", []),
            "HOST_OS_CONSTRAINTS": [],
        }
        source = build.read_text()
        for load in re.findall(r"load\([^\n]+", source):
            for name in re.findall(r'"([a-z][a-z0-9_]*)"', load):
                environment[name] = ignored
        environment["fuchsia_sync_detect_lock_cycles"] = False
        environment.update(rustc_library=capture("lib"), rustc_proc_macro=capture("macro"),
                           rustc_test=capture("test"))
        exec(compile(source, str(build), "exec"), environment)
        prefix = source_label(build.parent)
        for target in calls:
            target.update(directory=build.parent, label=prefix + ":" + target["name"])
            if target["kind"] == "test":
                tests.append(target)
            else:
                targets[target["label"]] = target
    return targets, tests


def crate_name(target):
    name = target.get("crate_name", target["name"])
    for suffix in ("-testutils", "-instrumented", "-benchmarks", "-loom", "-nostd"):
        name = name.removesuffix(suffix)
    return name.replace("-", "_")


def dependencies(target):
    return target.get("deps", []) + target.get("proc_macro_deps", [])


def toml_value(value):
    if isinstance(value, dict):
        return "{ " + ", ".join(k + " = " + toml_value(v) for k, v in value.items()) + " }"
    return json.dumps(value)


def generate():
    targets, tests = read_targets()
    source_cargo = tomllib.loads((SOURCE / "third-party-Cargo.toml").read_text())
    versions = source_cargo["dependencies"] | source_cargo.get("build-dependencies", {})
    for platform in source_cargo.get("target", {}).values():
        for name, version in platform.get("dependencies", {}).items():
            versions.setdefault(name, version)

    # Include the production closure plus its testutils variants and their
    # portable unit-test dependencies, while retaining one Cargo crate/type.
    queue = ["//src/connectivity/network/netstack3/core:netstack3-core"]
    selected, groups, attached_tests = set(), {}, set()
    while queue:
        label = queue.pop()
        if label == FIDL_ONLY or label.startswith("//third_party/") or label in selected:
            continue
        if label not in targets:
            raise RuntimeError(f"unhandled upstream dependency: {label}")
        target = targets[label]
        selected.add(label)
        name = crate_name(target)
        groups.setdefault(name, []).append(target)
        queue.extend(absolute_label(dep, label) for dep in dependencies(target))
        for other_label, other in targets.items():
            if crate_name(other) == name and other["name"].endswith("-testutils"):
                queue.append(other_label)
        for test in tests:
            if test["label"] in attached_tests or test["directory"] != target["directory"]:
                continue
            own = test.get("crate")
            if own and crate_name(targets[absolute_label(own, label)]) != name:
                continue
            if not own and (test.get("crate_name") or test["name"].split("-instrumented")[0].removesuffix("_test")).replace("-", "_") != name:
                continue
            attached_tests.add(test["label"])
            queue.extend(absolute_label(dep, test["label"]) for dep in dependencies(test))

    for name, variants in sorted(groups.items()):
        canonical = next((v for v in variants if not any(v["name"].endswith(s) for s in
                         ("-testutils", "-instrumented", "-benchmarks", "-loom", "-nostd"))), variants[0])
        manifest = HERE / "target/manifests" / name
        manifest.mkdir(parents=True, exist_ok=True)
        runtime, feature_deps, dev = {}, {}, {}

        def edge(dep, context):
            label = absolute_label(dep, context["label"])
            if label == FIDL_ONLY:
                return None
            if label.startswith("//third_party/"):
                key = label.rsplit(":", 1)[-1]
                value = versions[key]
                value = {"version": value} if isinstance(value, str) else dict(value)
                value["version"] = "=" + value["version"]
                # Monorepo-wide std/serde feature unification is not a
                # freestanding-runtime requirement. Proc macros stay host-side.
                if canonical["kind"] != "macro" and key not in ("proptest", "criterion", "rand_xorshift", "test-case", "parking_lot", "parking_lot_core", "crossbeam", "itertools"):
                    value["default-features"] = False
                    if "features" in value:
                        value["features"] = [f for f in value["features"] if f not in ("std", "serde")]
                if key == "rand" and context["kind"] == "test":
                    value["features"] = sorted(set(value.get("features", []) + ["sys_rng"]))
                if key == "lru":
                    # Its no_std HashMap is selected by a default feature.
                    value["features"] = sorted(set(value.get("features", []) + ["hashbrown"]))
                return key, value
            other = targets[label]
            key = crate_name(other)
            value = {"path": "../" + key}
            features = [f for f in other.get("crate_features", []) if f != "std" or canonical["kind"] == "macro"]
            if key == "net_types" and canonical["kind"] == "macro":
                features.append("std")
            if features:
                value["features"] = sorted(set(features))
            return key, value

        for dep in dependencies(canonical):
            item = edge(dep, canonical)
            if item:
                runtime[item[0]] = item[1]
        for variant in variants:
            if variant["name"].endswith("-testutils"):
                for dep in dependencies(variant):
                    item = edge(dep, variant)
                    if item:
                        feature_deps[item[0]] = item[1]
        for test in tests:
            if test["label"] not in attached_tests or test["directory"] != canonical["directory"]:
                continue
            own = test.get("crate")
            if own and crate_name(targets[absolute_label(own, canonical["label"])]) != name:
                continue
            for dep in dependencies(test):
                item = edge(dep, test)
                if item and item[0] != name:
                    dev[item[0]] = item[1]

        features = []
        for key, value in feature_deps.items():
            if key not in runtime:
                runtime[key] = dict(value, optional=True)
                features.append("dep:" + key)
            elif "features" in value:
                # Variants are represented as Cargo features on the same crate.
                for feature in value["features"]:
                    if feature not in runtime[key].get("features", []):
                        features.append(key + "/" + feature)
        text = '[package]\nname = ' + json.dumps(name) + '\nversion = "0.0.0"\nedition = "2024"\npublish = false\n'
        lib_path = canonical["directory"] / canonical.get("crate_root", "src/lib.rs")
        text += '[lib]\npath = ' + json.dumps(str(lib_path.relative_to(manifest, walk_up=True))) + '\n'
        if canonical["kind"] == "macro":
            text += 'proc-macro = true\n'
        text += '[features]\nstd = []\ninstrumented = []\nrc-debug-names = []\ntestutils = ' + toml_value(sorted(set(features))) + '\n'
        text += '[lints.rust]\nunexpected_cfgs = { level = "warn", check-cfg = ["cfg(loom)", "cfg(no_lock_order)", "cfg(benchmark)", "cfg(detect_lock_cycles)"] }\n'
        for section, entries in (("dependencies", runtime), ("dev-dependencies", dev)):
            text += '[' + section + ']\n'
            for key, value in sorted(entries.items()):
                text += key + ' = ' + toml_value(value) + '\n'
        (manifest / "Cargo.toml").write_text(text)
        if name == "netstack3_core":
            # Matches core/BUILD.bazel's no_lock_order setting for production
            # and host unit tests; this omits compile-time lock-order checking,
            # not the actual synchronization primitives.
            (manifest / "build.rs").write_text('fn main() { println!("cargo:rustc-cfg=no_lock_order"); }\n')
    summary = {"revision": PIN, "crates": sorted(groups), "omitted_fidl_helper": FIDL_ONLY}
    (HERE / "target/extraction.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Prepared {len(groups)} Cargo crates at {PIN}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="regenerate manifests from an existing fetched subset")
    args = parser.parse_args()
    SOURCE.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, DIRS.items()))
        (SOURCE / "third-party-Cargo.toml").write_bytes(base64.b64decode(download(BASE + "third_party/rust_crates/Cargo.toml?format=TEXT")))
        upstream_license = base64.b64decode(download(BASE + "LICENSE?format=TEXT"))
        if upstream_license != (HERE / "FUCHSIA-LICENSE").read_bytes():
            raise RuntimeError("pinned Fuchsia license differs from the reviewed notice")
    for patch in sorted((HERE / "patches").glob("*.patch")):
        # An already applied patch is verified by reverse dry-run.
        applied = subprocess.run(["patch", "--force", "--fuzz=0", "--dry-run", "-R", "-p1", "-i", str(patch)], cwd=SOURCE, capture_output=True)
        if applied.returncode:
            subprocess.run(["patch", "--force", "--fuzz=0", "--batch", "-p1", "-i", str(patch)], cwd=SOURCE, check=True)
    generate()


if __name__ == "__main__":
    main()
