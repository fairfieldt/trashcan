#!/bin/bash
# CPU tests of the original archived candidate, independent of the live engine.
# Default: GDN preparation tests. Pass other cargo-test arguments after the script,
# e.g. link::tests -- --nocapture, or a historical benchmark's --ignored filter.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
candidate_tmp=$(mktemp -d /tmp/trashcan-offline-cpu.XXXXXX)
trap 'rm -rf "$candidate_tmp"' EXIT
tar -xzf "$experiment/source.tar.gz" -C "$candidate_tmp"
if [[ $# == 0 ]]; then set -- preparation_ -- --nocapture; fi
cargo test --offline --release --manifest-path "$candidate_tmp/Cargo.toml" "$@"
