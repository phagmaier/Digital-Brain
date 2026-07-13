#!/usr/bin/env bash
# Reproducibility guard for the whole built pipeline.
#
# The unit tests check that pieces are deterministic; this checks that the
# actual `brain` binary, run end to end, produces byte-identical artefacts on
# repeated runs. This is the Phase 0 exit criterion, enforced.
#
# Usage:
#   scripts/check-determinism.sh [config.json]
#
# With no argument it runs the built-in default config. With a config path it
# passes that through to the binary, so you can pin any run.
#
# Exit 0 = identical, exit 1 = a reproducibility regression (a real bug).
set -euo pipefail

cd "$(dirname "$0")/.."

ARTEFACTS=(raster.csv metrics.csv neurons.csv synapses.csv)
CONFIG_ARG=("$@")

echo "==> building"
zig build

run_and_hash() {
  # Runs the binary, then prints "<sha256>  <name>" for each artefact.
  ./zig-out/bin/brain "${CONFIG_ARG[@]}" >/dev/null
  sha256sum "${ARTEFACTS[@]}"
}

echo "==> run 1"
H1="$(run_and_hash)"
echo "==> run 2"
H2="$(run_and_hash)"

echo
if [[ "$H1" == "$H2" ]]; then
  echo "$H1"
  echo
  echo "PASS: byte-identical across runs."
  exit 0
else
  echo "run 1:"; echo "$H1"
  echo "run 2:"; echo "$H2"
  echo
  echo "FAIL: artefacts differ between runs — reproducibility is broken." >&2
  diff <(echo "$H1") <(echo "$H2") || true
  exit 1
fi
