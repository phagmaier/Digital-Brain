#!/usr/bin/env bash
# Cross-version Phase-1 baseline guard. Unlike check-determinism.sh, this
# compares the current default run against a committed, human-auditable result.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOLDEN="$ROOT/reproducibility"
WORK="${TMPDIR:-/tmp}/brain-golden-check-$$"
ARTEFACTS=(raster.csv metrics.csv neurons.csv synapses.csv run_meta.json)

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

echo "==> building"
(cd "$ROOT" && zig build)

echo "==> running built-in Phase-1 baseline"
(cd "$WORK" && "$ROOT/zig-out/bin/brain" > summary.txt)
(cd "$WORK" && sha256sum "${ARTEFACTS[@]}" > actual.sha256)

diff -u "$GOLDEN/default-baseline.sha256" "$WORK/actual.sha256"
diff -u "$GOLDEN/default-summary.txt" "$WORK/summary.txt"

echo "PASS: default artefacts and summary match the committed golden baseline."
echo "      provenance: $GOLDEN/default-baseline.meta"
