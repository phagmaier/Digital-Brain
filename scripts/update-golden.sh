#!/usr/bin/env bash
# Intentionally refresh the committed Phase-1 golden after an approved dynamics
# or toolchain change. Review the resulting diff before committing it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GOLDEN="$ROOT/reproducibility"
WORK="${TMPDIR:-/tmp}/brain-golden-update-$$"
ARTEFACTS=(raster.csv metrics.csv neurons.csv synapses.csv run_meta.json)

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

(cd "$ROOT" && zig build)
(cd "$WORK" && "$ROOT/zig-out/bin/brain" > "$GOLDEN/default-summary.txt")
(cd "$WORK" && sha256sum "${ARTEFACTS[@]}" > "$GOLDEN/default-baseline.sha256")

PARENT_COMMIT="$(cd "$ROOT" && git rev-parse HEAD)"
ZIG_VERSION="$(zig version)"
cat > "$GOLDEN/default-baseline.meta" <<EOF
schema_version=1
baseline_parent_commit=$PARENT_COMMIT
baseline_change=intentional-golden-update
config_source=src/main.zig built-in defaults
master_seed=12648430
prng_algorithm=xoshiro256++
prng_impl_version=2
zig_version=$ZIG_VERSION
artifact_manifest=default-baseline.sha256
expected_summary=default-summary.txt
EOF

echo "Updated $GOLDEN. Review all golden diffs before committing."
