#!/usr/bin/env bash
# Turn the TileLang DSL in vadd.py into the objects this target ships.
#
#   TileLangIR -> NPU IR -> MIX -> CCE -> LLVM -> .o
#
# Run it on a machine with the lowering toolchain, not on the NPU host: the objects it
# writes are what ship. Output lands in ../artifacts/, where runtime.support expects it.
#
#   ./build.sh                # all shapes in vadd.py's SHAPES
#   FORCE_STAGE0=1 ./build.sh # re-emit the TileLang IR from the DSL first
#
# Tools are resolved as env var > local absolute path > bare name on PATH, so the same
# script runs here and on a host that has them installed:
#   TILE_OPT        default ~/work/tilelang-toolchain/build/bin/tile-opt
#   TILE_TRANSLATE  default ~/work/tilelang-toolchain/build/bin/tile-translate
#   CCEC            default CANN 9.1.0's ccec
#   TILELANG_PYTHON default /home/zwh/work/venv/bin/python (the lowering venv, 3.14)
#
# The stages are the ones the in-tree kernels are known to lower through; the parameters
# that matter are the arch the objects are built for (the vector core, so dav-c310-vec)
# and no mix mode (an add has no cube half).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

is_exec() { [ -x "$1" ] || command -v "$1" >/dev/null 2>&1; }

# env override > local absolute path > bare name on PATH
pick() {
  if [ -n "${1:-}" ]; then printf '%s' "$1"; return; fi
  if [ -x "$2" ]; then printf '%s' "$2"; return; fi
  printf '%s' "$3"
}

OPT="$(pick "${TILE_OPT:-}" /home/zwh/work/tilelang-toolchain/build/bin/tile-opt tile-opt)"
TRANS="$(pick "${TILE_TRANSLATE:-}" /home/zwh/work/tilelang-toolchain/build/bin/tile-translate tile-translate)"
CCEC="$(pick "${CCEC:-}" /home/zwh/work/cann/9.1.0/toolkit/cann-9.1.0/tools/bisheng_compiler/bin/ccec ccec)"
PY="${TILELANG_PYTHON:-/home/zwh/work/venv/bin/python}"

CCEC_ARCH="${CCEC_ARCH:-dav-c310-vec}"
FORCE_STAGE0="${FORCE_STAGE0:-0}"
ARTIFACTS="$HERE/../artifacts"
SHAPES="128 256"

for spec in "TILE_OPT:$OPT" "TILE_TRANSLATE:$TRANS" "CCEC:$CCEC"; do
  name="${spec%%:*}"; val="${spec#*:}"
  is_exec "$val" || { echo "ERROR: $name not usable: $val" >&2; exit 1; }
done
if [ "$FORCE_STAGE0" = 1 ]; then
  is_exec "$PY" || { echo "ERROR: TILELANG_PYTHON not usable: $PY" >&2; exit 1; }
fi

# Each step is checked on its own: this function runs inside `if ( ... )`, where bash
# ignores `set -e`, so a failure would otherwise fall through to the next step.
lower_one() { # $1=inner extent
  local k="$1" b="vadd_f32_k$1"
  local src="1_${b}_tilelangir.mlir"

  if [ "$FORCE_STAGE0" = 1 ] || [ ! -f "$src" ]; then
    "$PY" vadd.py || { echo "FAIL [$b] stage0 emit" >&2; return 1; }
  fi

  echo "-- [$b] TileLangIR -> NPU IR"
  # The producer's MLIR bindings spell a binary op `linalg.elementwise <add>`; the
  # conversion only knows the older `linalg.add`. Normalize rather than fail.
  sed -E 's/linalg\.elementwise <([a-zA-Z_]+)>/linalg.\1/' "$src" > "1_${b}.norm.mlir" \
    || { echo "FAIL [$b] sed normalize" >&2; return 1; }
  "$OPT" "1_${b}.norm.mlir" --convert-tilelang-to-npu --npu-normalize -o "2_${b}_npu.mlir" \
    || { echo "FAIL [$b] stage1 convert-tilelang-to-npu" >&2; return 1; }

  echo "-- [$b] NPU IR -> MIX"
  "$OPT" "2_${b}_npu.mlir" --npu-split-dataflow --npu-plan-memory --npu-split-mix-kernel \
    --npu-sync-pipeline -o "3_${b}_mix_npu.mlir" \
    || { echo "FAIL [$b] stage2 split" >&2; return 1; }

  echo "-- [$b] MIX NPU -> CCE"
  "$OPT" --cce-pipeline="target=dav-351x" "3_${b}_mix_npu.mlir" -o "4_${b}_cce.mlir" \
    || { echo "FAIL [$b] stage3 cce-pipeline" >&2; return 1; }

  echo "-- [$b] CCE -> LLVM"
  "$TRANS" --cce-to-backend -allow-unregistered-dialect "4_${b}_cce.mlir" -o "5_${b}.ll" \
    || { echo "FAIL [$b] stage4 cce-to-backend" >&2; return 1; }

  echo "-- [$b] ccec -> $ARTIFACTS/$b.o (arch=$CCEC_ARCH)"
  mkdir -p "$ARTIFACTS" || { echo "FAIL [$b] mkdir $ARTIFACTS" >&2; return 1; }
  "$CCEC" --cce-aicore-arch="$CCEC_ARCH" --cce-aicore-only -O2 -cce-bitcode-is-aicore -c \
    "5_${b}.ll" -o "$ARTIFACTS/$b.o" \
    || { echo "FAIL [$b] stage5 ccec" >&2; return 1; }

  # The carrier is looked up by this symbol, so the object has to export it.
  nm -a "$ARTIFACTS/$b.o" | grep -q ' T vadd_kernel$' \
    || { echo "FAIL [$b] object does not export vadd_kernel" >&2; return 1; }
  return 0
}

rc=0
for k in $SHAPES; do
  echo "############ inner extent k$k ############"
  if ! ( lower_one "$k" ); then
    echo "FAIL lower: vadd_f32_k$k（the .o was not updated; do not use a stale one）" >&2
    rc=1
  fi
done
exit $rc
