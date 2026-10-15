"""Which precompiled object serves which call.

An object is built from a TileLang DSL program for one element type and one inner
extent, so the mapping is a support *matrix* rather than a formula: a call outside it has
no object, and the builder says so instead of compiling something at call time. Shapes
stay dynamic because the tile the object was built for is fixed while the number of
tiles is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

# The directory ships beside this module and is listed in the package data, so an
# installed wheel carries the objects the same way the source tree does.
_ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Rows one block handles. Part of how an object was built, not a per-call choice, so a
# call whose row count is not a multiple of it has no object.
BLOCK_ROWS = 16


@dataclass(frozen=True)
class Artifact:
    """One precompiled object and the tile it was built for.

    ``symbol`` is the name the object exports, which is also what the Triton carrier is
    named: the runtime locates the kernel by that name.
    """

    symbol: str
    path: Path
    dtype: torch.dtype
    inner: int
    cv_mode: str


def _artifact(symbol: str, filename: str, dtype: torch.dtype, inner: int) -> Artifact:
    return Artifact(
        symbol=symbol,
        path=_ARTIFACTS / filename,
        dtype=dtype,
        inner=inner,
        # Every add object here is a vector kernel; the cube and mix cores are for the
        # matmul-shaped ops.
        cv_mode="aiv",
    )


# One row per shipped object. Adding a dtype or an inner extent is one line here plus the
# object it names.
_MATRIX: dict[tuple[torch.dtype, int], Artifact] = {
    (torch.float32, 128): _artifact("vadd_kernel", "vadd_f32_k128.o", torch.float32, 128),
    (torch.float32, 256): _artifact("vadd_kernel", "vadd_f32_k256.o", torch.float32, 256),
}


def vadd_artifact(dtype: torch.dtype, inner: int) -> Artifact | None:
    """The object serving ``input + other`` for *dtype* over an inner extent of *inner*.

    ``None`` when the matrix has no such row or the object it names is missing, which is
    what lets a builder answer "not available" instead of failing at launch.
    """
    artifact = _MATRIX.get((dtype, inner))
    if artifact is None or not artifact.path.is_file():
        return None
    return artifact


def supported() -> list[tuple[torch.dtype, int]]:
    """The (dtype, inner extent) rows whose objects are actually present, for messages."""
    return sorted(
        (dtype, inner) for (dtype, inner), artifact in _MATRIX.items() if artifact.path.is_file()
    )
