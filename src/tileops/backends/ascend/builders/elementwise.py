"""Builders for the binary arithmetic elementwise ops on the ``ascend`` target.

One builder per op this target serves — today ``AddFwdOp`` alone. An op with no builder
is refused by the op layer rather than served, which is what ``MulFwdOp`` gets: the
target has no object for it, and no in-tree kernel runs on an NPU to fall back to.

A builder matches its op's manifest ``signature``: a positional
:class:`~tileops.backend.protocol.TensorSpec` per input (``input``, ``other``), then the
params by keyword (``alpha``). Each returns a callable the op runs the whole call
with — here, one that loads and launches a precompiled NPU kernel.

A builder decides; the runtime does. So this module states what it can serve and names
nothing about objects, carriers or the environment: it asks
:mod:`tileops.backends.ascend.runtime.support` whether a row exists and hands the row it
gets back to :func:`tileops.backends.ascend.runtime.launch.launch_vadd`.
"""

from __future__ import annotations

import math

from tileops.backend import OpNotAvailableError, TensorSpec
from tileops.backends.ascend.runtime import launch, support


def _refuse(op: str, why: str) -> "OpNotAvailableError":
    """The error a builder raises for a call this target has no kernel for.

    Raising, rather than returning something uncallable, keeps the reason: the op layer
    can only report "not callable", while the caller needs to know which axis missed.
    """
    served = ", ".join(f"{dtype}~k{inner}" for dtype, inner in support.supported()) or "nothing"
    return OpNotAvailableError(
        f"the ascend target serves {op} for {why}; it ships {served}. "
        f"No in-tree kernel runs on an NPU, so the call cannot fall back."
    )


def build_add(input: TensorSpec, other: TensorSpec, *, alpha: int | float = 1):
    """Return the kernel serving ``input + alpha * other`` on the NPU.

    Raises:
        OpNotAvailableError: The call is outside what this target ships — a non-default
            ``alpha``, two element types or two shapes, or an inner extent or row count
            no object was built for.
    """
    # ``alpha`` is baked into the object rather than passed to it, so only the value the
    # shipped objects were built with is served. ``alpha=1.0`` runs the alpha=1 object,
    # which is the point of comparing by value.
    if alpha != 1:
        raise _refuse("AddFwdOp", f"alpha=1 (got {alpha})")
    if input.dtype != other.dtype:
        raise _refuse("AddFwdOp", f"one element type (got {input.dtype} and {other.dtype})")
    if tuple(input.shape) != tuple(other.shape):
        # The manifest allows broadcast; the objects do not implement it yet, so a
        # broadcast pair is refused instead of silently computing the wrong shape.
        raise _refuse(
            "AddFwdOp",
            f"equal shapes (got {tuple(input.shape)} and {tuple(other.shape)})",
        )
    if not input.shape:
        raise _refuse("AddFwdOp", "at least one dimension")

    # The object walks rows of ``inner`` elements, ``BLOCK_ROWS`` rows to a block, so the
    # two are what a call has to divide into — not the total size.
    inner = input.shape[-1]
    artifact = support.vadd_artifact(input.dtype, inner)
    if artifact is None:
        raise _refuse("AddFwdOp", f"inner extent {inner} in {input.dtype}")

    # Every axis but the last flattens into rows, which is what the object walks: the last
    # axis is the inner extent, and the stride between rows is exactly it.
    rows = math.prod(input.shape[:-1])
    if rows % support.BLOCK_ROWS:
        raise _refuse("AddFwdOp", f"a row count divisible by {support.BLOCK_ROWS} (got {rows})")

    def run(a, b):
        """Run ``a + b`` on the NPU, returning a fresh tensor."""
        return launch.launch_vadd(artifact, a, b)

    return run
