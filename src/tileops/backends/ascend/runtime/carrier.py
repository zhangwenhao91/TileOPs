"""Triton launch carriers for the ascend objects.

A carrier exists so Triton produces the launch — grid, argument marshalling, the address
of each buffer — while the object Triton would have compiled is replaced by a
precompiled one (:mod:`tileops.backends.ascend.runtime.launch`). The carrier's *body* is
therefore not what runs; what matters is that its name and its runtime arguments match
the object's exported symbol and parameter list:

- name: the symbol the object exports, here ``vadd_kernel``;
- runtime arguments: one per buffer, then the integer slot the runtime fills with the
  block index. That slot is why the row count is passed at all — the object indexes by
  block, so the value is the runtime's, not the caller's.

Triton is imported here rather than by the backend package: the backend is loaded on
every machine that installs TileOPs, Triton only exists on the ones with an NPU.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def vadd_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    block_index,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """The carrier for ``out = a + b`` over a ``(BM, K)`` tile at row ``block_index``.

    ``block_index`` is unused: the runtime supplies the block index through that slot,
    and the object reads it there. The body mirrors the DSL program the object was built
    from, so the two agree on layout.
    """
    pid = tl.program_id(axis=0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BK)
    a = tl.load(a_ptr + rows[:, None] * K + cols[None, :])
    b = tl.load(b_ptr + rows[:, None] * K + cols[None, :])
    tl.store(out_ptr + rows[:, None] * K + cols[None, :], a + b)
