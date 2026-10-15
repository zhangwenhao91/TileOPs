"""The TileLang DSL source of the objects this target ships for ``AddFwdOp``.

One program per shipped ``(dtype, inner extent)`` row: the row's *inner extent* is the
``K`` here, and ``BR`` is the row count per block, which is
:data:`tileops.backends.ascend.runtime.support.BLOCK_ROWS`. Both are baked into the
object — the tile it walks and the stride between its rows — while the block index stays
the runtime's, which is why one object serves every row count that divides by ``BR``.

``build.sh`` beside this file runs the program and lowers what it writes; see there for
the toolchain it needs. TileLang is imported inside the function so that reading or
packaging this module does not require it.
"""

from __future__ import annotations

import os

# Rows one block handles. Must equal ``runtime.support.BLOCK_ROWS``: the object's tile is
# what the carrier's grid is divided by.
BR = 16

# The ``(dtype, inner extent)`` rows to emit, as ``(M, K)`` for the DSL. ``M`` only has to
# be a multiple of ``BR`` and is not baked into the object; ``K`` is.
SHAPES = ((32, 128), (32, 256))


def emit(M: int, K: int, path: str, br: int = BR) -> str:
    """Lower ``OUT = A + B`` over ``(M, K)`` float32 and write the TileLang IR to *path*."""
    import tilelang
    import tilelang.language as T

    assert M % br == 0, f"M must be a multiple of BR ({br}), got M={M}"

    dtype = "float32"

    @T.prim_func
    def vadd_kernel(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([M, K], dtype),
        OUT: T.Tensor([M, K], dtype),
    ):
        with T.Kernel(M // br) as bx:
            a_shared = T.alloc_shared((br, K), dtype)
            b_shared = T.alloc_shared((br, K), dtype)
            c_shared = T.alloc_shared((br, K), dtype)
            T.copy(A[bx * br : (bx + 1) * br, 0:K], a_shared)
            T.copy(B[bx * br : (bx + 1) * br, 0:K], b_shared)

            with T.SimdVF():
                a_frag = T.alloc_frag((br, K), dtype)
                b_frag = T.alloc_frag((br, K), dtype)
                c_frag = T.alloc_frag((br, K), dtype)
                T.copy(a_shared[0:br, 0:K], a_frag)
                T.copy(b_shared[0:br, 0:K], b_frag)
                T.vadd(a_frag, b_frag, c_frag)
                T.copy(c_frag, c_shared[0:br, 0:K])

            T.copy(c_shared, OUT[bx * br : (bx + 1) * br, 0:K])

    with open(path, "w") as handle:
        handle.write(tilelang.lower(vadd_kernel, target="tile").kernel_source)
    return path


if __name__ == "__main__":
    # One IR per shape, named the way build.sh looks for it. Only the IR is written here;
    # lowering it is build.sh's job.
    where = os.path.dirname(os.path.abspath(__file__))
    for m, k in SHAPES:
        out = os.path.join(where, f"1_vadd_f32_k{k}_tilelangir.mlir")
        print("wrote", emit(m, k, out))
