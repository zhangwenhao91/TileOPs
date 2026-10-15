"""Running one object on the NPU.

The runtime learns which object to load from the environment, because that is the only
door the patched ``triton-opentile`` backend offers. The variables are process-wide, so
they are set around the launch and put back afterwards rather than left behind for
whatever runs next.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch

from tileops.backends.ascend.runtime.support import BLOCK_ROWS, Artifact


@contextmanager
def _runtime_pointed_at(artifact: Artifact) -> Iterator[None]:
    """Point the runtime at *artifact* for the duration of the block.

    ``KERNEL_PATH`` names the object to load in place of the one Triton would compile,
    and ``KERNEL_CV_MODE`` which core type it was built for.
    """
    keys = ("KERNEL_PATH", "KERNEL_CV_MODE")
    saved = {key: os.environ.get(key) for key in keys}
    os.environ["KERNEL_PATH"] = str(artifact.path)
    os.environ["KERNEL_CV_MODE"] = artifact.cv_mode
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def launch_vadd(artifact: Artifact, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run ``a + b`` on the NPU through *artifact*, returning a fresh tensor.

    Both operands are contiguous and equally shaped — the builder refuses the rest — so
    the leading axes flatten into rows without copying: the object walks rows of
    ``inner`` elements, ``BLOCK_ROWS`` to a block, which is exactly this layout.
    """
    # Imported here, not at module scope: the carrier module imports Triton, which is
    # installed only where an NPU is.
    from tileops.backends.ascend.runtime.carrier import vadd_kernel

    inner = a.shape[-1]
    rows = a.numel() // inner
    out = torch.empty_like(a)
    # The runtime reads the object's location from the environment, so the variables are
    # set for the launch and put back afterwards.
    with _runtime_pointed_at(artifact):
        # One block per BLOCK_ROWS rows. The row count fills the object's block-index slot;
        # the runtime overwrites it with the index of each block in turn.
        vadd_kernel[(rows // BLOCK_ROWS,)](a, b, out, rows, K=inner, BM=BLOCK_ROWS, BK=inner)
    return out
