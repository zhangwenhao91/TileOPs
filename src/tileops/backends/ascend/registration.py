"""What the ``ascend`` target registers: how it recognizes its devices, and one builder
per op it serves.

Importing this module is the whole act of joining TileOPs: module-level calls fill the
registry. It compiles nothing — a builder is only run when an op asks for a kernel.
"""

from __future__ import annotations

import torch

from tileops.backend import register_detector, register_kernel_builder
from tileops.backends.ascend.builders.attention import build_mha_decode_paged
from tileops.backends.ascend.builders.elementwise import build_add


def detects_ascend_npu(device: torch.device) -> bool:
    """Whether *device* is an Ascend NPU.

    ``device.type`` reads ``"npu"`` once ``torch_npu`` has claimed ``PrivateUse1`` under
    that name. Every other device — CPU, CUDA, and the plain-``torch`` ``"privateuseone"``
    spelling before ``torch_npu`` renames it — is not ours, and answers ``False`` rather
    than raising, which is what a detector owes the dispatcher.
    """
    return device.type == "npu"


register_detector("ascend", detects_ascend_npu)

# One line per op this target serves. ``op`` is the manifest key, which is also the name
# the op class is registered under, so the two never drift.
register_kernel_builder("AddFwdOp", "ascend", build_add)
register_kernel_builder("MHADecodePagedWithKVCacheFwdOp", "ascend", build_mha_decode_paged)
