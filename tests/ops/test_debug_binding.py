"""Temporary process-level probe: why does the op fall to the in-tree path on NPU?

Run: pytest tests/ops/test_debug_binding.py --tileops-device npu -v -s
Delete once the binding mystery is solved.
"""

import torch
import pytest

import workloads.device as device_mod
from tileops.backend import registry
from tileops.backend import dispatch
from tileops.ops import MHADecodePagedWithKVCacheFwdOp


@pytest.mark.smoke
def test_debug_binding() -> None:
    print("\n=== in-pytest registry state ===")
    registry.ensure_loaded()
    print("DETECTORS:", registry.DETECTORS)
    print("BUILDERS:", sorted(registry.BUILDERS))
    print("default_target:", registry.default_target)
    d = torch.device("npu", 0)
    print("run_device:", device_mod.run_device())
    print("npu_is_available:", device_mod.npu_is_available())
    print("detect_target(npu):", dispatch.detect_target(d))
    print("select_target(None, npu):", dispatch.select_target(None, d))

    op = MHADecodePagedWithKVCacheFwdOp(page_size=128, is_causal=False)
    print("op.target:", op.target)
    print("op._target_builder is _UNRESOLVED:",
          op._target_builder is type(op).__mro__[0]
          or op._target_builder)  # print whatever it is

    dev = torch.device(device_mod.run_device())
    B, S_Q, H, D = 1, 1, 2, 128
    PAGE, W = 128, 4
    q = torch.randn(B, S_Q, H, D, device=dev, dtype=torch.float16)
    kp = torch.randn(W * PAGE, H, D, device=dev, dtype=torch.float16)
    vp = torch.randn(W * PAGE, H, D, device=dev, dtype=torch.float16)
    real = torch.tensor([PAGE], dtype=torch.int32)
    table = torch.arange(W, dtype=torch.int32).view(B, W)
    print("=== direct op call (no verify) ===")
    out = op(q, kp, vp, real, table)
    print("direct call OK, out:", out.shape, "finite:", torch.isfinite(out).all().item())
    print("op._serving_target after call:", op._serving_target)
