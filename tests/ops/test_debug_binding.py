"""Temporary process-level probe: why does the op fall to the in-tree path on NPU?

Run: pytest tests/ops/test_debug_binding.py --tileops-device npu -v -s
Delete once the binding mystery is solved.
"""

import torch
import pytest

import workloads.device as device_mod
import tileops.ops.op_base as op_base
from tileops.backend import registry
from tileops.backend import dispatch
from tileops.ops import MHADecodePagedWithKVCacheFwdOp

_orig_select = dispatch.select_target
_orig_detect = dispatch.detect_target


def _spy_select(requested, device):
    got = _orig_select(requested, device)
    print(f"[SPY] select_target(requested={requested!r}, device={device!r}) -> {got!r}")
    return got


def _spy_detect(device):
    got = _orig_detect(device)
    print(f"[SPY] detect_target(device={device!r}) -> {got!r}")
    return got


@pytest.mark.smoke
def test_debug_binding(monkeypatch) -> None:
    print("\n=== in-pytest registry state ===")
    registry.ensure_loaded()
    print("DETECTORS:", registry.DETECTORS)
    print("BUILDERS:", sorted(registry.BUILDERS))
    print("default_target:", registry.default_target)
    print("run_device:", device_mod.run_device())
    print("npu_is_available:", device_mod.npu_is_available())
    d = torch.device("npu", 0)
    print("detect_target(npu):", dispatch.detect_target(d))
    print("select_target(None, npu):", dispatch.select_target(None, d))

    monkeypatch.setattr(dispatch, "select_target", _spy_select)
    monkeypatch.setattr(dispatch, "detect_target", _spy_detect)
    # op_base 调的是 `from ... import select_target` 进来的名字，两处都替换
    monkeypatch.setattr(op_base, "select_target", _spy_select)

    op = MHADecodePagedWithKVCacheFwdOp(page_size=128, is_causal=False)
    print("op.target:", op.target)

    dev = torch.device(device_mod.run_device())
    B, S_Q, H, D = 1, 1, 2, 128
    PAGE, W = 128, 4
    q = torch.randn(B, S_Q, H, D, device=dev, dtype=torch.float16)
    kp = torch.randn(W * PAGE, H, D, device=dev, dtype=torch.float16)
    vp = torch.randn(W * PAGE, H, D, device=dev, dtype=torch.float16)
    real = torch.tensor([PAGE], dtype=torch.int32, device=dev)
    table = torch.arange(W, dtype=torch.int32, device=dev).view(B, W)
    print("=== direct op call (no verify) ===")
    try:
        out = op(q, kp, vp, real, table)
    except Exception as exc:
        print(f"[SPY] direct call RAISED: {type(exc).__name__}: {exc}")
        print("op._serving_target after failure:", op._serving_target)
        raise
    print("direct call OK, out:", tuple(out.shape), "finite:", torch.isfinite(out).all().item())
    print("op._serving_target after call:", op._serving_target)
