"""Temporary process-level probe: why does the op fall to the in-tree path on NPU?

Logs to /tmp/tileops_spy.log (immune to tail truncation).
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

_LOG = "/tmp/tileops_spy.log"


def log(msg: str) -> None:
    with open(_LOG, "a") as f:
        f.write(msg + "\n")


_orig_select = dispatch.select_target
_orig_detect = dispatch.detect_target
_orig_bind = op_base.Op._bind_target


def _spy_select(requested, device):
    got = _orig_select(requested, device)
    log(f"select_target(requested={requested!r}, device={device!r}) -> {got!r}")
    return got


def _spy_detect(device):
    got = _orig_detect(device)
    log(f"detect_target(device={device!r}) -> {got!r}")
    return got


def _spy_bind(self, args, kwargs, device=None):
    _orig_bind(self, args, kwargs, device)
    log(
        f"_bind_target: device={device!r} self.target={self.target!r} -> "
        f"serving={self._serving_target!r} builder={self._target_builder!r}"
    )


@pytest.mark.smoke
def test_debug_binding(monkeypatch) -> None:
    open(_LOG, "w").close()  # truncate
    log("=== in-pytest registry state ===")
    registry.ensure_loaded()
    log(f"DETECTORS: {registry.DETECTORS}")
    log(f"BUILDERS: {sorted(registry.BUILDERS)}")
    log(f"default_target: {registry.default_target}")
    log(f"run_device: {device_mod.run_device()}")
    log(f"npu_is_available: {device_mod.npu_is_available()}")
    d = torch.device("npu", 0)
    log(f"detect_target(npu): {dispatch.detect_target(d)!r}")
    log(f"select_target(None, npu): {dispatch.select_target(None, d)!r}")

    monkeypatch.setattr(dispatch, "select_target", _spy_select)
    monkeypatch.setattr(dispatch, "detect_target", _spy_detect)
    monkeypatch.setattr(op_base, "select_target", _spy_select)
    monkeypatch.setattr(op_base.Op, "_bind_target", _spy_bind)

    op = MHADecodePagedWithKVCacheFwdOp(page_size=128, is_causal=False)
    log(f"op.target: {op.target!r}")

    dev = torch.device(device_mod.run_device())
    B, S_Q, H, D = 1, 1, 2, 128
    PAGE, W = 128, 4
    q = torch.randn(B, S_Q, H, D, device=dev, dtype=torch.float16)
    kp = torch.randn(W * PAGE, H, D, device=dev, dtype=torch.float16)
    vp = torch.randn(W * PAGE, H, D, device=dev, dtype=torch.float16)
    real = torch.tensor([PAGE], dtype=torch.int32, device=dev)
    table = torch.arange(W, dtype=torch.int32, device=dev).view(B, W)
    log("=== direct op call ===")
    try:
        out = op(q, kp, vp, real, table)
    except Exception as exc:
        log(f"direct call RAISED: {type(exc).__name__}: {exc}")
        log(f"serving_target after failure: {op._serving_target!r}")
        raise
    log(f"direct call OK, out={tuple(out.shape)} finite={torch.isfinite(out).all().item()}")
    log(f"serving_target after call: {op._serving_target!r}")
