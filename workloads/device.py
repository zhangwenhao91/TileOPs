"""The device a test or benchmark run places its tensors on.

The run names it (``--tileops-device`` in ``tests/conftest.py``); every workload and every test
that is not ``cuda_only`` or ``npu_only`` reads it here. Read it when a tensor is built: a module
constant or a default argument would keep the value from before the run set it.
"""

from __future__ import annotations

from importlib.util import find_spec

import torch

_device: torch.device | str = "cuda"


def run_device() -> torch.device | str:
    """The device this run places tensors on; ``"cuda"`` unless the run named another."""
    return _device


def run_device_is_cuda() -> bool:
    """Whether this run places tensors on a CUDA device."""
    return torch.device(_device).type == "cuda"


def npu_is_available() -> bool:
    """Whether an Ascend NPU is reachable from this process.

    ``torch_npu`` is what claims the ``npu`` device type, so its presence is the question:
    until it is imported torch has no such device to ask about, and importing it here would
    load the driver only to decide which tests this run can serve.
    """
    return find_spec("torch_npu") is not None


def set_run_device(device: torch.device | str) -> None:
    """Place this run's tensors on *device* from now on."""
    global _device
    _device = device


def run_device_available() -> bool:
    """Whether this process can place tensors on the run device."""
    kind = torch.device(_device).type
    if kind in ("cpu", "meta"):
        return True
    return getattr(torch, kind).is_available()
