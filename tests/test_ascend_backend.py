"""What the ``ascend`` backend promises.

The two halves a target owns: routing, which the op layer drives and which is observable
on any machine, and the object matrix, which is a table of files. Launching needs an NPU,
so it sits behind ``npu_only`` at the end and is deselected wherever there is none.

Refusals are checked as carefully as the successes. A target that quietly returned
something for a call it cannot serve would compute the wrong answer; one that raised the
right error tells the caller which axis missed.
"""

import os

import pytest
import torch

from tileops.backend import OpNotAvailableError, TensorSpec
from tileops.backends.ascend.builders.elementwise import build_add
from tileops.backends.ascend.runtime import launch, support
from tileops.ops.elementwise.arithmetic import AddFwdOp, MulFwdOp

pytestmark = pytest.mark.smoke

# An ELF magic number: what a compiled object starts with, and what a placeholder or a
# truncated file does not. Read here rather than by invoking `nm`, so the check needs no
# toolchain on the machine running it.
ELF_MAGIC = b"\x7fELF"


def _spec(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> TensorSpec:
    """A tensor description the builder accepts; the device is the op layer's business."""
    return TensorSpec(torch.device("cpu"), dtype, shape)


def _refuses(**args) -> bool:
    """Whether ``build_add`` refuses this call."""
    try:
        build_add(**args)
    except OpNotAvailableError:
        return True
    return False


# --------------------------------------------------------------------------------------
# The object matrix
# --------------------------------------------------------------------------------------


def test_every_row_the_matrix_reports_names_a_real_compiled_object():
    """A row is only worth listing if the object it names was built and shipped."""
    rows = support.supported()

    assert rows == [(torch.float32, 128), (torch.float32, 256)]
    for dtype, inner in rows:
        artifact = support.vadd_artifact(dtype, inner)
        assert artifact is not None
        assert artifact.path.name == f"vadd_f32_k{inner}.o"
        assert artifact.path.read_bytes().startswith(ELF_MAGIC)


def test_a_dtype_and_an_inner_extent_that_were_not_built_have_no_object():
    assert support.vadd_artifact(torch.float16, 128) is None
    assert support.vadd_artifact(torch.float32, 64) is None


# --------------------------------------------------------------------------------------
# What the builder serves
# --------------------------------------------------------------------------------------


def test_a_shipped_shape_gets_a_kernel():
    assert callable(build_add(_spec((32, 128)), _spec((32, 128))))


def test_the_row_count_is_free_because_the_object_indexes_by_block():
    """The tile is baked into the object; the number of blocks is the runtime's."""
    assert callable(build_add(_spec((4096, 128)), _spec((4096, 128))))
    assert callable(build_add(_spec((4096, 256)), _spec((4096, 256))))


def test_the_inner_extent_is_not_free_because_it_is_the_object_stride():
    """K is the stride the object was built with, so only the built ones are served."""
    assert callable(build_add(_spec((32, 256)), _spec((32, 256))))
    assert _refuses(input=_spec((32, 64)), other=_spec((32, 64)))


def test_a_default_alpha_written_as_a_float_is_the_same_call():
    """``alpha`` is compared by value: 1 and 1.0 pick the same object."""
    assert callable(build_add(_spec((32, 128)), _spec((32, 128)), alpha=1.0))


# --------------------------------------------------------------------------------------
# What the builder refuses
# --------------------------------------------------------------------------------------


def test_a_non_default_alpha_is_refused_because_it_is_baked_in():
    """The object computes ``a + b``; an alpha it was not built with has no object."""
    assert _refuses(input=_spec((32, 128)), other=_spec((32, 128)), alpha=0.5)
    assert _refuses(input=_spec((32, 128)), other=_spec((32, 128)), alpha=2)


def test_a_broadcast_pair_is_refused_rather_than_computed_wrong():
    """The manifest allows broadcast. Refusing says so; broadcasting here would not."""
    assert _refuses(input=_spec((1024, 4096)), other=_spec((1, 4096)))


def test_two_element_types_are_refused():
    assert _refuses(input=_spec((32, 128), torch.float32), other=_spec((32, 128), torch.float16))


def test_a_row_count_the_tile_does_not_divide_is_refused():
    assert _refuses(input=_spec((24, 128)), other=_spec((24, 128)))


def test_a_zero_dimensional_input_is_refused():
    assert _refuses(input=_spec(()), other=_spec(()))


# --------------------------------------------------------------------------------------
# Pointing the runtime at an object
# --------------------------------------------------------------------------------------


def test_the_runtime_is_pointed_at_the_object_and_put_back():
    """The variables are process-wide, so the block must not leave them set."""
    artifact = support.vadd_artifact(torch.float32, 128)
    assert artifact is not None
    before = dict(os.environ)

    with launch._runtime_pointed_at(artifact):
        assert os.environ["KERNEL_PATH"] == str(artifact.path)
        assert os.environ["KERNEL_CV_MODE"] == artifact.cv_mode

    assert os.environ.get("KERNEL_PATH") == before.get("KERNEL_PATH")
    assert os.environ.get("KERNEL_CV_MODE") == before.get("KERNEL_CV_MODE")


def test_a_nested_block_restores_the_outer_object():
    """The value that was there comes back: the block is nestable, not merely clearing."""
    outer = support.vadd_artifact(torch.float32, 128)
    inner = support.vadd_artifact(torch.float32, 256)
    assert outer is not None and inner is not None

    with launch._runtime_pointed_at(outer):
        with launch._runtime_pointed_at(inner):
            assert os.environ["KERNEL_PATH"] == str(inner.path)
        assert os.environ["KERNEL_PATH"] == str(outer.path)


# --------------------------------------------------------------------------------------
# Routing: the op layer's half
# --------------------------------------------------------------------------------------


def test_the_op_runs_what_the_builder_returned(monkeypatch):
    """The op calls the returned kernel with the manifest inputs, in order."""
    seen = {}

    def fake_launch(artifact, a, b):
        seen["artifact"] = artifact
        seen["tensors"] = (a, b)
        return torch.zeros_like(a)

    monkeypatch.setattr(launch, "launch_vadd", fake_launch)
    a = torch.zeros(32, 128)
    b = torch.ones(32, 128)

    out = AddFwdOp(target="ascend")(a, b)

    assert seen["artifact"] is support.vadd_artifact(torch.float32, 128)
    assert seen["tensors"] == (a, b)
    assert out.shape == a.shape and out.dtype == a.dtype


def test_an_op_this_target_does_not_serve_is_refused_and_does_not_fall_back():
    """``MulFwdOp`` has in-tree kernels and no ascend one; the call must not reach them."""
    with pytest.raises(OpNotAvailableError):
        MulFwdOp(target="ascend")(torch.zeros(32, 128), torch.zeros(32, 128))


# --------------------------------------------------------------------------------------
# On hardware
# --------------------------------------------------------------------------------------


@pytest.mark.npu_only
@pytest.mark.parametrize(
    "shape",
    [(32, 128), (4096, 128), (64, 256)],
    ids=["one-block", "many-blocks", "inner-256"],
)
def test_the_op_adds_on_an_npu(shape):
    """Route, load the object, and agree with torch: the whole path on hardware.

    Tensors go on the ``npu`` device this mark says the machine has, rather than on
    ``run_device()``: this case needs an NPU whatever the run was pointed at, which is the
    same carve-out ``cuda_only`` makes for CUDA.

    The row count is the axis the matrix does not fix, so it is the axis worth varying:
    one block of rows and 256 of them go through the same object, because the tile is
    baked in and the block index is the runtime's. A second inner extent is a second
    object, which is why it is a separate case rather than a fourth size.
    """
    pytest.importorskip("torch_npu")  # claims the device type named below

    npu = torch.device("npu")
    a = torch.randn(shape, dtype=torch.float32, device=npu)
    b = torch.randn(shape, dtype=torch.float32, device=npu)

    out = AddFwdOp(target="ascend")(a, b)

    assert out.device == npu
    torch.testing.assert_close(out, a + b)
