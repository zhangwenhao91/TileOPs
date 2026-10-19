"""The builder of ``MHADecodePagedWithKVCacheFwdOp`` on the ``ascend`` target.

The returned callable runs one call end to end on the NPU: it sanitizes the page pools,
builds the additive attention bias the kernel consumes, launches the TileLang Ascend
kernel and reshapes its output. Compiling the kernel is the builder's job, so the op
layer's per-signature cache keeps it to one compile per shape.

Two responsibilities live here rather than in the kernel because the kernel walks whole
128-row tiles:

* **Pools whose unread rows are safe.** Pool rows no request reads may hold NaN; a tile
  the cache length cuts would otherwise poison its scores and its ``P @ V`` through
  masked-but-nonzero lanes. When the read mask proves such rows exist, the builder
  zeroes every unread row of a clone; a pool every request reads in full is passed to
  the kernel as is, skipping the pool-sized copy. Both the mask and the full-read
  verdict only vary with the cache lengths and the page table, so calls sharing them
  reuse the built mask — the full-read path then runs no mask op at all.
* **The bias.** The kernel adds a float32 bias per query row and key tile instead of
  masking after the scale: a causal query ``i`` sees keys up to ``i + real_seqlen_kv -
  seqlen_q``, and a query that sees no key must output zeros, which the caller-side
  zero of its row's finished output provides. The bias only varies with the cache
  lengths, so calls sharing ``real_seqlen_kv`` reuse the built one.
"""

from __future__ import annotations

import torch

from tileops.backend import OpNotAvailableError, TensorSpec

# Query rows the kernel stages per block, and key rows per tile. Baked into the
# compiled program; a call outside them has no kernel.
BR = 16
BC = 128

_DTYPE_TO_STR = {torch.float16: "float16", torch.bfloat16: "bfloat16"}


def _refuse(why: str) -> OpNotAvailableError:
    """The error for a call this target has no kernel for."""
    served = (
        f"float16 Q and KV, seqlen_q <= {BR}, dim a multiple of 64, page_size "
        f"{BC} or a multiple of it"
    )
    return OpNotAvailableError(
        f"the ascend target serves MHADecodePagedWithKVCacheFwdOp for {served}; "
        f"this call asks for {why}. No in-tree kernel runs on an NPU to fall back to."
    )


def build_mha_decode_paged(
    q: TensorSpec,
    k: TensorSpec,
    v: TensorSpec,
    real_seqlen_kv: TensorSpec,
    block_table: TensorSpec,
    *,
    page_size: int,
    is_causal: bool | None = False,
):
    """Return the kernel serving one paged MHA decode call shape on the NPU."""
    batch, seqlen_q, heads, dim = q.shape
    pool_rows = k.shape[0]
    max_pages_per_req = block_table.shape[1]

    if q.dtype not in _DTYPE_TO_STR:
        raise _refuse(f"Q dtype {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise _refuse(f"one element type across Q and KV (got {q.dtype}, {k.dtype}, {v.dtype})")
    if seqlen_q < 1 or seqlen_q > BR:
        raise _refuse(f"1 <= seqlen_q <= {BR} (got {seqlen_q})")
    if dim % 64:
        raise _refuse(f"dim a multiple of 64 (got {dim})")
    if page_size % BC:
        raise _refuse(f"page_size a multiple of {BC} (got {page_size})")
    if pool_rows != v.shape[0]:
        raise _refuse(f"K and V pools of one height (got {pool_rows} and {v.shape[0]})")

    causal = bool(is_causal)
    dtype_str = _DTYPE_TO_STR[q.dtype]
    from tileops.backends.ascend.kernels.mha_decode_paged import compile_kernel

    kernel = compile_kernel(
        batch, seqlen_q, heads, dim, page_size, max_pages_per_req, pool_rows, dtype_str
    )

    # Bias 只随 real_seqlen_kv 变化；同缓长的重复调用（解码逐步推进前的连续请求、
    # 基准循环）直接复用已构造的 fp32 张量。按 (rl 元组, 设备) 作键，超出上限整表
    # 清空防无限增长。
    bias_cache: dict[tuple, torch.Tensor] = {}
    # read mask 与全读判定只随 (real_seqlen_kv, block_table) 变化：命中时连 mask
    # 的约 6 个 device 算子都省掉，全读路径只剩 kernel 启动。条目为
    # (read_mask, full_read)，全读时 mask 记 None；mask 张量构造后只读，可跨调用复用。
    mask_cache: dict[tuple, tuple] = {}

    def run(q_t, k_t, v_t, rl_t, bt_t):
        """Run one decode call on the NPU, returning a fresh ``(B, S_q, H, D)`` tensor."""
        dev = q_t.device
        rl_cpu = rl_t.to("cpu", torch.int32)
        bt_cpu = bt_t.to("cpu", torch.int32)
        mask_key = (tuple(rl_cpu.tolist()), tuple(bt_cpu.reshape(-1).tolist()), str(dev))
        cached = mask_cache.get(mask_key)
        if cached is not None:
            read, full_read = cached
        else:
            if len(mask_cache) >= 8:
                mask_cache.clear()
            # Pool rows no request reads become zeros, so a tile the cache length cuts
            # contributes an exact zero rather than propagating whatever the pool held.
            # The pool-sized clone is the expensive part: run it only when the mask
            # proves at least one row is unread, and hand the untouched pools over
            # when every request reads them in full.
            num_pages = pool_rows // page_size
            pos = torch.arange(max_pages_per_req * page_size, device=dev)
            rows = bt_t[:, pos // page_size] * page_size + pos % page_size
            read = torch.zeros(pool_rows, dtype=torch.bool, device=dev)
            read[rows[pos[None, :] < rl_t[:, None]]] = True
            full_read = not bool((~read).any())
            read = None if full_read else read
            mask_cache[mask_key] = (read, full_read)
        if full_read:
            k_safe, v_safe = k_t, v_t
        else:
            unread = ~read
            k_safe = k_t.clone()
            v_safe = v_t.clone()
            k_safe[unread] = 0
            v_safe[unread] = 0

        # Additive bias per query row and key tile: 0 where the key is visible,
        # -1e30 where the causal end or the cache length cuts it.
        num_kv = max_pages_per_req * (page_size // BC)
        key = (tuple(rl_cpu.tolist()), str(dev))
        bias = bias_cache.get(key)
        if bias is None:
            if len(bias_cache) >= 8:
                bias_cache.clear()
            kv_pos = torch.arange(num_kv * BC, device=dev, dtype=torch.float32).view(1, 1, -1)
            row_idx = torch.arange(BR, device=dev, dtype=torch.float32).view(1, BR, 1)
            rl_f = rl_t.to(torch.float32).view(batch, 1, 1)
            if causal:
                last_key = row_idx + rl_f - seqlen_q
                visible = kv_pos <= last_key
            else:
                visible = (kv_pos < rl_f).expand(batch, BR, num_kv * BC)
            bias = torch.where(visible, 0.0, -1.0e30).view(batch * BR, num_kv * BC)
            bias_cache[key] = bias

        # The kernel's block store fills every o_buf row, so no zero fill here.
        o_buf = torch.empty(
            batch, heads, BR, dim, dtype=q_t.dtype, device=dev
        )
        kernel(q_t, bt_t, rl_t, bias, k_safe, v_safe, o_buf)

        out = o_buf[:, :, :seqlen_q, :].permute(0, 2, 1, 3).contiguous()
        if causal:
            # A query that sees no key outputs zeros; its softmax denominator is 0.
            q_rows = torch.arange(seqlen_q, device=dev)
            no_key = (q_rows[None, :] + rl_t[:, None] - seqlen_q) < 0
            out[no_key] = 0
        return out

    return run
