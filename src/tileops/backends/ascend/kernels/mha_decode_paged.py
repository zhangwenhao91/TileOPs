"""The TileLang DSL of the paged MHA decode kernel for ``MHADecodePagedWithKVCacheFwdOp``.

The program is compiled per ``(batch, seqlen_q, heads, dim, page_size, max_pages_per_req,
pool_rows, dtype)`` specialization on the NPU through TileLang's Ascend backend. One
block serves one ``(batch, head)`` pair: it scores every 128-row key tile of the
request's cache with an online softmax and accumulates the output in float32.

A key tile never straddles a page: pages of 256 rows walk as two ``BC=128`` sub-blocks
whose page id ``block_table[b, kv // SP]`` and in-page offset ``kv % SP * BC`` the block
computes at run time. Rows the cache length cuts, and pool rows no request reads, are the
builder's business: it hands the kernel a pool whose unread rows hold zeros, so a masked
tile contributes an exact zero instead of a poisoned NaN.

TileLang is imported inside :func:`compile_kernel` so that reading or packaging this
module does not require it.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=64)
def compile_kernel(
    batch: int,
    seqlen_q: int,
    heads: int,
    dim: int,
    page_size: int,
    max_pages_per_req: int,
    pool_rows: int,
    dtype: str = "float16",
):
    """Compile the decode kernel for one shape specialization and return the JIT object."""
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.layout import make_ascend_compact_nz_layout
    from tilelang.ascend.language import simd as S

    B = batch
    S_Q = seqlen_q
    H = heads
    D = dim
    PAGE = page_size
    W = max_pages_per_req
    POOL_ROWS = pool_rows
    BC = 128  # key-tile rows; a page walks as SP sub-blocks of this height
    SP = PAGE // BC
    NUM_KV = W * SP
    BR = 16  # query rows a block stages; the first S_Q are real queries
    ROWS = BR // 2  # rows per AIV of the dual-lane vector ops
    VL = 64  # lanes one PAT_ALL vector operation reads or writes
    out_dtype = dtype
    accum_dtype = "float32"
    # NZ 打包的转换目标与 vsstb 重解释类型：两者都是 16 位，位宽布局一致，
    # 仅 vcvt 的目标名和最后一步 reinterpret 的向量类型名不同（example core.py 同款）。
    pack_dtype = "bfloat16" if dtype == "bfloat16" else "float16"
    pack_vec = pack_dtype + "x128"
    scale = dim**-0.5

    assert page_size % BC == 0, f"page_size must be a multiple of {BC}, got {page_size}"
    assert seqlen_q <= BR, f"seqlen_q must be at most {BR}, got {seqlen_q}"
    assert dim % VL == 0, f"dim must be a multiple of {VL}, got {dim}"

    @T.macro
    def qk(base, h, KP, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub):
        T.copy(KP[base:base + BC, h, 0:D], K_shared[0:BC, 0:D])
        T.copy(Q_shared[0:BR, 0:D], qk_a_l0)
        T.copy(K_shared[0:BC, 0:D], qk_b_l0)
        T.gemm(qk_a_l0, qk_b_l0, qk_acc_l0c, transpose_B=True, clear_accum=True)
        T.dual_copy(qk_acc_l0c, S_ub[0:ROWS, 0:BC])

    @T.macro
    def softmax(S_ub, P_nz_ub, bias_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        with T.SimdVF(latency=735):
            softmax_full = S.pset(32, "PAT_ALL")
            softmax_one = S.pset(32, "PAT_VL1")
            softmax_b16 = S.pset(16, "PAT_VL128")
            dst_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC),
                pack_dtype,
            )

            S.vsts(old_m_ub[0], S.vld(m_ub[0], "NORM"), softmax_full, "NORM_B32")

            for r in range(0, ROWS, 2):
                b0lo = S.vld(bias_ub[r, 0], "NORM")
                b0hi = S.vld(bias_ub[r, VL], "NORM")
                b1lo = S.vld(bias_ub[r + 1, 0], "NORM")
                b1hi = S.vld(bias_ub[r + 1, VL], "NORM")
                s00 = S.vadd(S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), softmax_full),
                             b0lo, softmax_full)
                s01 = S.vadd(S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), softmax_full),
                             b0hi, softmax_full)
                s10 = S.vadd(S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), softmax_full),
                             b1lo, softmax_full)
                s11 = S.vadd(S.vmuls(S.vld(S_ub[r + 1, VL], "NORM"), T.float32(scale), softmax_full),
                             b1hi, softmax_full)
                S.vsts(S_ub[r, 0], s00, softmax_full, "NORM_B32")
                S.vsts(S_ub[r, VL], s01, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, 0], s10, softmax_full, "NORM_B32")
                S.vsts(S_ub[r + 1, VL], s11, softmax_full, "NORM_B32")

                max0 = S.vcmax(S.vmax(s00, s01, softmax_full), softmax_full)
                max1 = S.vcmax(S.vmax(s10, s11, softmax_full), softmax_full)
                S.vsts(m_ub[r], max0, softmax_one, "ONEPT_B32")
                S.vsts(m_ub[r + 1], max1, softmax_one, "ONEPT_B32")

            S.mem_bar("VST_VLD")
            merged_max = S.vmax(S.vld(m_ub[0], "NORM"), S.vld(old_m_ub[0], "NORM"), softmax_full)
            S.vsts(m_ub[0], merged_max, softmax_full, "NORM_B32")
            S.mem_bar("VST_VLD")

            for r in range(0, ROWS, 2):
                max0 = S.vld(m_ub[r], "BRC_B32")
                max1 = S.vld(m_ub[r + 1], "BRC_B32")
                p00, p01 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")
                e00 = S.vexpdif(p00, max0, softmax_full)
                e10 = S.vexpdif(p10, max1, softmax_full)
                e01 = S.vexpdif(p01, max0, softmax_full)
                e11 = S.vexpdif(p11, max1, softmax_full)
                row_sum0 = S.vcadd(S.vadd(e00, e01, softmax_full), softmax_full)
                row_sum1 = S.vcadd(S.vadd(e10, e11, softmax_full), softmax_full)
                S.vsts(row_sum_ub[r], row_sum0, softmax_one, "ONEPT_B32")
                S.vsts(row_sum_ub[r + 1], row_sum1, softmax_one, "ONEPT_B32")

                even_pk_0 = S.vcvt(e00, pack_dtype, softmax_full, sat=False, part=0)
                even_pk_1 = S.vcvt(e10, pack_dtype, softmax_full, sat=False, part=0)
                odd_pk_0 = S.vcvt(e01, pack_dtype, softmax_full, sat=False, part=1)
                odd_pk_1 = S.vcvt(e11, pack_dtype, softmax_full, sat=False, part=1)
                merged_bits0 = S.vor(
                    T.reinterpret(even_pk_0, "uint16x128"),
                    T.reinterpret(odd_pk_0, "uint16x128"),
                    softmax_b16,
                )
                merged_bits1 = S.vor(
                    T.reinterpret(even_pk_1, "uint16x128"),
                    T.reinterpret(odd_pk_1, "uint16x128"),
                    softmax_b16,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits0, pack_vec),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits1, pack_vec),
                    dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1),
                    softmax_b16,
                    update=True,
                )

            S.mem_bar("VST_VLD")
            alpha_reg = S.vexpdif(S.vld(old_m_ub[0], "NORM"), S.vld(m_ub[0], "NORM"), softmax_full)
            S.vsts(alpha_ub[0], alpha_reg, softmax_full, "NORM_B32")
            l_scaled = S.vmul(S.vld(l_ub[0], "NORM"), alpha_reg, softmax_full)
            l_updated = S.vadd(l_scaled, S.vld(row_sum_ub[0], "NORM"), softmax_full)
            S.vsts(l_ub[0], l_updated, softmax_full, "NORM_B32")

    @T.macro
    def pack_p(P_nz_ub, P_shared):
        T.dual_copy(P_nz_ub[0:ROWS, 0:BC], P_shared[0:BR, 0:BC])

    @T.macro
    def pv(base, h, VP, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub):
        T.copy(VP[base:base + BC, h, 0:D], V_shared[0:D, 0:BC], transpose=True)
        T.copy(P_shared[0:BR, 0:BC], pv_a_l0)
        T.copy(V_shared[0:D, 0:BC], pv_b_l0)
        T.gemm(pv_a_l0, pv_b_l0, pv_acc_l0c, transpose_B=True, clear_accum=True)
        T.dual_copy(pv_acc_l0c, O_tmp_ub[0:ROWS, 0:D])

    @T.macro
    def accumulate_output(O_tmp_ub, O_ub, alpha_ub):
        with T.SimdVF(latency=318):
            rescale_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    prev = S.vld(O_ub[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, rescale_mask), cur, rescale_mask)
                    S.vsts(O_ub[i, c], updated, rescale_mask, "NORM_B32")

    @T.macro
    def init_m_l(m_ub, l_ub):
        with T.SimdVF(latency=400):
            init_mask = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), accum_dtype, init_mask)
            initial_m = S.vdup(-T.infinity(accum_dtype), accum_dtype, init_mask)
            S.vsts(l_ub[0], zero, init_mask, "NORM_B32")
            S.vsts(m_ub[0], initial_m, init_mask, "NORM_B32")

    @T.macro
    def clear_O_ub(O_ub):
        with T.SimdVF(latency=142):
            clear_mask = S.pset(32, "PAT_ALL")
            output_zero = S.vdup(T.float32(0.0), accum_dtype, clear_mask)
            for i in range(ROWS):
                for c in range(0, D, VL):
                    S.vsts(O_ub[i, c], output_zero, clear_mask, "NORM_B32")

    @T.macro
    def kv_body(b, kv, base, h, KP, VP, Q_shared, K_shared, V_shared, P_shared,
                qk_a_l0, qk_b_l0, pv_a_l0, pv_b_l0, qk_acc_l0c, pv_acc_l0c,
                S_ub, P_nz_ub, bias_ub, O_tmp_ub, O_ub, m_ub, l_ub, alpha_ub,
                old_m_ub, row_sum_ub, BIAS):
        T.copy(BIAS[b * BR:(b + 1) * BR, kv * BC:(kv + 1) * BC], bias_ub[0:BR, 0:BC])
        qk(base, h, KP, Q_shared, K_shared, qk_a_l0, qk_b_l0, qk_acc_l0c, S_ub)
        softmax(S_ub, P_nz_ub, bias_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub)
        pack_p(P_nz_ub, P_shared)
        pv(base, h, VP, P_shared, V_shared, pv_a_l0, pv_b_l0, pv_acc_l0c, O_tmp_ub)
        accumulate_output(O_tmp_ub, O_ub, alpha_ub)

    @T.macro
    def finalize_cast(O, b, h, O_ub, O_cast_ub, l_ub):
        with T.SimdVF(latency=1780):
            normalize_mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                denominator = S.vld(l_ub[i], "BRC_B32")
                for c in range(0, D, VL):
                    normalized = S.vdiv(S.vld(O_ub[i, c], "NORM"), denominator, normalize_mask)
                    S.vsts(O_ub[i, c], normalized, normalize_mask, "NORM_B32")
        with T.SimdVF(latency=193):
            cast_full = S.pset(32, "PAT_ALL")
            for r in range(ROWS):
                for c in range(0, D, VL):
                    ob = S.vcvt(S.vld(O_ub[r, c], "NORM"), out_dtype, cast_full)
                    S.vsts(O_cast_ub[r, c], ob, cast_full, "PK_B32")
        T.dual_copy(O_cast_ub[0:ROWS, 0:D], O[b, h, 0:BR, 0:D])

    @T.prim_func
    def mha_decode_paged(
        Q: T.Buffer((B, S_Q, H, D), dtype),
        BT: T.Buffer((B, W), "int32"),
        RL: T.Buffer((B,), "int32"),
        BIAS: T.Buffer((B * BR, NUM_KV * BC), accum_dtype),
        KP: T.Buffer((POOL_ROWS, H, D), dtype),
        VP: T.Buffer((POOL_ROWS, H, D), dtype),
        O: T.Buffer((B, H, BR, D), out_dtype),
    ):
        with T.Kernel(B * H) as bx:
            b: T.int32 = bx // H
            h: T.int32 = bx % H
            # W and POOL_ROWS appear only in the shape annotations above; the
            # annotation evaluator resolves names through the function's closure,
            # so pin them with these (otherwise dead) TIR scalars.
            page_width: T.int32 = W
            pool_rows: T.int32 = POOL_ROWS
            Q_shared = T.alloc_l1((BR, D), dtype)
            K_shared = T.alloc_l1((BC, D), dtype)
            V_shared = T.alloc_l1((D, BC), dtype)
            P_shared = T.alloc_l1((BR, BC), dtype)

            qk_a_l0 = T.alloc_l0a((BR, D), dtype)
            qk_b_l0 = T.alloc_l0b((BC, D), dtype)
            pv_a_l0 = T.alloc_l0a((BR, BC), dtype)
            pv_b_l0 = T.alloc_l0b((D, BC), dtype)
            qk_acc_l0c = T.alloc_l0c((BR, BC), accum_dtype)
            pv_acc_l0c = T.alloc_l0c((BR, D), accum_dtype)

            S_ub = T.alloc_shared((ROWS, BC), accum_dtype)
            P_nz_ub = T.alloc_shared((ROWS + 1, BC), dtype)
            T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
            bias_ub = T.alloc_shared((BR, BC), accum_dtype)
            O_tmp_ub = T.alloc_shared((ROWS, D), accum_dtype)
            O_ub = T.alloc_shared((ROWS, D), accum_dtype)
            m_ub = T.alloc_shared((VL,), accum_dtype)
            l_ub = T.alloc_shared((VL,), accum_dtype)
            alpha_ub = T.alloc_shared((VL,), accum_dtype)
            old_m_ub = T.alloc_shared((VL,), accum_dtype)
            row_sum_ub = T.alloc_shared((VL,), accum_dtype)
            T.annotate_buffer_versions(
                {
                    K_shared: 3,
                    V_shared: 3,
                    qk_acc_l0c: 2,
                    pv_acc_l0c: 2,
                    S_ub: 3,
                    P_nz_ub: 2,
                    alpha_ub: 3,
                    O_tmp_ub: 2,
                }
            )

            T.copy(Q[b, 0:S_Q, h, 0:D], Q_shared[0:S_Q, 0:D])
            init_m_l(m_ub, l_ub)
            clear_O_ub(O_ub)

            rl = RL[b]
            for kv in T.serial(NUM_KV):
                with T.If(kv * BC < rl):
                    with T.Then():
                        pg = kv // SP
                        off = (kv % SP) * BC
                        base = BT[b, pg] * PAGE + off
                        kv_body(b, kv, base, h, KP, VP, Q_shared, K_shared, V_shared, P_shared,
                                qk_a_l0, qk_b_l0, pv_a_l0, pv_b_l0, qk_acc_l0c, pv_acc_l0c,
                                S_ub, P_nz_ub, bias_ub, O_tmp_ub, O_ub, m_ub, l_ub, alpha_ub,
                                old_m_ub, row_sum_ub, BIAS)

            O_cast_ub = T.alloc_shared((ROWS, D), out_dtype)
            finalize_cast(O, b, h, O_ub, O_cast_ub, l_ub)

    return tilelang.compile(mha_decode_paged)
