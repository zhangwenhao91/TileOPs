"""The TileLang DSL of the MLA decode kernel for ``MLADecodeWithKVCacheFwdOp``.

The program is compiled per ``(batch, heads, dim, pe_dim, seqlen_kv, dtype)``
specialization on the NPU through TileLang's Ascend backend. Weights are absorbed:
``V`` is the nope half of ``K`` itself, so one block scores the request's whole cache
with an online softmax and streams ``P @ K`` straight out — no separate value pool.

One block serves ``GQ = min(128, heads)`` query heads of one request. Each key tile is
loaded once and reused by every 16-row head tile of the block, which is what keeps the
arithmetic intensity near the machine balance point. Head tiles keep one online-softmax
state slot and one float32 output slot each (the multi-slot layout probe 3 validated);
a final pass normalizes by the accumulated denominator and casts to the store dtype.

``D`` never enters L0B in full: both gemms walk it in 128-wide chunks, and the PE tail
appends into the same accumulator. The QK gemm's B operand uses the transposed L0B
layout and PV the forward one — mutually exclusive forms that cannot share a buffer,
so QK halves its key tile (two 64-row passes over one 16 KB buffer) while PV keeps a
32 KB one (56 KB of the 64 KB L0B budget in total).

TileLang is imported inside :func:`compile_kernel` so that reading or packaging this
module does not require it.
"""

from __future__ import annotations

import functools

BR = 16  # query rows one head tile stages
BC = 128  # key rows one tile scores
VL = 64  # lanes one PAT_ALL vector operation reads or writes
ROWS = BR // 2  # rows per AIV of the dual-lane vector ops


@functools.lru_cache(maxsize=64)
def compile_kernel(
    batch: int,
    heads: int,
    dim: int,
    pe_dim: int,
    seqlen_kv: int,
    dtype: str = "float16",
):
    """Compile the decode kernel for one shape specialization and return the JIT object."""
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S
    from tilelang.layout import make_ascend_compact_nz_layout

    B = batch
    H = heads
    D_ = dim
    PE_ = pe_dim
    SKV = seqlen_kv
    GQ = min(128, H)     # query heads one block eats
    NT = GQ // BR        # head tiles = state slots
    NBLK = H // GQ       # head blocks per request
    NUM_KV = SKV // BC
    accum_dtype = "float32"
    out_dtype = dtype
    # NZ 打包的转换目标与 vsstb 重解释类型：两者都是 16 位，位宽布局一致，
    # 仅 vcvt 的目标名和最后一步 reinterpret 的向量类型名随 dtype 换（MHA 同款）。
    pack_dtype = dtype
    pack_vec = dtype + "x128"
    scale = (dim + pe_dim) ** -0.5

    assert H % GQ == 0 and GQ % BR == 0, f"heads={H} must divide into {BR}-row tiles"
    assert seqlen_kv % BC == 0, f"seqlen_kv must be a multiple of {BC}, got {seqlen_kv}"
    assert dim % 128 == 0, f"dim must be a multiple of 128, got {dim}"

    @T.macro
    def qk_tile(Q_shared, QPE_shared, K_shared, KPE_shared,
                qk_a, qk_b, qk_a_pe, qk_b_pe, qk_acc, S_ub):
        # BC=128 拆两个 64 行半块复用同一 qk_b（16KB）：qk 用转置布局、pv 用正向布局，
        # 两者不能共用 L0B（layout_inference 报 Get different layout），只能各自独立缓冲
        T.copy(QPE_shared[0:BR, 0:PE_], qk_a_pe)
        for kh in range(2):
            for d0 in range(0, D_, 128):
                T.copy(Q_shared[0:BR, d0:d0 + 128], qk_a)
                T.copy(K_shared[kh * 64:(kh + 1) * 64, d0:d0 + 128], qk_b)
                T.gemm(qk_a, qk_b, qk_acc, transpose_B=True, clear_accum=(d0 == 0))
            T.copy(KPE_shared[kh * 64:(kh + 1) * 64, 0:PE_], qk_b_pe)
            T.gemm(qk_a_pe, qk_b_pe, qk_acc, transpose_B=True, clear_accum=False)
            T.dual_copy(qk_acc, S_ub[0:ROWS, kh * 64:(kh + 1) * 64])

    @T.macro
    def softmax_tile(S_ub, P_nz_ub, m_ub, l_ub, alpha_ub, old_m_ub, row_sum_ub):
        """与 MHA 单块 softmax 逐条同构（去掉 bias vadd），全部从索引 0 寻址。"""
        with T.SimdVF(latency=735):
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            b16 = S.pset(16, "PAT_VL128")
            dst_nz_ptr = S.make_ubuf_ptr(
                T.access_ptr(P_nz_ub[0, 0], "w", ROWS + 1, BC),
                pack_dtype,
            )
            S.vsts(old_m_ub[0], S.vld(m_ub[0], "NORM"), full, "NORM_B32")

            for r in range(0, ROWS, 2):
                s00 = S.vmuls(S.vld(S_ub[r, 0], "NORM"), T.float32(scale), full)
                s01 = S.vmuls(S.vld(S_ub[r, VL], "NORM"), T.float32(scale), full)
                s10 = S.vmuls(S.vld(S_ub[r + 1, 0], "NORM"), T.float32(scale), full)
                s11 = S.vmuls(S.vld(S_ub[r + 1, VL], "NORM"), T.float32(scale), full)
                S.vsts(S_ub[r, 0], s00, full, "NORM_B32")
                S.vsts(S_ub[r, VL], s01, full, "NORM_B32")
                S.vsts(S_ub[r + 1, 0], s10, full, "NORM_B32")
                S.vsts(S_ub[r + 1, VL], s11, full, "NORM_B32")

                max0 = S.vcmax(S.vmax(s00, s01, full), full)
                max1 = S.vcmax(S.vmax(s10, s11, full), full)
                S.vsts(m_ub[r], max0, one, "ONEPT_B32")
                S.vsts(m_ub[r + 1], max1, one, "ONEPT_B32")

            S.mem_bar("VST_VLD")
            merged = S.vmax(S.vld(m_ub[0], "NORM"), S.vld(old_m_ub[0], "NORM"), full)
            S.vsts(m_ub[0], merged, full, "NORM_B32")
            S.mem_bar("VST_VLD")

            for r in range(0, ROWS, 2):
                max0 = S.vld(m_ub[r], "BRC_B32")
                max1 = S.vld(m_ub[r + 1], "BRC_B32")
                p00, p01 = S.vld2(S_ub[r, 0], "DINTLV_B32")
                p10, p11 = S.vld2(S_ub[r + 1, 0], "DINTLV_B32")
                e00 = S.vexpdif(p00, max0, full)
                e10 = S.vexpdif(p10, max1, full)
                e01 = S.vexpdif(p01, max0, full)
                e11 = S.vexpdif(p11, max1, full)
                row_sum0 = S.vcadd(S.vadd(e00, e01, full), full)
                row_sum1 = S.vcadd(S.vadd(e10, e11, full), full)
                S.vsts(row_sum_ub[r], row_sum0, one, "ONEPT_B32")
                S.vsts(row_sum_ub[r + 1], row_sum1, one, "ONEPT_B32")

                even_pk_0 = S.vcvt(e00, pack_dtype, full, sat=False, part=0)
                even_pk_1 = S.vcvt(e10, pack_dtype, full, sat=False, part=0)
                odd_pk_0 = S.vcvt(e01, pack_dtype, full, sat=False, part=1)
                odd_pk_1 = S.vcvt(e11, pack_dtype, full, sat=False, part=1)
                merged_bits0 = S.vor(
                    T.reinterpret(even_pk_0, "uint16x128"),
                    T.reinterpret(odd_pk_0, "uint16x128"),
                    b16,
                )
                merged_bits1 = S.vor(
                    T.reinterpret(even_pk_1, "uint16x128"),
                    T.reinterpret(odd_pk_1, "uint16x128"),
                    b16,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits0, pack_vec), dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1), b16, update=True,
                )
                dst_nz_ptr = S.vsstb(
                    T.reinterpret(merged_bits1, pack_vec), dst_nz_ptr,
                    T.int32(((ROWS + 1) << 16) | 1), b16, update=True,
                )

            S.mem_bar("VST_VLD")
            alpha_reg = S.vexpdif(S.vld(old_m_ub[0], "NORM"), S.vld(m_ub[0], "NORM"), full)
            S.vsts(alpha_ub[0], alpha_reg, full, "NORM_B32")
            l_updated = S.vadd(
                S.vmul(S.vld(l_ub[0], "NORM"), alpha_reg, full),
                S.vld(row_sum_ub[0], "NORM"),
                full,
            )
            S.vsts(l_ub[0], l_updated, full, "NORM_B32")

    @T.macro
    def pack_p(P_nz_ub, P_shared):
        T.dual_copy(P_nz_ub[0:ROWS, 0:BC], P_shared[0:BR, 0:BC])

    @T.macro
    def pv_tile(P_shared, K_shared, pv_a, pv_b, pv_acc, O_tmp_ub):
        T.copy(P_shared[0:BR, 0:BC], pv_a)
        for d2 in range(0, D_, 128):
            T.copy(K_shared[0:BC, d2:d2 + 128], pv_b)
            T.gemm(pv_a, pv_b, pv_acc, clear_accum=True)
            T.dual_copy(pv_acc, O_tmp_ub[0:ROWS, d2:d2 + 128])

    @T.macro
    def accumulate_tile(O_tmp_ub, O_slot, alpha_ub):
        with T.SimdVF(latency=318):
            mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                alpha = S.vld(alpha_ub[i], "BRC_B32")
                for c in range(0, D_, VL):
                    prev = S.vld(O_slot[i, c], "NORM")
                    cur = S.vld(O_tmp_ub[i, c], "NORM")
                    updated = S.vadd(S.vmul(alpha, prev, mask), cur, mask)
                    S.vsts(O_slot[i, c], updated, mask, "NORM_B32")

    @T.macro
    def init_m_l(m_ub, l_ub):
        with T.SimdVF(latency=400):
            init_mask = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), accum_dtype, init_mask)
            initial_m = S.vdup(-T.infinity(accum_dtype), accum_dtype, init_mask)
            S.vsts(l_ub[0], zero, init_mask, "NORM_B32")
            S.vsts(m_ub[0], initial_m, init_mask, "NORM_B32")

    @T.macro
    def finalize_tile(O, b_idx, hb, ht, O_slot, O_cast_ub, l_ub):
        with T.SimdVF(latency=1780):
            mask = S.pset(32, "PAT_ALL")
            for i in range(ROWS):
                den = S.vld(l_ub[i], "BRC_B32")
                for c in range(0, D_, VL):
                    normalized = S.vdiv(S.vld(O_slot[i, c], "NORM"), den, mask)
                    S.vsts(O_slot[i, c], normalized, mask, "NORM_B32")
        with T.SimdVF(latency=193):
            cast_mask = S.pset(32, "PAT_ALL")
            for r in range(ROWS):
                for c in range(0, D_, VL):
                    ob = S.vcvt(S.vld(O_slot[r, c], "NORM"), out_dtype, cast_mask)
                    S.vsts(O_cast_ub[r, c], ob, cast_mask, "PK_B32")
        # dual_copy dst 必须是干净 2D 切片（标量索引掉维）：
        # 4D O 的 [b_idx, hb, a:b, 0:D] 两个标量索引正好掉掉前两维
        T.dual_copy(O_cast_ub[0:ROWS, 0:D_],
                    O[b_idx, hb, ht * BR:(ht + 1) * BR, 0:D_])

    @T.prim_func
    def mla_decode(
        Q: T.Buffer((B, H, D_), dtype),
        QPE: T.Buffer((B, H, PE_), dtype),
        K: T.Buffer((B, SKV, 1, D_), dtype),
        KPE: T.Buffer((B, SKV, 1, PE_), dtype),
        O: T.Buffer((B, NBLK, GQ, D_), out_dtype),
    ):
        h_total: T.int32 = H  # 仅注解引用的形状名必须钉成闭包自由变量
        with T.Kernel(B * NBLK) as bx:
            b_idx: T.int32 = bx // NBLK
            hb: T.int32 = bx % NBLK
            head0: T.int32 = hb * GQ
            skv: T.int32 = SKV

            Q_shared = T.alloc_l1((BR, D_), dtype)
            QPE_shared = T.alloc_l1((BR, PE_), dtype)
            K_shared = T.alloc_l1((BC, D_), dtype)
            KPE_shared = T.alloc_l1((BC, PE_), dtype)
            P_shared = T.alloc_l1((BR, BC), dtype)

            qk_a = T.alloc_l0a((BR, 128), dtype)
            qk_a_pe = T.alloc_l0a((BR, PE_), dtype)
            pv_a = T.alloc_l0a((BR, BC), dtype)
            # L0B 64KB 预算：qk 半块化 16+8KB（转置布局）与 pv_b 32KB（正向布局）
            # 分离——qk/pv 布局形态互斥，共用于同一缓冲会报 Get different layout
            qk_b = T.alloc_l0b((64, 128), dtype)
            qk_b_pe = T.alloc_l0b((64, PE_), dtype)
            pv_b = T.alloc_l0b((BC, 128), dtype)
            qk_acc = T.alloc_l0c((BR, 64), accum_dtype)
            pv_acc = T.alloc_l0c((BR, 128), accum_dtype)

            S_ub = T.alloc_shared((ROWS, BC), accum_dtype)
            P_nz_ub = T.alloc_shared((ROWS + 1, BC), dtype)
            T.annotate_layout({P_nz_ub: make_ascend_compact_nz_layout(P_nz_ub)})
            O_tmp_ub = T.alloc_shared((ROWS, D_), accum_dtype)
            O_cast_ub = T.alloc_shared((ROWS, D_), out_dtype)

            # 每 tile 独立状态槽：Python 列表分配，宏调用按 ht 展开（trace 期解包）
            m_bufs = [T.alloc_shared((VL,), accum_dtype) for _ in range(NT)]
            l_bufs = [T.alloc_shared((VL,), accum_dtype) for _ in range(NT)]
            old_bufs = [T.alloc_shared((VL,), accum_dtype) for _ in range(NT)]
            alpha_bufs = [T.alloc_shared((VL,), accum_dtype) for _ in range(NT)]
            rs_bufs = [T.alloc_shared((VL,), accum_dtype) for _ in range(NT)]
            o_slots = [T.alloc_shared((ROWS, D_), accum_dtype) for _ in range(NT)]

            # 初始化：m=-inf、l=0、O=0（逐 tile 槽）
            for ht in list(range(NT)):
                init_m_l(m_bufs[ht], l_bufs[ht])
            with T.SimdVF(latency=142):
                clear_mask = S.pset(32, "PAT_ALL")
                output_zero = S.vdup(T.float32(0.0), accum_dtype, clear_mask)
                for ht in list(range(NT)):
                    for i in range(ROWS):
                        for c in range(0, D_, VL):
                            S.vsts(o_slots[ht][i, c], output_zero, clear_mask, "NORM_B32")

            for kv in T.serial(NUM_KV):
                kb0 = kv * BC
                # K 每 kv 迭代只加载一次，NT 个头 tile 依次复用（保住算术强度）
                T.copy(K[b_idx, kb0:kb0 + BC, 0, 0:D_], K_shared[0:BC, 0:D_])
                T.copy(KPE[b_idx, kb0:kb0 + BC, 0, 0:PE_], KPE_shared[0:BC, 0:PE_])
                for ht in list(range(NT)):
                    T.copy(Q[b_idx, head0 + ht * BR:head0 + (ht + 1) * BR, 0:D_],
                           Q_shared[0:BR, 0:D_])
                    T.copy(QPE[b_idx, head0 + ht * BR:head0 + (ht + 1) * BR, 0:PE_],
                           QPE_shared[0:BR, 0:PE_])
                    qk_tile(Q_shared, QPE_shared, K_shared, KPE_shared,
                            qk_a, qk_b, qk_a_pe, qk_b_pe, qk_acc, S_ub)
                    softmax_tile(S_ub, P_nz_ub, m_bufs[ht], l_bufs[ht], alpha_bufs[ht],
                                 old_bufs[ht], rs_bufs[ht])
                    pack_p(P_nz_ub, P_shared)
                    pv_tile(P_shared, K_shared, pv_a, pv_b, pv_acc, O_tmp_ub)
                    accumulate_tile(O_tmp_ub, o_slots[ht], alpha_bufs[ht])

            for fb in list(range(NT)):
                finalize_tile(O, b_idx, hb, fb, o_slots[fb], O_cast_ub, l_bufs[fb])

    return mla_decode
