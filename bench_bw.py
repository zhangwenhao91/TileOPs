"""d806 HBM 带宽探针：校准 bench_mha.py 的峰值带宽（替换占位值）。

口径：
- copy    ：整块 clone（读+写对称，纯 DMA 路径）；
- mul     ：in-place mul_（读+写对称，向量单元路径，与 copy 互为下界校验）；
- strided ：out.copy_(x3[:, h, :])——按 kernel 真实访问模式跨步读（256B 连续段、
            行间 8KB 步长）+ 连续写。注意只读 1/32 的张量，字节按实际量计。

末行输出 PEAK_GBPS=<值>（copy/mul 取大者），供 BENCH_PEAK_BW_GBPS 使用。
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

DEV = torch.device("npu", 0)
REPEAT = 20


def timed(fn):
    fn()  # 预热
    torch.npu.synchronize()
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    n = 1024 * 1024 * 1024  # 2 GiB fp16
    x = torch.randn(n, dtype=torch.float16, device=DEV)
    y = torch.empty_like(x)
    # kernel 的 K/V 页读模式：32 头、D=128，取 h=4 一列 = 256B 段 + 8KB 行步长
    x3 = x.view(-1, 32, 128)
    h = 4
    out = torch.empty(x3.shape[0], 128, dtype=torch.float16, device=DEV)

    t_copy = timed(lambda: y.copy_(x))
    t_mul = timed(lambda: x.mul_(1.0))
    t_strided = timed(lambda: out.copy_(x3[:, h, :]))

    gb_total = n * 2 / 1e9           # 2 GiB 张量的字节数
    gb_slice = n * 2 / 32 / 1e9      # strided 口径只触达 1/32
    copy_gbps = 2 * gb_total / t_copy
    mul_gbps = 2 * gb_total / t_mul
    strided_gbps = 2 * gb_slice / t_strided
    print(f"copy   : {copy_gbps:7.1f} GB/s  (2x{gb_total:.1f}GB 读+写, {t_copy*1e3:.2f}ms)")
    print(f"mul    : {mul_gbps:7.1f} GB/s  (2x{gb_total:.1f}GB 读+写, {t_mul*1e3:.2f}ms)")
    print(f"strided: {strided_gbps:7.1f} GB/s  (2x{gb_slice:.3f}GB 跨步读+写, {t_strided*1e3:.2f}ms)")
    print(f"PEAK_GBPS={max(copy_gbps, mul_gbps):.0f}")


if __name__ == "__main__":
    main()
