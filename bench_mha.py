"""MHA decode ascend backend 基线基准：固定形状重复调用取中位耗时，算有效带宽与 MBU。

在 ~/TileOPs 目录下运行（bench 依赖已 pip install -e 的 tileops）：
    python3 bench_mha.py

口径说明：
- 首调计时含 tilelang JIT 编译，单列一栏，不计入稳态统计；
- per-call 中位：每次调用前后各一次 torch.npu.synchronize()，取 20 次的中位数——
  含 builder 每次 call 的 host 开销（bias 构造 + NaN 池 sanitize 的池克隆 + 启动），
  这是当前实现的诚实基线；
- amortized 平均：批量调用首尾各一次 sync，反映流水稳态吞吐；
- 有效字节 = K/V 实读（按 real_seqlen 求和，K+V 各一份）+ Q 读 + O 写，fp16/bf16 均 2B；
  builder 构造的 BIAS fp32 掩码是实现的额外真实流量，不计入有效字节（只压低 MBU，
  正好暴露 padding 开销）；
- MBU = 有效带宽 / HBM 峰值。峰值用 BENCH_PEAK_BW_GBPS 环境变量覆盖（默认 3200）。

输出：人类可读表格 + 末尾一行 "BENCH_JSON: {...}" 便于机器解析。
"""

import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401

from tileops.ops import MHADecodePagedWithKVCacheFwdOp

DEV = torch.device("npu", 0)
PEAK_BW_GBPS = float(os.environ.get("BENCH_PEAK_BW_GBPS", "3200"))
NUM_STAGES = os.environ.get("TILEOPS_MHA_NUM_STAGES", "1")

WARMUP = 3          # 首调之后的预热次数
REPEAT = 20         # per-call 与 amortized 共用的重复次数
# (名称, B, H, kv_rows, page_size, real 填充率, dtype)
SHAPES = [
    ("s1_B1H8_kv2048_p128",      1,  8, 2048, 128, 1.0, torch.float16),
    ("s2_B8H8_kv2048_p128",      8,  8, 2048, 128, 1.0, torch.float16),
    ("s3_B16H8_kv1024_p128",    16,  8, 1024, 128, 1.0, torch.float16),
    ("s4_B8H32_kv4096_p256",     8, 32, 4096, 256, 1.0, torch.float16),
    ("s5_B8H8_kv2048_p128_half", 8,  8, 2048, 128, 0.5, torch.float16),
    ("s6_B4H8_kv4096_p256_half", 4,  8, 4096, 256, 0.5, torch.float16),
    ("s7_bf16_B8H8_kv2048_p128", 8,  8, 2048, 128, 1.0, torch.bfloat16),
    # 大池形状：K+V 池 256MB，远超 L2，排除缓存加成拿 HBM 诚实口径
    ("s8_B2H32_kv16384_p256",    2, 32, 16384, 256, 1.0, torch.float16),
]
D = 128  # head dim，全部形状统一


def make_inputs(B, H, kv_rows, page_size, fill, dtype):
    """按契约构造一组输入：顺序页表 + 按 fill 比例截断的 real_seqlen。"""
    W = (kv_rows + page_size - 1) // page_size
    pool_rows = W * page_size
    torch.manual_seed(0)
    q = torch.randn(B, 1, H, D, device=DEV, dtype=dtype) * 0.5
    kp = torch.randn(pool_rows, H, D, device=DEV, dtype=dtype) * 0.5
    vp = torch.randn(pool_rows, H, D, device=DEV, dtype=dtype) * 0.5
    full = W * page_size
    real = torch.full((B,), max(1, int(full * fill)), device=DEV, dtype=torch.int32)
    table = torch.arange(W, device=DEV, dtype=torch.int32).repeat(B, 1)
    return q, kp, vp, real, table


def bench_one(name, B, H, kv_rows, page_size, fill, dtype):
    q, kp, vp, real, table = make_inputs(B, H, kv_rows, page_size, fill, dtype)
    op = MHADecodePagedWithKVCacheFwdOp(page_size, target="ascend")

    torch.npu.synchronize()
    t0 = time.perf_counter()
    op(q, kp, vp, real, table)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3  # 含 JIT 编译

    for _ in range(WARMUP):
        op(q, kp, vp, real, table)
    torch.npu.synchronize()

    per_call = []
    for _ in range(REPEAT):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        op(q, kp, vp, real, table)
        torch.npu.synchronize()
        per_call.append((time.perf_counter() - t0) * 1e3)

    t0 = time.perf_counter()
    for _ in range(REPEAT):
        op(q, kp, vp, real, table)
    torch.npu.synchronize()
    amort_ms = (time.perf_counter() - t0) * 1e3 / REPEAT

    med_ms = statistics.median(per_call)
    # 有效字节：K/V 实读（K+V）+ Q 读 + O 写
    real_sum = int(real.sum().item())
    eff_bytes = real_sum * H * D * 2 * 2 + B * H * D * 2 * 2
    med_us = med_ms * 1e3
    amort_us = amort_ms * 1e3
    med_gbps = eff_bytes / (med_us * 1e-6) / 1e9
    amort_gbps = eff_bytes / (amort_us * 1e-6) / 1e9
    row = {
        "name": name,
        "dtype": str(dtype).replace("torch.", ""),
        "first_ms": round(first_ms, 1),
        "median_ms": round(med_ms, 3),
        "amortized_ms": round(amort_ms, 3),
        "eff_GB": round(eff_bytes / 1e9, 4),
        "median_Gbps": round(med_gbps, 1),
        "amortized_Gbps": round(amort_gbps, 1),
        "median_MBU": round(med_gbps / PEAK_BW_GBPS * 100, 2),
        "amortized_MBU": round(amort_gbps / PEAK_BW_GBPS * 100, 2),
    }
    print(
        f"{name:<28} {row['dtype']:<9} first={row['first_ms']:>8}ms "
        f"med={row['median_ms']:>8}ms amort={row['amortized_ms']:>8}ms "
        f"eff={row['eff_GB']:>7}GB med={row['median_Gbps']:>8}GB/s "
        f"({row['median_MBU']:>6}%) amort={row['amortized_Gbps']:>8}GB/s "
        f"({row['amortized_MBU']:>6}%)"
    )
    return row


def main():
    print(f"peak_bw={PEAK_BW_GBPS}GB/s warmup={WARMUP} repeat={REPEAT} "
          f"num_stages={NUM_STAGES} "
          f"per-call 口径含 builder host 开销（rl/bt D2H + launch + contiguous）")
    header = (f"{'shape':<28} {'dtype':<9} {'first(ms)':>10} {'median(ms)':>11} "
              f"{'amort(ms)':>10} {'eff(GB)':>9} {'med GB/s':>12} {'amort GB/s':>12} {'MBU%':>7}")
    print(header)
    rows = []
    for spec in SHAPES:
        try:
            rows.append(bench_one(*spec))
        except Exception as e:  # 单形状失败不拖垮整表
            import traceback
            traceback.print_exc()
            print(f"{spec[0]:<28} FAILED: {e}")
    print("BENCH_JSON: " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
