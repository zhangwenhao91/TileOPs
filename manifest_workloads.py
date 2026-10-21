"""MHADecodePagedWithKVCacheFwdOp 官方 manifest workload 全量覆盖。

manifest/spec/attention.yaml 里该 op 的 7 个 workload 原样实例化：输入由官方
MHADecodePagedWorkload.gen_inputs 生成（乱序页表 make_fragmented_block_table），
kv_lens 覆写 real_seqlen_kv，输出与官方 ref_program + verification 比对，
另附中位耗时供参考。官方 benchmarks 框架计时走 CUDA events/CUPTI、conftest
硬拒 npu，故 NPU 侧以本脚本作为 workload 覆盖载体。

在 ~/TileOPs 目录下运行（依赖 repo 内 workloads 包与已装的 tileops）：
    python3 manifest_workloads.py
"""

import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "/home/z30086261/TileOPs")  # repo 内 workloads 包

from workloads.device import set_run_device  # noqa: E402

set_run_device(torch.device("npu", 0))

from workloads.attention.mha import MHADecodePagedWorkload  # noqa: E402
from workloads.numerics import compare_outputs  # noqa: E402
from tileops.ops import MHADecodePagedWithKVCacheFwdOp  # noqa: E402

# 与 manifest/spec/attention.yaml 的 7 条 workload 一一对应
# (label, B, H, S_q, N_kv, D, page_size, is_causal, kv_lens, dtype)
CASES = [
    ("serving-p128",       1, 16, 1,   512,   128, 128, False, [512],                        torch.float16),
    ("serving-b2",         2,  8, 1,  1024,    64, 256, False, [1024, 1024],                 torch.float16),
    ("serving-kv1k",       1,  8, 1,  1024,    64, 256, False, [1024],                       torch.float16),
    ("serving-kv512",      1,  8, 1,   512,    64, 256, False, [512],                        torch.float16),
    ("speculative-decode", 2,  8, 4,  1024,    64, 256, True,  [1000, 1000],                 torch.float16),
    ("opt-1.3b-b1",        1, 32, 1,  2048,    64, 128, False, [2048],                       torch.float16),
    ("opt-1.3b-b16",      16, 32, 1, 32768,    64, 128, False, [2048] * 16,                  torch.float16),
]

WARMUP = 3
REPEAT = 10


def _diff_report(got, expected, tag):
    """失败时的定位信息：逐 batch max_abs/max_rel + 全局最差元素坐标。"""
    g, e = got.float(), expected.float()
    absd = (g - e).abs()
    rel = absd / e.abs().clamp_min(1e-6)
    for b in range(g.shape[0]):
        print(f"    [{tag}] batch{b}: max_abs={absd[b].max().item():.3e} "
              f"max_rel={rel[b].max().item():.3e}")
    idx = absd.flatten().argmax().item()
    strides = [int(torch.tensor(g.shape[i:]).prod().item()) for i in range(g.ndim)]
    coord = []
    r = idx
    for i in range(g.ndim):
        strides[i] = strides[i] // g.shape[i]
        coord.append(r // strides[i])
        r %= strides[i]
    print(f"    [{tag}] worst elem at {coord}: got={g.flatten()[idx].item():.6f} "
          f"expected={e.flatten()[idx].item():.6f} absd={absd.flatten()[idx].item():.3e}")


def run_case(label, B, H, S_q, N_kv, D, page_size, is_causal, kv_lens, dtype, isolate=False):
    # 官方 workload：乱序页表 + 满长 real，随后按 manifest kv_lens 覆写 real_seqlen
    wl = MHADecodePagedWorkload(B, H, S_q, N_kv, D, page_size, is_causal, dtype)
    q, k, v, real, table = wl.gen_inputs()
    real = torch.tensor(kv_lens, dtype=torch.int32, device=real.device)
    table = table.contiguous()

    op = MHADecodePagedWithKVCacheFwdOp(page_size, target="ascend")

    torch.npu.synchronize()
    t0 = time.perf_counter()
    got = op(q, k, v, real, table)
    torch.npu.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3  # 含 JIT 编译

    for _ in range(WARMUP):
        op(q, k, v, real, table)
    torch.npu.synchronize()
    per_call = []
    for _ in range(REPEAT):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        op(q, k, v, real, table)
        torch.npu.synchronize()
        per_call.append((time.perf_counter() - t0) * 1e3)
    med_ms = statistics.median(per_call)

    expected = wl.ref_program(q, k, v, real, table)
    # 正规消费口：结构检查 + Custom validator（atol=1e-3 + cosine>0.99）
    try:
        compare_outputs(got, expected, wl.verification(q, k, v, real, table))
    except AssertionError:
        _diff_report(got, expected, label)
        if isolate and is_causal:
            # 归因对照：同输入关掉 causal 再验一次，区分"causal 路径问题"与"基础路径问题"
            wl0 = MHADecodePagedWorkload(B, H, S_q, N_kv, D, page_size, False, dtype)
            exp0 = wl0.ref_program(q, k, v, real, table)
            got0 = MHADecodePagedWithKVCacheFwdOp(page_size, target="ascend")(q, k, v, real, table)
            try:
                compare_outputs(got0, exp0, wl0.verification(q, k, v, real, table))
                print(f"    [{label}] non-causal 同输入 PASS -> 问题定位在 causal 掩码路径")
            except AssertionError:
                print(f"    [{label}] non-causal 同输入也 FAIL -> 基础路径（页表/累积）问题")
                _diff_report(got0, exp0, label + "-noncausal")
        raise
    print(f"  {label:<20} PASS  first={first_ms:7.1f}ms med={med_ms:.3f}ms")
    return {"label": label, "first_ms": round(first_ms, 1), "median_ms": round(med_ms, 3)}


def main():
    print("MHADecodePagedWithKVCacheFwdOp manifest workloads (official gen/ref/verification):")
    rows = []
    for spec in CASES:
        try:
            rows.append(run_case(*spec, isolate=True))
        except Exception as e:  # 单 case 失败不拖垮整表
            import traceback
            traceback.print_exc()
            print(f"  {spec[0]:<20} FAILED: {e}")
    ok = len(rows)
    print(f"manifest_workloads: {ok}/{len(CASES)} PASS")
    sys.exit(0 if ok == len(CASES) else 1)


if __name__ == "__main__":
    main()
