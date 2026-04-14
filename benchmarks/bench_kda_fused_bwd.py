#!/usr/bin/env python3
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
bench_kda_fused_bwd.py - Benchmark: cuLA fused KDA backward vs FLA Triton backward.

The benchmark builds one forward graph for each implementation, then times only
the backward pass with CUDA events and retain_graph=True.

Usage:
  python benchmarks/bench_kda_fused_bwd.py --mode fixed
  python benchmarks/bench_kda_fused_bwd.py --mode both --heads 16 --iters 20
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("FLA_USE_FAST_OPS", os.getenv("CULA_USE_FAST_MATH", "1"))

import torch
from fla.ops.kda import chunk_kda as fla_chunk_kda

from benchmarks.utils import (
    SEED,
    build_varlen_configs,
    exclusive_cumsum,
    prepare_safe_gate_inputs,
    set_seed,
)
from cula.utils import get_device_sm_version, get_kda_fused_fwd


_device = torch.device("cuda")
_major, _minor = get_device_sm_version(_device)
_SM_TAG = f"sm{_major}{_minor}"
cula_kda_fused = get_kda_fused_fwd(_device)

WARMUP = 10
N_ITERS = 30
NCU_MODE = False


def accuracy_stats(ref, out):
    ref_f = ref.float()
    out_f = out.float()
    diff = (ref_f - out_f).abs()
    rmse = diff.pow(2).mean().sqrt().item()
    max_diff = diff.max().item()
    denom = ref_f.abs().max().item()
    rel_max = max_diff / denom if denom > 0 else 0.0
    return rmse, rel_max, diff.mean().item()


def clone_inputs(inputs, requires_init_state):
    cloned = {}
    for name in ("q", "k", "v", "g", "beta", "A_log", "dt_bias"):
        cloned[name] = inputs[name].detach().clone().requires_grad_(True)
    cloned["init_state"] = (
        inputs["init_state"].detach().clone().requires_grad_(True) if requires_init_state and inputs["init_state"] is not None else None
    )
    return cloned


def run_fla(inp, scale, cu_seqlens, lower_bound):
    return fla_chunk_kda(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        scale=scale,
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        initial_state=inp["init_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        transpose_state_layout=True,
    )


def run_cula(inp, scale, cu_seqlens, lower_bound):
    return cula_kda_fused(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        g=inp["g"],
        beta=inp["beta"],
        scale=scale,
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        initial_state=inp["init_state"],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
    )


def zero_grads(inp):
    for tensor in inp.values():
        if tensor is not None:
            tensor.grad = None


def collect_grads(inp):
    names = ["q", "k", "v", "g", "beta", "A_log", "dt_bias"]
    if inp["init_state"] is not None:
        names.append("init_state")
    return {name: inp[name].grad.detach().clone() for name in names if inp[name].grad is not None}


def backward_once(outputs, grads, retain_graph=True):
    torch.autograd.backward(outputs, grads, retain_graph=retain_graph)


def make_backward_case(common, has_init_state):
    fla_inp = clone_inputs(common["inputs"], has_init_state)
    cula_inp = clone_inputs(common["inputs"], has_init_state)

    o_fla, ht_fla = run_fla(fla_inp, common["scale"], common["cu_seqlens"], common["lower_bound"])
    o_cula, ht_cula = run_cula(cula_inp, common["scale"], common["cu_seqlens"], common["lower_bound"])

    do = torch.randn_like(o_fla)
    dht = torch.randn_like(ht_fla)
    torch.cuda.synchronize()

    outputs_fla, outputs_cula = (o_fla, ht_fla), (o_cula, ht_cula)
    grads = (do, dht)
    return fla_inp, cula_inp, outputs_fla, outputs_cula, grads


def time_backward(inp, outputs, grads, warmup, iters):
    for _ in range(warmup):
        zero_grads(inp)
        backward_once(outputs, grads, retain_graph=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        zero_grads(inp)
        backward_once(outputs, grads, retain_graph=True)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def grad_accuracy(fla_inp, cula_inp, outputs_fla, outputs_cula, grads):
    zero_grads(fla_inp)
    zero_grads(cula_inp)
    backward_once(outputs_fla, grads, retain_graph=True)
    backward_once(outputs_cula, grads, retain_graph=True)
    torch.cuda.synchronize()

    fla_grads = collect_grads(fla_inp)
    cula_grads = collect_grads(cula_inp)
    stats = {}
    for name, ref in fla_grads.items():
        if name in cula_grads:
            stats[name] = accuracy_stats(ref, cula_grads[name])
    return stats


def make_common(B, T, H, D, cu_seqlens, has_init_state):
    inputs = prepare_safe_gate_inputs(B, T, H, D, _device, cu_seqlens=cu_seqlens, has_init_state=has_init_state)
    return {
        "inputs": inputs,
        "scale": inputs["scale"],
        "cu_seqlens": cu_seqlens,
        "lower_bound": inputs["lower_bound"],
    }


def summarize_grad_stats(stats):
    tracked = ["q", "k", "v", "g", "beta", "init_state"]
    rmse = max((stats[name][0] for name in tracked if name in stats), default=0.0)
    rel = max((stats[name][1] for name in tracked if name in stats), default=0.0)
    mean = max((stats[name][2] for name in tracked if name in stats), default=0.0)
    return rmse, rel, mean


def bench_fixed(configs, H, D, has_init_state, warmup, iters):
    print("\n" + "=" * 110)
    print(f" Fixed-Length Backward Benchmark: cuLA fused ({_SM_TAG}) vs FLA Triton")
    print("=" * 110)
    results = []
    for B, T in configs:
        set_seed(SEED)
        torch.cuda.empty_cache()
        cu_seqlens = torch.tensor(exclusive_cumsum([T] * B), dtype=torch.int32, device=_device)
        common = make_common(B, T, H, D, cu_seqlens, has_init_state)
        fla_inp, cula_inp, out_fla, out_cula, grads = make_backward_case(common, has_init_state)

        stats = grad_accuracy(fla_inp, cula_inp, out_fla, out_cula, grads)
        rmse, rel, mean = summarize_grad_stats(stats)
        ms_fla = time_backward(fla_inp, out_fla, grads, warmup, iters)
        ms_cula = time_backward(cula_inp, out_cula, grads, warmup, iters)
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")

        results.append({"B": B, "T": T, "rmse": rmse, "rel": rel, "mean": mean, "ms_fla": ms_fla, "ms_cula": ms_cula, "speedup": speedup})
        del fla_inp, cula_inp, out_fla, out_cula, grads, common
        torch.cuda.empty_cache()
    return results


def bench_varlen(configs, H, D, has_init_state, warmup, iters):
    print("\n" + "=" * 110)
    print(f" Varlen Backward Benchmark: cuLA fused ({_SM_TAG}) vs FLA Triton")
    print("=" * 110)
    results = []
    for seq_lens, total_len, dist in configs:
        set_seed(SEED)
        torch.cuda.empty_cache()
        cu_seqlens = torch.tensor(exclusive_cumsum(seq_lens), dtype=torch.int32, device=_device)
        common = make_common(1, total_len, H, D, cu_seqlens, has_init_state)
        fla_inp, cula_inp, out_fla, out_cula, grads = make_backward_case(common, has_init_state)

        stats = grad_accuracy(fla_inp, cula_inp, out_fla, out_cula, grads)
        rmse, rel, mean = summarize_grad_stats(stats)
        ms_fla = time_backward(fla_inp, out_fla, grads, warmup, iters)
        ms_cula = time_backward(cula_inp, out_cula, grads, warmup, iters)
        speedup = ms_fla / ms_cula if ms_cula > 0 else float("inf")
        tag = f"{dist:>7s} {len(seq_lens):>2d}seqs T={total_len} [{min(seq_lens)}..{max(seq_lens)}]"

        results.append({"tag": tag, "rmse": rmse, "rel": rel, "mean": mean, "ms_fla": ms_fla, "ms_cula": ms_cula, "speedup": speedup})
        del fla_inp, cula_inp, out_fla, out_cula, grads, common
        torch.cuda.empty_cache()
    return results


def print_report(fixed, varlen, H, D, has_init_state, warmup, iters):
    sep = "=" * 112
    print(f"\n\n{sep}")
    print("                  BENCHMARK REPORT: cula_kda_fused_bwd")
    print(f"                  H={H}  D={D}  dtype=bf16  safe_gate=True  has_init_state={has_init_state}")
    print(f"                  Warmup={warmup}  Iters={iters}")
    print(sep)
    if fixed:
        print("\n  [Fixed-Length]")
        print(f"  {'B':>3s}  {'T':>6s}  |  {'grad_RMSE':>10s}  {'grad_rel':>10s}  {'grad_mean':>10s}  |  {'FLA bwd(ms)':>11s}  {'cuLA bwd(ms)':>12s}  {'Speedup':>8s}")
        for row in fixed:
            print(f"  {row['B']:3d}  {row['T']:6d}  |  {row['rmse']:10.6f}  {row['rel']:10.6f}  {row['mean']:10.6f}  |  {row['ms_fla']:11.4f}  {row['ms_cula']:12.4f}  {row['speedup']:7.2f}x")
    if varlen:
        print("\n  [Varlen]")
        print(f"  {'Config':>44s}  |  {'grad_RMSE':>10s}  {'grad_rel':>10s}  {'grad_mean':>10s}  |  {'FLA bwd(ms)':>11s}  {'cuLA bwd(ms)':>12s}  {'Speedup':>8s}")
        for row in varlen:
            print(f"  {row['tag']:>44s}  |  {row['rmse']:10.6f}  {row['rel']:10.6f}  {row['mean']:10.6f}  |  {row['ms_fla']:11.4f}  {row['ms_cula']:12.4f}  {row['speedup']:7.2f}x")
    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(description="bench_kda_fused_bwd: cuLA fused KDA backward vs FLA Triton backward")
    parser.add_argument("--mode", choices=["fixed", "varlen", "both"], default="fixed")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=N_ITERS)
    parser.add_argument("--init_state", action="store_true")
    parser.add_argument("--ncu", action="store_true", help="Use warmup=1 and iters=1")
    args = parser.parse_args()

    warmup = 1 if args.ncu else args.warmup
    iters = 1 if args.ncu else args.iters
    print(
        f"[Device] {torch.cuda.get_device_name(0)}  compute capability {_SM_TAG}  using {cula_kda_fused.__module__}.{cula_kda_fused.__name__}"
    )

    fixed_configs = [(1, 512), (1, 1024), (1, 2048), (2, 512), (2, 1024)]
    varlen_configs = build_varlen_configs(
        num_seqs_list=(8, 16),
        total_lens=(1024, 2048),
        dists=("uniform", "random", "skewed"),
    )

    fixed, varlen = [], []
    if args.mode in ("fixed", "both"):
        fixed = bench_fixed(fixed_configs, args.heads, args.head_dim, args.init_state, warmup, iters)
    if args.mode in ("varlen", "both"):
        varlen = bench_varlen(varlen_configs, args.heads, args.head_dim, args.init_state, warmup, iters)
    print_report(fixed, varlen, args.heads, args.head_dim, args.init_state, warmup, iters)
    return fixed, varlen


if __name__ == "__main__":
    main()
