import argparse
import torch
import time
parser = argparse.ArgumentParser()
parser.add_argument("--is-nsys-profile", action="store_true")
parser.add_argument("--warm-iter", type=int, default=10)
parser.add_argument("--perf-iter", type=int, default=100)
parser.add_argument("--safe-gate", action="store_true")
parser.add_argument("--mode", type=str, choices=["fla", "cula"], default="fla")
parser.add_argument("--seq-num", type=int, default=10)
args = parser.parse_args()

if args.mode == "fla":
    from fla.ops.kda import chunk_kda
elif args.mode == "cula":
    # from cula.kda import chunk_kda
    from cula.utils import get_device_sm_version, get_kda_fused_fwd
    _device = torch.device("cuda")
    _major, _minor = get_device_sm_version(_device)
    _SM_TAG = f"sm{_major}{_minor}"
    chunk_kda = get_kda_fused_fwd(_device)
else:
    raise ValueError(f"Invalid mode: {args.mode}")

def gen_uniform(N, T):
    """All sequences have equal length."""
    per = T // N
    lens = [per] * N
    lens[0] += T - per * N  # absorb remainder
    return lens




is_nsys_profile = args.is_nsys_profile
if is_nsys_profile:
    args.warm_iter = 0
    args.perf_iter = 1

B, T, H, K, V = 1, 4096, 64, 128, 128
seq_num = args.seq_num
device = 'cuda'

q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1   # gate (log space)
beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16).sigmoid()
A_log = torch.randn(H, device=device, dtype=torch.float32) * 0.01
dt_bias = torch.zeros(H * K, device=device, dtype=torch.float32)
init_state = torch.zeros(seq_num, H, K, V, device=device, dtype=torch.float32)

seq_lens = torch.tensor(gen_uniform(seq_num, T), device=device, dtype=torch.int32)
cu_seqlens = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), seq_lens.cumsum(dim=0, dtype=torch.int32)], dim=0)
print(cu_seqlens)

o, final_state = chunk_kda(
    q=q, k=k, v=v, g=g, beta=beta,
    A_log=A_log,
    dt_bias=dt_bias,
    initial_state=init_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    safe_gate=args.safe_gate,
    lower_bound=-5.0 if args.safe_gate else None,
    cu_seqlens=cu_seqlens,
)
do = torch.randn_like(o)

for _ in range(args.warm_iter):
    o, final_state = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=args.safe_gate,
        lower_bound=-5.0 if args.safe_gate else None,
        cu_seqlens=cu_seqlens,
    )
    o.backward(do)

print("Running FLA chunk_kda...")
torch.cuda.synchronize()
if is_nsys_profile:
    torch.cuda.cudart().cudaProfilerStart()
else:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

all_time = 0
# Forward
for _ in range(args.perf_iter):
    o, final_state = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=args.safe_gate,
        lower_bound=-5.0 if args.safe_gate else None,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    if not is_nsys_profile:
        start_event.record()
        o.backward(do)
        end_event.record()
        torch.cuda.synchronize()
        all_time += start_event.elapsed_time(end_event)
    else:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        
        o.backward(do)
        
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

torch.cuda.synchronize()
if is_nsys_profile:
    torch.cuda.cudart().cudaProfilerStop()
else:
    print(f"Time taken: {all_time / args.perf_iter} milliseconds")


