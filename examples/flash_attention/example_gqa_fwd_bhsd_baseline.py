"""GQA Flash Attention BHSD baseline with separate seq_q / seq_kv support.

Based on example_mha_fwd_bhsd_baseline.py with groups parameter for GQA.
When groups=1 this is identical to MHA.

GQA optimization: Q heads sharing the same KV head are packed into the seq
dimension via reshape [B, H_q, S_q, D] -> [B, H_kv, S_q*groups, D]. This
is a zero-copy view in BHSD layout. The kernel runs standard MHA on the
packed tensors, loading KV only once per KV head.
"""
import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.kernel_report import report_kernel_resources

tilelang.disable_cache()

def get_configs():
    iter_params = dict(block_M=[128], block_N=[128], num_stages=[2], threads=[256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, groups=1,
              block_M=128, block_N=128, num_stages=1, threads=256):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    seq_q_eff = seq_q * groups  # packed Q length: groups Q heads merged into seq dim
    q_shape = [batch, head_kv, seq_q_eff, dim]
    kv_shape = [batch, head_kv, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q_eff, block_M), head_kv, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
                if is_causal and groups == 1
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = (bx * block_M + i) % seq_q + past_len
                        k_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main


def ref_program(Q, K, V, is_causal, groups=1, seq_q_orig=None):
    # Q arrives as [B, head_kv, seq_q*groups, D] (packed layout from kernel)
    # Unpack to [B, heads_q, seq_q, D] for standard attention
    B, head_kv, seq_q_eff, dim = Q.shape
    if seq_q_orig is None:
        seq_q_orig = seq_q_eff // groups
    Q = Q.reshape(B, head_kv, groups, seq_q_orig, dim).reshape(B, head_kv * groups, seq_q_orig, dim)
    K = K.repeat_interleave(groups, dim=1)
    V = V.repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q_orig, seq_kv, device=scores.device), seq_kv - seq_q_orig)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    # Pack back to [B, head_kv, seq_q*groups, D]
    output = output.reshape(B, head_kv, groups, seq_q_orig, dim).reshape(B, head_kv, seq_q_eff, dim)
    return output


def main(
    batch: int = 8,
    heads: int = 32,
    seq_q: int = 4096,
    seq_kv: int = 4096,
    dim: int = 128,
    is_causal: bool = False,
    groups: int = 1,
    tune: bool = False,
    tracekernel: bool = False,
    no_ref: bool = False,
    block_M: int = 128,
    block_N: int = 128,
    num_stages: int = 2,
    threads: int = 256,
):
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    if not tune:
        kernel = flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, groups=groups,
                           block_M=block_M, block_N=block_N, num_stages=num_stages, threads=threads)
        report_kernel_resources(kernel, threads)
        if tracekernel:
            kernel.get_profiler().do_bench(n_warmup=0, n_repeat=1)
            return
        profiler = kernel.get_profiler()
        if not no_ref:
            ref_program_processed = partial(ref_program, is_causal=is_causal, groups=groups, seq_q_orig=seq_q)
            profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
            print("All checks pass.")
            latency = profiler.do_bench(ref_program_processed, n_warmup=50, n_repeat=200)
            print("Ref: {:.2f} ms".format(latency))
            print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(n_warmup=50, n_repeat=200)
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        best_result = flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, groups=groups)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GQA Flash Attention BHSD baseline (supports seq_q != seq_kv)")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="query heads")
    parser.add_argument("--seq_q", type=int, default=4096, help="query sequence length")
    parser.add_argument("--seq_kv", type=int, default=4096, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="head dimension")
    parser.add_argument("--is_causal", action="store_true", help="causal attention")
    parser.add_argument("--groups", type=int, default=1, help="GQA groups (heads/groups = kv heads)")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    parser.add_argument("--tracekernel", action="store_true", help="Run kernel once for nsys/ncu tracing")
    parser.add_argument("--no_ref", action="store_true", help="Skip torch reference check and comparison")
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=256)
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.is_causal, args.groups,
         args.tune, args.tracekernel, args.no_ref, args.block_M, args.block_N, args.num_stages, args.threads)
