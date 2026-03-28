"""Flash Attention BHSD baseline with separate seq_q / seq_kv support.

Based on example_mha_fwd_bhsd.py with:
  - block_M/N, num_stages, threads exposed as CLI args
  - report_kernel_resources() for shmem/regs/occupancy
  - Larger default problem size (b8h32s4096d128) for benchmarking
  - Correctness verified against torch naive attention
"""
import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial
import ctypes, tempfile, os, re

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
def flashattn(batch, heads, seq_q, seq_kv, dim, is_causal,
              block_M=128, block_N=128, num_stages=1, threads=256):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
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
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
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
                if is_causal else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
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


def report_kernel_resources(kernel, threads_per_block):
    """Report kernel resource usage and occupancy."""
    props = torch.cuda.get_device_properties(0)

    actual_shmem = 0
    try:
        tir_src = str(kernel.artifact.device_mod)
        m = re.search(r'"dyn_shared_memory_buf":\s*(\d+)', tir_src)
        if m:
            actual_shmem = int(m.group(1))
    except Exception:
        pass

    regs_per_thread, max_cta = None, None
    try:
        torch.zeros(1, device='cuda')
        dev_mod = kernel.artifact.rt_mod.imports_[0]
        cubin_path = os.path.join(tempfile.gettempdir(), '_tilelang_query.cubin')
        dev_mod.write_to_file(cubin_path, fmt='cubin')
        tir_src = str(kernel.artifact.device_mod)
        m = re.search(r'def (\w+_kernel)\(', tir_src)
        func_name = m.group(1).encode() if m else b'main_kernel'
        cuda = ctypes.CDLL('libcuda.so.1')
        module = ctypes.c_void_p()
        if cuda.cuModuleLoad(ctypes.byref(module), cubin_path.encode()) == 0:
            func = ctypes.c_void_p()
            if cuda.cuModuleGetFunction(ctypes.byref(func), module, func_name) == 0:
                val = ctypes.c_int()
                cuda.cuFuncGetAttribute(ctypes.byref(val), 4, func)
                regs_per_thread = val.value
                cuda.cuFuncSetAttribute(func, 8, actual_shmem)
                num_blocks = ctypes.c_int()
                cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    ctypes.byref(num_blocks), func, threads_per_block, ctypes.c_size_t(actual_shmem))
                max_cta = num_blocks.value
            cuda.cuModuleUnload(module)
        os.unlink(cubin_path)
    except Exception:
        pass

    shmem_per_sm = props.shared_memory_per_multiprocessor
    regs_per_sm = props.regs_per_multiprocessor
    max_warps_per_sm = props.max_threads_per_multi_processor // props.warp_size
    warps_per_cta = threads_per_block // props.warp_size

    print("\n=== Kernel Resource Report ===")
    print(f"  shmem per CTA:    {actual_shmem/1024:.1f} KB")
    print(f"  warps per CTA:    {warps_per_cta}")
    if regs_per_thread is not None:
        print(f"  regs per thread:  {regs_per_thread}")
    print(f"  SM resources vs CTA demand:")
    max_cta_by_shmem = shmem_per_sm // actual_shmem if actual_shmem > 0 else 99
    max_cta_by_warps = max_warps_per_sm // warps_per_cta
    print(f"    shmem:     {shmem_per_sm/1024:.0f} KB / {actual_shmem/1024:.1f} KB = {max_cta_by_shmem} CTAs")
    print(f"    warps:     {max_warps_per_sm} / {warps_per_cta} = {max_cta_by_warps} CTAs")
    if regs_per_thread is not None:
        max_cta_by_regs = regs_per_sm // (regs_per_thread * threads_per_block)
        print(f"    registers: {regs_per_sm} / ({regs_per_thread} x {threads_per_block}) = {max_cta_by_regs} CTAs")
    if max_cta is not None:
        print(f"    -> {max_cta} concurrent CTA(s) per SM (cuOccupancy)")
    print()


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device), seq_kv - seq_q)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output


def main(
    batch: int = 8,
    heads: int = 32,
    seq_q: int = 4096,
    seq_kv: int = 4096,
    dim: int = 128,
    is_causal: bool = False,
    tune: bool = False,
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
        kernel = flashattn(batch, heads, seq_q, seq_kv, dim, is_causal,
                           block_M=block_M, block_N=block_N, num_stages=num_stages, threads=threads)
        report_kernel_resources(kernel, threads)
        ref_program_processed = partial(ref_program, is_causal=is_causal)

        profiler = kernel.get_profiler()
        profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(ref_program_processed, n_warmup=50, n_repeat=200)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(n_warmup=50, n_repeat=200)
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        best_result = flashattn(batch, heads, seq_q, seq_kv, dim, is_causal)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention BHSD baseline (supports seq_q != seq_kv)")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--seq_q", type=int, default=4096, help="query sequence length")
    parser.add_argument("--seq_kv", type=int, default=4096, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="head dimension")
    parser.add_argument("--is_causal", action="store_true", help="causal attention")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=256)
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_q, args.seq_kv, args.dim, args.is_causal, args.tune,
         args.block_M, args.block_N, args.num_stages, args.threads)
