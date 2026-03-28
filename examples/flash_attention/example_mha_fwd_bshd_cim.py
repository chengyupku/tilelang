"""
CIM-simulated Flash Attention (BSHD layout) using T.gemm(cim_simulate=True)
with micro_m/n/k ratio control.

Results are numerically incorrect — this is a latency-only CIM benchmark.
"""
import torch
import tilelang
import tilelang.language as T
import argparse

tilelang.disable_cache()

DTYPE_BYTES = {"float16": 2, "int8": 1}


def _query_kernel_resources(kernel, threads_per_block, dyn_shmem):
    """Query actual register count and occupancy via CUDA Driver API."""
    import ctypes, tempfile, os, re
    try:
        torch.zeros(1, device='cuda')
        dev_mod = kernel.artifact.rt_mod.imports_[0]
        cubin_path = os.path.join(tempfile.gettempdir(), '_tilelang_query.cubin')
        dev_mod.write_to_file(cubin_path, fmt='cubin')

        tir_src = str(kernel.artifact.device_mod)
        m = re.search(r'def (\w+_kernel)\(', tir_src)
        func_name = m.group(1).encode() if m else b'kernel_kernel'

        cuda = ctypes.CDLL('libcuda.so.1')
        CUmodule = ctypes.c_void_p
        CUfunction = ctypes.c_void_p

        module = CUmodule()
        if cuda.cuModuleLoad(ctypes.byref(module), cubin_path.encode()) != 0:
            return None, None
        func = CUfunction()
        if cuda.cuModuleGetFunction(ctypes.byref(func), module, func_name) != 0:
            cuda.cuModuleUnload(module)
            return None, None

        val = ctypes.c_int()
        cuda.cuFuncGetAttribute(ctypes.byref(val), 4, func)
        regs = val.value

        cuda.cuFuncSetAttribute(func, 8, dyn_shmem)
        num_blocks = ctypes.c_int()
        cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(num_blocks), func, threads_per_block, ctypes.c_size_t(dyn_shmem))
        occ = num_blocks.value

        cuda.cuModuleUnload(module)
        os.unlink(cubin_path)
        return regs, occ
    except Exception:
        return None, None


def report_cim_capacity(cim_buffers, num_stages, kernel, threads_per_block):
    """Report CIM macro capacity at multiple levels."""
    import re
    props = torch.cuda.get_device_properties(0)

    actual_shmem = 0
    try:
        tir_src = str(kernel.artifact.device_mod)
        m = re.search(r'"dyn_shared_memory_buf":\s*(\d+)', tir_src)
        if m:
            actual_shmem = int(m.group(1))
    except Exception:
        pass

    regs_per_thread, max_cta = _query_kernel_resources(kernel, threads_per_block, actual_shmem)

    n_bufs = len(cim_buffers)
    n_sm = props.multi_processor_count

    print("\n=== CIM Macro Capacity Report ===")

    print(f"  [Block Tile]  {n_bufs} CIM buffer(s) per tile:")
    cim_per_tile = 0
    for name, shape, dtype in cim_buffers:
        elem_bytes = DTYPE_BYTES[dtype]
        n_elems = 1
        for d in shape:
            n_elems *= d
        buf_bytes = n_elems * elem_bytes
        cim_per_tile += buf_bytes
        shape_str = "x".join(str(d) for d in shape)
        print(f"    {name:12s}  {shape_str:>12s} x {dtype:>7s} = {buf_bytes:>8d} B ({buf_bytes/1024:.1f} KB)")
    print(f"    {'':12s}  {'tile total':>12s}           = {cim_per_tile:>8d} B ({cim_per_tile/1024:.1f} KB)")

    cim_per_cta = cim_per_tile * num_stages
    print(f"  [Per CTA]     {n_bufs} tile(s) x {num_stages} stage(s) = {cim_per_cta/1024:.1f} KB CIM")
    print(f"                total shmem (CIM + non-CIM) = {actual_shmem/1024:.1f} KB")

    shmem_per_sm = props.shared_memory_per_multiprocessor
    max_threads_per_sm = props.max_threads_per_multi_processor
    regs_per_sm = props.regs_per_multiprocessor
    warps_per_cta = threads_per_block // props.warp_size
    max_warps_per_sm = max_threads_per_sm // props.warp_size

    max_cta_by_shmem = shmem_per_sm // actual_shmem if actual_shmem > 0 else 99
    max_cta_by_warps = max_warps_per_sm // warps_per_cta if warps_per_cta > 0 else 99
    if regs_per_thread is not None and regs_per_thread > 0:
        regs_per_cta = regs_per_thread * threads_per_block
        max_cta_by_regs = regs_per_sm // regs_per_cta
    else:
        regs_per_cta = None
        max_cta_by_regs = 99

    print(f"  [Per SM]      SM resources vs CTA demand → max concurrent CTAs:")
    print(f"    shmem:      {shmem_per_sm/1024:.0f} KB / {actual_shmem/1024:.1f} KB per CTA = {max_cta_by_shmem} CTAs")
    print(f"    warps:      {max_warps_per_sm} / {warps_per_cta} per CTA = {max_cta_by_warps} CTAs")
    if regs_per_thread is not None:
        print(f"    registers:  {regs_per_sm} / ({regs_per_thread} x {threads_per_block}) per CTA = {max_cta_by_regs} CTAs")
    if max_cta is not None:
        print(f"    → {max_cta} concurrent CTA(s) (cuOccupancy)")
        cim_per_sm = cim_per_cta * max_cta
        print(f"    → CIM capacity = {max_cta} x {cim_per_cta/1024:.1f} KB = {cim_per_sm/1024:.1f} KB")

        cim_total = cim_per_sm * n_sm
        print(f"  [Device]      {n_sm} SMs x {cim_per_sm/1024:.1f} KB = {cim_total/1024/1024:.1f} MB CIM total")
    print()


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn_cim(batch, heads, seq_len, dim, is_causal,
                  block_M=128, block_N=128, num_stages=1, threads=256,
                  micro_m=0, micro_n=0, micro_k=0):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
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

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                if is_causal else T.ceildiv(seq_len, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow,
                       cim_simulate=True,
                       cim_micro_m=micro_m, cim_micro_n=micro_n, cim_micro_k=micro_k)

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

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow,
                       cim_simulate=True,
                       cim_micro_m=micro_m, cim_micro_n=micro_n, cim_micro_k=micro_k)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


def main(
    batch: int = 8,
    heads: int = 32,
    seq_len: int = 4096,
    dim: int = 128,
    is_causal: bool = False,
    micro_m: int = 0,
    micro_n: int = 0,
    micro_k: int = 0,
    block_M: int = 128,
    block_N: int = 128,
    num_stages: int = 1,
    threads: int = 256,
):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    kernel = flashattn_cim(batch, heads, seq_len, dim, is_causal,
                           block_M=block_M, block_N=block_N, num_stages=num_stages, threads=threads,
                           micro_m=micro_m, micro_n=micro_n, micro_k=micro_k)

    # CIM capacity report: K and V matrices live in CIM
    report_cim_capacity(
        cim_buffers=[
            ("K_shared", (block_N, dim), "float16"),
            ("V_shared", (block_N, dim), "float16"),
        ],
        num_stages=num_stages,
        kernel=kernel,
        threads_per_block=threads,
    )

    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti", n_warmup=50, n_repeat=200)
    micro_str = f"micro={micro_m}/{micro_n}/{micro_k}" if any([micro_m, micro_n, micro_k]) else "micro=default"
    print(f"CIM FA ({micro_str}): {latency:.2f} ms, {total_flops / latency * 1e-9:.2f} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIM Flash Attention with CIM instruction shape control")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument("--micro_m", type=int, default=0, help="CIM instruction M dim (0=default)")
    parser.add_argument("--micro_n", type=int, default=0, help="CIM instruction N dim (0=default)")
    parser.add_argument("--micro_k", type=int, default=0, help="CIM instruction K dim (0=default)")
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=256)
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal,
         args.micro_m, args.micro_n, args.micro_k,
         args.block_M, args.block_N, args.num_stages, args.threads)
