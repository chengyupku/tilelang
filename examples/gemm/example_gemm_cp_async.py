"""GEMM using cp.async + WGMMA (no TMA, no warp specialization)."""
import torch
import tilelang
import tilelang.language as T
from tilelang.contrib import nvcc
import argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.kernel_report import report_kernel_resources

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}
CUDA_TARGET = f"cuda -arch=sm_{nvcc.get_target_arch(nvcc.get_target_compute_version())}"

DTYPE_MAP = {"float16": T.float16, "int8": T.int8}
ACCUM_MAP = {"float16": T.float32, "int8": T.int32}
TORCH_DTYPE_MAP = {"float16": torch.float16, "int8": torch.int8}


@tilelang.jit(out_idx=[-1], target=CUDA_TARGET, pass_configs=PASS_CONFIGS)
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32,
           num_stages=3, threads=128):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype),
               C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel


def cupti_bench_torch(func, n_warmup=20, n_repeat=200):
    for _ in range(n_warmup):
        func()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, profile_memory=False,
    ) as prof:
        for _ in range(n_repeat):
            func()
        torch.cuda.synchronize()
    cuda_us = sum(e.self_device_time_total for e in prof.key_averages())
    return cuda_us / n_repeat / 1000


def main(M=8192, N=8192, K=4096, block_M=128, block_N=128, block_K=64,
         dtype="float16", num_stages=3, threads=128, tracekernel=False, no_ref=False):
    tl_dtype = DTYPE_MAP[dtype]
    accum_dtype = ACCUM_MAP[dtype]
    torch_dtype = TORCH_DTYPE_MAP[dtype]
    tops = 2 * M * N * K / 1e12
    unit = "TFlops" if dtype == "float16" else "TOPS"

    kernel = matmul(M, N, K, block_M, block_N, block_K, tl_dtype, accum_dtype, num_stages, threads)
    report_kernel_resources(kernel, threads)
    if tracekernel:
        profiler = kernel.get_profiler()
        ins = profiler._get_inputs()
        profiler.func(*ins)
        torch.cuda.synchronize()
        return
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(n_warmup=50, n_repeat=200)
    print(f"tilelang: {latency:.4f} ms, {tops/(latency/1e3):.1f} {unit}")

    if not no_ref:
        A = torch.randn(M, K, device="cuda", dtype=torch_dtype) if dtype == "float16" \
            else torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
        B = torch.randn(N, K, device="cuda", dtype=torch_dtype) if dtype == "float16" \
            else torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
        B_T = B.T
        if dtype == "int8":
            torch_latency = cupti_bench_torch(lambda: torch._int_mm(A, B_T))
        else:
            torch_latency = cupti_bench_torch(lambda: torch.matmul(A, B_T))
        print(f"torch:    {torch_latency:.4f} ms, {tops/(torch_latency/1e3):.1f} {unit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=256)
    parser.add_argument("--block_K", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "int8"])
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--tracekernel", action="store_true", help="Run kernel once for nsys/ncu tracing")
    parser.add_argument("--no_ref", action="store_true", help="Skip torch/cuBLAS reference comparison")
    args = parser.parse_args()
    main(args.M, args.N, args.K, args.block_M, args.block_N, args.block_K,
         args.dtype, args.num_stages, args.threads, args.tracekernel, args.no_ref)
