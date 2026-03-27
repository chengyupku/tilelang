"""GEMM baseline using T.gemm (high-level API)."""
import torch
import tilelang
import tilelang.language as T
import argparse

tilelang.disable_cache()

DTYPE_MAP = {"float16": T.float16, "int8": T.int8}
ACCUM_MAP = {"float16": T.float32, "int8": T.int32}
TORCH_DTYPE_MAP = {"float16": torch.float16, "int8": torch.int8}


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, num_stages=3):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype),
               C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.use_swizzle(panel_size=10)  # L2 rasterization optimization
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
         dtype="float16", num_stages=3):
    tl_dtype = DTYPE_MAP[dtype]
    accum_dtype = ACCUM_MAP[dtype]
    torch_dtype = TORCH_DTYPE_MAP[dtype]
    tops = 2 * M * N * K / 1e12
    unit = "TFlops" if dtype == "float16" else "TOPS"

    kernel = matmul(M, N, K, block_M, block_N, block_K, tl_dtype, accum_dtype, num_stages)
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(n_warmup=50, n_repeat=200)
    print(f"tilelang: {latency:.4f} ms, {tops/(latency/1e3):.1f} {unit}")

    # torch (cuBLAS) comparison
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
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--block_K", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "int8"])
    parser.add_argument("--num_stages", type=int, default=3)
    args = parser.parse_args()
    main(args.M, args.N, args.K, args.block_M, args.block_N, args.block_K,
         args.dtype, args.num_stages)
