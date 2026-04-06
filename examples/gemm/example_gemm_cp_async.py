import tilelang
import tilelang.language as T
from tilelang.contrib import nvcc


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}
CUDA_TARGET = f"cuda -arch=sm_{nvcc.get_target_arch(nvcc.get_target_compute_version())}"


@tilelang.jit(out_idx=[-1], target=CUDA_TARGET, pass_configs=PASS_CONFIGS)
def matmul_cp_async(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = 0
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def _print_cp_async_snippet(cuda_source: str):
    cp_async_lines = [
        line
        for line in cuda_source.splitlines()
        if "cp.async" in line or "cp_async" in line
    ]
    if not cp_async_lines:
        raise RuntimeError(
            "Expected cp.async-style async copy in generated CUDA source, "
            "but neither raw PTX `cp.async` nor TileLang helper calls "
            "like `tl::cp_async_gs` were found."
        )

    print("Detected cp.async-style async copy in generated CUDA source:")
    for line in cp_async_lines[:12]:
        print(line)


def main():
    M = 1024
    N = 1024
    K = 1024
    block_M = 128
    block_N = 128
    block_K = 32

    kernel = matmul_cp_async(M, N, K, block_M, block_N, block_K)

    import torch

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    c = kernel(a, b)
    ref_c = a @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    cuda_source = kernel.get_kernel_source()
    _print_cp_async_snippet(cuda_source)

    print("CUDA Source:")
    print(cuda_source)

    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    kernel = matmul_cp_async(1024, 1024, 1024, 128, 128, 32)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()
