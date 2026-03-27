import torch
import tilelang
import tilelang.testing
from tilelang.utils.tensor import map_torch_type
from example_gemm_intrinsic_kernel import tl_matmul

import os

tilelang.disable_cache()

# dtype → (A/B dtype, accum dtype, output dtype)
DTYPE_CONFIGS = {
    "int8":    ("int8",    "int32",   "int32"),
    "float16": ("float16", "float32", "float16"),
}

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(
    M,
    N,
    K,
    warp_row_tiles,
    warp_col_tiles,
    chunk,
    block_row_tiles,
    block_col_tiles,
    dtype,
    stage,
    tracekernel,
    use_shmem_writeback,
    use_zero_benchmark=False
):
    in_dtype, accum_dtype, out_dtype = DTYPE_CONFIGS[dtype]

    kernel = tl_matmul(
        M,
        N,
        K,
        warp_row_tiles,
        warp_col_tiles,
        chunk,
        block_row_tiles,
        block_col_tiles,
        in_dtype,
        in_dtype,
        accum_dtype,
        out_dtype,
        stage=stage,
        use_shmem_writeback=use_shmem_writeback,
    )

    # Get CUDA Source
    source = kernel.get_kernel_source()
    # print(source)

    A_torch = map_torch_type(in_dtype)
    B_torch = map_torch_type(in_dtype)
    C_torch = map_torch_type(out_dtype)

    if use_zero_benchmark:
        if A_torch in {torch.int8, torch.int32}:
            A = torch.zeros((M, K), dtype=torch.int8).to(A_torch).cuda()
        elif A_torch in {torch.float8_e4m3fn, torch.float8_e5m2}:
            A = torch.zeros(M, K).to(A_torch).cuda()
        else:
            A = torch.zeros(M, K).to(A_torch).cuda() - 0.5
        if B_torch in {torch.int8, torch.int32}:
            B = torch.zeros((N, K), dtype=torch.int8).to(B_torch).cuda()
        elif B_torch in {torch.float8_e4m3fn, torch.float8_e5m2}:
            B = torch.zeros(N, K).to(B_torch).cuda()
        else:
            B = torch.zeros(N, K).to(B_torch).cuda() - 0.5
    else:
        if A_torch in {torch.int8, torch.int32}:
            A = torch.randint(-128, 128, (M, K), dtype=torch.int8).to(A_torch).cuda()
        elif A_torch in {torch.float8_e4m3fn, torch.float8_e5m2}:
            A = torch.randn(M, K).to(A_torch).cuda()
        else:
            A = torch.randn(M, K).to(A_torch).cuda() - 0.5
        if B_torch in {torch.int8, torch.int32}:
            B = torch.randint(-128, 128, (N, K), dtype=torch.int8).to(B_torch).cuda()
        elif B_torch in {torch.float8_e4m3fn, torch.float8_e5m2}:
            B = torch.randn(N, K).to(B_torch).cuda()
        else:
            B = torch.randn(N, K).to(B_torch).cuda() - 0.5

    C = kernel(A, B)

    if tracekernel:
        return

    # benchmark
    if use_zero_benchmark:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Zero)
    else:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    latency = profiler.do_bench(backend="cupti", n_warmup=20, n_repeat=200)
    total_flops = 2 * M * N * K
    unit = "TFlops" if dtype == "float16" else "TOPS"
    print(f"tilelang: {latency:.4f} ms, {total_flops / latency * 1e-9:.1f} {unit}")

    # benchmark torch (CUPTI via torch.profiler)
    def cupti_bench_torch(func, n_warmup=20, n_repeat=200):
        for _ in range(n_warmup):
            func()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            for _ in range(n_repeat):
                func()
            torch.cuda.synchronize()
        cuda_us = sum(e.self_device_time_total for e in prof.key_averages())
        return cuda_us / n_repeat / 1000  # us -> ms

    B_T = B.T
    if A_torch == torch.int8:
        latency = cupti_bench_torch(lambda: torch._int_mm(A, B_T))
    else:
        latency = cupti_bench_torch(lambda: torch.matmul(A, B_T))
    print(f"torch:    {latency:.4f} ms, {total_flops / latency * 1e-9:.1f} {unit}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline intrinsic GEMM benchmark")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="int8", choices=["float16", "int8"])
    parser.add_argument("--warp_m", type=int, default=64)
    parser.add_argument("--warp_n", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--block_m", type=int, default=128)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--tracekernel", type=str_to_bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--use_shmem_writeback", type=str_to_bool, nargs='?',
                        const=True, default=False)

    args = parser.parse_args()

    main(
        M=args.M,
        N=args.N,
        K=args.K,
        warp_row_tiles=args.warp_m,
        warp_col_tiles=args.warp_n,
        chunk=args.chunk,
        block_row_tiles=args.block_m,
        block_col_tiles=args.block_n,
        dtype=args.dtype,
        stage=args.stage,
        tracekernel=args.tracekernel,
        use_shmem_writeback=args.use_shmem_writeback,
    )
