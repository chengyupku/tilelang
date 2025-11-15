import torch
import tilelang
import tilelang.testing
from tilelang.utils.tensor import map_torch_type
from example_gemm_intrinsic_kernel import tl_matmul

import os

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
    A_in_dtype,
    B_in_dtype,
    C_in_dtype,
    C_out_dtype,
    stage,
    tracekernel,
    use_shmem_writeback,
    use_zero_benchmark=False
):
    kernel = tl_matmul(
        M,
        N,
        K,
        warp_row_tiles,
        warp_col_tiles,
        chunk,
        block_row_tiles,
        block_col_tiles,
        A_in_dtype,
        B_in_dtype,
        C_in_dtype,
        C_out_dtype,
        stage=stage,
        use_shmem_writeback=use_shmem_writeback,
    )

    # Get CUDA Source
    source = kernel.get_kernel_source()
    print(source)

    A_in_dtype = map_torch_type(A_in_dtype)
    B_in_dtype = map_torch_type(B_in_dtype)
    C_out_dtype = map_torch_type(C_out_dtype)
    C_in_dtype = map_torch_type(C_in_dtype)

    if use_zero_benchmark:
        if A_in_dtype in {torch.int8, torch.int32}:
            A = torch.zeros((M, K), dtype=torch.int8).to(A_in_dtype).cuda()
        elif A_in_dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
            A = torch.zeros(M, K).to(A_in_dtype).cuda()
        else:
            A = torch.zeros(M, K).to(A_in_dtype).cuda() - 0.5
        if B_in_dtype in {torch.int8, torch.int32}:
            B = torch.zeros((N, K), dtype=torch.int8).to(B_in_dtype).cuda()
        elif B_in_dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
            B = torch.zeros(N, K).to(B_in_dtype).cuda()
        else:
            B = torch.zeros(N, K).to(B_in_dtype).cuda() - 0.5
    else:
        if A_in_dtype in {torch.int8, torch.int32}:
            A = torch.randint(-128, 128, (M, K), dtype=torch.int8).to(A_in_dtype).cuda()
        elif A_in_dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
            A = torch.randn(M, K).to(A_in_dtype).cuda()
        else:
            A = torch.randn(M, K).to(A_in_dtype).cuda() - 0.5
        if B_in_dtype in {torch.int8, torch.int32}:
            B = torch.randint(-128, 128, (N, K), dtype=torch.int8).to(B_in_dtype).cuda()
        elif B_in_dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
            B = torch.randn(N, K).to(B_in_dtype).cuda()
        else:
            B = torch.randn(N, K).to(B_in_dtype).cuda() - 0.5

    C = kernel(A, B)
    
    if tracekernel:
        return

    # # Get Reference Result
    # if in_dtype == torch.int8:
    #     ref_c = torch._int_mm(A, B.T)
    # else:
    #     ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(out_dtype)
    # # torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
    # # tilelang.testing.torch_assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
    # print("All check passed.")

    # benchmark
    if use_zero_benchmark:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Zero)
    else:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
    latency = profiler.do_bench(backend="cupti", n_warmup=20, n_repeat=200)
    print(f"tilelang Latency: {latency}ms")
    total_flops = 2 * M * N * K
    print(f"tilelang TFlops (or Tops): {total_flops / latency * 1e-9} TFlops")

    # # benchmark torch
    # def torch_bench(func, *args, **kwargs):
    #     # warmup
    #     for _ in range(10):
    #         func(*args, **kwargs)
    #     torch.cuda.synchronize()
    #     # bench
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     for _ in range(100):
    #         func(*args, **kwargs)
    #     end.record()
    #     torch.cuda.synchronize()
    #     return start.elapsed_time(end) / 100
    # if in_dtype == torch.int8:
    #     latency = torch_bench(torch._int_mm, A, B.T)
    # else:
    #     latency = torch_bench(torch.matmul, A, B.T)
    # print(f"torch Latency: {latency}ms")
    # print(f"torch TFlops (or Tops): {total_flops / latency * 1e-9} TFlops")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--warp_m", type=int, default=64)
    parser.add_argument("--warp_n", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--block_m", type=int, default=128)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--Atype", type=str, default="int8")
    parser.add_argument("--Wtype", type=str, default="int8")
    parser.add_argument("--Outtype", type=str, default="int32")
    parser.add_argument("--acctype", type=str, default="int32")
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
        A_in_dtype=args.Atype,
        B_in_dtype=args.Wtype,
        C_in_dtype=args.acctype,
        C_out_dtype=args.Outtype,
        stage=args.stage,
        tracekernel=args.tracekernel,
        use_shmem_writeback=args.use_shmem_writeback,
    )
