from tilelang import tvm as tvm
from tvm import DataType
import tilelang
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.intrinsics.mma_cim_macro_generator import (
    TensorCoreIntrinEmitter,)
from tilelang.transform import simplify_prim_func
from tilelang.profiler import do_bench
import torch
from typing import Callable

tilelang.disable_cache()

# @register_cuda_postproc_callback
# def tilelang_callback_cuda_postproc(code, _):
#     code = """"""
#     return code


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


data_map = {
    "float32": 32,
    "float16": 16,
    "int8": 8,
    "int4": 4,
}

def benchmark_cuda(func: Callable[[], None], warmup: int = 10, repeat: int = 100) -> float:
    import torch

    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        func()
    end.record()
    torch.cuda.synchronize()
    latency_ms = start.elapsed_time(end) / repeat
    return latency_ms

@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    })
@simplify_prim_func
def tl_matmul(
    M,
    N,
    K,
    # Faked CIM instruction size
    micro_size_m,
    micro_size_n,
    micro_size_k,
    fake_instr_m,
    fake_instr_n,
    fake_instr_k,
    warp_row_tiles,
    warp_col_tiles,
    chunk,
    block_row_tiles,
    block_col_tiles,
    A_in_dtype,
    B_in_dtype,
    C_in_dtype,
    C_out_dtype,
    stage=2,
):
    assert A_in_dtype in [
        "float16",
        "int8",
    ], "Currently only float16 and int8 are supported"
    assert B_in_dtype in [
        "int8",
        "int4",
        "float16",
    ], "Currently only int8, int4, and float16 are supported"
    assert C_in_dtype in [
        "float16",
        "float32",
    ], "Currently only float16, float32 are supported"

    # block_row_warps = 2
    # block_col_warps = 2
    # warp_row_tiles = 64
    # warp_col_tiles = 128
    # chunk = 32 if in_dtype == "float16" else 64
    # chunk = 64
    shared_scope = "shared.dyn"

    block_row_warps = block_row_tiles // warp_row_tiles
    block_col_warps = block_col_tiles // warp_col_tiles
    block_M = block_row_tiles
    block_N = block_col_tiles
    block_K = chunk

    A_shape = (M, K * data_map[A_in_dtype] // 16)
    B_shape = (N, K * data_map[B_in_dtype] // 16)
    A_shared_shape = (block_M, block_K * data_map[A_in_dtype] // 16)
    B_shared_shape = (block_N, block_K * data_map[B_in_dtype] // 16)
    # Use fake PTX instruction inner tile for safe staging to avoid OOB in stmatrix
    C_shared_shape = (
        block_M // micro_size_m,
        block_N // micro_size_n,
        fake_instr_m,
        fake_instr_n,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    # Local fragment sizes follow the real PTX instruction shape
    local_size_a = (micro_size_m * micro_size_k) // warp_size
    # local_size_b = (micro_size_n * micro_size_k) // warp_size
    local_size_c = (micro_size_m * micro_size_n) // warp_size

    warp_rows = warp_row_tiles // micro_size_m  # 64 // 1 = 64
    warp_cols = warp_col_tiles // micro_size_n  # 64 // 32 = 2
    
    print(f"fake_warp_rows: {warp_rows}, fake_warp_cols: {warp_cols}")

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype="float16",
        b_dtype="float16",
        accum_dtype="float32",
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        fake_instr_m=fake_instr_m,
        fake_instr_n=fake_instr_n,
        fake_instr_k=fake_instr_k,
        fake_warp_rows=warp_rows,
        fake_warp_cols=warp_cols,
    )

    @T.prim_func
    def gemm_intrinsics(
            A: T.Tensor(A_shape, "float16"),
            B: T.Tensor(B_shape, "float16"),
            C: T.Tensor((M, N), C_out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, "float16", scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, "float16", scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, "float16", scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a * data_map[A_in_dtype] // 16),
                                    "float16")
            # B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
            C_local = T.alloc_local(
                (warp_rows * warp_cols * local_size_c * data_map[C_in_dtype] // 32), "float32")

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(A_local, A_shared, ki)

                    # Do not need to use ldmatrix for B because
                    # weight is in CIM
                    # mma_emitter.ldmatrix_b(B_local, B_shared, ki)

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_shared, C_local, cim_simulate=True)

            # Perform STMatrix
            mma_emitter.stmatrix(C_local, C_shared)

            # Store shared into global. Use fake_instr_m/n as inner dims to match PTX store layout
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_m,
                    j // micro_size_n,
                    i % fake_instr_m,
                    j % fake_instr_n,
                ]

    return gemm_intrinsics

def ref_program_fp(A, B):
    return A @ B
def ref_program_int(A, B):
    return torch._int_mm(A, B)

def main(M=4096,
         N=4096,
         K=4096,
         micro_size_m=1,
         micro_size_n=64,
         micro_size_k=32,
         fake_instr_m=16,
         fake_instr_n=8,
         fake_instr_k=16,
         warp_row_tiles=64,
         warp_col_tiles=64,
         chunk=32,
         block_row_tiles=128,
         block_col_tiles=128,
         A_in_dtype="float16",
         B_in_dtype="float16",
         C_in_dtype="float32",
         C_out_dtype="float16",
         stage=2,
         tracekernel=False,
):
    tflops = 2 * M * N * K / 1e12
    kernel = tl_matmul(M, N, K, micro_size_m, micro_size_n, micro_size_k,
                       fake_instr_m,
                       fake_instr_n,
                       fake_instr_k,
                       warp_row_tiles,
                       warp_col_tiles,
                       chunk,
                       block_row_tiles,
                       block_col_tiles,
                       A_in_dtype, 
                       B_in_dtype,
                       C_in_dtype,
                       C_out_dtype,
                       stage=stage)
    print(kernel.get_kernel_source())
    # src_code = kernel.get_kernel_source()
    # # src_code is the generated cuda source
    # assert src_code is not None
    
    if tracekernel == True:
        A_shape = (M, K * data_map[A_in_dtype] // 16)
        B_shape = (N, K * data_map[B_in_dtype] // 16)
        a = torch.ones(A_shape, dtype=torch.float16).cuda()
        b = torch.ones(B_shape, dtype=torch.float16).cuda()
        _ = kernel(a, b)
        return

    A_shape = (M, K * data_map[A_in_dtype] // 16)
    B_shape = (N, K * data_map[B_in_dtype] // 16)
    a = torch.randn(A_shape, dtype=torch.float16).cuda()
    b = torch.randn(B_shape, dtype=torch.float16).cuda()
    latency = benchmark_cuda(lambda: kernel(a, b))

    print(f"CIM latency:{latency} ms")
    print(f"CIM TFLOPS: {tflops / (latency / 1e3)}")
    
    # if A_in_dtype == "float16":
    #     a = torch.randn((M, K), dtype=torch.float16).cuda()
    #     b = torch.randn((K, N), dtype=torch.float16).cuda()
    #     torch_latency = benchmark_cuda(lambda: ref_program_fp(a, b))
    # else:
    #     a = torch.randint(-128, 127, (M, K), dtype=torch.int8).cuda()
    #     b = torch.randint(-128, 127, (N, K), dtype=torch.int8).cuda()
    #     torch_latency = benchmark_cuda(lambda: ref_program_int(a, b))
    
    # print(f"torch latency:{torch_latency} ms")
    # print(f"torch TFLOPS: {tflops / (torch_latency / 1e3)}")

    # Ensure that the latency is not None
    assert latency is not None


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--micro_m", type=int, default=1)
    parser.add_argument("--micro_n", type=int, default=128)
    parser.add_argument("--micro_k", type=int, default=64)
    parser.add_argument("--fake_instr_m", type=int, default=16)
    parser.add_argument("--fake_instr_n", type=int, default=8)
    parser.add_argument("--fake_instr_k", type=int, default=16)
    parser.add_argument("--warp_m", type=int, default=2)
    parser.add_argument("--warp_n", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--block_m", type=int, default=64)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--Atype", type=str, default="int8")
    parser.add_argument("--Wtype", type=str, default="int4")
    parser.add_argument("--Outtype", type=str, default="float16")
    parser.add_argument("--acctype", type=str, default="float32")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--tracekernel", type=str_to_bool, nargs='?',
                        const=True, default=False)

    args = parser.parse_args()

    main(
        M=args.M,
        N=args.N,
        K=args.K,
        micro_size_m=args.micro_m,
        micro_size_n=args.micro_n,
        micro_size_k=args.micro_k,
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
    )
