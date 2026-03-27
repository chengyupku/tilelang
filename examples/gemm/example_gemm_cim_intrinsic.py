"""
CIM GEMM using intrinsic-level CIMTensorCoreIntrinEmitter.

Based on tilelang-old/examples/gemm/example_gemm_cim_simulate.py.
Changes from old version:
  - dtype controls actual MMA instruction (int8 → m16n8k32, fp16 → m16n8k16)
    instead of always using fp16 MMA proxy
  - Simplified interface: --dtype replaces --Atype/Wtype/Outtype/acctype/force_accum_float
  - fake_instr_m/n/k auto-derived from dtype (still overridable)
  - Tensors declared with real dtype instead of fp16 packing
"""
from tilelang import tvm as tvm
from tvm import DataType
import tilelang
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.intrinsics.mma_cim_macro_generator import (
    TensorCoreIntrinEmitter,)
import torch
from typing import Callable
import argparse

tilelang.disable_cache()

# dtype → (mma_m, mma_n, mma_k, accum_dtype, out_dtype)
MMA_CONFIGS = {
    "float16": (16, 8, 16, "float32", "float16"),
    "int8":    (16, 8, 32, "int32",   "int32"),
}

data_map = {
    "int32": 32,
    "float32": 32,
    "float16": 16,
    "int8": 8,
    "int4": 4,
}


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape
    from tvm import DataType
    row_bits = shape[-1] * DataType(dtype).bits
    # Swizzle when row is at least 512 bits (64 bytes)
    if row_bits < 512:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    })
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
    accum_dtype,
    C_out_dtype,
    stage=2,
    use_shmem_writeback=False,
    ldb=False,
    cim_stride_index=False,
):
    shared_scope = "shared.dyn"

    block_row_warps = block_row_tiles // warp_row_tiles
    block_col_warps = block_col_tiles // warp_col_tiles
    block_M = block_row_tiles
    block_N = block_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)

    # CIM micro-based warp tiling: each MMA = one CIM instruction
    cim_warp_rows = warp_row_tiles // micro_size_m
    cim_warp_cols = warp_col_tiles // micro_size_n

    # MMA Wrapper — ldmatrix uses hardware MMA dims (self.warp_rows/cols),
    # mma/stmatrix use fake_warp_rows/cols (CIM micro-based loop structure).
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=A_in_dtype,
        b_dtype=A_in_dtype,
        accum_dtype=accum_dtype,
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
        fake_warp_rows=cim_warp_rows,
        fake_warp_cols=cim_warp_cols,
        cim_micro_m=micro_size_m,
        cim_micro_n=micro_size_n,
        cim_micro_k=micro_size_k,
        cim_stride_index=cim_stride_index,
    )

    # C_shared tiled in CIM micro units — matches stmatrix indexing
    C_shared_shape = (
        block_M // micro_size_m,
        block_N // micro_size_n,
        micro_size_m,
        micro_size_n,
    )

    @T.prim_func
    def gemm_intrinsics(
            A: T.Tensor(A_shape, A_in_dtype),
            B: T.Tensor(B_shape, A_in_dtype),
            C: T.Tensor((M, N), C_out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, A_in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, A_in_dtype, scope=shared_scope)
            if use_shmem_writeback:
                C_shared = T.alloc_shared(C_shared_shape, A_in_dtype, scope=shared_scope)
            # A_local: warp_m × micro_k / warp_size — the A data per thread per ki.
            # C_local: warp_m × warp_n / warp_size — full output accumulator.
            # mma/stmatrix may OOB-access beyond these — acceptable for latency sim.
            A_local = T.alloc_local((T.max(warp_row_tiles * micro_size_k // 32, 1)), A_in_dtype)
            B_local = T.alloc_local((1), A_in_dtype)
            C_local = T.alloc_local((T.max(warp_row_tiles * warp_col_tiles // 32, 1)), accum_dtype)

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)
            warp_idx = T.get_thread_binding(0) // warp_size % block_col_warps

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # T.sync_threads()

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                # ki steps by CIM micro_k — controls A load frequency
                # Each ki loads the full A_local (warp_m × micro_k tile).
                # When micro_k > mma_k, multiple ldmatrix sub-passes fill A_local.
                k_sub_steps = micro_size_k // fake_instr_k  # micro_k / mma_k
                hw_load_size = mma_emitter.warp_rows * mma_emitter.local_size_a
                for ki in T.serial(0, (block_K // micro_size_k)):

                    for k_sub in T.serial(0, k_sub_steps):
                        mma_emitter.ldmatrix_a(
                            A_local, A_shared,
                            ki * k_sub_steps + k_sub,
                            a_local_offset=k_sub * hw_load_size)

                    if ldb and ki == 0 and ko == 0:
                        mma_emitter.ldmatrix_b(B_local, B_shared, ki)

                    # MMA loop: cim_warp_rows × cim_warp_cols iterations,
                    # each MMA = one CIM instruction
                    mma_emitter.mma(A_local, B_shared, C_local, cim_simulate=True,
                                    offset=(block_N // block_col_warps * warp_idx) * block_K)

            if use_shmem_writeback:
                mma_emitter.stmatrix(C_local, C_shared)

                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = C_shared[
                        i // micro_size_m,
                        j // micro_size_n,
                        i % micro_size_m,
                        j % micro_size_n,
                    ]
            else:
                mma_emitter.stmatrix(C_local, C, pid_m=by, pid_n=bx)

    return gemm_intrinsics


def main(M=8192, N=8192, K=4096, dtype="int8",
         micro_m=16, micro_n=8, micro_k=0,
         warp_m=64, warp_n=64, chunk=64,
         block_M=128, block_N=128, stage=3,
         use_shmem_writeback=False, ldb=True,
         cim_stride_index=False, tracekernel=False):

    mma_m, mma_n, mma_k, accum_dtype, out_dtype = MMA_CONFIGS[dtype]
    # fake_instr = real MMA shape (auto from dtype)
    fake_instr_m, fake_instr_n, fake_instr_k = mma_m, mma_n, mma_k
    if micro_k == 0:
        micro_k = mma_k

    tops = 2 * M * N * K / 1e12
    kernel = tl_matmul(M, N, K,
                       micro_m, micro_n, micro_k,
                       fake_instr_m, fake_instr_n, fake_instr_k,
                       warp_m, warp_n, chunk,
                       block_M, block_N,
                       A_in_dtype=dtype,
                       accum_dtype=accum_dtype,
                       C_out_dtype=out_dtype,
                       stage=stage,
                       use_shmem_writeback=use_shmem_writeback,
                       ldb=ldb,
                       cim_stride_index=cim_stride_index)

    src = kernel.get_kernel_source()
    print(src)

    import torch
    torch_dtype = torch.int8 if dtype == "int8" else torch.float16
    out_torch = torch.int32 if dtype == "int8" else torch.float16
    A = torch.zeros((M, K), dtype=torch_dtype, device="cuda")
    B = torch.zeros((N, K), dtype=torch_dtype, device="cuda")
    C = kernel(A, B)

    if tracekernel:
        return

    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti", n_warmup=50, n_repeat=200)
    unit = "TFlops" if dtype == "float16" else "TOPS"
    print(f"CIM intrinsic ({dtype}, micro={micro_m}/{micro_n}/{micro_k}): "
          f"{latency:.4f} ms, {tops/(latency/1e3):.1f} {unit}")


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
    parser = argparse.ArgumentParser(description="CIM GEMM (intrinsic) with real dtype MMA")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="int8", choices=["float16", "int8"])
    parser.add_argument("--micro_m", type=int, default=4)
    parser.add_argument("--micro_n", type=int, default=32)
    parser.add_argument("--micro_k", type=int, default=32, help="0=auto from dtype")
    parser.add_argument("--warp_m", type=int, default=64)
    parser.add_argument("--warp_n", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--use_shmem_writeback", type=str_to_bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--ldb", type=str_to_bool, nargs='?',
                        const=True, default=True)
    parser.add_argument("--cim_stride_index", type=str_to_bool, nargs='?',
                        const=True, default=False,
                        help="True: CIM micro stride (arch-accurate); False: hw cycling (GPU-fast)")
    parser.add_argument("--tracekernel", type=str_to_bool, nargs='?',
                        const=True, default=False,
                        help="Run kernel once for nsys/ncu tracing, then exit")
    args = parser.parse_args()
    main(args.M, args.N, args.K, args.dtype,
         args.micro_m, args.micro_n, args.micro_k,
         args.warp_m, args.warp_n, args.chunk,
         args.block_M, args.block_N, args.stage,
         use_shmem_writeback=args.use_shmem_writeback, ldb=args.ldb,
         cim_stride_index=args.cim_stride_index,
         tracekernel=args.tracekernel)
