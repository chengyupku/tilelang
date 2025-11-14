from tilelang import tvm as tvm
from tvm import DataType
import tilelang
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.intrinsics.mma_macro_generator import (
    TensorCoreIntrinEmitter,)
from tilelang.transform import simplify_prim_func


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


@tilelang.jit(out_idx=[2])
@simplify_prim_func
def tl_matmul(
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
    use_shmem_writeback,
):
    assert A_in_dtype in [
        "float16",
        "int8",
    ], "Currently only float16 and int8 are supported"
    assert B_in_dtype in [
        "float16",
        "int8",
    ], "Currently only float16 and int8 are supported"
    assert C_out_dtype in [
        "float16",
        "float32",
        "int32",
    ], "Currently only float16, float32 and int32 are supported"

    micro_size_x = micro_size_y = micro_size_k = 16

    if C_out_dtype == "int32":
        micro_size_k = 32

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
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=A_in_dtype,
        b_dtype=B_in_dtype,
        accum_dtype=C_in_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    @T.prim_func
    def gemm_intrinsics(
            A: T.Tensor(A_shape, A_in_dtype),
            B: T.Tensor(B_shape, B_in_dtype),
            C: T.Tensor((M, N), C_out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, A_in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, B_in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, C_out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a), A_in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b), B_in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), C_in_dtype)

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

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(B_local, B_shared, ki)

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_local, C_local)

            if use_shmem_writeback:
                # Perform STMatrix
                mma_emitter.stmatrix(
                    C_local,
                    C_shared,
                )

                # Store shared into global
                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = C_shared[
                        i // micro_size_x,
                        j // micro_size_y,
                        i % micro_size_x,
                        j % micro_size_y,
                    ]
            else:
                mma_emitter.stmatrix(
                    C_local,
                    C,
                    pid_m=by,
                    pid_n=bx,
                )

    return gemm_intrinsics
