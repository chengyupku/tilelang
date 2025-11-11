from tilelang import tvm as tvm
from tvm import DataType
import tilelang
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.intrinsics.mma_macro_generator import (
    TensorCoreIntrinEmitter,)
from tilelang.transform import simplify_prim_func

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
    micro_size_x,
    micro_size_y,
    micro_size_k,
    A_in_dtype,
    B_in_dtype,
    C_in_dtype,
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

    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 64
    warp_col_tiles = 64
    # chunk = 32 if in_dtype == "float16" else 64
    chunk = 32
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K * data_map[A_in_dtype] // 16)
    B_shape = (N, K * data_map[B_in_dtype] // 16)
    A_shared_shape = (block_M, block_K * data_map[A_in_dtype] // 16)
    B_shared_shape = (block_N, block_K * data_map[B_in_dtype] // 16)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    # local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size

    warp_rows = warp_row_tiles // micro_size_x  # 64 // 1 = 64
    warp_cols = warp_col_tiles // micro_size_y  # 64 // 32 = 2

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
        fake_instr_m=16,
        fake_instr_n=8,
        fake_instr_k=16,
        fake_warp_rows=warp_rows,
        fake_warp_cols=warp_cols,
    )

    @T.prim_func
    def gemm_intrinsics(
            A: T.Tensor(A_shape, "float16"),
            B: T.Tensor(B_shape, "float16"),
            C: T.Tensor((M, N), "float16"),
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

            # Store shared into global
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]

    return gemm_intrinsics


def main(M=4096,
         N=4096,
         K=4096,
         micro_size_x=1,
         micro_size_y=64,
         micro_size_k=32,
         A_in_dtype="float16",
         B_in_dtype="float16",
         C_in_dtype="float32"):
    kernel = tl_matmul(M, N, K, micro_size_x, micro_size_y, micro_size_k, A_in_dtype, B_in_dtype,
                       C_in_dtype)
    print(kernel.get_kernel_source())
    src_code = kernel.get_kernel_source()
    # src_code is the generated cuda source
    assert src_code is not None

    profiler = kernel.get_profiler()

    latency = profiler.do_bench(profiler.func, warmup=25)

    print(latency)

    # Ensure that the latency is not None
    assert latency is not None


if __name__ == "__main__":
    main(
        M=4096,
        N=4096,
        K=4096,
        micro_size_x=1,
        micro_size_y=64,
        micro_size_k=32,
        A_in_dtype="int8",
        B_in_dtype="int8",
        C_in_dtype="float32")
