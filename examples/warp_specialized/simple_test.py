# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tilelang
from tilelang import Profiler
import tilelang.language as T


def test(M, N, block_M, block_N, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def main_specialized(
            A: T.Buffer((M, N), dtype),
            B: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            tx = T.get_thread_binding(0)
            T.copy(A[bx * block_M, by * block_N], A_shared)
            with T.attr(T.iter_var(tx, T.Range(0, 128), "DataPar", ""), "warp_specialized", 1):
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A_shared[i, j] + 1
            with T.attr(T.iter_var(tx, T.Range(128, 256), "DataPar", ""), "warp_specialized", 1):
                for i, j in T.Parallel(block_M, block_N):
                    A_shared[i, j] = A_shared[i, j] + 2
            T.copy(A_shared, B[bx * block_M, by * block_N])

    @T.prim_func
    def main(
            A: T.Buffer((M, N), dtype),
            B: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            tx = T.get_thread_binding(0)
            T.copy(A[bx * block_M, by * block_N], A_shared)
            for i, j in T.Parallel(block_M, block_N):
                A_shared[i, j] = A_shared[i, j] + 1
            for i, j in T.Parallel(block_M, block_N):
                A_shared[i, j] = A_shared[i, j] + 2
            T.copy(A_shared, B[bx * block_M, by * block_N])

    return main_specialized


func = test(1024, 1024, 128, 128)
print(func)
rt_mod, params = tilelang.lower(func)

profiler = Profiler(rt_mod, params, result_idx=[1])

# import torch

# a = torch.randn(1024, 1024).cuda().half()
# c = profiler(a)

# ref_c = a + 3

# print(c)
# print(ref_c)

# torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
# print("All checks pass.")


# # Get CUDA Source
# print(rt_mod.imported_modules[0].get_source())
