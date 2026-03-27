"""CIM GEMM using T.gemm(cim_simulate=True) with CIM instruction shape control."""
import torch
import tilelang
import tilelang.language as T
import argparse

tilelang.disable_cache()

DTYPE_MAP = {"float16": T.float16, "int8": T.int8}
ACCUM_MAP = {"float16": T.float32, "int8": T.int32}


@tilelang.jit(out_idx=[-1])
def matmul_cim(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32,
               num_stages=3, micro_m=0, micro_n=0, micro_k=0):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype),
               C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True, cim_simulate=True,
                       cim_micro_m=micro_m, cim_micro_n=micro_n, cim_micro_k=micro_k)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel


def main(M=8192, N=8192, K=4096, block_M=128, block_N=128, block_K=64,
         dtype="int8", num_stages=3, micro_m=0, micro_n=0, micro_k=0):
    tl_dtype = DTYPE_MAP[dtype]
    accum_dtype = ACCUM_MAP[dtype]
    tops = 2 * M * N * K / 1e12
    kernel = matmul_cim(M, N, K, block_M, block_N, block_K, tl_dtype, accum_dtype,
                        num_stages, micro_m, micro_n, micro_k)
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(n_warmup=50, n_repeat=200)
    unit = "TFlops" if dtype == "float16" else "TOPS"
    micro_str = f"micro={micro_m}/{micro_n}/{micro_k}" if any([micro_m, micro_n, micro_k]) else "micro=default"
    print(f"CIM GEMM ({dtype}, {micro_str}): {latency:.4f} ms, {tops/(latency/1e3):.1f} {unit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIM GEMM with CIM instruction shape control")
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--block_K", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "int8"])
    parser.add_argument("--num_stages", type=int, default=3)
    parser.add_argument("--micro_m", type=int, default=16, help="CIM instruction M dim (0=hardware default)")
    parser.add_argument("--micro_n", type=int, default=8, help="CIM instruction N dim (0=hardware default)")
    parser.add_argument("--micro_k", type=int, default=16, help="CIM instruction K dim (0=hardware default)")
    args = parser.parse_args()
    main(args.M, args.N, args.K, args.block_M, args.block_N, args.block_K,
         args.dtype, args.num_stages, args.micro_m, args.micro_n, args.micro_k)
