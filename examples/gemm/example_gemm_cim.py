"""CIM GEMM using T.gemm(cim_simulate=True) with CIM instruction shape control."""
import torch
import tilelang
import tilelang.language as T
import argparse

tilelang.disable_cache()

DTYPE_MAP = {"float16": T.float16, "int8": T.int8}
ACCUM_MAP = {"float16": T.float32, "int8": T.int32}
DTYPE_BYTES = {"float16": 2, "int8": 1}


def _query_kernel_resources(kernel, threads_per_block, dyn_shmem):
    """Query actual register count and occupancy via CUDA Driver API.

    Exports the compiled cubin, loads it, and queries cuFuncGetAttribute
    and cuOccupancyMaxActiveBlocksPerMultiprocessor.

    Returns: (regs_per_thread, max_active_blocks_per_sm) or (None, None) on failure.
    """
    import ctypes, tempfile, os, re
    try:
        # Ensure CUDA context is initialized
        torch.zeros(1, device='cuda')
        # Export cubin from TVM module
        dev_mod = kernel.artifact.rt_mod.imports_[0]
        cubin_path = os.path.join(tempfile.gettempdir(), '_tilelang_query.cubin')
        dev_mod.write_to_file(cubin_path, fmt='cubin')

        # Find kernel function name from TIR
        tir_src = str(kernel.artifact.device_mod)
        m = re.search(r'def (\w+_kernel)\(', tir_src)
        func_name = m.group(1).encode() if m else b'kernel_kernel'

        # CUDA Driver API
        cuda = ctypes.CDLL('libcuda.so.1')
        CUmodule = ctypes.c_void_p
        CUfunction = ctypes.c_void_p

        module = CUmodule()
        if cuda.cuModuleLoad(ctypes.byref(module), cubin_path.encode()) != 0:
            return None, None

        func = CUfunction()
        if cuda.cuModuleGetFunction(ctypes.byref(func), module, func_name) != 0:
            cuda.cuModuleUnload(module)
            return None, None

        # Query registers per thread
        val = ctypes.c_int()
        cuda.cuFuncGetAttribute(ctypes.byref(val), 4, func)  # CU_FUNC_ATTRIBUTE_NUM_REGS
        regs = val.value

        # Set max dynamic shmem and query occupancy
        cuda.cuFuncSetAttribute(func, 8, dyn_shmem)  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        num_blocks = ctypes.c_int()
        cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(num_blocks), func, threads_per_block, ctypes.c_size_t(dyn_shmem))
        occ = num_blocks.value

        cuda.cuModuleUnload(module)
        os.unlink(cubin_path)
        return regs, occ
    except Exception:
        return None, None


def report_cim_capacity(cim_buffers, num_stages, kernel, threads_per_block):
    """Report CIM macro capacity at multiple levels.

    Args:
        cim_buffers: list of (name, shape_tuple, dtype_str) for buffers in CIM.
        num_stages: pipeline stages (multi-buffering factor).
        kernel: compiled tilelang kernel (for actual shmem/register usage).
        threads_per_block: threads per CTA.
    """
    import re
    props = torch.cuda.get_device_properties(0)

    # Get actual shmem from compiled kernel's TIR func_attr
    actual_shmem = 0
    try:
        tir_src = str(kernel.artifact.device_mod)
        m = re.search(r'"dyn_shared_memory_buf":\s*(\d+)', tir_src)
        if m:
            actual_shmem = int(m.group(1))
    except Exception:
        pass

    # Query actual registers and occupancy from CUDA Driver API
    regs_per_thread, max_cta = _query_kernel_resources(kernel, threads_per_block, actual_shmem)

    n_bufs = len(cim_buffers)
    n_sm = props.multi_processor_count

    print("\n=== CIM Macro Capacity Report ===")

    # Level 1: individual CIM block tiles
    print(f"  [Block Tile]  {n_bufs} CIM buffer(s) per tile:")
    cim_per_tile = 0
    for name, shape, dtype in cim_buffers:
        elem_bytes = DTYPE_BYTES[dtype]
        n_elems = 1
        for d in shape:
            n_elems *= d
        buf_bytes = n_elems * elem_bytes
        cim_per_tile += buf_bytes
        shape_str = "x".join(str(d) for d in shape)
        print(f"    {name:12s}  {shape_str:>12s} x {dtype:>7s} = {buf_bytes:>8d} B ({buf_bytes/1024:.1f} KB)")
    print(f"    {'':12s}  {'tile total':>12s}           = {cim_per_tile:>8d} B ({cim_per_tile/1024:.1f} KB)")

    # Level 2: per CTA (× num_stages for pipeline double/triple buffering)
    cim_per_cta = cim_per_tile * num_stages
    print(f"  [Per CTA]     {n_bufs} tile(s) x {num_stages} stage(s) = {cim_per_cta/1024:.1f} KB CIM")
    print(f"                total shmem (CIM + non-CIM) = {actual_shmem/1024:.1f} KB")

    # Per-SM resource breakdown
    shmem_per_sm = props.shared_memory_per_multiprocessor
    max_threads_per_sm = props.max_threads_per_multi_processor
    regs_per_sm = props.regs_per_multiprocessor
    warps_per_cta = threads_per_block // props.warp_size
    max_warps_per_sm = max_threads_per_sm // props.warp_size

    max_cta_by_shmem = shmem_per_sm // actual_shmem if actual_shmem > 0 else 99
    max_cta_by_warps = max_warps_per_sm // warps_per_cta if warps_per_cta > 0 else 99
    if regs_per_thread is not None and regs_per_thread > 0:
        regs_per_cta = regs_per_thread * threads_per_block
        max_cta_by_regs = regs_per_sm // regs_per_cta
    else:
        regs_per_cta = None
        max_cta_by_regs = 99

    print(f"  [Per SM]      SM resources vs CTA demand → max concurrent CTAs:")
    print(f"    shmem:      {shmem_per_sm/1024:.0f} KB / {actual_shmem/1024:.1f} KB per CTA = {max_cta_by_shmem} CTAs")
    print(f"    warps:      {max_warps_per_sm} / {warps_per_cta} per CTA = {max_cta_by_warps} CTAs")
    if regs_per_thread is not None:
        print(f"    registers:  {regs_per_sm} / ({regs_per_thread} x {threads_per_block}) per CTA = {max_cta_by_regs} CTAs")
    if max_cta is not None:
        print(f"    → {max_cta} concurrent CTA(s) (cuOccupancy)")
        cim_per_sm = cim_per_cta * max_cta
        print(f"    → CIM capacity = {max_cta} x {cim_per_cta/1024:.1f} KB = {cim_per_sm/1024:.1f} KB")

        cim_total = cim_per_sm * n_sm
        print(f"  [Device]      {n_sm} SMs x {cim_per_sm/1024:.1f} KB = {cim_total/1024/1024:.1f} MB CIM total")
    print()


@tilelang.jit(out_idx=[-1])
def matmul_cim(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32,
               num_stages=3, micro_m=0, micro_n=0, micro_k=0, cim_stride_index=False):
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
                       cim_micro_m=micro_m, cim_micro_n=micro_n, cim_micro_k=micro_k,
                       cim_stride_index=cim_stride_index)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel


def main(M=8192, N=8192, K=4096, block_M=128, block_N=128, block_K=64,
         dtype="int8", num_stages=3, micro_m=0, micro_n=0, micro_k=0,
         cim_stride_index=False):
    tl_dtype = DTYPE_MAP[dtype]
    accum_dtype = ACCUM_MAP[dtype]
    tops = 2 * M * N * K / 1e12
    kernel = matmul_cim(M, N, K, block_M, block_N, block_K, tl_dtype, accum_dtype,
                        num_stages, micro_m, micro_n, micro_k, cim_stride_index)

    # CIM capacity report: B matrix lives in CIM
    report_cim_capacity(
        cim_buffers=[("B_shared", (block_N, block_K), dtype)],
        num_stages=num_stages,
        kernel=kernel,
        threads_per_block=128,
    )

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
    parser.add_argument("--cim_stride_index", action="store_true",
                        help="Use CIM micro-based strides for A/C indexing (arch-accurate, slower on GPU)")
    args = parser.parse_args()
    main(args.M, args.N, args.K, args.block_M, args.block_N, args.block_K,
         args.dtype, args.num_stages, args.micro_m, args.micro_n, args.micro_k,
         args.cim_stride_index)
