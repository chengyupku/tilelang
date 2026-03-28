"""Kernel resource reporting utilities for tilelang examples.

Provides:
  - query_kernel_resources(): regs/thread + occupancy via CUDA Driver API
  - report_kernel_resources(): print shmem, warps, regs, CTAs/SM
  - report_cim_capacity(): print CIM tile/CTA/SM/device capacity breakdown

All data is obtained through proper software interfaces (TVM IR API + CUDA
Driver API), not string matching.
"""
import ctypes
import tempfile
import os
import torch

DTYPE_BYTES = {"float16": 2, "int8": 1}


def _get_device_func(kernel):
    """Get the first device function and its name from the compiled kernel's TIR module.

    Returns:
        (func_name_str, tir_func) or (None, None) on failure.
    """
    try:
        mod = kernel.artifact.device_mod
        for gv in mod.functions:
            return gv.name_hint, mod.functions[gv]
    except Exception:
        pass
    return None, None


def get_kernel_shmem(kernel):
    """Get dynamic shared memory size (bytes) from compiled kernel's TIR func_attr."""
    _, func = _get_device_func(kernel)
    if func is not None and func.attrs and "dyn_shared_memory_buf" in func.attrs:
        return int(func.attrs["dyn_shared_memory_buf"])
    return 0


def query_kernel_resources(kernel, threads_per_block, dyn_shmem):
    """Query actual register count and occupancy via CUDA Driver API.

    Exports the compiled cubin, loads it with cuModuleLoad, and queries
    cuFuncGetAttribute + cuOccupancyMaxActiveBlocksPerMultiprocessor.

    Returns:
        (regs_per_thread, max_active_blocks_per_sm) or (None, None) on failure.
    """
    try:
        torch.zeros(1, device="cuda")  # ensure CUDA context

        # Export cubin
        dev_mod = kernel.artifact.rt_mod.imports_[0]
        cubin_path = os.path.join(tempfile.gettempdir(), "_tilelang_query.cubin")
        dev_mod.write_to_file(cubin_path, fmt="cubin")

        # Get kernel function name from TIR module (not string matching)
        func_name, _ = _get_device_func(kernel)
        if func_name is None:
            return None, None
        func_name_bytes = func_name.encode()

        # CUDA Driver API
        cuda = ctypes.CDLL("libcuda.so.1")
        module = ctypes.c_void_p()
        if cuda.cuModuleLoad(ctypes.byref(module), cubin_path.encode()) != 0:
            return None, None
        func = ctypes.c_void_p()
        if cuda.cuModuleGetFunction(ctypes.byref(func), module, func_name_bytes) != 0:
            cuda.cuModuleUnload(module)
            return None, None

        val = ctypes.c_int()
        cuda.cuFuncGetAttribute(ctypes.byref(val), 4, func)  # CU_FUNC_ATTRIBUTE_NUM_REGS
        regs = val.value

        cuda.cuFuncSetAttribute(func, 8, dyn_shmem)  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        num_blocks = ctypes.c_int()
        cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(num_blocks), func, threads_per_block, ctypes.c_size_t(dyn_shmem)
        )
        occ = num_blocks.value

        cuda.cuModuleUnload(module)
        os.unlink(cubin_path)
        return regs, occ
    except Exception:
        return None, None


def report_kernel_resources(kernel, threads_per_block):
    """Print kernel resource usage and SM occupancy."""
    props = torch.cuda.get_device_properties(0)
    actual_shmem = get_kernel_shmem(kernel)
    regs_per_thread, max_cta = query_kernel_resources(kernel, threads_per_block, actual_shmem)

    shmem_per_sm = props.shared_memory_per_multiprocessor
    regs_per_sm = props.regs_per_multiprocessor
    max_warps_per_sm = props.max_threads_per_multi_processor // props.warp_size
    warps_per_cta = threads_per_block // props.warp_size

    print("\n=== Kernel Resource Report ===")
    print(f"  shmem per CTA:    {actual_shmem / 1024:.1f} KB")
    print(f"  warps per CTA:    {warps_per_cta}")
    if regs_per_thread is not None:
        print(f"  regs per thread:  {regs_per_thread}")
    print(f"  SM resources vs CTA demand:")
    max_cta_by_shmem = shmem_per_sm // actual_shmem if actual_shmem > 0 else 99
    max_cta_by_warps = max_warps_per_sm // warps_per_cta
    print(f"    shmem:     {shmem_per_sm / 1024:.0f} KB / {actual_shmem / 1024:.1f} KB = {max_cta_by_shmem} CTAs")
    print(f"    warps:     {max_warps_per_sm} / {warps_per_cta} = {max_cta_by_warps} CTAs")
    if regs_per_thread is not None:
        max_cta_by_regs = regs_per_sm // (regs_per_thread * threads_per_block)
        print(f"    registers: {regs_per_sm} / ({regs_per_thread} x {threads_per_block}) = {max_cta_by_regs} CTAs")
    if max_cta is not None:
        print(f"    -> {max_cta} concurrent CTA(s) per SM (cuOccupancy)")
    print()


def report_cim_capacity(cim_buffers, num_stages, kernel, threads_per_block):
    """Print CIM macro capacity at block tile / CTA / SM / device levels.

    Args:
        cim_buffers: list of (name, shape_tuple, dtype_str) for buffers in CIM.
        num_stages: pipeline stages (multi-buffering factor).
        kernel: compiled tilelang kernel.
        threads_per_block: threads per CTA.
    """
    props = torch.cuda.get_device_properties(0)
    actual_shmem = get_kernel_shmem(kernel)
    regs_per_thread, max_cta = query_kernel_resources(kernel, threads_per_block, actual_shmem)

    n_bufs = len(cim_buffers)
    n_sm = props.multi_processor_count
    shmem_per_sm = props.shared_memory_per_multiprocessor
    regs_per_sm = props.regs_per_multiprocessor
    max_warps_per_sm = props.max_threads_per_multi_processor // props.warp_size
    warps_per_cta = threads_per_block // props.warp_size

    print("\n=== CIM Macro Capacity Report ===")

    # Level 1: block tiles
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
        print(f"    {name:12s}  {shape_str:>12s} x {dtype:>7s} = {buf_bytes:>8d} B ({buf_bytes / 1024:.1f} KB)")
    print(f"    {'':12s}  {'tile total':>12s}           = {cim_per_tile:>8d} B ({cim_per_tile / 1024:.1f} KB)")

    # Level 2: per CTA
    cim_per_cta = cim_per_tile * num_stages
    print(f"  [Per CTA]     {n_bufs} tile(s) x {num_stages} stage(s) = {cim_per_cta / 1024:.1f} KB CIM")
    print(f"                total shmem (CIM + non-CIM) = {actual_shmem / 1024:.1f} KB")

    # Level 3: per SM
    max_cta_by_shmem = shmem_per_sm // actual_shmem if actual_shmem > 0 else 99
    max_cta_by_warps = max_warps_per_sm // warps_per_cta if warps_per_cta > 0 else 99
    if regs_per_thread is not None and regs_per_thread > 0:
        max_cta_by_regs = regs_per_sm // (regs_per_thread * threads_per_block)
    else:
        max_cta_by_regs = 99

    print(f"  [Per SM]      SM resources vs CTA demand:")
    print(f"    shmem:      {shmem_per_sm / 1024:.0f} KB / {actual_shmem / 1024:.1f} KB = {max_cta_by_shmem} CTAs")
    print(f"    warps:      {max_warps_per_sm} / {warps_per_cta} = {max_cta_by_warps} CTAs")
    if regs_per_thread is not None:
        print(f"    registers:  {regs_per_sm} / ({regs_per_thread} x {threads_per_block}) = {max_cta_by_regs} CTAs")
    if max_cta is not None:
        print(f"    -> {max_cta} concurrent CTA(s) per SM (cuOccupancy)")
        cim_per_sm = cim_per_cta * max_cta
        print(f"    -> CIM capacity = {max_cta} x {cim_per_cta / 1024:.1f} KB = {cim_per_sm / 1024:.1f} KB")
        cim_total = cim_per_sm * n_sm
        print(f"  [Device]      {n_sm} SMs x {cim_per_sm / 1024:.1f} KB = {cim_total / 1024 / 1024:.1f} MB CIM total")
    print()
