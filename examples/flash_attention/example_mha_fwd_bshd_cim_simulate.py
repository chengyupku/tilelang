"""
CIM-simulated Flash Attention (BSHD layout).

Both K and V are treated as "in-memory" (CIM): they stay in shared memory and
are fed to the MMA instruction directly via address, simulating a weight-
stationary CIM architecture.  Results are numerically INCORRECT -- this is a
latency-only benchmark.
"""

from tilelang import tvm as tvm
from tvm import DataType
import tilelang
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.intrinsics.mma_cim_macro_generator import (
    TensorCoreIntrinEmitter,)
import torch

tilelang.disable_cache()

# ── helpers ──────────────────────────────────────────────────────────────────


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




# ── kernel ───────────────────────────────────────────────────────────────────


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def flashattn_cim(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    # CIM micro-tile sizes (used for local buffer sizing)
    micro_size_m,
    micro_size_n,
    micro_size_k,
    # Fake PTX instruction shape
    fake_instr_m,
    fake_instr_n,
    fake_instr_k,
    # Tiling
    block_M,
    block_N,
    # Warp config
    block_row_warps,
    block_col_warps,
    stage=2,
    ldb=False,
):
    dtype = "float16"
    accum_dtype = "float32"
    shared_scope = "shared.dyn"
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)

    warp_size = 32
    threads = warp_size * block_row_warps * block_col_warps

    # ── derived tiling ───────────────────────────────────────────────────
    warp_row_tiles = block_M // block_row_warps

    # MMA0 (Q * K^T): output [block_M, block_N], reduction over dim
    warp_col_tiles_0 = block_N // block_col_warps
    chunk_0 = dim

    # MMA1 (S * V): output [block_M, dim], reduction over block_N
    warp_col_tiles_1 = dim // block_col_warps
    chunk_1 = block_N

    # Fake warp tiling for the CIM emitter
    fake_warp_rows = warp_row_tiles // micro_size_m
    fake_warp_cols_0 = warp_col_tiles_0 // micro_size_n
    fake_warp_cols_1 = warp_col_tiles_1 // micro_size_n

    # Per-thread local sizes (driven by micro_size, same for both stages)
    local_size_a = (micro_size_m * micro_size_k) // warp_size
    local_size_c = (micro_size_m * micro_size_n) // warp_size

    # ── MMA emitters ─────────────────────────────────────────────────────
    #
    # MMA0: Q[block_M, dim] * K^T[block_N, dim]  → acc_s[block_M, block_N]
    #       A = Q (shared),  B = K (shared, CIM),  b_transposed = True
    #
    mma0 = TensorCoreIntrinEmitter(
        a_dtype=dtype, b_dtype=dtype, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True,
        block_row_warps=block_row_warps, block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles, warp_col_tiles=warp_col_tiles_0,
        chunk=chunk_0,
        fake_instr_m=fake_instr_m, fake_instr_n=fake_instr_n,
        fake_instr_k=fake_instr_k,
        fake_warp_rows=fake_warp_rows, fake_warp_cols=fake_warp_cols_0,
    )

    # MMA1: S[block_M, block_N] * V[block_N, dim]  → acc_o[block_M, dim]
    #       A = S (fragment, Route B),  B = V (shared, CIM),  b_transposed = True
    #       Route B: acc_s_cast fragment is passed directly to mma as A operand,
    #       skipping the S_shared round-trip.  fake_warp_rows=1 matches the
    #       fragment per-ki stride (local_size_a elements per ki step).
    #
    mma1 = TensorCoreIntrinEmitter(
        a_dtype=dtype, b_dtype=dtype, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True,
        block_row_warps=block_row_warps, block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles, warp_col_tiles=warp_col_tiles_1,
        chunk=chunk_1,
        fake_instr_m=fake_instr_m, fake_instr_n=fake_instr_n,
        fake_instr_k=fake_instr_k,
        fake_warp_rows=1, fake_warp_cols=fake_warp_cols_1,
    )

    # ── buffer shapes ────────────────────────────────────────────────────
    # For fp16 the packing factor (data_map // 16) is 1, so all dims are natural.
    a_local_size = T.max(fake_warp_rows * local_size_a, 1)

    in_shape = (batch, seq_len, heads, dim)

    @T.prim_func
    def main(
        Q: T.Tensor(in_shape, dtype),
        K: T.Tensor(in_shape, dtype),
        V: T.Tensor(in_shape, dtype),
        Output: T.Tensor(in_shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_len, block_M), heads, batch, threads=threads
        ) as (bx, by, bz):

            # ── shared memory ────────────────────────────────────────────
            Q_shared = T.alloc_shared([block_M, dim], dtype, scope=shared_scope)
            K_shared = T.alloc_shared([block_N, dim], dtype, scope=shared_scope)
            V_shared = T.alloc_shared([block_N, dim], dtype, scope=shared_scope)
            O_shared = T.alloc_shared([block_M, dim], dtype, scope=shared_scope)

            # ── local / fragment buffers ─────────────────────────────────
            A_local_0 = T.alloc_local(a_local_size, dtype)
            B_local = T.alloc_local(1, dtype)  # placeholder (CIM skips B load)

            # Fragment accumulators (shaped for softmax / rescale ops)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)

            # Softmax state
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.annotate_layout({
                Q_shared: make_swizzle_layout(Q_shared),
                K_shared: make_swizzle_layout(K_shared),
                V_shared: make_swizzle_layout(V_shared),
            })

            T.use_swizzle(panel_size=10)
            warp_idx = T.get_thread_binding(0) // warp_size % block_col_warps

            # ── initialise ───────────────────────────────────────────────
            T.clear(acc_o)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # Load Q once (stays resident across all k-iterations)
            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_N),
                    T.ceildiv((bx + 1) * block_M, block_N),
                )
                if is_causal
                else T.ceildiv(seq_len, block_N)
            )

            for ko in T.Pipelined(loop_range, num_stages=stage):

                # ═══════════════════════════════════════════════════════
                # MMA0:  Q * K^T  →  acc_s  [block_M × block_N]
                # K stays in shared (CIM); Q loaded via ldmatrix_a
                # ═══════════════════════════════════════════════════════
                T.copy(K[bz, ko * block_N:(ko + 1) * block_N, by, :], K_shared)

                T.clear(acc_s)
                for ki in T.serial(chunk_0 // micro_size_k):
                    mma0.ldmatrix_a(A_local_0, Q_shared, ki)
                    if ldb and ki == 0 and ko == 0:
                        mma0.ldmatrix_b(B_local, K_shared, ki)
                    mma0.mma(
                        A_local_0, K_shared, acc_s,
                        cim_simulate=True,
                        offset=(warp_col_tiles_0 * warp_idx) * dim,
                    )

                # ═══════════════════════════════════════════════════════
                # Online softmax
                # ═══════════════════════════════════════════════════════
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_M + i >= ko * block_N + j,
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                # ═══════════════════════════════════════════════════════
                # Rescale previous acc_o
                # ═══════════════════════════════════════════════════════
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                # ═══════════════════════════════════════════════════════
                # MMA1:  S * V  →  acc_o  [block_M × dim]
                # Route B: acc_s_cast fragment passed directly as A operand
                # V stays in shared (CIM)
                # ═══════════════════════════════════════════════════════
                T.copy(V[bz, ko * block_N:(ko + 1) * block_N, by, :], V_shared)

                for ki in T.serial(chunk_1 // micro_size_k):
                    mma1.mma(
                        acc_s_cast, V_shared, acc_o,
                        k_inner=ki,
                        cim_simulate=True,
                        offset=(warp_col_tiles_1 * warp_idx) * block_N,
                    )

            # ── finalise: acc_o / logsum → global output ─────────────────
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

    return main


# ── main ─────────────────────────────────────────────────────────────────────


def main(
    batch=8,
    heads=32,
    seq_len=4096,
    dim=128,
    is_causal=False,
    micro_size_m=16,
    micro_size_n=8,
    micro_size_k=16,
    fake_instr_m=16,
    fake_instr_n=8,
    fake_instr_k=16,
    block_M=128,
    block_N=128,
    block_row_warps=4,
    block_col_warps=2,
    stage=1,
    ldb=False,
):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    kernel = flashattn_cim(
        batch, heads, seq_len, dim, is_causal,
        micro_size_m, micro_size_n, micro_size_k,
        fake_instr_m, fake_instr_n, fake_instr_k,
        block_M, block_N,
        block_row_warps, block_col_warps,
        stage=stage,
        ldb=ldb,
    )

    # print(kernel.get_kernel_source())

    profiler = kernel.get_profiler()
    latency = profiler.do_bench(n_warmup=50, n_repeat=200)
    print(f"CIM Flash Attention latency: {latency:.4f} ms")
    print(f"CIM Flash Attention TFLOPs:  {total_flops / (latency / 1e3) / 1e12:.2f}")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CIM-simulated Flash Attention benchmark (latency only)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--is_causal", type=str_to_bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--micro_m", type=int, default=16)
    parser.add_argument("--micro_n", type=int, default=8)
    parser.add_argument("--micro_k", type=int, default=16)
    parser.add_argument("--fake_instr_m", type=int, default=16)
    parser.add_argument("--fake_instr_n", type=int, default=8)
    parser.add_argument("--fake_instr_k", type=int, default=16)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--block_row_warps", type=int, default=4)
    parser.add_argument("--block_col_warps", type=int, default=2)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--ldb", type=str_to_bool, nargs='?',
                        const=True, default=False)

    args = parser.parse_args()

    main(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        dim=args.dim,
        is_causal=args.is_causal,
        micro_size_m=args.micro_m,
        micro_size_n=args.micro_n,
        micro_size_k=args.micro_k,
        fake_instr_m=args.fake_instr_m,
        fake_instr_n=args.fake_instr_n,
        fake_instr_k=args.fake_instr_k,
        block_M=args.block_M,
        block_N=args.block_N,
        block_row_warps=args.block_row_warps,
        block_col_warps=args.block_col_warps,
        stage=args.stage,
        ldb=args.ldb,
    )
