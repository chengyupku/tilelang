import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse


@tilelang.jit(out_idx=[3, 4])
def flashattn_fwd(batch, heads, seq_len, dim_qk, dim_v, is_causal, block_M, block_N, groups=1):
    scale = (1.0 / dim_qk)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def flash_fwd(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            K: T.Tensor(k_shape, dtype),  # type: ignore
            V: T.Tensor(v_shape, dtype),  # type: ignore
            Output: T.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=256) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
            K_shared = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})
            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            loop_range = (
                T.ceildiv(
                    (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N:(k + 1) * block_N, by // groups, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                     -T.infinity(acc_s.dtype))
                else:
                    T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(V[bz, k * block_N:(k + 1) * block_N, by // groups, :], V_shared)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(block_M, dim_v):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])

    return flash_fwd


@tilelang.jit(out_idx=[2])
def flashattn_bwd_preprocess(batch, heads, seq_len, dim_v):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim_v]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
            O: T.Tensor(shape, dtype),  # type: ignore
            dO: T.Tensor(shape, dtype),  # type: ignore
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim_v, blk)):
                T.copy(O[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], o)
                T.copy(dO[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk:(by + 1) * blk])

    return flash_bwd_prep


def make_dq_layout(dQ):
    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(dQ.shape,
                    lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2])


@tilelang.jit(out_idx=[1])
def flashattn_bwd_postprocess(batch, heads, seq_len, dim_qk):
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim_qk]
    blk = 64

    @T.prim_func
    def flash_bwd_post(
            dQ: T.Tensor(shape, accum_dtype),  # type: ignore
            dQ_out: T.Tensor(shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.annotate_layout({dQ: make_dq_layout(dQ)})
            T.copy(
                dQ[bz, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk:(bx + 1) * blk, by, :],
            )

    return flash_bwd_post


@tilelang.jit
def flashattn_bwd(batch, heads, seq_len, dim_qk, dim_v, is_causal, block_M, block_N, groups=1):
    sm_scale = (1.0 / dim_qk)**0.5
    scale = (1.0 / dim_qk)**0.5 * 1.44269504  # log2(e)
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def flash_bwd(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            K: T.Tensor(k_shape, dtype),  # type: ignore
            V: T.Tensor(v_shape, dtype),  # type: ignore
            dO: T.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
            dQ: T.Tensor(q_shape, accum_dtype),  # type: ignore
            dK: T.Tensor(k_shape, dtype),  # type: ignore
            dV: T.Tensor(v_shape, dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=128) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim_qk], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim_v], dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            dk = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dq = T.alloc_fragment([block_N, dim_qk], accum_dtype)
            dv_shared = T.alloc_shared([block_N, dim_v], dtype)
            dk_shared = T.alloc_shared([block_N, dim_qk], dtype)

            T.annotate_layout({
                dQ: make_dq_layout(dQ),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
            })

            T.copy(K[bz, by * block_M:(by + 1) * block_M, bx // groups, :], K_shared)
            T.copy(V[bz, by * block_M:(by + 1) * block_M, bx // groups, :], V_shared)
            T.clear(dv)
            T.clear(dk)
            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q)
                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j, qkT[i, j],
                                                   0)
                T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do)
                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta)

                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                for i, j in T.Parallel(block_N, dim_qk):
                    if k * block_N + i < seq_len:
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

            for i, j in T.Parallel(block_M, dim_v):
                T.atomic_add(dV[bz, by * block_M + i, bx // groups, j], dv[i, j])
            for i, j in T.Parallel(block_M, dim_qk):
                T.atomic_add(dK[bz, by * block_M + i, bx // groups, j], dk[i, j])

    return flash_bwd


@torch.compile
class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, groups=1):
        BATCH, N_CTX, H, D_HEAD_QK = q.shape
        D_HEAD_V = v.shape[-1]
        block_M = 128
        block_N = 64
        mod = flashattn_fwd(BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, causal, block_M, block_N, groups)
        o, lse = mod(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        BATCH, N_CTX, H, D_HEAD_QK = q.shape
        HEAD_KV, D_HEAD_V, = v.shape[-2], v.shape[-1]
        groups = H // HEAD_KV

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        do, q, k, v, o = [maybe_contiguous(x) for x in (do, q, k, v, o)]
        block_M = 64
        block_N = 32
        mod_prep = flashattn_bwd_preprocess(BATCH, H, N_CTX, D_HEAD_V)
        mod_post = flashattn_bwd_postprocess(BATCH, H, N_CTX, D_HEAD_QK)
        delta = mod_prep(o, do)
        kernel = flashattn_bwd(BATCH, H, N_CTX, D_HEAD_QK, D_HEAD_V, ctx.causal, block_M, block_N,
                               groups)
        shape_q = [BATCH, N_CTX, H, D_HEAD_QK]
        shape_k = [BATCH, N_CTX, HEAD_KV, D_HEAD_QK]
        shape_v = [BATCH, N_CTX, HEAD_KV, D_HEAD_V]
        dq = torch.zeros(shape_q, dtype=torch.float32, device=q.device)
        dk = torch.zeros(shape_k, dtype=torch.float16, device=q.device)
        dv = torch.zeros(shape_v, dtype=torch.float16, device=q.device)
        kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = mod_post(dq)
        return dq, dk, dv, None, None


attention = _attention.apply


def ref_program(Q, K, V, is_causal, groups=1):
    # Q: [B, T, HQ, D_QK]
    # K: [B, T, HK, D_QK]
    # V: [B, T, HV, D_V]
    # HQ = HKV * groups
    assert Q.size(2) == K.size(
        2) * groups, f"Q.size(2): {Q.size(2)}, K.size(2): {K.size(2)}, groups: {groups}"
    assert Q.size(2) == V.size(
        2) * groups, f"Q.size(2): {Q.size(2)}, V.size(2): {V.size(2)}, groups: {groups}"

    dim_qk = Q.size(-1)
    K = K.repeat_interleave(groups, dim=2)
    V = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim_qk, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    return output


def main(BATCH: int = 1,
         H: int = 32,
         N_CTX: int = 256,
         D_HEAD_QK: int = 192,
         D_HEAD_V: int = 128,
         groups: int = 16,
         causal: bool = False):
    flops_per_qk = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD_QK
    flops_per_v = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD_V
    total_flops = 3 * flops_per_qk + 2 * flops_per_v
    if causal:
        total_flops *= 0.5
    Q = (
        torch.empty(BATCH, N_CTX, H, D_HEAD_QK, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())

    head_kv = H // groups
    K = (
        torch.empty(BATCH, N_CTX, head_kv, D_HEAD_QK, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())
    V = (
        torch.empty(BATCH, N_CTX, head_kv, D_HEAD_V, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())
    dO = (
        torch.empty(BATCH, N_CTX, H, D_HEAD_V, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())
    O = attention(Q, K, V, causal, groups)
    O.backward(dO, retain_graph=True)
    dQ, Q.grad = Q.grad.clone(), None
    dK, K.grad = K.grad.clone(), None
    dV, V.grad = V.grad.clone(), None

    O_ref = ref_program(Q, K, V, causal, groups)
    O_ref.backward(dO, retain_graph=True)
    dQ_ref, Q.grad = Q.grad.clone(), None
    dK_ref, K.grad = K.grad.clone(), None
    dV_ref, V.grad = V.grad.clone(), None

    assert torch.allclose(O, O_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(dV, dV_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dV, dV_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dK, dK_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(dQ, dQ_ref, rtol=1e-2, atol=1e-2)

    def run():
        O_ref.backward(dO, retain_graph=True)

    def run1():
        O.backward(dO, retain_graph=True)

    from tilelang.profiler import do_bench

    latency = do_bench(run, warmup=500)
    print("torch: {:.2f} ms".format(latency))
    print("torch: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = do_bench(run1, warmup=500)
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--h', type=int, default=32, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=1024, help='Context size')
    parser.add_argument('--d_head_qk', type=int, default=192, help='Head dimension for Q/K')
    parser.add_argument('--d_head_v', type=int, default=128, help='Head dimension for V')
    parser.add_argument('--causal', type=bool, default=False, help='Causal flag')
    parser.add_argument('--groups', type=int, default=16, help='groups')
    args = parser.parse_args()
    main(args.batch, args.h, args.n_ctx, args.d_head_qk, args.d_head_v, args.groups, args.causal)
