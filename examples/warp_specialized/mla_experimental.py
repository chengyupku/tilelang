import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from einops import rearrange, einsum

num_split = 1


def flashattn(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim, block_N, block_H):
    scale = (1.0 / (dim + pe_dim))**0.5 * 1.44269504  # log2(e)
    dtype = "float16"
    accum_dtype = "float"
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.macro
    def flash_attn(
            Q: T.Buffer([batch, heads, dim], dtype),
            Q_pe: T.Buffer([batch, heads, pe_dim], dtype),
            KV: T.Buffer([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Buffer([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            Output: T.Buffer([batch, heads, dim], dtype),
    ):
        with T.Kernel(heads // min(block_H, kv_group_num), batch, threads=256) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            scores_scale_shared = T.alloc_shared([block_H], dtype, scope="shared")
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_scale_cast = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            bid = by
            hid = bx
            cur_kv_head = hid // (kv_group_num // block_H)

            T.use_swizzle(10)
            T.annotate_layout({
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
            })

            tx = T.get_thread_binding(0)

            T.copy(Q[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(KV[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, k * block_N:(k + 1) * block_N, cur_kv_head, :], K_pe_shared)

                if tx < 128:
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                    T.gemm(
                        Q_pe_shared,
                        K_pe_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, S_shared)
                    T.copy(scores_scale, scores_scale_shared)
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale_cast[i]
                    T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

                elif tx < 256:
                    T.copy(S_shared, acc_s_cast)
                    T.copy(scores_scale_shared, scores_scale_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale_cast[i]
                    T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bid, hid * VALID_BLOCK_H:(hid + 1) * VALID_BLOCK_H, :])

    @T.prim_func
    def non_split_kv(
            Q: T.Buffer([batch, heads, dim], dtype),
            Q_pe: T.Buffer([batch, heads, pe_dim], dtype),
            KV: T.Buffer([batch, seqlen_kv, kv_head_num, dim], dtype),
            K_pe: T.Buffer([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
            Output: T.Buffer([batch, heads, dim], dtype),
    ):
        flash_attn(Q, Q_pe, KV, K_pe, Output)

    return non_split_kv

def ref_program(q, q_pe, kv, k_pe):
    #     """
    #     Inputs:
    #     - q (Tensor): [batch, heads, dim]
    #     - q_pe (Tensor): [batch, heads, pe_dim]
    #     - kv (Tensor): [batch, seqlen_kv, kv_head_num, dim]
    #     - k_pe (Tensor): [batch, seqlen_kv, kv_head_num, pe_dim]
    #     - glse (Tensor): [batch, heads, num_split]
    #     - Output_partial (Tensor): [batch, heads, num_split, dim]
    #     Outputs:
    #     - output (Tensor): [batch, heads, dim]
    #     """
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    num_head_groups = q.shape[1] // kv.shape[2]
    scale = (dim + pe_dim)**0.5
    q = rearrange(
        q, 'b (h g) d -> b g h d', g=num_head_groups)  # [batch_size, num_head_groups, groups, dim]

    q_pe = rearrange(
        q_pe, 'b (h g) d -> b g h d',
        g=num_head_groups)  # [batch_size, num_head_groups, groups, pe_dim]

    kv = rearrange(kv, 'b n h d -> b h n d')  # [batch_size, groups, seqlen_kv, dim]

    k_pe = rearrange(k_pe, 'b n h d -> b h n d')  # [batch_size, num_head_groups, groups, pe_dim]

    query = torch.concat([q, q_pe], dim=-1)
    key = torch.concat([kv, k_pe], dim=-1)

    scores = einsum(
        query, key,
        'b g h d, b h s d -> b g h s')  # [batch_size, num_head_groups, groups, seqlen_kv]

    attention = F.softmax(
        scores / scale, dim=-1)  # [batch_size, num_head_groups, groups, seqlen_kv]

    out = einsum(attention, kv,
                 'b g h s, b h s d -> b g h d')  # [batch_size, num_head_groups, groups, dim]
    out = rearrange(out, 'b g h d -> b (h g) d')  # [batch_size, heads, dim]
    return out

if __name__ == "__main__":
    BATCH, H_Q, KV_H, KV_CTX, D_HEAD, DPE = 128, 128, 1, 8192, 512, 64
    qk_flops = 2 * BATCH * H_Q * KV_CTX * (D_HEAD + DPE)
    pv_flops = 2 * BATCH * H_Q * KV_CTX * D_HEAD
    total_flops = qk_flops + pv_flops
    BLOCK_N = 64  # if D_HEAD <= 128 else 32
    BLOCK_H = 64

    program = flashattn(BATCH, H_Q, KV_H, KV_CTX, D_HEAD, DPE, BLOCK_N, BLOCK_H)
    mod, params = tilelang.lower(program)
    mod = tilelang.Profiler(mod, params, [4], tilelang.TensorSupplyType.Normal)
    # mod.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    # print("All close")
    # latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="torch")
    # print("Tile-lang: {:.2f} ms".format(latency))
    # print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    
# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT License.

# import tilelang
# from tilelang import Profiler
# import tilelang.language as T


# def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

#     @T.prim_func
#     def main(
#             A: T.Buffer((M, K), dtype),
#             B: T.Buffer((K, N), dtype),
#             C: T.Buffer((M, N), dtype),
#     ):
#         with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
#             A_shared = T.alloc_shared((block_M, block_K), dtype)
#             B_shared = T.alloc_shared((block_K, block_N), dtype)
#             C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

#             tx = T.get_thread_binding(0)

#             T.clear(C_local)
#             for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
#                 T.copy(A[by * block_M, k * block_K], A_shared)
#                 T.copy(B[k * block_K, bx * block_N], B_shared)
#                 # T.gemm(A_shared, B_shared, C_local)
#                 if tx < 128:
#                     T.gemm(A_shared, B_shared, C_local)
#                 elif tx < 256:
#                     pass
#                     # T.gemm(A_shared, B_shared, C_local)

#             T.copy(C_local, C[by * block_M, bx * block_N])

#     return main


# func = matmul(1024, 1024, 1024, 128, 128, 32)

# print(func)

# rt_mod, params = tilelang.lower(func)

# profiler = Profiler(rt_mod, params, result_idx=[2])

# import torch

# a = torch.randn(1024, 1024).cuda().half()
# b = torch.randn(1024, 1024).cuda().half()

# c = profiler(a, b)

# ref_c = a @ b

# print(c)
# print(ref_c)

# torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

# # Get CUDA Source
# print(rt_mod.imported_modules[0].get_source())
