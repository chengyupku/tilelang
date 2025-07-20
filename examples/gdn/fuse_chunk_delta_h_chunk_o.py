# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from inspect import getattr_static
import tilelang
import tilelang.language as T
import sys
import os

# Add your fla repository path to sys.path
# You can set the FLA_REPO_PATH environment variable to point to your fla repository
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae

# sys.path.insert(0, "/root/workspace/flash-linear-attention")
# import fla
# print(fla.__file__)

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
import torch
from tilelang.engine.callback import register_cuda_postproc_callback



torch.random.manual_seed(0)

tilelang.disable_cache()

def ref_program(q, k, w, u, g, initial_state, output_final_state, cu_seqlens, scale):
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state

def prepare_input(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
):
    BS = S // chunk_size
    # torch.empty(*shape, device=device, dtype=dtype).normal_(-1.0, 1.0)
    ext = 0.1
    Q = torch.empty(B, S, H, DK, dtype=input_dtype).cuda().normal_(-ext, ext)
    K = torch.empty(B, S, H, DK, dtype=input_dtype).cuda().normal_(-ext, ext)
    W = torch.empty(B, S, H, DK, dtype=input_dtype).cuda().normal_(-ext, ext)
    U = torch.empty(B, S, H, DV, dtype=input_dtype).cuda().normal_(-ext, ext)
    G = torch.empty(B, S, H, dtype=gate_dtype).cuda().normal_(-ext, ext)
    # G = torch.ones(B, S, H, dtype=gate_dtype).cuda() * 2
    initial_state = torch.empty(B, H, DK, DV, dtype=input_dtype).cuda().normal_(-ext, ext)
    return Q, K, W, U, G, initial_state


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
    state_dtype,
):
    BS = S // chunk_size
    h = torch.empty(B, BS, H, DK, DV, dtype=output_dtype).cuda()
    final_state = torch.empty(B, H, DK, DV, dtype=state_dtype).cuda()
    V_new = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return h, final_state, V_new


def tilelang_chunk_gated_delta_rule_chunk_o_fwd(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    use_g=True,
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    scale=1.0,
    # kernel config
    block_DK=64,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    block_S = chunk_size
    # Should support cu_seqlen
    BS = S // block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    W_shape = (B, S, H, DK)
    U_shape = (B, S, H, DV)
    O_shape = (B, S, H, DV)
    G_shape = (B, S, H)
    h_shape = (B, BS, H, DK, DV)
    initial_state_shape = (B, H, DK, DV)
    final_state_shape = (B, H, DK, DV)

    @T.prim_func
    def kernel(
        Q: T.Tensor(K_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        U: T.Tensor(U_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        initial_state: T.Tensor(initial_state_shape, dtype=input_dtype),
        h: T.Tensor(h_shape, dtype=output_dtype),
        final_state: T.Tensor(final_state_shape, dtype=state_dtype),
        V_new: T.Tensor(V_shape, dtype=output_dtype),
        O: T.Tensor(O_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H
            Q_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            O_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)

            b_h_shared = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            b_h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)

            U_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            G_last_local = T.alloc_local((1), dtype=gate_dtype)
            G_shared = T.alloc_shared((block_S, block_DV), dtype=gate_dtype, scope="shared")
            G_fragment = T.alloc_fragment((block_S, block_DV), dtype=gate_dtype)
            A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            G_diff_local = T.alloc_fragment((block_S, block_S), dtype=gate_dtype)

            # T.no_set_max_nreg()

            T.annotate_layout({
                b_h_shared: tilelang.layout.make_swizzled_layout(b_h_shared),
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                V_new_shared: tilelang.layout.make_swizzled_layout(V_new_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                # G_shared: tilelang.layout.make_swizzled_layout(G_shared),
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                # G_diff_local: T.Fragment(G_diff_local.shape, forward_thread_fn=lambda i: i // 16 * 32 + i % 8 * 4),
            })

            if use_initial_state:
                T.copy(initial_state[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV], b_h_shared)
                T.copy(b_h_shared, b_h_fragment)
            else:
                T.clear(b_h_fragment)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # Store previous result to the hidden tensor, like the epilogue
                T.copy(b_h_shared, h[bb, i_s, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

                # Cast the intermediate result to input dtype
                # T.copy(b_h, b_h_input_dtype)

                # Recurrence
                # T.clear(V_new_fragment)
                T.copy(W[bb, i_s * block_S:(i_s + 1) * block_S, bh, 0:DK], W_shared)
                T.gemm(W_shared, b_h_shared, V_new_fragment, clear_accum=True)

                # U - W * S
                T.copy(U[bb, i_s * block_S:(i_s + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV], U_shared)
                T.copy(U_shared, U_fragment)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    V_new_fragment[i_s2, i_v] = - V_new_fragment[i_s2, i_v] + U_fragment[i_s2, i_v]

                # Save V_new
                if save_new_value:
                    T.copy(V_new_fragment, V_shared)
                    T.copy(V_shared, V_new[bb, i_s * block_S:(i_s + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV])

                T.copy(K[bb, i_s * block_S:(i_s + 1) * block_S, bh, 0:DK], K_shared)
                # use_g
                if use_g:
                    G_last_local[0] = G[bb, (i_s + 1) * block_S - 1, bh]
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        G_shared[i_s2, i_v] = G[bb, i_s * block_S + i_s2, bh]
                    T.copy(G_shared, G_fragment)
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        # with T.If(G_last_local[0] - G_shared[i_s2, i_v] <= 0):
                        with T.If(G_last_local[0] - G_fragment[i_s2, i_v] <= 0):
                            with T.Then():
                                V_new_fragment[i_s2, i_v] = V_new_fragment[i_s2, i_v] * T.exp(G_last_local[0] - G_fragment[i_s2, i_v])
                                # V_new_fragment[i_s2, i_v] = V_new_fragment[i_s2, i_v] * T.exp(G_last_local[0] - G_shared[i_s2, i_v])
                            with T.Else():
                                V_new_fragment[i_s2, i_v] = 0
                    G_last_local[0] = T.exp(G_last_local[0])
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_h_fragment[i_k, i_v] *= G_last_local[0]
                        # b_h_shared[i_k, i_v] *= G_last_local[0]

                    # T.copy(b_h_shared, b_h_fragment)

                # Update intermediate results
                T.copy(V_new_fragment, V_new_shared)
                T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)

                T.clear(A_fragment)
                T.clear(O_fragment)
                T.copy(Q[bb, i_s * block_S:(i_s + 1) * block_S, bh, 0:DK], Q_shared)
                T.gemm(Q_shared, b_h_shared, O_fragment)
                T.gemm(Q_shared, K_shared, A_fragment, transpose_B=True)

                T.copy(b_h_fragment, b_h_shared)

                if use_g:
                    for p_s, p_v in T.Parallel(block_S, block_DV):
                        O_fragment[p_s, p_v] = O_fragment[p_s, p_v] * T.exp(G[bb, i_s * block_S + p_s, bh])
                    for p_s1, p_s2 in T.Parallel(block_S, block_S):
                        G_diff_local[p_s1, p_s2] = G[bb, i_s * block_S + p_s1, bh] - G[bb, i_s * block_S + p_s2, bh]
                    for p_s1, p_s2 in T.Parallel(block_S, block_S):
                        with T.If(G_diff_local[p_s1, p_s2] <= 0):
                            with T.Then():
                                A_fragment[p_s1, p_s2] *= T.exp(G_diff_local[p_s1, p_s2])
                            with T.Else():
                                A_fragment[p_s1, p_s2] = 0

                for p_s1, p_s2 in T.Parallel(block_S, block_S):
                    with T.If(p_s1 < p_s2):
                        with T.Then():
                            A_fragment[p_s1, p_s2] = 0

                T.copy(A_fragment, A_shared)
                T.gemm(A_shared, V_shared, O_fragment)

                for p_s, p_v in T.Parallel(block_S, block_DV):
                    O_fragment[p_s, p_v] = O_fragment[p_s, p_v] * scale

                T.copy(O_fragment, O_shared)
                T.copy(O_shared, O[bb, i_s * block_S:(i_s + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV])

            # Save final state
            if store_final_state:
                T.copy(b_h_fragment, final_state[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

    return kernel


def do_bench(fn, *args, warmup=10, rep=10, **kwargs):
    """
    Do benchmark for a function.
    """
    import time
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    for i in range(warmup):
        fn(*args, **kwargs)


    start_time = time.time()
    torch.cuda.synchronize()
    for i in range(rep):
        start_event[i].record()
        fn(*args, **kwargs)
        end_event[i].record()
    torch.cuda.synchronize()
    end_time = time.time()

    # Record clocks
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )

    # return (end_time - start_time) * 1000 / rep
    return times.mean().item()

def run_test(
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    use_g=True,
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    scale=1.0,
    block_DK=64,
    block_DV=32,
    threads=128,
    num_stages=0,
):
    Q, K, W, U, G, initial_state = prepare_input(
        B, S, H, DK, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, output_dtype), getattr(torch, accum_dtype), getattr(torch, gate_dtype)
    )
    # h_ref, final_state_ref, V_new_ref = prepare_output(
    #     B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, state_dtype)
    # )
    # h_tilelang, final_state_tilelang, V_new_tilelang = prepare_output(
    #     B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, state_dtype)
    # )

    # fla ref
    # h_ref, V_new_ref, final_state_ref = chunk_gated_delta_rule_fwd_h(
    #     K, W, U, None, initial_state, store_final_state, chunk_size, save_new_value
    # )
    o_ref, final_state_ref = ref_program(Q, K, W, U, G, initial_state, output_final_state=True, cu_seqlens=None, scale=scale)

    # tilelang
    h_tilelang = torch.empty(B, S // chunk_size, H, DK, DV, dtype=getattr(torch, output_dtype)).cuda()
    V_new_tilelang = torch.empty(B, S, H, DV, dtype=getattr(torch, output_dtype)).cuda()
    o_tilelang = torch.empty(B, S, H, DV, dtype=getattr(torch, output_dtype)).cuda()
    final_state_tilelang = torch.empty(B, H, DK, DV, dtype=getattr(torch, state_dtype)).cuda()

    program = tilelang_chunk_gated_delta_rule_chunk_o_fwd(
        B, S, H, DK, DV, input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype, chunk_size, use_g, use_initial_state, store_final_state, save_new_value, scale, block_DK, block_DV, threads, num_stages
    )
    kernel = tilelang.compile(program)
    # kernel = tilelang.compile(program, pass_configs={"tl.disable_warp_specialized" : True})
    # kernel = tilelang.compile(program, pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})
    print(kernel.get_kernel_source())
    kernel(Q, K, W, U, G, initial_state, h_tilelang, final_state_tilelang, V_new_tilelang, o_tilelang)

    print("tilelang:", o_tilelang)
    print("o_ref:", o_ref)

    print("tilelang:", final_state_tilelang)
    print("final_state_ref:", final_state_ref)

    # fla_time = do_bench(chunk_gated_delta_rule_fwd_h, K, W, U, G, initial_state, store_final_state, chunk_size, save_new_value)
    fla_time = do_bench(ref_program, Q, K, W, U, G, initial_state, store_final_state, None, 1.0)
    tilelang_time = do_bench(kernel, Q, K, W, U, G, initial_state, h_tilelang, final_state_tilelang, V_new_tilelang, o_tilelang)

    print(f"tilelang time: {tilelang_time} ms")
    print(f"fla time: {fla_time} ms")

    # check
    try:
        torch.testing.assert_close(o_ref, o_tilelang, rtol=1e-2, atol=1e-2, equal_nan=True)
        print("tilelang chunk gated delta rule fwd o passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule fwd o failed ✗")
        # print("ref h:", h_ref)
        # print("tilelang h:", h_tilelang)
        print(e)

    try:
        torch.testing.assert_close(final_state_ref, final_state_tilelang, rtol=1e-2, atol=1e-2, equal_nan=True)
        print("tilelang chunk gated delta rule fwd final_state passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule fwd final_state failed ✗")
        # print("ref final state:", final_state_ref)
        # print("tilelang final state:", final_state_tilelang)
        print(e)

    # try:
    #     torch.testing.assert_close(V_new_ref, V_new_tilelang, rtol=1e-2, atol=1e-2, equal_nan=True)
    #     print("tilelang chunk gated delta rule fwd V_new passed √")
    # except Exception as e:
    #     print("tilelang chunk gated delta rule fwd V_new failed ✗")
    #     # print("ref V_new:", V_new_ref)
    #     # print("tilelang V_new:", V_new_tilelang)
    #     print(e)


if __name__ == "__main__":
    run_test(
        B=1,
        S=32768,
        H=32,
        DK=128,
        DV=128,
        input_dtype="bfloat16",
        output_dtype="bfloat16",
        accum_dtype="float32",
        gate_dtype="float32",
        state_dtype="float32",
        chunk_size=64,
        use_g=True,
        use_initial_state=True,
        store_final_state=True,
        save_new_value=True,
        scale=1.0,
        block_DK=64,
        block_DV=32,
        threads=128,
        num_stages=1,
    )
    # run_test(
    #     B=1,
    #     S=256,
    #     H=1,
    #     DK=64,
    #     DV=32,
    #     input_dtype="bfloat16",
    #     output_dtype="bfloat16",
    #     accum_dtype="float32",
    #     gate_dtype="float32",
    #     state_dtype="float32",
    #     chunk_size=64,
    #     use_g=True,
    #     use_initial_state=True,
    #     store_final_state=True,
    #     save_new_value=True,
    #     block_DK=64,
    #     block_DV=32,
    #     threads=128,
    #     num_stages=2,
    # )
    # run_test(
    #     B=1,
    #     S=1024,
    #     H=2,
    #     DK=128,
    #     DV=128,
    #     input_dtype="bfloat16",
    #     output_dtype="bfloat16",
    #     accum_dtype="float32",
    #     gate_dtype="float32",
    #     state_dtype="float32",
    #     chunk_size=64,
    #     use_g=True,
    #     use_initial_state=True,
    #     store_final_state=True,
    #     save_new_value=True,
    #     block_DK=64,
    #     block_DV=32,
    #     threads=128,
    #     num_stages=2,
    # )