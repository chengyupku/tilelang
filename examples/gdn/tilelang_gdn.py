import warnings
from typing import Optional

import torch
from einops import rearrange

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_o import chunk_bwd_dv_local
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import tilelang
import tilelang.language as T
from chunk_scaled_dot_kkt import tilelang_chunk_scaled_dot_kkt_fwd
from chunk_delta_h import tilelang_chunk_gated_delta_rule_fwd_h
from chunk_o import tilelang_chunk_fwd_o
from chunk_delta_bwd import tilelang_chunk_gated_delta_rule_bwd_dhu
from wy_fast import tilelang_recompute_w_u_fwd
from chunk_o_bwd import tilelang_chunk_o_bwd_dqkwg

tilelang.disable_cache()

def tilelang_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None
):
    B, S, H, DK = k.shape
    DV = v.shape[-1]
    input_dtype="bfloat16"
    output_dtype="bfloat16"
    accum_dtype="float32"
    gate_dtype="float32"
    state_dtype="float32"
    chunk_size=64

    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    # obtain WY representation. u is actually the new v.
    A = tilelang_chunk_scaled_dot_kkt_fwd(B, S, H, DK)(
        k,
        beta,
        g,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    w, u = tilelang_recompute_w_u_fwd(B, S, H, DK, DV, input_dtype, output_dtype, gate_dtype, accum_dtype, chunk_size, block_S=chunk_size, block_DK=64, block_DV=32, threads=128, num_stages=3)(
        k,
        v,
        beta,
        g,
        A,
    )
    h, final_state, v_new = tilelang_chunk_gated_delta_rule_fwd_h(B, S, H, DK, DV, input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype, chunk_size, use_g=True, use_initial_state=True, store_final_state=True, save_new_value=True, block_DK=64, block_DV=32, threads=128, num_stages=1)(
        k,
        w,
        u,
        g,
        initial_state,
    )
    o = tilelang_chunk_fwd_o(B, S, H, DK, DV, input_dtype, output_dtype, accum_dtype, gate_dtype, chunk_size, scale, use_g=True)(
        q,
        k,
        v_new,
        h,
        g,
    )
    return g, o, A, final_state

def tilelang_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, S, H, DK = k.shape
    DV = v.shape[-1]
    input_dtype="bfloat16"
    output_dtype="bfloat16"
    accum_dtype="float32"
    gate_dtype="float32"
    state_dtype="float32"
    chunk_size=64
    use_g= True
    
    w, u = tilelang_recompute_w_u_fwd(B, S, H, DK, DV, input_dtype, output_dtype, gate_dtype, accum_dtype, chunk_size, block_S=chunk_size, block_DK=64, block_DV=32, threads=128, num_stages=3)(
        k,
        v,
        beta,
        g,
        A,
    )
    h, _, v_new = tilelang_chunk_gated_delta_rule_fwd_h(B, S, H, DK, DV, input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype, chunk_size, use_g=use_g, use_initial_state=True, store_final_state=True, save_new_value=True, block_DK=64, block_DV=32, threads=128, num_stages=1)(
        k,
        w,
        u,
        g,
        initial_state,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    dh, dh0, dv = tilelang_chunk_gated_delta_rule_bwd_dhu(B, S, H, DK, DV, input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype, chunk_size, scale, use_g=use_g, use_initial_state=True, use_final_state_gradient=True, block_DV=32, threads=128, num_stages=1)(
        q,
        k,
        w,
        g,
        initial_state,
        dht,
        do,
        dv,
    )
    dq, dk, dw, dg = tilelang_chunk_o_bwd_dqkwg(B, S, H, DK, DV, input_dtype, output_dtype, accum_dtype, gate_dtype, state_dtype, chunk_size, scale, use_g=use_g, use_dw=True, block_DK=64, block_DV=64, threads=128, num_stages=0)(
        q,
        k,
        v_new,
        h,
        g,
        do,
        dh,
        dv,
        w,
    )
    if use_g:
        dg = dg.sum(dim=0)
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    assert dg.dtype == torch.float32, "dg should be fp32"
    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0
    
class TilelangChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False
    ):
        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state = tilelang_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q_orig, k_orig, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        if ctx.use_qk_l2norm_in_kernel:
            q, q_orig = l2norm_fwd(q), q
            k, k_orig = l2norm_fwd(k), k
        dq, dk, dv, db, dg, dh0 = tilelang_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q_orig, dq)
            dk = l2norm_bwd(k_orig, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None
        # return None, None, None, None, None, None, None, None, None, None



@torch.compiler.disable
def tilelang_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, beta, g))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = TilelangChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel
    )
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, final_state
    