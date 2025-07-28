from typing import Optional
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from utils import do_bench, assert_similar
import torch
import torch.nn.functional as F
from tilelang_gdn import tilelang_chunk_gated_delta_rule

torch.random.manual_seed(0)

def test_chunk_gdn(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
    gate_logit_normalizer: float,
    device: torch.device,
):
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid().fill_(1)
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    g = g / gate_logit_normalizer
    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))


    def run_tilelang_fwd():
        o_tilelang, final_state_tilelang = tilelang_chunk_gated_delta_rule(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
        )
        return o_tilelang, final_state_tilelang

    # forward
    def run_fwd():
        tri, tri_ht = chunk_gated_delta_rule(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            scale=scale,
            output_final_state=True,
            initial_state=h0.clone(),
        )
        return tri, tri_ht

    tri, tri_ht = run_fwd()
    o_tilelang, final_state_tilelang = run_tilelang_fwd()
    assert_similar(o_tilelang, tri, name="o_tilelang", raise_assert=False)
    assert_similar(final_state_tilelang, tri_ht, name="final_state_tilelang", raise_assert=False)

    
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    # backward
    def run_bwd():
        q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None
        ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
        tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
        return tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0

    def run_tilelang_bwd():
        q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None
        ((o_tilelang * do).sum() + (final_state_tilelang * dht).sum()).backward(retain_graph=True)
        tl_dq, tl_dk, tl_dv, tl_dbeta, tl_dg, tl_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
        return tl_dq, tl_dk, tl_dv, tl_dbeta, tl_dg, tl_dh0

    fla_fwd_latency = do_bench(run_fwd, warmup=500)
    print("fla fwd: {:.2f} ms".format(fla_fwd_latency))
    tilelang_fwd_latency = do_bench(run_tilelang_fwd, warmup=500)
    print("tilelang fwd: {:.2f} ms".format(tilelang_fwd_latency))

    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = run_bwd()
    tl_dq, tl_dk, tl_dv, tl_dbeta, tl_dg, tl_dh0 = run_tilelang_bwd()
    assert_similar(tri_dq, tl_dq, name="dq", raise_assert=False)
    assert_similar(tri_dk, tl_dk, name="dk", raise_assert=False)
    assert_similar(tri_dv, tl_dv, name="dv", raise_assert=False)
    assert_similar(tri_dbeta, tl_dbeta, name="dbeta", raise_assert=False)
    assert_similar(tri_dg, tl_dg, name="dg", raise_assert=False)
    assert_similar(tri_dh0, tl_dh0, name="dh0", raise_assert=False)
    
    
    fla_bwd_latency = do_bench(run_bwd, warmup=500)
    print("fla bwd: {:.2f} ms".format(fla_bwd_latency))
    tilelang_bwd_latency = do_bench(run_tilelang_bwd, warmup=500)
    print("tilelang bwd: {:.2f} ms".format(tilelang_bwd_latency))


test_chunk_gdn(
    B=1,
    T=32768,
    H=8,
    D=128,
    dtype=torch.bfloat16,
    scale=1.0,
    gate_logit_normalizer=1.0,
    device=torch.device("cuda"),
)