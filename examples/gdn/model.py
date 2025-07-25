from typing import Optional
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from utils import do_bench, assert_similar
import torch
import torch.nn.functional as F
# torch.set_printoptions(profile="full")
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from chunk_scaled_dot_kkt import tilelang_chunk_scaled_dot_kkt_fwd
from chunk_delta_h import tilelang_chunk_gated_delta_rule_fwd_h
from chunk_o import tilelang_chunk_fwd_o


from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
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
    assert_similar(o_tilelang, tri, name="o_tilelang")
    assert_similar(final_state_tilelang, tri_ht, name="final_state_tilelang")

    
    # do = torch.randn_like(v)
    # dht = torch.randn_like(h0)

    # # backward
    # def run_bwd():
    #     ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    #     return do, dht

    # fwd_latency = do_bench(run_fwd, warmup=500)
    # print("fwd: {:.2f} ms".format(fwd_latency))
    # bwd_latency = do_bench(run_bwd, warmup=500)
    # print("bwd: {:.2f} ms".format(bwd_latency))

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