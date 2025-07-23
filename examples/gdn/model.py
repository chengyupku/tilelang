from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule
from utils import do_bench
import torch
import torch.nn.functional as F
# torch.set_printoptions(profile="full")
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
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    # backward
    def run_bwd():
        ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
        return do, dht

    fwd_latency = do_bench(run_fwd, warmup=500)
    print("fwd: {:.2f} ms".format(fwd_latency))
    bwd_latency = do_bench(run_bwd, warmup=500)
    print("bwd: {:.2f} ms".format(bwd_latency))

test_chunk_gdn(
    B=3,
    T=32768,
    H=8,
    D=128,
    dtype=torch.bfloat16,
    scale=1.0,
    gate_logit_normalizer=1.0,
    device=torch.device("cuda"),
)