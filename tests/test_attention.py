import math

import pytest
import torch
import torch.nn.functional as F

from tco.attention import GatedAttention, rope

B, T, D, H = 2, 24, 32, 4


def explicit(attn, x):
    """Reference path: build the T x T penalty and run a masked softmax by hand."""
    q, k, v = attn.qkv(x).view(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
    if attn.pos == "rope":
        q, k = rope(q), rope(k)
    logits = q @ k.transpose(-1, -2) / math.sqrt(D // H)
    pen = torch.zeros(B, H, T, T)
    if attn.gate != "none":
        b, S = attn.gates(x)
        pen = torch.einsum("bhjr,bhir->bhij", b, S) - (b * S).sum(-1)[:, :, None, :]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    w = (logits - pen).masked_fill(~causal, float("-inf")).softmax(-1)
    return attn.out((w @ v).transpose(1, 2).reshape(B, T, D)), pen


def make(gate, **kw):
    torch.manual_seed(0)
    attn = GatedAttention(D, H, gate=gate, rank=3, kill_bias=-1.0, sig_bias=0.0, **kw)
    return attn, torch.randn(B, T, D)


@pytest.mark.parametrize("gate", ["none", "scalar", "typed"])
@pytest.mark.parametrize("pos", ["rope", "none"])
def test_folded_dot_product_matches_explicit_penalty(gate, pos):
    attn, x = make(gate, pos=pos)
    ref, _ = explicit(attn, x)
    assert torch.allclose(attn(x), ref, atol=1e-5)


@pytest.mark.parametrize("gate", ["scalar", "typed"])
def test_penalty_is_zero_on_self_nonnegative_and_monotone(gate):
    attn, x = make(gate)
    _, pen = explicit(attn, x)
    assert torch.allclose(pen.diagonal(dim1=-2, dim2=-1), torch.zeros(B, H, T), atol=1e-5)
    low = torch.ones(T, T, dtype=torch.bool).tril()
    assert (pen[..., low] >= -1e-5).all()
    # for a fixed key j the penalty never drops as the reader i moves forward
    step = pen[:, :, 1:, :] - pen[:, :, :-1, :]
    assert (step[..., low[:-1]] >= -1e-5).all()


def test_closed_gates_reduce_to_plain_attention():
    attn, x = make("typed")
    plain = GatedAttention(D, H, gate="none")
    plain.qkv, plain.out = attn.qkv, attn.out
    torch.nn.init.zeros_(attn.kill.weight)
    torch.nn.init.constant_(attn.kill.bias, -50.0)
    assert torch.allclose(attn(x), plain(x), atol=1e-5)


def test_scalar_gate_equals_fox_cumulative_log_forget():
    attn, x = make("scalar")
    _, pen = explicit(attn, x)
    log_f = F.logsigmoid(-attn.kill(x)).view(B, T, H).transpose(1, 2)  # log forget gate
    D_cum = log_f.cumsum(-1)
    fox = -(D_cum[:, :, :, None] - D_cum[:, :, None, :])  # -sum_{l=j+1..i} log f_l
    low = torch.ones(T, T, dtype=torch.bool).tril()
    assert torch.allclose(pen[..., low], fox[..., low], atol=1e-4)
