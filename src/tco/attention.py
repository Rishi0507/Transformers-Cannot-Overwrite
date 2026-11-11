import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(x, base=10000.0):
    # x: (B, H, T, d) with d even
    T, d = x.shape[-2], x.shape[-1]
    half = d // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freq[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GatedAttention(nn.Module):
    """Causal self-attention with optional forget gates.

    logit(i, j) = q_i . k_j - b_j . (S_i - S_j),  S = cumsum of kill vectors u.

    gate="none":   plain attention.
    gate="scalar": one gate per head, b = 1 (FoX-style).
    gate="typed":  rank-r kill vectors u and signatures b per head.
    """

    def __init__(self, dim, heads, gate="none", rank=8, pos="rope", kill_bias=-5.0, sig_bias=-1.0):
        super().__init__()
        assert dim % heads == 0 and gate in ("none", "scalar", "typed")
        self.heads, self.gate, self.pos = heads, gate, pos
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        if gate != "none":
            r = 1 if gate == "scalar" else rank
            self.kill = nn.Linear(dim, heads * r)
            nn.init.constant_(self.kill.bias, kill_bias)
        if gate == "typed":
            self.sig = nn.Linear(dim, heads * r)
            nn.init.constant_(self.sig.bias, sig_bias)

    def gates(self, x):
        # returns signatures b and clocks S, both (B, H, T, r)
        B, T, _ = x.shape
        u = F.softplus(self.kill(x)).view(B, T, self.heads, -1).transpose(1, 2)
        S = u.float().cumsum(dim=2)
        if self.gate == "typed":
            b = F.softplus(self.sig(x)).view(B, T, self.heads, -1).transpose(1, 2)
        else:
            b = torch.ones_like(u)
        return b, S

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if self.pos == "rope":
            q, k = rope(q), rope(k)
        q = q / math.sqrt(q.shape[-1])
        if self.gate != "none":
            b, S = self.gates(x)
            # fold the penalty into the dot product
            q = torch.cat([q, -S, torch.ones_like(S[..., :1])], dim=-1)
            k = torch.cat([k, b, (b * S).sum(-1, keepdim=True)], dim=-1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        return self.out(y.transpose(1, 2).reshape(B, T, D))
