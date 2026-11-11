import torch.nn as nn

from tco.attention import GatedAttention


class Block(nn.Module):
    def __init__(self, dim, heads, **attn_kw):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = GatedAttention(dim, heads, **attn_kw)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class RegisterModel(nn.Module):
    """Tiny transformer for the K-register task. One token per operation."""

    def __init__(self, n_keys, n_vals, dim=64, layers=1, heads=2, **attn_kw):
        super().__init__()
        self.op = nn.Embedding(3, dim)
        self.key = nn.Embedding(n_keys, dim)
        self.val = nn.Embedding(n_vals + 1, dim)  # last id hides the value at reads
        self.blocks = nn.ModuleList(Block(dim, heads, **attn_kw) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_vals)

    def forward(self, op, key, val):
        x = self.op(op) + self.key(key) + self.val(val)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))
