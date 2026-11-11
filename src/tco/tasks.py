import torch

WRITE, READ, IGNORE = 0, 1, 2


def make_batch(B, T, n_keys, n_vals, p_write=0.3, p_read=0.3, gen=None):
    """K-register stream. Each token is (op, key, value).

    WRITE sets register[key] = value. READ asks for register[key]. IGNORE changes nothing.
    Returns op, key, val_in, target (-100 off reads), depth (writes so far to the read key),
    age (tokens since the last write to the read key).
    """
    probs = torch.tensor([p_write, p_read, 1.0 - p_write - p_read])
    op = torch.multinomial(probs, B * T, replacement=True, generator=gen).view(B, T)
    key = torch.randint(n_keys, (B, T), generator=gen)
    val = torch.randint(n_vals, (B, T), generator=gen)

    ar = torch.arange(B)
    state = torch.full((B, n_keys), -1)
    count = torch.zeros((B, n_keys), dtype=torch.long)
    last = torch.zeros((B, n_keys), dtype=torch.long)
    target = torch.full((B, T), -100)
    depth = torch.zeros((B, T), dtype=torch.long)
    age = torch.zeros((B, T), dtype=torch.long)
    for t in range(T):
        k = key[:, t]
        cur = state[ar, k]
        # a read of an unset key becomes a write
        op[:, t] = torch.where((op[:, t] == READ) & (cur < 0), WRITE, op[:, t])
        w, r = op[:, t] == WRITE, op[:, t] == READ
        state[ar[w], k[w]] = val[w, t]
        count[ar[w], k[w]] += 1
        last[ar[w], k[w]] = t
        target[r, t] = cur[r]
        depth[r, t] = count[ar[r], k[r]]
        age[r, t] = t - last[ar[r], k[r]]
    val_in = torch.where(op == READ, n_vals, val)
    return op, key, val_in, target, depth, age
