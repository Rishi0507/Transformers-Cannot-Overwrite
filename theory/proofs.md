# Theory Notes

Draft proofs for the four theoretical claims. Status of each result appears in Section 7.

## 0. Scope

Every theorem here covers one attention layer. Queries, keys, values, and gates are functions of a single token (and its position), with no dependence on other tokens. Section 6 states the deep-model versions as conjectures and lists what experiments must show. Lower bounds for deep transformers are a known open problem, and these notes do not claim to solve it.

## 1. Setup

**Stream.** A sequence of operations `x_1, ..., x_n`. Each is `write(k, u)`, `read(k)`, or a filler token. A register task asks, at `read(k)`, for the value of the latest earlier `write(k, .)`.

**Head.** For reader position `i` and position `j <= i`, a head has logit `l(i, j)`, weight `alpha_ij = exp(l(i, j)) / sum_{j' <= i} exp(l(i, j'))`, and output `o_i = sum_j alpha_ij v_j`.

**Assumptions used throughout.**

- (A1) Content logits are bounded: the gate-free part of every logit lies in `[-B, B]`.
- (A2) Value vectors are bounded: `||v_j|| <= V`.
- (A3) Everything after the attention layer is a `Lambda`-Lipschitz map `R` from the summed head outputs to output logits.
- (A4) One layer: the logit `l(i, j)` depends on the tokens at `i` and `j` and on their positions. It does not depend on any other token.

**Margin.** A model is `gamma`-correct on a stream if the output logit of the correct value exceeds every other value's logit by at least `gamma`.

**Three head types.**

- Plain: `l(i, j) = c(i, j)`, with `c` in `[-B, B]`. Any positional encoding is allowed inside `c`.
- Scalar gate (FoX form): `l(i, j) = c(i, j) - (D_i - D_j)`, `D_t = sum_{l <= t} phi(x_l)`, `phi >= 0`.
- Typed gate: `l(i, j) = c(i, j) - b_j . (S_i - S_j)`, `S_t = sum_{l <= t} u_l`, with `u_l, b_j` in `R^r`, entrywise non-negative.

**Lemma 0 (folding).** With `q~_i = [q_i, -S_i, 1]` and `k~_j = [k_j, b_j, b_j . S_j]`, `q~_i . k~_j = q_i . k_j - b_j . (S_i - S_j)`. A typed-gate head is a plain dot-product head on augmented vectors, so any scaled-dot-product kernel computes it unchanged. The unit test `test_folded_dot_product_matches_explicit_penalty` checks this numerically.

## 2. Theorem 1: plain attention cannot isolate the latest write

**Statement.** Take a one-layer model with `H` plain heads under (A1) to (A4). Fix any positions `j_1 < ... < j_c` before a reader `read(k)`, and two distinct values `v, v'`. Then there exist two streams that differ only in the value of the latest `write(k, .)`, with different correct answers, on which the model cannot be `gamma`-correct on both unless

```
gamma <= 4 Lambda V H (2B + ln c) / (c - 1).
```

So the margin decays as `O(HB / c)` in the number `c` of writes to the key, for every positional encoding.

**Proof.**

*Step 1, a ladder of contexts.* For a pattern `P = (val_1, ..., val_c)` in `{v, v'}^c` and `t` in `{1, ..., c}`, let `C_t(P)` be the stream in which positions `j_1, ..., j_t` hold `write(k, val_m)` and positions `j_{t+1}, ..., j_c` hold fillers. All other tokens are fixed. The latest write in `C_t(P)` sits at `j_t`.

*Step 2, weights do not depend on the context.* By (A4), the unnormalized weight of the write at `j_m` with value `u`, as seen by the reader, is a number `a_m(u)` in `[e^{-B}, e^{B}]`. It is the same in every context that contains that write.

*Step 3, telescoping.* Fix one head. Let `A_t(P) = sum_{m <= t} a_m(val_m)` and `s^_t(P) = a_t(val_t) / A_t(P)`. The true attention on the latest write in `C_t(P)` is at most `s^_t(P)`, since other tokens only add to the denominator. Because `1 - s^_t = A_{t-1} / A_t`,

```
prod_{t=2..c} (1 - s^_t(P)) = A_1 / A_c >= e^{-B} / (c e^{B}) = e^{-2B} / c.
```

Taking logs and using `-ln(1 - s) >= s`:

```
sum_{t=2..c} s^_t(P) <= 2B + ln c        for every pattern P.        (1)
```

*Step 4, the flipped context.* Let `C'_t(P)` be `C_t(P)` with the value at `j_t` flipped. Its correct answer differs from that of `C_t(P)`. Let `P^(t)` be `P` with coordinate `t` flipped. Since `A_{t-1}` depends on `P` only through coordinates below `t`, the attention on the latest write in `C'_t(P)` is at most `s^_t(P^(t))`.

*Step 5, averaging.* Draw `P` uniformly. The map `P -> P^(t)` is a bijection, so `E[s^_t(P^(t))] = E[s^_t(P)]`. With (1),

```
E_P sum_{t=2..c} [ s^_t(P) + s^_t(P^(t)) ] <= 2 (2B + ln c)        per head.
```

Summing over `H` heads, some pattern `P` and some `t` satisfy

```
sum_h [ s_h + s'_h ] <= 2 H (2B + ln c) / (c - 1),                     (2)
```

where `s_h, s'_h` are head `h`'s attention on the latest write in `C_t(P)` and `C'_t(P)`.

*Step 6, outputs are close.* In head `h`, write the output as `o = (1 - s) o_rest + s w`, where `w` is the value vector of the latest write and `o_rest` is the attention average over all other tokens. The two contexts share every other token and every other unnormalized weight, so `o_rest` is the same in both. Hence `||o - o'|| <= 2V (s_h + s'_h)`. Summing over heads and applying (2) and (A3), the output logits of the two contexts differ by at most `Lambda 4 V H (2B + ln c) / (c - 1)` in every coordinate.

*Step 7, margin.* If the model were `gamma`-correct on both, then `z_v - z_{v'} >= gamma` on one and `z'_{v'} - z'_v >= gamma` on the other. Adding gives `2 gamma <= 2 ||z - z'||_inf`. QED.

**Tightness.** A linear recency profile `c(i, j) = B - 2B (i - j) / n` gives consecutive weight ratio `e^{2B/c}` for evenly spaced writes and attention about `2B / c` on the latest write. The bound is tight up to the `ln c` term.

**Reading the bound.** With `B = 20`, a head gives the latest of 400 writes at most 11.5% of its attention in the worst context of the ladder. The bound does not involve `n`. Filler tokens do not enter. Only competitors that match the key do. This matches the 2026 measurements that length is cheap and competition is not.

## 3. Theorem 2: a scalar-gate head serves one register

**Statement.** Take a one-layer model with `H` scalar-gate heads under (A1) to (A4), where the gate `phi_h(x)` depends on the token `x`. Fix a target margin `gamma* > 0`. If the model is `gamma*`-correct on all register streams over `K` keys, with no bound on the number of writes, then `K <= H`.

**Definitions.** Let `eps0 = gamma* / (16 Lambda V H)`, assumed at most 1. For head `h`, let `F_h` be the set of keys `y` such that some value `u` has `phi_h(write(y, u)) >= eps0`. Head `h` serves key `x` if `F_h = {x}`.

**Proof.** Suppose key `x` is served by no head. Each head is then of one of two types.

- Type 1: `x` is not in `F_h`. Every `write(x, .)` emits forgetting below `eps0`.
- Type 2: `F_h` contains some key `y_h != x`, with a value `u_h` whose write emits forgetting at least `eps0`.

*The stream.* Place `c` consecutive writes to `x`, then for each type-2 head a block of `c'` copies of `write(y_h, u_h)`, then `read(x)`. Use the ladder of Theorem 1 on the `c` writes to `x`: contexts `C_t(P)` and their flips `C'_t(P)`.

*Type-2 heads.* Every write to `x` precedes the block for head `h`, so its penalty is at least `c' eps0`. The reader attends to itself with penalty zero and logit at least `-B`. Each token before the block therefore has attention at most `e^{2B - c' eps0}`. Flipping one value among the writes to `x` changes this head's output by at most `4 V n e^{2B - c' eps0}`.

*Type-1 heads.* Consecutive writes to `x` differ in penalty by less than `eps0`. Repeat Step 3 of Theorem 1 with `a~_m = exp(c_m + D_{j_m})`. Tokens after `j_t` add a common penalty to all earlier writes, so ratios among them do not depend on the context. Now `A~_1 / A~_c >= e^{-2B - (c - 1) eps0} / c`, which gives

```
sum_{t=2..c} s^_t(P) <= 2B + ln c + (c - 1) eps0.
```

Steps 4 and 5 go through unchanged, because every quantity at rung `t` depends on `P` through coordinates up to `t`. The flip also changes the gate emitted at `j_t` by at most `eps0`, which rescales all earlier weights by a factor within `e^{+-eps0}` and moves the output by at most `2V (e^{eps0} - 1) <= 4 V eps0`.

*Combining.* Some pattern and rung give a pair of streams with different answers whose output logits differ by at most

```
Lambda [ 4 V H (2B + ln c) / (c - 1) + 8 V H eps0 + 4 V H n e^{2B - c' eps0} ].
```

Let `c` and `c'` grow. The first and third terms vanish (the length `n` grows linearly in `c + H c'`). The middle term equals `gamma* / 2`. By Step 7 of Theorem 1 the model is not `gamma*`-correct on both streams. So every key is served by some head. A head serves at most one key, hence `K <= H`. QED.

**Reading the result.** A scalar gate forgets in order of age. To resolve rewrites of `x` it must forget on writes to `x`, and that forgetting hits every older token in the head, including live values of other keys. Content logits can absorb at most `2B` of penalty, so a second key survives only about `2B / eps0` rewrites.

## 4. Theorem 3: a typed-gate head serves r registers

**Statement.** For `K <= r` keys, one typed-gate head with bounded parameters solves the register task with attention at least `1 - 2 delta` on the latest write, for every stream of length at most `n` and every number of writes. The parameters depend on `n` and `delta` through logarithms. They do not depend on the number of writes.

**Construction.** Give key `k` the basis vector `e_k`. Set

- `u(write(k, .)) = theta e_k`, and `u = 0` for reads and fillers,
- `b(write(k, .)) = e_k`, and `b = 0` for reads and fillers,
- content logit `beta` between `read(k)` and any `write(k, .)`, and `-beta` otherwise.

At `read(k)`, the latest write to `k` has penalty zero. A write to `k` with `m` later writes to `k` has penalty `m theta`. Writes to other keys add nothing to it. The attention on the latest write is at least

```
1 / ( 1 + sum_{m >= 1} e^{-m theta} + n e^{-2 beta} ) = 1 / ( 1 + 1/(e^theta - 1) + n e^{-2 beta} ).
```

Choose `theta = ln(1 + 1/delta)` and `beta = (1/2) ln(n / delta)`. QED.

**Remarks.**

- The `ln n` in `beta` is the usual cost of softmax retrieval among `n` non-matching tokens. Plain attention pays it too.
- With softplus gates, exact zeros are not available. A bias of `-ln(n / delta)` keeps the total leak below `delta`.
- Capacity per layer is `H r` registers for typed gates against `H` for scalar gates.

**Proposition 3b (rank is necessary for exact service).** If non-negative signatures and kill vectors satisfy `b_x . u_x > 0` for all keys and `b_x . u_y = 0` for all `x != y`, then `r >= K`. Proof: pick a coordinate `i_x` in the support of both `u_x` and `b_x`. For `y != x`, `b_y . u_x = 0` with non-negative entries forces `i_x` outside the support of `b_y`, while `i_y` lies inside it. So the `i_x` are distinct. QED. With a bounded number of interleaved writes, small overlaps are tolerable and `K` can exceed `r`.

## 5. Theorem 4: sound selective eviction

**Statement.** In a typed-gate head under (A1) and (A2), the penalty `pi_j(i) = b_j . (S_i - S_j)` is non-decreasing in `i`. Let `E` be a set of evicted positions, where `j` was evicted at time `t_j`, and suppose

```
sum_{j in E} e^{2B - pi_j(t_j)} <= eps < 1.
```

Then at every later reader position, the head output with `E` removed differs from the exact output by at most `2 V eps / (1 - eps)`.

**Proof.** Monotonicity: `S_i - S_j` is a sum of entrywise non-negative vectors that grows with `i`, and `b_j >= 0`. For any `i >= t_j`, the weight of `j` is at most `e^{B - pi_j(i)}`, and the reader's own weight is at least `e^{-B}` since its penalty is zero. So `alpha_ij <= e^{2B - pi_j(t_j)}`, and the total evicted mass `alpha_E` is at most `eps`. Removing mass `alpha_E` and renormalizing moves a convex combination of vectors of norm at most `V` by at most `2 V alpha_E / (1 - alpha_E)`. QED.

**Remarks.**

- The eviction test is a running sum of one scalar per evicted token.
- Under a scalar gate, penalties are ordered by age within a head, so the evictable set is always a prefix. Typed gates can evict a newer dead token and keep an older live one.
- The bound is per layer. A whole-network bound needs Lipschitz constants for the layers above. Experiments will report measured end-to-end error.
- (A1) needs a real bound `B`. A retrofit can clamp content logits at a value above the range seen in the pretrained model.

## 6. Conjectures for deep models and matching experiments

| Claim | One layer | Deep models | Experiment |
|-------|-----------|-------------|------------|
| Plain attention loses the latest write as writes pile up | Theorem 1 | Conjecture. A stale token's representation cannot learn that it is stale, because attention is causal. | Accuracy against write count, at lengths beyond training. |
| Scalar gates serve about one register per head | Theorem 2 | Conjecture: at most `L H` registers. | Sweep `K` against `H` with one and two layers. |
| Typed gates serve `H r` registers | Theorem 3 | Construction extends layer by layer. | Same sweep with typed gates. |
| Eviction is sound | Theorem 4 | Per-layer bound holds in any layer. | Measure output error and memory after eviction. |

## 7. Status

| Result | State |
|--------|-------|
| Lemma 0 | Proved. Unit-tested. |
| Theorem 1 | Proof drafted. Needs an independent check. |
| Theorem 2 | Proof drafted. The limit argument needs explicit constants. Needs an independent check. |
| Theorem 3 | Proved by construction. |
| Proposition 3b | Proved. |
| Theorem 4 | Proved. |
| Deep-model versions | Open. Conjectures only. |
