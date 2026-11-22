# Theory Notes

Proofs for the four theoretical claims. Section 7 gives the status of each result. Section 8 logs what changed in the review of 2026-11-22.

## 0. Scope

Every theorem here covers one attention layer. Queries, keys, values, and gates are functions of a single token and its position, with no dependence on other tokens. Section 6 states the deep-model versions as conjectures. Lower bounds for deep transformers are a known open problem, and these notes do not claim to solve it.

## 1. Setup

**Stream.** A sequence of tokens `x_1, ..., x_n`. Each token is `write(k, u)`, `read(k)`, or a filler. Keys come from a set of size `K`, values from a set of size at least 2. A register task asks, at `read(k)`, for the value of the latest earlier `write(k, .)`.

**Head.** For reader position `i` and position `j <= i`, a head has logit `l(i, j)`, weight `alpha_ij = exp(l(i, j)) / sum_{j' <= i} exp(l(i, j'))`, and output `o_i = sum_j alpha_ij v_j`. The sum includes `j = i`.

**Assumptions.**

- (A1) Bounded content logits. The gate-free part `c(i, j)` of every logit, including `c(i, i)`, lies in `[-B, B]`.
- (A2) Bounded values. `||v_j|| <= V` (Euclidean norm).
- (A3) Lipschitz readout. The output logits at position `i` are `z = R(x_i, o^1_i, ..., o^H_i)`, and for fixed `x_i`, `||z - z'||_inf <= Lambda sum_h ||o^h_i - o'^h_i||`. This covers output projection, residual connection, MLP, and unembedding.
- (A4) Pairwise content and position. `c(i, j)`, `v_j`, and any gate emitted at `j` depend on the token at `j`, the token at `i`, and the positions `i, j`. Nothing else. Gates in Theorem 2 depend on the token only, not on its position.

(A4) admits every data-independent positional scheme: absolute, learned, sinusoidal, RoPE, ALiBi, NoPE. It excludes schemes whose position signal depends on other tokens: CoPE, PaTH, Selective RoPE, stick-breaking attention, and forget gates themselves.

**Margin.** A model is `gamma`-correct on a stream if, at the final read, the output logit of the correct value exceeds every other value's logit by at least `gamma`.

**Three head types.**

- Plain: `l(i, j) = c(i, j)`.
- Scalar gate (FoX form): `l(i, j) = c(i, j) - (D_i - D_j)`, `D_t = sum_{l <= t} phi(x_l)`, `phi >= 0`.
- Typed gate: `l(i, j) = c(i, j) - b_j . (S_i - S_j)`, `S_t = sum_{l <= t} u_l`, with `u_l, b_j` in `R^r`, entrywise non-negative.

The penalty on `j` counts gates emitted at positions `j+1, ..., i`, including the reader's own. The penalty on `j = i` is zero.

**Lemma 0 (folding).** With `q~_i = [q_i, -S_i, 1]` and `k~_j = [k_j, b_j, b_j . S_j]`, `q~_i . k~_j = q_i . k_j - b_j . (S_i - S_j)`. A typed-gate head is a plain dot-product head on augmented vectors, so any scaled-dot-product kernel computes it unchanged. The unit test `test_folded_dot_product_matches_explicit_penalty` checks this numerically. The augmented key depends on `S_j`, so the folded head does not satisfy (A4).

**Fact F (rescaling a subset).** Take a softmax average of vectors with norm at most `V`. Multiply the unnormalized weights of a subset `S` by `rho` in `[e^{-eps}, e^{eps}]`. If `p` and `p'` are the mass of `S` before and after, then `|p' - p| = p (1 - p) |rho - 1| / (rho p + 1 - p) <= (e^{eps} - 1) / 4`, and the average moves by at most `2 V |p' - p| <= V (e^{eps} - 1) / 2`. Proof: the averages within `S` and within its complement do not change. The mass formula is direct algebra. For `rho >= 1` the denominator is at least 1. For `rho < 1` it is at least `rho`, and `(1 - rho) / rho <= e^{eps} - 1`. Use `p (1 - p) <= 1/4`.

## 2. Theorem 1: plain attention cannot isolate the latest write

**Statement.** Take a one-layer model with `H` plain heads under (A1) to (A4). Fix a key `k`, two distinct values `v, v'`, a reader position `i`, and `2c` positions `q_1 < ... < q_c < j_1 < ... < j_c < i`, with `c >= 2`. Then there exist two streams with exactly `c` writes to `k` each, identical except for the value of the latest write to `k`, such that the model is not `gamma`-correct on both unless

```
gamma <= 4 Lambda V H (2B + ln c) / (c - 1).
```

So the worst-case margin for the latest of `c` writes is `O(Lambda V H (B + ln c) / c)`, under every positional scheme that (A4) admits.

**Proof.**

*Step 1, a ladder of contexts.* For a pattern `P = (val_1, ..., val_c)` in `{v, v'}^c` and a rung `t` in `{2, ..., c}`, let `C_t(P)` be the stream in which:

- `j_1, ..., j_t` hold `write(k, val_1), ..., write(k, val_t)`,
- `q_{t+1}, ..., q_c` hold `write(k, v)` (padding, so the count is exactly `c`),
- `q_1, ..., q_t` and `j_{t+1}, ..., j_c` hold fillers,
- position `i` holds `read(k)`, and every other position holds a fixed filler.

The latest write to `k` in `C_t(P)` sits at `j_t`, with value `val_t`.

*Step 2, weights do not depend on the context.* By (A4), in head `h` the unnormalized weight of `write(k, u)` at `j_m`, seen from the reader, is a number `a_m(u)` in `[e^{-B}, e^{B}]`. It is the same in every context that has that token at `j_m`.

*Step 3, telescoping.* Fix one head. Let `A_t(P) = sum_{m <= t} a_m(val_m)` and `s^_t(P) = a_t(val_t) / A_t(P)`. The attention on `j_t` in `C_t(P)` is at most `s^_t(P)`, because the padding, the fillers, and the reader only add to the denominator. Since `1 - s^_t = A_{t-1} / A_t`,

```
prod_{t=2..c} (1 - s^_t(P)) = A_1 / A_c >= e^{-B} / (c e^{B}) = e^{-2B} / c.
```

With `-ln(1 - s) >= s`:

```
sum_{t=2..c} s^_t(P) <= 2B + ln c        for every pattern P.        (1)
```

*Step 4, the flipped context.* Let `P^(t)` be `P` with coordinate `t` flipped. The padding depends on `t` only, so `C_t(P^(t))` equals `C_t(P)` with the value at `j_t` flipped. The two streams have different correct answers. The attention on `j_t` in `C_t(P^(t))` is at most `s^_t(P^(t))`.

*Step 5, averaging.* Draw `P` uniformly. The map `P -> P^(t)` is a bijection, so `E[s^_t(P^(t))] = E[s^_t(P)]`. With (1), per head,

```
E_P sum_{t=2..c} [ s^_t(P) + s^_t(P^(t)) ] <= 2 (2B + ln c).
```

Sum over the `H` heads and divide by the `c - 1` rungs. Some pattern `P` and rung `t` satisfy

```
sum_h [ s_h + s'_h ] <= 2 H (2B + ln c) / (c - 1),                     (2)
```

where `s_h, s'_h` are head `h`'s attention on `j_t` in `C_t(P)` and `C_t(P^(t))`.

*Step 6, outputs are close.* In head `h`, write the output as `o = (1 - s) o_rest + s w`, where `w` is the value at `j_t` and `o_rest` is the attention average over all other positions. The two streams agree at every other position, and by (A4) every other unnormalized weight and value agrees, so `o_rest` is the same in both. Hence

```
||o - o'|| <= |s - s'| ||o_rest|| + s ||w|| + s' ||w'|| <= 2 V (s_h + s'_h).
```

The reader token is the same in both streams. By (A3) and (2), `||z - z'||_inf <= 4 Lambda V H (2B + ln c) / (c - 1)`.

*Step 7, margin.* Let the answers be `v` in the first stream and `v'` in the second. If the model were `gamma`-correct on both, then `z_v - z_{v'} >= gamma` and `z'_{v'} - z'_v >= gamma`. Adding gives `2 gamma <= (z_v - z'_v) + (z'_{v'} - z_{v'}) <= 2 ||z - z'||_inf`. QED.

**Remarks.**

- The bound does not involve the stream length `n`. Fillers only lower the attention on the latest write, so they cannot help. Only competitors that the head weights like the target enter.
- Compare the dispersion argument for bounded-logit softmax (Velickovic et al. 2024): every weight among `c` competitors is within `e^{2B}` of the largest, so the largest is at most `e^{2B} / c`. Theorem 1 replaces `e^{2B}` with `2B + ln c`. The price is that it holds for some stream in the ladder, not for every stream.
- The bound is a worst case over value patterns. A model can be correct on many `c`-write streams. It cannot be correct with margin above the bound on all of them.
- Tightness is open. The earlier draft claimed that a linear recency profile attains the bound up to `ln c`. That example gives attention about `2B / c` on the latest write, but the remaining mass sits on older writes whose values can all disagree with the latest one, so that model has negative margin on some streams. It shows only that the attention share in Step 3 can reach order `B / c`. Whether some model attains margin of order `B / c` on all `c`-write streams is open.

**Reading the bound.** With `B = 20` and `c = 400`, `(2B + ln c) / (c - 1) = 0.115`. In the worst stream of the ladder, the heads together give the latest of 400 writes at most `11.5% x 2H` of summed attention.

## 3. Theorem 2: a scalar-gate head serves one register

**Statement.** Take a one-layer model with `H` scalar-gate heads under (A1) to (A4), where gate `phi_h(x)` depends on the token `x` only. Fix a target margin `gamma* > 0`. If the model is `gamma*`-correct on every register stream over `K` keys, with no bound on the number of writes or the length, then `K <= H`.

**Definitions.** Let `eps0 = min(1, gamma* / (10 Lambda V H))`. For head `h`, let `F_h` be the set of keys `y` such that some value `u` has `phi_h(write(y, u)) >= eps0`. Head `h` serves key `x` if `F_h = {x}`. A head serves at most one key.

**Proof.** Suppose key `x` is served by no head. Each head is one of two types.

- Type 1: `x` is not in `F_h`. Every `write(x, .)` emits a gate in `[0, eps0)`.
- Type 2: `F_h` contains some key `y_h != x`, with a value `u_h` whose write emits a gate of at least `eps0`.

*The stream.* Use the ladder of Theorem 1 without padding, on consecutive positions `j_1, ..., j_c`, so no filler sits between two ladder writes. Positions `j_1, ..., j_c` hold `write(x, val_1), ..., write(x, val_t)` and then fillers. After `j_c`, place one block of `c'` copies of `write(y_h, u_h)` for each type-2 head `h`, then `read(x)`. The length is `n = c + H2 c' + 1`, where `H2` counts type-2 heads.

*Type-2 heads.* Every token at or before `j_c` precedes head `h`'s block, so its penalty is at least `c' eps0`. The reader attends to itself with penalty zero and logit at least `-B`. So each such token has attention at most `delta = e^{2B - c' eps0}`, and together at most `n delta`. A flip at `j_t` changes no gate, weight, or value after `j_c`, and the attention average over positions after `j_c` does not change. As in Step 6 of Theorem 1, the head output moves by at most `2 V (p + p') <= 4 V n delta`, where `p, p'` are the masses at or before `j_c`.

*Type-1 heads.* Let `g_l = phi_h(write(x, val_l))`, in `[0, eps0)`. In `C_t(P)`, tokens after `j_t` add the same penalty to every write at or before `j_t`, so ratios among those writes depend only on `P` through coordinates up to `t`. Define

```
a~_m = exp( c_m(val_m) - sum_{l = m+1..t} g_l ),    A~_t = sum_{m <= t} a~_m,    s^_t = a~_t / A~_t.
```

The attention on `j_t` is at most `s^_t`. At rung `t`, the earlier writes carry the extra factor `e^{-g_t}` relative to rung `t - 1`, so `1 - s^_t = e^{-g_t} A~_{t-1} / A~_t`. The product over `t = 2..c` is `e^{-sum g} A~_1 / A~_c >= e^{-2B - (c - 1) eps0} / c`, so

```
sum_{t=2..c} s^_t(P) <= 2B + ln c + (c - 1) eps0.
```

The flipped stream is `C_t(P^(t))`, so Steps 4 and 5 of Theorem 1 apply unchanged. Step 6 needs one change. The flip changes the gate at `j_t` by less than `eps0`, which multiplies the weights of all positions before `j_t` by a common `rho` in `(e^{-eps0}, e^{eps0})` relative to positions after it. By Fact F applied to `o_rest`, `||o_rest - o'_rest|| <= V (e^{eps0} - 1) / 2 <= V eps0`, using `eps0 <= 1`. So `||o - o'|| <= 2 V (s + s') + V eps0`.

*Combining.* Averaging over `P` and `t` as in Step 5 gives a pair of streams with different answers and

```
||z - z'||_inf <= Lambda [ 4 V H (2B + ln c) / (c - 1) + 4 V H eps0 + V H eps0 + 4 V H n e^{2B - c' eps0} ].
```

The middle two terms sum to `5 Lambda V H eps0 <= gamma* / 2`. Take `c' = c`, so `n <= (H + 1) c + 1`. Choose `c` large enough that

```
4 Lambda V H (2B + ln c) / (c - 1) < gamma* / 4    and    4 Lambda V H ((H + 1) c + 1) e^{2B - c eps0} < gamma* / 4.
```

Both hold for all large `c`, because the left sides tend to 0. Then `||z - z'||_inf < gamma*`, and by Step 7 of Theorem 1 the model is not `gamma*`-correct on both streams. This contradicts the hypothesis, so every key is served by some head. A head serves at most one key, hence `K <= H`. QED.

**Reading the result.** A scalar gate forgets in order of age. To resolve rewrites of `x` a head must forget on writes to `x`, and that forgetting hits every older token in the head, including live values of other keys. Content logits absorb at most `2B` of penalty, so in a head that forgets at rate `eps0` per write, a second key survives about `2B / eps0` rewrites. With a bounded number of writes, one head can serve more than one key. The theorem concerns unbounded write counts at a fixed margin.

## 4. Theorem 3: a typed-gate head serves r registers

**Statement.** Let `K <= r`, `n >= 1`, and `0 < delta < 1/4`. One typed-gate head with `|c(i, j)| <= (1/2) ln(n / delta)`, gate entries in `{0, ln(1 + 1/delta)}`, and an identity readout is `(1 - 4 delta)`-correct on every register stream of length at most `n`, for every number of writes.

**Construction.** Give key `k` the basis vector `e_k` of `R^r`. Let `theta = ln(1 + 1/delta)` and `beta = (1/2) ln(n / delta)`.

- Kill vectors: `u(write(k, .)) = theta e_k`. Reads and fillers emit `u = 0`.
- Signatures: `b(write(k, .)) = e_k`. Reads and fillers carry `b = 0`.
- Content logits: `c = beta` between `read(k)` and any `write(k, .)`, and `c = -beta` for every other pair, including the reader with itself. With keys in `R^{K+1}`, take `k(write(k', .)) = e_{k'}`, `k(read or filler) = e_0`, and `q(read(k)) = 2 beta e_k - beta (e_0 + sum_{k'} e_{k'})`.
- Values: `v(write(k, u)) = e_u`, one-hot over values, and `v = 0` otherwise. The readout is the identity, so `Lambda = 1` and `V = 1`.

**Proof.** At `read(k)`, the latest write to `k` has penalty zero and logit `beta`. A write to `k` followed by `m` later writes to `k` has penalty `m theta`, since writes to other keys and reads and fillers add nothing to coordinate `k`. Every other position has logit at most `-beta`. The attention on the latest write is at least

```
1 / ( 1 + sum_{m >= 1} e^{-m theta} + n e^{-2 beta} ) = 1 / ( 1 + 1/(e^theta - 1) + n e^{-2 beta} ) = 1 / (1 + 2 delta) >= 1 - 2 delta.
```

The output coordinate of the correct value is at least `1 - 2 delta`. Every other coordinate is at most `2 delta`. The margin is at least `1 - 4 delta`. QED.

**Remarks.**

- The `ln n` in `beta` is the usual cost of softmax retrieval among `n` non-matching tokens. Plain attention pays it too. No parameter depends on the number of writes.
- Theorem 1 does not apply because the penalty depends on the tokens between `j` and `i`, which violates (A4).
- With softplus gates, exact zeros are not available. A gate bias of `-ln(n / delta)` keeps each leaked entry below `delta / n`. This variant is not proved here in full.
- Capacity per layer is `H r` registers for typed gates against `H` for scalar gates (Theorem 2).

**Proposition 3b (rank is necessary for exact service).** Suppose each key `x` has a kill vector `u_x` and signature `b_x`, entrywise non-negative in `R^r`, with `b_x . u_x > 0` for all `x` and `b_x . u_y = 0` for all `x != y`. Then `r >= K`.

Proof: pick a coordinate `i_x` in the support of both `u_x` and `b_x`. For `y != x`, `b_y . u_x = 0` with non-negative entries forces `i_x` outside the support of `b_y`, while `i_y` lies inside it. So the `i_x` are distinct. QED.

Proposition 3b covers exact zero cross-talk only. It is not a lower bound for typed gates in general. Packing `K > r` keys with small overlaps under a bounded number of interleaved writes is a conjecture.

## 5. Theorem 4: sound selective eviction

**Statement.** In a typed-gate head under (A1) and (A2), the penalty `pi_j(i) = b_j . (S_i - S_j)` is non-decreasing in `i`. Let `E` be a set of evicted positions, where `j` was evicted at time `t_j`, and suppose

```
sum_{j in E} e^{2B - pi_j(t_j)} <= eps.
```

Then at every reader position `i >= max_{j in E} t_j`, the head output with `E` removed differs from the exact output by at most `2 V eps`.

**Proof.** Monotonicity: `S_i - S_j` is a sum of entrywise non-negative vectors that grows with `i`, and `b_j >= 0`. For `i >= t_j`, the unnormalized weight of `j` is at most `e^{B - pi_j(i)} <= e^{B - pi_j(t_j)}`, and the reader's own weight is at least `e^{-B}` since its penalty is zero. So `alpha_ij <= e^{2B - pi_j(t_j)}`, and the evicted mass `alpha_E` is at most `eps`. Write the exact output as `(1 - alpha_E) o_keep + alpha_E o_E`, where `o_keep` is the output with `E` removed. The difference is `alpha_E ||o_E - o_keep|| <= 2 V eps`. QED.

**Remarks.**

- The earlier draft gave `2 V eps / (1 - eps)` and required `eps < 1`. The direct bound `2 V eps` is tighter and needs no condition on `eps`.
- The eviction test is a running sum of one scalar per evicted token.
- Under a scalar gate, penalties are ordered by age within a head, so a threshold test evicts a prefix. Typed gates can evict a newer dead token and keep an older live one.
- The bound is per head and per layer. A whole-network bound needs Lipschitz constants for the layers above.
- (A1) needs a real bound `B`. A retrofit can clamp content logits at a value above the range seen in the pretrained model.

## 6. Conjectures for deep models and matching experiments

| Claim | One layer | Deep models | Experiment |
|-------|-----------|-------------|------------|
| Plain attention loses the latest write as writes pile up | Theorem 1 | Conjecture. Later layers can mark stale tokens, so the one-layer argument does not transfer. | Accuracy against write count at fixed age and position. |
| Scalar gates serve about one register per head | Theorem 2 | Conjecture: at most `L H` registers. | Sweep `K` against `H` with one and two layers. |
| Typed gates serve `H r` registers | Theorem 3 | Achievable: a deep model can contain the one-layer construction, with later layers passing it through the residual stream. A matching upper bound is open. | Same sweep with typed gates. |
| Eviction is sound | Theorem 4 | Per-layer bound holds in each layer. End-to-end bound open. | Measure output error and memory after eviction. |

## 7. Status

| Result | State |
|--------|-------|
| Lemma 0, Fact F | Proved. Lemma 0 unit-tested. |
| Theorem 1 | Proved for one layer. Checked line by line on 2026-11-22 by the same author. Needs an independent reader. Tightness open. |
| Theorem 2 | Proved for one layer, with explicit constants. Checked line by line on 2026-11-22 by the same author. Needs an independent reader. |
| Theorem 3 | Proved by construction, with exact-zero gates. |
| Proposition 3b | Proved. Covers exact service only. |
| Theorem 4 | Proved. |
| Deep-model versions | Open. Conjectures only. |

## 8. Review log, 2026-11-22

| Location | Problem found | Fix |
|----------|---------------|-----|
| Setup, (A4) | "Any positional encoding" was too broad. Content-dependent schemes (CoPE, PaTH, Selective RoPE, stick-breaking, forget gates) violate (A4). | (A4) now names the admitted and excluded schemes. |
| Setup, (A3) | Norms and the role of the reader token were unstated. | (A3) now fixes the norms and holds the reader token fixed. |
| Theorem 1, statement | The ladder streams had `t <= c` writes, so the proof covered "at most `c` writes", not exactly `c`. | Added `c` padding writes before the ladder. Every stream now has exactly `c` writes. |
| Theorem 1, rate | The `ln c` term was dropped in the summary `O(HB / c)`. | Rate now reads `O(Lambda V H (B + ln c) / c)`. |
| Theorem 1, tightness | The recency example does not reach the margin bound: older writes can outvote the latest. | Claim withdrawn. Tightness marked open. |
| Theorem 2, Step 6 | The rescaling bound for type-1 heads was asserted without proof. | Added Fact F and a proof. |
| Theorem 2, limit | No explicit choice of `c, c'`. | Explicit sufficient conditions on `c` with `c' = c`. `eps0` now `min(1, gamma* / (10 Lambda V H))`. |
| Theorem 2, gates | Position dependence of `phi` was not excluded. | Gates depend on the token only. |
| Theorem 2, stream | Fixed fillers between ladder writes could emit large gates and break the telescoping bound. | Ladder writes now sit at consecutive positions. |
| Theorem 3 | Content logits, values, and readout were not specified, so "solves" had no margin. | Full construction with margin `1 - 4 delta`. |
| Proposition 3b | Text implied rank `K` is needed in general. | Scoped to exact service. Packing beyond `r` marked as a conjecture. |
| Theorem 4 | Bound `2 V eps / (1 - eps)` was loose. | Replaced with `2 V eps`. |
| Section 6 | Deep plain models: the stated reason ("a stale token cannot learn it is stale") is false, since a later layer can mark it. | Reason replaced. Deep typed-gate capacity stated as an upper bound only. |
