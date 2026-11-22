# Prior Art Notes: Attention Mechanisms Closest to Typed Forget Gates

Status: DRAFT IN PROGRESS (2026-11-11). Sections marked PENDING are not yet verified. Do not cite PENDING content.

## Purpose

These notes record what the closest prior work states, so that the novelty claims for typed forget gates stay exact. Typed forget gates use the logit `logit(i, j) = q_i . k_j - b_j . (S_i - S_j)` with `S_t = sum_{l <= t} u_l`, `u_l >= 0` in R^r, and `b_j >= 0` in R^r. Each entry below comes from a fetch of the arXiv full text through a small-model reader. Formulas appear as the reader returned them, transcribed to plain text. The "Unverified items" section lists every gap.

## Summary Table

PENDING. Filled after all papers are read.

## 1. Forgetting Transformer (FoX)

### 1.1 Citation

- Title: Forgetting Transformer: Softmax Attention with a Forget Gate.
- Authors: Zhixuan Lin, Evgenii Nikishin, Xu Owen He, Aaron Courville.
- Venue: ICLR 2025. The arXiv comments field reads "Published as a conference paper at ICLR 2025; Fixed an issue with the attention map visualization".
- arXiv: 2503.02130 (v1 on 3 March 2025, v2 on 31 March 2025).

### 1.2 Mechanism

- Forget gate: `f_t = sigmoid(w_f^T x_t + b_f)`, a real scalar. Quote: "we first compute a scalar forget gate f_t for each timestep t".
- Decay matrix: `F_ij = prod_{l=j+1}^{i} f_l` and `D_ij = log F_ij = sum_{l=j+1}^{i} log f_l`.
- Attention: `O = softmax(Q K^T + D) V`.
- Implementation form: "compute the cumulative sum c_i = sum_{l=1}^{i} log f_l" and "This allows us to compute D_ij = c_i - c_j easily later."
- Gates are per head: "we maintain H instances of forget gate parameters".
- Suppression type: (a). One scalar per head per token, computed from `x_t` alone. The gate at token l lowers the logit of every earlier key j < l by the same amount `log f_l`, for every later query. The gate does not read the content of earlier keys and does not read the query.
- The authors call Forgetting Attention "a data-dependent and learnable version of ALiBi". A fixed gate equals ALiBi.

### 1.3 Cost

- The paper gives no big-O statement for FoX. Quote: "since the forget gates are scalars instead of vectors, the additional computation and parameters introduced are negligible."
- Kernel: FoX "can be computed with a simple modification to the FlashAttention (Dao, 2023) algorithm." The authors wrote the kernel in Triton on top of Flag Attention, "without significant optimization". FoX does not run on an unchanged FlashAttention or SDPA kernel in the paper.
- T x T accumulation: none in memory. Quote: "This algorithm avoids instantiating the L x L D matrix in the HBM." The kernel builds `D` blocks on the fly from the cumulative sums `c`.
- The paper does not state that `c_i - c_j` can fold into augmented queries and keys for a stock kernel.
- Throughput on 4 NVIDIA L40S GPUs: FoX (Pro) 27k tokens/sec against 30k for Transformer (Pro). FoX (LLaMA) 30k against 38k for Transformer (LLaMA), which uses the official CUDA FlashAttention.

### 1.4 Training Setup

- All models train from scratch. The paper reports no conversion of a pretrained model.
- Data: LongCrawl64, "a long-sequence subset of RedPajama-v2".
- Main model: 760M non-embedding parameters, about 48B tokens, training context 16384 tokens, validation sequences of 65536 tokens.
- Analysis models: 360M parameters on about 7.5B tokens, 125M on about 2.7B tokens, 760M on about 16B tokens.
- Appendix F.3 adds a SlimPajama short-context run.
- Compute: the paper names 4 NVIDIA L40S GPUs for throughput. It gives no GPU-hour total.
- "Pro" block: output gates, output normalization, QK-norm, and a data-dependent KV-shift.

### 1.5 Evaluations

- Long-context language modeling on LongCrawl64 (per-token loss, perplexity).
- Length extrapolation: validation at 65536 tokens against a 16384-token training context.
- Needle-in-the-haystack (standard and easy mode).
- Short-context tasks: Wikitext, LAMBADA, PiQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, Copa, SciQA, OpenbookQA, BoolQ.
- Long-context tasks: 14 LongBench tasks (HotpotQA, 2WikiMultihopQA, MuSiQue, MultiFieldQA-en, NarrativeQA, Qasper, GovReport, QMSum, MultiNews, TriviaQA, SAMSum, TREC, LCC, RepoBench-P).
- Flip-flop: not tested. State tracking: not tested. Key-value overwrite: not tested. Proactive interference: not tested. The reader found none of these terms in the text.
- Length extrapolation: tested. Long-context benchmarks: tested.

### 1.6 KV Cache Eviction

- No method. One future-work sentence: "we could potentially prune computation (e.g., KV-cache eviction) adaptively based on the forget gate values".
- No error guarantee in this paper. Paper 2 supplies one.

### 1.7 Limitations and Future Work

- "our main experiments only use models up to 760M parameters, 48B tokens".
- "an important direction for future work is to test FoX at larger scales".
- "It would be interesting to extend Forgetting Attention to the non-causal case."
- "forget gates introduce a recency bias".
- Appendix: "length extrapolation is sensitive to hyperparameters."

### 1.8 Relation to Typed Forget Gates

- FoX is the r = 1 case of our logit with a constant signature. Set `b_j = 1` and `u_l = -log f_l >= 0`. Then `b_j . (S_i - S_j) = -(c_i - c_j) = -D_ij`.
- FoX already contains: a non-negative, monotone, prefix-sum penalty; a cumulative-sum implementation; the link to ALiBi; the idea of KV eviction from gate values.
- FoX does not contain: a key-side signature `b_j`, a vector gate, or any selectivity among earlier tokens. The paper says the gates "are scalars instead of vectors" as a cost argument, so the authors considered and set aside vector gates in the sense of LSTM-style per-dimension gates. The reader found no per-key or typed decay.
- FoX does not state that the penalty folds into augmented queries and keys. It uses a modified kernel.
- FoX states no theorem on register counts or on recency-ordered suppression. It names the recency bias in one sentence.

### 1.9 Evidence Grade

FULL. Read through the HTML in two windows (characters 0 to 100000 and 100000 to 129785). Not verified by direct reading: exact hyperparameter tables in Appendix B, and the numeric results tables.

## 2. Adaptive Computation Pruning for the Forgetting Transformer (ACP)

### 2.1 Citation

- Title: Adaptive Computation Pruning for the Forgetting Transformer.
- Authors: Zhixuan Lin, Johan Obando-Ceron, Xu Owen He, Aaron Courville.
- Venue: COLM 2025 (arXiv comments field: "Published as a conference paper at COLM 2025").
- arXiv: 2504.06949 (v1 on 9 April 2025, v2 on 11 August 2025).

### 2.2 Mechanism

- Same FoX logit. `D_ij = sum_{l=j+1}^{i} log f_l`, and in matrix form `D = c 1^T - 1 c^T` with `c_i = sum_{l=1}^{i} log f_l`.
- Pruning rule: drop terms with `D_ij < delta`.
- Threshold: `delta = -2U - log L + log eps`, with `U` an upper bound on `|q_i^T k_j| / sqrt(d)`. With QK-norm, `U` comes from the learned norm scales. The paper sets `eps = e^(-10)`.
- Suppression type: (a), inherited from FoX. Scalar gate per head per token.

### 2.3 Cost

- Pruning acts on FlashAttention computation blocks: "ACP operates at the block level".
- The pruned set is a staircase: "the set of computation blocks to be pruned constitutes a consecutive region on the lower-left part of the M x N grid."
- Boundary search costs linear time in the grid dimensions. The paper does not claim sub-quadratic attention.
- Reported savings: "ACP consistently prunes around 70% of the FLOPs and memory accesses in softmax attention in all cases", "a roughly 50% to 70% reduction in attention runtime", and "a roughly 10% to 40% increase in end-to-end training throughput."
- Needs the modified FoX kernel. No T x T matrix in memory.

### 2.4 Training Setup

- FoX (Pro) models trained from scratch on LongCrawl64, with and without ACP.
- Sizes: 125M parameters on 2.7B tokens, 360M on 7.5B, 760M on 16B. Context lengths 4k, 8k, 16k.
- Throughput measured on 4 NVIDIA L40S GPUs.
- No pretrained-model conversion.

### 2.5 Evaluations

- Validation loss, WikiText perplexity, LAMBADA perplexity and accuracy.
- PIQA, HellaSwag, Winogrande, ARC-easy, ARC-challenge, COPA, OpenBookQA, SciQ, BoolQ.
- Easy-mode needle-in-a-haystack.
- Flip-flop, state tracking, key-value overwrite, proactive interference: not tested. Length extrapolation and long-context benchmarks: the reader listed none.

### 2.6 KV Cache Eviction

- Guarantee: for any i, `sum_j 1{D_ij < delta} A_ij < eps`. Appendix A proves that `D_ij < delta` implies `A_ij < eps / L`. The abstract calls this "provably safe pruning".
- The bound covers total pruned attention weight per query row. It relies on a logit bound `U`.
- Eviction proposal for decoding: "Since j never decreases, we can discard any KV-cache entries beyond the pruning boundary, thus reducing both memory consumption and memory accesses".
- The paper did not implement eviction. Its decoding test used "a naive implementation that does not perform explicit KV cache eviction", and "we do not see a clear improvement in end-to-end decoding throughput."

### 2.7 Limitations and Future Work

- "A more thorough investigation of inference-time ACP is left to future work."
- Decoding is "likely bottlenecked by the kernel launch overheads".
- The main results cover pretraining only.

### 2.8 Relation to Typed Forget Gates

- ACP already contains the core eviction argument: a monotone decay bias plus bounded content logits gives a per-row bound on dropped attention mass, and the monotone boundary permits KV eviction. Our eviction bound must cite ACP as the scalar case.
- ACP evicts in recency order only. Its boundary is one index per head: all keys older than the boundary go. A typed gate would evict by key signature, so an old key with a non-matching signature can stay while a newer matching key goes. ACP does not contain per-key eviction.
- ACP bounds attention mass, not output error, as far as the reader reported.

### 2.9 Evidence Grade

FULL. The HTML fit in one window. Not verified by direct reading: the exact wording of the Appendix A proof and the numeric tables.

## 3. Selective Attention Improves Transformer

### 3.1 Citation

- Title: Selective Attention Improves Transformer.
- Authors: Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research).
- Venue: PENDING check of the arXiv comments field.
- arXiv: 2410.02703 (v2 on 24 April 2025).

### 3.2 Mechanism

- Attention: `softmax(Q K^T / sqrt(d_k) - F) V`. Quote: "The matrix F = Accumulate(Constrain(S)) is then simply subtracted from the attention logits before applying the softmax."
- Selection matrix: "we instead simply reuse the result of one of the existing heads of the attention module." Code: `S = attn_logits[:, 0]  # Select head 0.`
- Constraints: "Zero out negative values (i.e. applying ReLU), only reducing attention, never boosting it." "Zero out the first column, so as not to mask the <BOS> token." "Zero out the diagonal, so as not to let a token mask itself."
- Accumulation: `F_{i,j} = sum_{k <= i-1} S_{k,j}`. Code: `F = np.cumsum(S, axis=-2)  # Accumulate.` then `attn_logits -= F[:, None]`.
- The same `F` applies to all heads of the layer.
- Suppression type: (b). Token k masks token j by an amount that depends on the content of both k and j. The reading query i does not set the mask. Quote: "the masking by a token would not affect its own attention operation."
- `F` never decreases along i because the summands pass through ReLU.

### 3.3 Cost

- "adds no new parameters and only a negligible amount of computation". A footnote compares O(h n^2) for the extra work with O(d n^2) for attention.
- The method needs the T x T matrix `S` and its column-wise cumulative sum `F`.
- No FlashAttention kernel. Future work: "selective attention can be implemented in a GPU-aware way, similar to Flash Attention."

### 3.4 Training Setup

- From scratch only: "We only experimented with pre-training models with selective attention."
- Decoder-only models with d from 8 to 28. Table 8 lists 33,603,584 to 1,108,645,888 parameters.
- Data: C4 with an 8K vocabulary. WikiText and lm1b gave "similar results".
- Context sizes 512, 1,024, 2,048. Trained on TPUv4s for 524,288 steps. No total compute figure.

### 3.5 Evaluations

- C4 validation perplexity.
- ARC (Easy), ARC (Challenge), HellaSwag, PiQA, CommonSenseQA, OpenBookQA.
- T5 span corruption (Appendix A.9).
- Synthetic: Variable Assignment, Copy, Parity*.
- Key-value overwrite: tested. Variable Assignment is an overwrite task. Quote: "for the input: y=7; x=1; x=3; z=5; x=? the output is 3." Setup: "3 variables, 1,000 possible values, and 128 assignments", d = 3. Selective attention "reaches a validation loss of 0.002 (and 100% accuracy)". The standard transformer "only achieves a validation log-perplexity loss of 3.18 and 26% accuracy after 1,000 step."
- Flip-flop: not tested under that name. State tracking beyond Variable Assignment and Parity*: not tested. Proactive interference: not tested. Length extrapolation: not tested. Long-context benchmarks: not tested.

### 3.6 KV Cache Eviction

- Method: fixed per-layer budgets. "when processing the first K_l tokens in layer l we drop nothing, and for each following token we drop the single not-yet-dropped past token with the highest F value."
- Budgets come from a greedy search with step 8 that stops when "model performance reaches a predefined threshold."
- An auxiliary memory loss raises the savings.
- Reported attention-memory reductions for d = 12: 5X, 7X, 8X at context 512, 1,024, 2,048 with the standard loss, and 16X, 25X, 47X with the memory loss.
- No error guarantee. The paper states no bound.

### 3.7 Limitations and Future Work

- "there are still important variants we did not test, notably transformers with multi-query ... and grouped-query ... attention, as well as models much larger than 1B parameters".
- "We did not further train the models after removing elements".
- "It is interesting to investigate how it could be applied in a fine-tuning step to existing models".
- "Reducing the size of the context as in Section 4 improves inference efficiency but not training efficiency."

### 3.8 Relation to Typed Forget Gates

- This paper is the closest prior mechanism. It already contains: content-selective suppression of specific earlier tokens by later tokens; independence from the reading query; a non-negative, monotone penalty; accumulation by cumulative sum; a demonstration on a key-value overwrite task; selection-driven context pruning.
- Our logit equals Selective Attention with a factorized selection function. Selective Attention uses `S_{k,j} = ReLU(q_k^0 . k_j^0)` from head 0. We use `S_{k,j} = b_j . u_k` with `b_j, u_k >= 0`. Moving the non-negativity inside the bilinear form makes the sum over k separable, so `F_{i,j} = b_j . (S_i - S_j)` needs O(T r) memory and no T x T accumulation.
- The paper tested "a separate bilinear form" for the selection function (Appendix A.3). It did not test a non-negative low-rank factorization, a prefix-sum form, or a FlashAttention-compatible form.
- One index difference: Selective Attention sums maskers `k <= i-1`. Our `S_i` includes `u_i`.
- Selective Attention shares one mask across all heads of a layer. Our penalty can differ per head.
- The paper gives no eviction bound and no theory on register counts.

### 3.9 Evidence Grade

FULL. The HTML fit in one window. Not verified: venue, and numeric tables.

## 4. PaTH Attention

PENDING. Verified so far: title "PaTH Attention: Position Encoding via Accumulating Householder Transformations"; authors Songlin Yang, Yikang Shen, Kaiyue Wen, Shawn Tan, Mayank Mishra, Liliang Ren, Rameswar Panda, Yoon Kim; arXiv 2505.16381. Logit: `A_ij proportional to exp(k_j^T (prod_{s=j+1}^{i} H_s) q_i)` with `H_t = I - beta_t w_t w_t^T`.

## 5. Contextual Position Encoding (CoPE)

PENDING. Verified so far: gate `g_ij = sigmoid(q_i^T k_j)`, position `p_ij = sum_{k=j}^{i} g_ik`, logit `q_i^T k_j + z_i[p_ij]`. The gate depends on the query.

## 6. Scaling Stick-Breaking Attention

PENDING. Verified so far: `A_{i,j} = beta_{i,j} prod_{i<k<j} (1 - beta_{k,j})` with `beta_{i,j} = sigmoid(z_{i,j})`.

## 7. Selective RoPE

PENDING.

## 8. GRAPE

PENDING.

## 9. Flip-Flop Language Modeling

PENDING.

## 10. Simulating Hard Attention Using Soft Attention

PENDING.

## 11. softmax is not enough

PENDING.

## Search for Closer Prior Art

PENDING.

## Threats to Novelty

PENDING.

## Unverified Items

PENDING.
