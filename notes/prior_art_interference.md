# Prior art on proactive interference in LLMs

Status: DRAFT IN PROGRESS, written 2026-11-11. Sections marked PENDING are not yet filled.

## Purpose

These notes record what the diagnosis papers, benchmarks, and existing cures for proactive interference (PI) in LLMs report, so that the typed forget gate project can replicate the right benchmarks and state its novelty against verified prior work. Each claim below comes from the source text read on 2026-11-11. Quotes replace source dashes with hyphens. Items I could not verify sit in the last section.

## Paper 1: Unable to Forget (PI-LLM), arXiv 2506.08184

### Citation

- Title: Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length.
- Authors: Chupei Wang (University of Virginia), Jiaqiu Vince Sun (New York University, Center for Neuroscience). Equal contribution.
- Venue: ICML 2025 Workshop on Long Context Foundation Models (arXiv comment field). The dataset card also claims an AAAI 2026 workshop oral (LaMAS, January 2026, Singapore) for an updated version with an interpretability perspective.
- arXiv id: 2506.08184. Latest version v3, dated 2025-07-31 (v1 dated 2025-06-09). Paper license CC BY 4.0.

### Task and benchmark specification

- Structure of each input: (1) instruction naming the keys to track, (2) update stream of key value pairs, (3) query for the final value of each tracked key.
- Keys: 46 category names. Values: 400 words or short phrases per category. Source file `testing_data/dict_category_double-word_46-400_v1-1.json` in the code repository (46 keys, 400 values each, confirmed by loading the JSON).
- Key list (from the dictionary file): bird, material, African place, profession, dessert, built environment, music, bug, body, climate, American place, mythology, emotion, plant, weapon, furnishing, illness, aircraft, condiment, action, clothing, gaming, dance, programming, law, water body, European place, measurement, landform, tools, visual art, Tech, sport, beverage, land vehicle, fruit, festival, cosmos, Asian place, watercraft, quodruped (spelling as in source), physics, electronics, dish, chemical, aquatic animal.
- Stream order: updates for different keys are interleaved at random. The paper states the same key does not appear in two consecutive pairs. Each key receives the same number of updates. Values within a prompt are all different (dataset card).
- Prompt template, copied from row 0 of the `core` config of the Hugging Face dataset (single user message, no system message):
  - Opening: `As my secretary, I need you to carefully read a text stream where the values of multiple keys are being continuously updated.The 46 keys to track include <key1>, <key2>, ..., <key46>. I will ask you to identify the value of each key later.\n\nThe text stream starts on the next line.\n `
  - Stream: `tools: bee brush; profession: dermatologist; landform: Juneau Icefield; ...` Each pair has the form `<key>: <value>; `.
  - Query: `\n\nWhat are the current value of each key (<key1>, <key2>, ..., <key46>) you are tracking? End your response with: 'The current value of <key> is <value>.'`
  - The value length experiment appends: `Provide the exact current value string without modification or breaking it into pieces.`
  - The missing space after "updated." is in the source.
- Expected answer format: one line per key, `The current value of <key> is <value>.`
- Experiments and grids (paper sections 2 to 4, code configs, and dataset rows agree):
  - Updates per key (`exp_updates`, config `test1_updates`): 46 keys, updates per key in [2, 3, 4, 6, 8, 12, 17, 24, 34, 48, 68, 97, 139, 197, 281, 400]. The paper text says 3 to 400.
  - Number of keys (`exp_keys`, config `test2_nkeys`): keys in [2, 3, 4, 5, 6, 8, 11, 15, 20, 26, 35, 46], updates fixed at 125 or 350.
  - Tracked keys at fixed length (config `test3_ntracked`): 46 keys updated, tracked keys in the same 12 point grid, updates 125 or 350. This experiment is in the code but not in the Hugging Face dataset.
  - Value length (`exp_valuelength`, config `test4_itemlen`): 46 keys, updates 4 or 20, value length in [1, 2, 3, 4, 6, 10, 16, 25, 40] words. Values are several dictionary words with spaces removed and first letters capitalised, concatenated (example in paper: AppleOrangeBanana). The paper reports the 20 update setting.
  - Sequential order (`extra_exp_updates_randomoff`, config `test5_updates_randomoff`): same 16 point update grid, `random_update: 0`.
- Context lengths (dataset column `n_tokens`, tokenizer o200k_base per the card code): `exp_updates` runs from about 0.8K tokens at 2 updates to about 100.7K tokens at 400 updates. `exp_keys` runs from about 1.3K to 88.2K. `exp_valuelength` runs from about 1.3K to 102.5K.
- Dataset size: `core` has 580 rows (160 updates, 240 keys, 180 value length), 10 sessions per condition. `sequential_additional` has 160 rows.
- Decoding in the code configs: temperature 0, max_tokens 1024. Number of sessions set by a CI95 width threshold when `n_sessions: 0`.
- Metric: per prompt accuracy = fraction of tracked keys whose parsed answer equals the final value by exact string match. The grader parses `The current value of X is Y` patterns and `X: Y` lines.
- Aggregate metric in the paper: Interference Endurance Score (IES) = area under the accuracy curve across log scaled update counts.
- Aggregate metric on the dataset card: `auc_log1.5` = weighted mean accuracy with weight log base 1.5 of max(n_updates, 2). For `exp_keys` and `exp_valuelength` the card also reports easy (updates at most 125, or at most 4) and hard (at least 350, or at least 20) splits. Total score = mean of per experiment `auc_log1.5`.

### Models tested and headline numbers

- More than 30 models, 0.6B (Qwen3-0.6B) to 637B (DeepSeek V3) parameters, plus GPT, Claude, Gemini, Grok. Appendix B lists: gemini-2.5-flash-preview-04-17, gemini-2.0-flash, gemini-1.5-flash-8b, gemini-2.0-flash-thinking-exp-01-21, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o-2024-11-20, gpt-4o-mini, deepseek-chat, deepseek-reasoner, qwen2.5-72b-instruct, qwen3 0.6b, 1.7b, 4b, 8b, 14b, 32b, 30b-a3b, 235b-a22b (each with and without thinking), llama-4-maverick, llama-4-scout, llama-3.1-405b, llama-3.2-90b-vision, llama-3.1-70b, llama-3.1-8b, grok-3-beta, grok-3-mini-beta, claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, mistral-small-2503, three NVIDIA Nemotron models. Experiments ended 2025-05-05.
- Headline result: accuracy falls log linearly with updates per key for every model (Figures 1 and 3). The paper gives curves, not a table. No per model numbers appear in the text for Qwen3 small models.
- Numbers stated in the text:
  - At 400 updates per key, M group models fall to 0% with one exception. L group models stay higher (Figure 3).
  - llama4-maverick: near 100% with 2 tracked keys, below 5% with 46 tracked keys at fixed input length (Figure 9).
  - Value length: 10 words pushes every model below 40%. 40 words pushes accuracy under 5%, except Grok-3-beta at about 0.1 (Figure 10, 20 updates).
  - Regression of IES on size class and context length, N = 30 open models: size class t = 3.03, p = 0.005. Context length t = -0.144, p = 0.886. R squared 0.261. For models with 128K to 131K context, Spearman rho squared = 0.673, p = 0.0016 (Figure 6).
- MoE models match or trail dense models of similar total size (Figure 7).
- Sequential order (Figure 4): accuracy stays near perfect until a model specific threshold, then drops to near zero.
- Dataset card claims beyond the paper: GPT-5, Grok-4, Gemini 2.5 Pro also fail. In sequential mode, models under about 600B confuse values after 50 to 100 updates. The card gives no numbers for these claims.

### Reasoning or thinking modes

- Tested in section 6 and Figure 14 with four pairs: gemini-2.0-flash versus gemini-2.0-flash-thinking (and gemini-2.5-flash with CoT), deepseek-V3 versus deepseek-R1, and two NVIDIA Nemotron pairs.
- Result: in three of four pairs the reasoning variant is equal or worse. deepseek-R1 trails deepseek-V3 beyond 12 updates. Reasoning helps only nvidia-llama-3.1-nemotron-ultra, the weakest base model.
- Reasoning variants were tested only up to about 100 updates because the thinking output overflowed.
- Reasoning models were excluded from the IES regression because latency exceeded 200 seconds per task.
- Appendix B lists qwen3 thinking variants for all sizes, but the text reports no numbers for them.

### Mitigations tried and their effect

- Five prompt interventions inserted into the stream (Figures 11 and 12), tested with 46 keys:
  - Per key forget: "Forget all the previous updates to key". Gain under 10 percentage points at 100 updates. It hurt gpt-4.1-mini. Errors clustered at the position of the instruction (Figures 13 and 20).
  - Forward focus (example: "pay close attention to the following updates"): limited effect.
  - Relevance meta prompt: no gain for any model, harmful for gpt-4.1-mini.
  - Soft session reset (verbal cue that a new session begins): largely ineffective.
  - Mock QA reset: a fake user query, a fabricated assistant answer, and a new user turn, inserted 120 updates before the final query. This gave a substantial gain for all tested models. Example: Gemini 2.0 Flash at 150 updates matched its baseline at 30 to 45 updates. Accuracy still declined with more updates.
- No fine tuning, no architecture change, no attention intervention.
- The authors note the mock QA reset discards all earlier content, so it cannot serve tasks that need selective access to history.

### Mechanistic claims

- The paper is behavioural. It reports no attention or activation analysis.
- Error analysis (Figures 5 and 19): at low load errors come from updates just before the final one. At medium load errors come from much earlier updates. At high load models return values that never appeared, and many remaining errors anchor on the earliest bins. The authors call this a persistent primacy bias.
- The authors hypothesise that LLMs lack an unbinding or gating mechanism of the kind human working memory has (Oberauer and Vockenberg 2009).
- The authors read the mock QA reset result as evidence that a gating mechanism could reduce interference.
- Section 9.2.1 names induction heads as a candidate mechanism without testing it.

### Released artifacts

- Code: https://github.com/zhuangziGiantfish/Unable-to-Forget (MIT license, last push 2025-11-04). Contains `core/run_pi.py`, `core/pi_flow_upgrade.py`, five YAML test configs, and the word dictionary.
- Dataset: https://huggingface.co/datasets/Cog2ai/pi-llm-bench (MIT license). Configs `core` (core.parquet) and `sequential_additional` (sequential_additional.parquet). Columns: prompt, answer_natural_language, answer_formatted, experiment, session_id, n_updates, n_keys, value_length, randomize_mode, n_tokens, n_chars, n_words. The quick start code on the card loads `giantfish-fly/pi-llm`, an older repository id.
- Demo site: https://sites.google.com/view/cog4llm (not read).

### Open questions stated by the authors

- Natural language control of memory and attention "remain an open challenge".
- The authors call for architectures that can adjust memory representations and retrieval in response to cues, and for "targeted advances in anti-interference mechanisms or executive control within model architectures".
- Selective suppression: the reset trick drops all earlier content, and real tasks need selective access to history.
- The dataset card says mechanistic research is ongoing.

### Evidence grade

FULL for the paper (v3 HTML read end to end), the dataset card, the code configs, and the dictionary file. Not verified: figure contents (curves are images, so no per model numbers), the AAAI 2026 workshop version, the demo site. One discrepancy: the card says the sequential config uses strict per key blocks, but the three rows I sampled from `sequential_additional` through the datasets server API show interleaved keys. Check the parquet before relying on that config.

## Paper 2: LLMs Remember First, Forget Last, arXiv 2603.00270

### Citation

- Title: LLMs Remember First, Forget Last: Dual-Process Interference in Large Language Models.
- Authors: Sourav Chattaraj, Kanak Raj (Thomson Reuters, Bengaluru).
- Venue: arXiv comment field says NPIR, Springer Lecture Notes in Networks and Systems. Year 2026.
- arXiv id: 2603.00270. v1 dated 2026-02-27. Latest version v3, dated 2026-08-10.

### Task and benchmark specification

- Paradigm: AB-AC paired associates. The same input serves two probes. RI probe: return the first value assigned to each category. PI probe: return the most recent value.
- Keys: 46 semantic categories (examples: visual art, musical instrument, programming language). Values: 478 to 624 real instances per category.
- Each category appears N times, N in {3, 10, 50, 100, 200, 300}: one initial assignment plus N-1 interleaved updates. Consecutive updates never target the same key.
- Pair format shown in the text: `visual art: impressionism`. The paper does not print the full prompt or query wording (Figure 2 is an image).
- Decoding: temperature 0. Each session draws a fresh random sequence.
- Sessions: run until the 95% bootstrap CI (10,000 resamples) is narrower than 0.10 for two consecutive sessions, minimum 5, cap 50 (cap 20 for reasoning models). Sessions that extract under 50% of categories count as parse failures and are excluded.
- Metric: accuracy per N. Summary scores RIES and PIES = integral of accuracy (percentage points) over log10(N+1) from N = 3 to 300. Maximum about 188.
- Context lengths are not stated in tokens.

### Models tested and headline numbers

- 54 models tested, 39 with complete data, 1B to 2.5T parameters (proprietary sizes are estimates). Includes GPT-5, Claude 4.5, Gemini 2.5, o1, o3, Llama, Qwen, DeepSeek.
- PI accuracy drops from 94% at N = 3 to 3% at N = 300, averaged over 39 models (Figure 1).
- RI accuracy: 93.9% at N = 3, 96.0% at N = 50, 72.9% at N = 100, 45.5% at N = 200, 24.6% at N = 300.
- All 39 models have RIES above PIES. Mean gap 69.5 points, paired t(38) = 10.71, Cohen d = 1.73.
- RIES and PIES correlate weakly: r = 0.21, p = 0.201.
- Size predicts RIES (R squared 0.491) but not PIES (R squared 0.06, p = 0.14). Context window predicts neither.
- PIES values from Table 4 (higher is better, maximum about 188): Claude-4.5-opus 170.3, Claude-4-sonnet 137.8, Claude-3-opus 131.5, Claude-4-opus 131.0, Claude-4.5-haiku 128.5, Claude-3.5-haiku 126.4, Claude-4.5-sonnet 124.2, Gemini-2.5-pro 123.6, Qwen-3-VL-235b 120.2, GPT-3.5-turbo 120.0, GPT-4.1 118.6, GPT-5-mini 109.1, GPT-5 99.2, Qwen-3-coder-30b 89.3, Llama-3.1-8b 57.1, Llama-3.2-11b 47.6, GPT-5-nano 21.0, Llama-3.2-3b 16.8, Llama-3.2-1b 1.0.
- No Qwen3 model under 30B appears in Table 4.

### Reasoning or thinking modes

- Six reasoning models: o1, o1-preview, o3, o3-mini, o4-mini, DeepSeek-R1.
- RI: reasoning models average RIES 178.0 against 150.3 for standard models (t(37) = 3.54, p = 0.001).
- PI: no benefit. PIES values: o1 22.0, o4-mini 22.0, DeepSeek-R1 55.6, o3-mini 75.6, o3 76.0, o1-preview 76.0. o1 ranks 1st on RIES and 35th on PIES. Table 3 labels reasoning models as lowest on PI resistance.

### Mitigations tried and their effect

- None tested. The authors list recency aware attention re-weighting, deliberate context ordering, and positional debiasing as future work.

### Mechanistic claims

- Claim: causal self attention gives early tokens extra influence (attention sink), which protects first values and hurts recent ones. The authors call this primacy protection.
- Claim: PI is a winner take all competition in softmax attention. Evidence: at N of 200 or more, PI sessions are bimodal, either above 85% or below 5%.
- Error analysis over 9,753 PI errors and 5,409 RI errors (Table 2): PI errors are 56.1% same key intrusions, 42.5% retrieval failures, 0.7% cross key, 0.2% hallucination. RI errors are 50.8% retrieval failures and 46.0% same key intrusions.
- PI intrusions come from early positions: mean normalised position 0.12 at N = 3, with 14% returning the first value. Positions converge toward the midpoint as N grows (Figure 6).
- 8.6% of PI retrieval failures contain an explicit refusal, against 0% for RI.
- The authors state these accounts are post hoc and unconfirmed by attention probing.
- Difference from Paper 1: Paper 1 reports many never-seen values at high load. This paper reports 0.2% hallucination among PI errors.

### Released artifacts

- None found. The HTML contains no code or dataset link for this work.

### Open questions stated by the authors

- Whether the PI over RI asymmetry is architectural or learned (they propose attention head ablation and training checkpoint comparisons on open models).
- Whether attention re-weighting or positional debiasing reduces PI without hurting RI.
- Transfer to naturalistic tasks such as clinical logs and legal filings.

### Evidence grade

FULL (v3 HTML read end to end, including Table 4). Not verified: figure contents, exact prompt and query wording, the value pool.

## Paper 3: SleepGate, arXiv 2603.14517

### Citation

- Title: Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models.
- Author: Ying Xie (Kennesaw State University).
- Venue: arXiv preprint, 2026. No venue listed.
- arXiv id: 2603.14517. Only version v1, dated 2026-03-15.

### Task and benchmark specification

- Not the PI-LLM dataset. A synthetic token level analogue.
- Vocabulary 1024 synthetic tokens. 100 entity keys, 500 values.
- Each episode: updates for a single entity (e, v1), (e, v2), ..., (e, vn), then a query for the most recent value.
- PI depths n in {1, 2, 5, 10, 15, 20, 30}, 200 episodes per depth.
- Maximum sequence length 1024.
- Metrics: exact match retrieval accuracy, stale retrieval rate (prediction equals a superseded value), PI slope (slope of accuracy against log n).
- No prompt template. Inputs are token sequences for a model trained from scratch.

### Models tested and headline numbers

- One model: 4 layer causal transformer, d_model 128, 4 heads, feed forward 256, trained from scratch. Base 793,344 parameters. Tagger 16,576, forgetting gate 74,241, consolidation 33,152. Total 917,313 (15.6% overhead).
- Table 2, accuracy / stale rate in percent:

| Method | n=1 | n=2 | n=5 | n=10 | n=15 | n=20 | n=30 |
|---|---|---|---|---|---|---|---|
| SleepGate | 82.5 / 0.0 | 99.0 / 0.5 | 99.5 / 0.0 | 97.0 / 3.5 | 73.5 / 23.5 | 30.5 / 55.5 | 16.5 / 62.0 |
| StreamingLLM | 17.5 / 0.0 | 18.0 / 9.0 | 10.0 / 28.5 | 6.0 / 19.5 | 6.0 / 22.5 | 4.0 / 31.5 | 5.5 / 29.0 |
| Sliding window | 12.5 / 0.0 | 12.5 / 7.0 | 8.0 / 23.5 | 8.5 / 23.0 | 6.5 / 23.0 | 4.0 / 22.5 | 2.0 / 30.0 |
| Decay only | 9.5 / 0.0 | 12.5 / 7.0 | 7.0 / 22.0 | 7.5 / 23.0 | 8.5 / 25.0 | 3.5 / 23.5 | 4.5 / 22.5 |
| Full KV cache | 8.0 / 0.0 | 9.0 / 7.5 | 3.5 / 21.0 | 2.0 / 17.0 | 3.0 / 20.0 | 1.5 / 18.5 | 2.5 / 19.5 |
| H2O | 6.5 / 0.0 | 7.5 / 5.5 | 1.0 / 7.5 | 4.0 / 11.0 | 3.0 / 11.5 | 0.0 / 10.0 | 0.5 / 9.0 |

- The full cache baseline scores 8.0% at n = 1, where no interference exists. The baseline model has not learned the retrieval task, so the comparison says little about pretrained LLMs.
- No pretrained LLM is tested. No PI-LLM dataset numbers.

### Reasoning or thinking modes

- Not tested.

### Mitigations tried and their effect

- SleepGate is the mitigation. Three modules over the KV cache:
  - Conflict aware temporal tagger: a learned projection of each key vector plus a local pool gives a semantic signature (d_s = 64). An entry is flagged superseded when a later entry has cosine similarity above delta = 0.85.
  - Forgetting gate: 2 layer MLP (hidden 128, GeLU, sigmoid) maps features [key, value, age encoding, signature, superseded flag, cumulative attention, context summary] to a retention score r in [0, 1].
  - Consolidation module: merges entries marked for compression into one summary entry per cluster. Hard eviction variant only.
- The experiments use the soft biasing variant only: additive pre-softmax bias b_i = beta * log(max(r_i, eps)), beta = 5, applied in a second forward pass. Hard eviction and consolidation are specified but not evaluated.
- Trigger: attention entropy above running mean plus 1.5 standard deviations, or superseded fraction above 0.4, or every 128 tokens.
- Training: loss = wake LM loss + 0.5 sleep retrieval loss + 0.1 compression loss + 0.3 gate alignment loss. Gate alignment uses binary cross entropy against the tagger flags. 45 epochs: 10 warm start, 5 gate pre-training on ground truth supersession labels, 30 joint. AdamW, lr 3e-4, batch 16. Depth curriculum from n at most 5 up to 30.
- All base model parameters train jointly (Algorithm 2 backpropagates through the transformer, the gate, and the tagger).
- Effect: 97.0% to 99.5% for n from 2 to 10. Collapse beyond n = 15: 73.5% at 15, 30.5% at 20, 16.5% at 30, with stale rate 62.0% at 30.
- The gate reaches 99.3% accuracy on supersession labels after 5 epochs of pre-training.

### Mechanistic claims

- Claim: with n same key entries, attention weights are about uniform, so p(retrieve v_n) is about 1/n. The paper offers this as the explanation of log linear decay. No measurement supports it.
- Theorem 1: if the gate identifies superseded entries with probability at least 1 - eps and cycles run every N tokens, the expected count of stale entries is O(max(N, eps * n)). Proof sketch only. This is a bound on a count of entries, not on output error.
- Corollary 1: p(retrieve v_n) is at least 1 / (1 + O(N)).
- Failure analysis: soft bias saturation and limited signature capacity (d_s = 64) explain the collapse at n of 15 or more. This is the author's reading. No ablation tests it.
- H2O does worst because cumulative attention is anti-correlated with freshness under PI (author's reading).

### Released artifacts

- None. The HTML has no code or data link.

### Open questions stated by the author

- "Whether the mechanism transfers to pre-trained models at scale and generalizes to naturalistic PI patterns (e.g., document corrections, evolving facts) remains to be validated."
- Future direction 1: integrate into pretrained models (Llama-3, Mistral) by post hoc fine tuning.
- The depth 15 cliff.
- Multi-entity PI, long document QA with corrections, RULER and LongBench checks that non-PI performance holds.
- Interpretability of what the gate learns.
- Composition with GQA, PagedAttention, quantised caches.

### Evidence grade

FULL (v1 HTML read end to end, including Appendix A and B). Not verified: Figure 2, Appendix C (neuroscience related work, skipped).

## Paper 4: Length Is Cheap, Competition Is Not, arXiv 2611.04169

### Citation

- Title: Length Is Cheap, Competition Is Not: Attention Predicts Context Rot in Retrieval.
- Author: Yinuo Liu (UC San Diego).
- Venue: arXiv preprint, 2026. 15 pages.
- arXiv id: 2611.04169. Only version v1, dated 2026-11-03.

### Task and benchmark specification

- Raw text prompt, no chat template. Structure from the dataset card and `harness/measure_c2.py`:
  - Prefix: `You are a careful assistant. Read the document and answer the question.`
  - Evidence: ` The secret code of the {key} is {val}.`
  - Query: `\nQuestion: what is the secret code of the {key}? Answer: The secret code of the {key} is`
  - Scored continuation: ` {val}`
- Keys: 51 common nouns: apple river stone cloud paper window garden silver motor yellow forest candle ladder bottle mirror pencil castle rocket marble tunnel violin anchor basket cactus dragon falcon harbor island jacket kettle lemon magnet needle orange pillow quartz saddle tiger umbrella velvet walnut zipper bridge camera desert engine feather glacier hammer insect jungle.
- Values: four digit integers drawn from 1000 to 9998.
- Four conditions per trial with shared evidence, query, and length L:
  - short: no filler.
  - neutral: random words, random numbers, or the repeated sentence "The weather report for the coastal region was unremarkable."
  - competing (`near_distractors`): statements in the evidence template with keys drawn from the 51 nouns. About 1 in 51 restates the queried key with a different value (a rival). The dataset card says this was not designed.
  - masked: filler hidden by an attention mask, positions preserved.
- Explicit distractors: 0 to 5 extra statements with a different key (1 to 4 in the attention runs).
- Evidence depth: uniform over the context (0.1 to 0.9 in `measure_c2.py`).
- Lengths: L in {512, 1024, 2048} for all models, up to 32,768 where the window allows.
- Rival counts: mean 0.8 at 512 tokens, 3.2 at 2,048, 50.5 at 32,768. Every context of 4,096 tokens or more has at least one rival.
- Metrics:
  - Loss in nats: log p(short) minus log p(condition), where p is the teacher forced probability of the value, summed over value tokens.
  - Accuracy: a trial is correct when every value token is the top 1 prediction.
  - a*: attention from the final query token to the evidence tokens, summed over evidence tokens, averaged over the 8 (layer, head) pairs with most evidence attention in the short context. No renormalisation for the attention sink.
  - Predictor comparison: R squared of univariate least squares fits of loss on a*, log(1/a*), and log L.
- Hardware and precision: float16, PyTorch SDPA attention.

### Models tested and headline numbers

- Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, OLMo-2-0425-1B-Instruct, Llama-3.2-3B (base). Qwen2.5-7B-Instruct only in the MMLU replication.
- Neutral filler: accuracy 0.97 to 1.00 on four of five models. Qwen2.5-1.5B at 8K, 16K, 24K, 32K: 1.00, 0.99, 1.00, 0.98 (Table 2). OLMo-2-1B is the exception (0.91 at 2,048).
- Competing filler on Qwen2.5-1.5B at 8K, 16K, 24K, 32K: median loss 2.82, 3.94, 4.03, 4.76 nats, accuracy 0.11 to 0.21. Qwen2.5-0.5B: loss 1.79 to 4.07 nats, accuracy 0.17 to 0.40.
- Mean loss in nats without and with a same key rival, L at most 2,048 (Table 3): Qwen2.5-0.5B 0.175 and 1.812. Qwen2.5-1.5B 0.020 and 2.238. SmolLM2-1.7B 0.022 and 1.381. OLMo-2-1B 0.058 and 2.796. Llama-3.2-3B 0.344 and 3.239.
- Accuracy 0.97 to 1.00 without a rival, 0.25 to 0.52 with at least one.
- Dose response on Qwen2.5-1.5B at 2,048 tokens for 0 to 6 or more rivals: 0.03, 1.37, 2.24, 2.48, 2.71, 3.26, 3.45 nats.
- Format only distractors explain at most 0.3% of loss variance.
- Predictors of loss (Table 4), R squared for log(1/a*) and log L: Qwen2.5-0.5B 0.35 and 0.03 (n = 32,318). Qwen2.5-1.5B 0.58 and 0.05 (n = 40,632). SmolLM2-1.7B 0.41 and 0.07 (n = 9,325). OLMo-2-1B 0.35 and 0.02 (n = 42,005). Llama-3.2-3B 0.45 and 0.08 (n = 2,193). Total 126,473 trials.
- a* falls from 0.51 to 0.77 in the short context to 0.17 to 0.31 with competing filler.
- Threshold on Qwen2.5-1.5B: loss stays near zero until a* falls to about 0.35.
- Masked filler on Qwen2.5-1.5B at 32K: accuracy 0.52, loss 1.20 nats, against 0.98 with the same filler visible.
- The arXiv listing abstract gives 0.16 to 0.33 and 45 points where the HTML and dataset card give 0.11 to 0.21 and 46 points.

### Reasoning or thinking modes

- Not tested. Scoring uses teacher forced log probability with no generation.

### Mitigations tried and their effect

- None. The paper recommends curating the context (deduplicate and resolve conflicting statements before they reach the model).
- The paper predicts that methods which sharpen attention at long range help most when they keep a* above the threshold. It cites a list of such methods, including arXiv 2503.02130, whose title ends "with a Forget Gate".

### Mechanistic claims

- Claim: in retrieval, loss tracks the attention mass left on the evidence. Evidence: the R squared values above over 126,473 trials, and the ordering holds within each (length, depth) cell.
- Once log(1/a*) is known, the rival count adds at most 0.04 to R squared.
- Slopes of loss on log(1/a*) exceed 1 on every model (1.40 to 3.97), so loss grows faster than single head softmax dilution implies.
- The evidence is predictive, not causal. The author ran no intervention on attention.

### Released artifacts

- Code: https://github.com/yliusci/length-is-cheap (MIT). Contains `harness/attn_probe.py`, `harness/measure_c2.py`, `experiments/ctxrot_v3.py`, `scripts/rival_split.py`. The probe docstring says it covers per head q_norm as used in Qwen3.
- Data: https://huggingface.co/datasets/yliusci/length-is-cheap (MIT). Six parquet files, 197,000 rows. Main config `c2_attention` has 126,473 rows. `c2_nconf_sample` has 20,193 rows with rebuilt rival counts.

### Open questions stated by the author

- No causal test of a*.
- Format only competitors above 2K tokens are not measured, because rivals are present in every long context.
- Models above 3B are untested on the retrieval task.
- Whether length hurts reasoning tasks (whitespace padding lowers MMLU on Qwen2.5-1.5B to 36.0% at 4,096 tokens).

### Evidence grade

FULL (v1 HTML, dataset card, and two harness source files read). Not verified: figures, `rival_split.py`, the parquet contents.

## Paper 9: Flip-Flop Language Modeling, arXiv 2306.00946

### Citation

- Title: Exposing Attention Glitches with Flip-Flop Language Modeling.
- Authors: Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, Cyril Zhang.
- Venue: NeurIPS 2023 (arXiv comment: camera ready plus data release).
- arXiv id: 2306.00946. Latest version v2, dated 2023-10-30 (v1 dated 2023-06-01).

### Task and benchmark specification

- Vocabulary: {w, r, i, 0, 1}. Strings alternate instruction and data bit, for even T of at least 4. Example: "w 0 i 1 i 0 r 0".
- Rule: the bit after each r must equal the bit after the most recent w. Strings start with w and end with r.
- FFL(T, p): first instruction w, last instruction r, other instructions i.i.d. with probabilities (p_w, p_r, p_i). Bits after w or i are uniform.
- Shorthand FFL(p_i): T = 512, p_w = p_r = (1 - p_i) / 2.
- Training distribution: FFL(0.8), online samples.
- Test sets: in distribution, 1,000 sequences from FFL(0.8) with 26,508 reads. Sparse tail, 100,000 sequences from FFL(0.98) with 353,875 reads. Dense tail, 3,000 sequences from FFL(0.1) with 345,781 reads.
- Mode used in the paper: deterministic ("clean"). The model is scored only on the token after each r.
- Metric: read error rate (glitch rate) on the sparse and dense tails.
- Baseline training: 6 layers, d = 512, 8 heads, 19M parameters. AdamW, betas (0.9, 0.999), lr 3e-4, weight decay 0.1, 50 warmup steps, linear decay, 10,000 steps, batch 16.
- Natural language version for pretrained LLMs (Appendix B.2.1): write = "Alice turns the light off" or "Alice turns the light on". Ignore = "Bob eats a banana" or "Bob eats an apple". Query prompt: "The light is turned ". Each prompt starts with a random length 16 flip-flop sequence as an in context demonstration. 500 sequences per length.

### Models tested and headline numbers

- Transformers trained from scratch: depth in {2, 4, 6, 8}, width in {128, 256, 512, 1024}, heads in {2, 4, 8, 16}. 10,625 models in total. 500 replicates of the baseline.
- R1: transformers show a long irregular tail of read errors on both sparse and dense sequences, with high variance across seeds.
- R2: a 1 layer LSTM (hidden 128, 133K parameters, 500 steps) reaches zero out of distribution error in 100 of 100 runs.
- R3: GPT-2, GPT-2-large, GPT-2-xl, Pythia-12B, GPT-NeoX-20B on the light switch version. Accuracy falls as sequence length grows. Larger models do better (Figure 3, no table).
- GPT-4 through ChatGPT (2023-05-22): correct up to about length 400, wrong at length 1000 (Figure 8, anecdotal).

### Reasoning or thinking modes

- Not tested. The authors note chain of thought could externalise the state and cite prior evidence that it works on synthetic recurrences.

### Mitigations tried and their effect

- R4: training on a uniform mixture of FFL with p_i in {0.9, 0.98, 0.1} works best. 6 of 25 runs had zero detected errors.
- R5: more data, more steps, larger models help by far less, with tradeoffs between sparse and dense errors. No monotonic relation between size and extrapolation.
- R6: embedding dropout 0.5 and attention sharpening losses (entropy or negative p-norm penalties on attention weights) give the largest gains on sparse sequences. Non-trainable position embeddings help most on dense sequences.
- R7: nothing except training on the tail distributions removes glitches.
- R8: attention sharpening produces hard attention, but errors persist. Sharpened heads can attend with confidence to the wrong write, for example the initial write.

### Mechanistic claims

- Proposition 3: soft attention with bounded weight norms dilutes as T grows (Appendix C.2).
- Proposition 4: hard attention with linear position encodings can attend to the wrong index unless weights meet an orthogonality condition (Appendix C.3).
- Figure 5(d): a sharpened head focuses on an initial write. The authors attribute this to causal masks that overemphasise early positions.
- Hypothesis: attention glitches cause part of closed domain hallucination in natural LLMs. The authors leave this untested.

### Released artifacts

- Data: https://huggingface.co/datasets/synthseq/flipflop. 16M FFL(0.8) training sequences, 16K FFL(0.8) test sequences, 160K sparse FFL(0.98) sequences, 4K dense FFL(0.1) sequences. License not checked.
- Model code: the paper links https://github.com/lucidrains/x-transformers as the library. No dedicated repository found in the HTML.

### Open questions stated by the authors

- A practical architecture with LSTM-like robustness and transformer strengths is "a grand open challenge".
- Reconciling synthetic and natural settings, for example by fine tuning pretrained LLMs on flip-flop sequences, is left as future work.

### Evidence grade

FULL for the main text and Appendix B.2 to B.3. Appendices A, B.4 to B.6, and C not read. Not verified: figure values, dataset license, dataset file format.

## Paper 5: Context Poisoning as Extreme-Value Attention Interference, arXiv 2609.22101

### Citation

- Title: Context Poisoning as Extreme-Value Attention Interference in Long-Context Language Models.
- Authors: Meysam Ghaffari, Nina Fatehi, Bhaskar Sen, Nasim Sabetpour, Carlos Morato.
- Venue: arXiv preprint, 2026. No venue listed.
- arXiv id: 2609.22101. Only version v1, dated 2026-08-12 in the arXiv API record.

### Task and benchmark specification

- Three behavioural studies plus an attention diagnostic. None is a key update task.
- Hard negative decay: one target fact inside N sentences from a public domain novel, with several decoy sentences that state a different value in the same phrasing. N in {100, 300, 500, 1000}, 10 trials each, random target position. Metric: accuracy. Prompt lengths 8,952 to 84,668 characters.
- Distractor confusability: key value retrieval over 300 records (about 20K to 21K tokens), one gold record, temperature 0. The model returns the answer and cites the record ID. Four distractor types: unrelated filler, same domain with a different attribute, same format with a different entity, near alias entity (shared surname). Metrics: answer accuracy and cited record recall.
- Gate test (Appendix F): RULER qa_2 built from HotpotQA, 100 questions, lengths 4K to 512K tokens (o200k_base). Raw context against BM25 top K with K = 16. Metrics: EM, F1, Recall@K, EM conditional on full support retrieval.
- Attention diagnostic (Appendix D): Gemma-3-4B-IT, needle prompt with one evidence sentence stating a random secret code among N distractor sentences, N in {20, 50, 100}, 10 trials each.
- The main text does not print prompt templates. It says the appendix holds them (Appendix C, not read).

### Models tested and headline numbers

- GPT-4o, hard negative decay (Table 3): 100% at N = 100, 80% at N = 300, 500, and 1000. Every error returned a decoy value present in the prompt.
- Fixed length confusability (Table 4), accuracy for unrelated, same domain, same format, near alias: Claude Sonnet 4 1.000, 1.000, 0.889, 1.000. GPT-4.1 1.000, 1.000, 0.967, 0.983. Gemma 3 12B 1.000, 1.000, about 0.85, about 0.89. The Gemma endpoint rejected 96% of calls.
- Gate test on GPT-4.1 (Table 14): raw EM 0.66 at 4K, 0.65 at 32K, 0.59 at 128K, 0.57 at 512K. Gated EM 0.64, 0.65, 0.64, 0.59. No confidence interval excludes zero.

### Reasoning or thinking modes

- Not reported in the sections read.

### Mitigations tried and their effect

- Tested: BM25 retrieve then reason gate with K = 16. No significant net gain. On instances where the gate kept all support, gated EM beat raw EM by 0.070 at 128K and 0.114 at 512K.
- Proposed but untested: alias resistant position codes with a margin regulariser, a contrastive anti-poison training loss with hard negatives, verifier mediated memory, hierarchical memory.

### Mechanistic claims

- Theorem 1: with one evidence logit bounded by gamma and N i.i.d. Gaussian distractor logits, accuracy is at most a0 + (1 - a0) * Phi((gamma + tau * log((1 - rho) / rho)) / sigma)^N. Holding accuracy needs gamma to grow as sigma * sqrt(2 log N).
- Proposition 1: a gate that keeps K candidates and misses evidence with probability eta replaces N by K and adds eta to the bound.
- Appendix D on Gemma-3-4B-IT: tests reject Gaussianity at every N. Independence holds at N = 20 and 50 and fails at N = 100. The empirical mean maximum is 2.31, 4.18, 3.02 against the predicted 2.45, 2.80, 3.04. The authors call the theorem a diagnostic baseline, not a calibrated bound.
- The main studies do not observe attention.

### Released artifacts

- None found. The HTML has no code or data link.

### Open questions stated by the authors

- Connect internal score margins to end to end errors.
- Cross context length with distractor construction.
- "temporally conflicting documents may exhibit additional failure modes".
- Compare fixed with adaptive retrieval budgets.

### Evidence grade

PARTIAL. Read: abstract, sections from "Formalizing context poisoning" to the conclusion, parts of Appendices B, D, and F. Not read: introduction, related work, Appendix A (proofs), Appendix C (protocol and prompts), Appendix E.

## Paper 6: The First Drop of Ink, arXiv 2605.10828

### Citation

- Title: The First Drop of Ink: Nonlinear Impact of Distracting Information in Long-Context Reasoning.
- Authors: Muhan Gao, Zih-Ching Chen, Kuan-Hao Huang (corresponding address at Texas A&M, College Station).
- Venue: not stated in the arXiv record. The HTML keywords line reads "Machine Learning, ICML". Year 2026.
- arXiv id: 2605.10828. v1 dated 2026-05-11. Latest version v2, dated 2026-08-20.

### Task and benchmark specification

- Multi-document QA. Datasets: Natural Questions, TriviaQA, PopQA, HotpotQA. Each sample has a question, a gold answer, and a gold passage.
- Distractor types, each normalised to about 100 to 150 tokens:
  - Easy: repeats of "The grass is green. The sky is blue. The sun is yellow..."
  - Random: passages from the Wikipedia 2019-08-01 KILT dump.
  - Hard: BM25 retrieved Wikipedia passages on the topic that do not contain the answer, filtered with gpt-4o-mini.
- Input: gold passage plus distractors that fill a target length. Hard proportion p of the distractors, the rest easy (for example `nq_easy`) or random (`nq_random`). Passages shuffled.
- Context lengths: 4K, 8K, 16K, 32K, 64K, 128K. Hard proportions in percent: 0, 1, 2, 3, 5, 10, 20, 40, 60, 80, 90, 100 (Table 2).
- 200 examples per setting.
- Metric: accuracy judged by gpt-4o-mini. Appendix C prints the judge prompt, which begins "You are an expert evaluator for question-answering systems." and ends with a CORRECT or INCORRECT verdict.
- Drop ratio = (Acc(0%) - Acc(10%)) / (Acc(0%) - Acc(100%)). Linear decay gives 0.1.
- The QA prompt given to the tested model is not printed in the sections read.

### Models tested and headline numbers

- Model names found in the text: Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen3-Next-80B-Instruct.
- Qwen2.5-7B-Instruct on `nq_easy` at 128K: drop ratio 0.58.
- Llama-3.2-1B-Instruct, Natural Questions with easy filler, 128K (Table 2): 55.5 at 0% hard, 35.0 at 1%, 29.0 at 10%, 23.0 at 100%. At 4K: 76.0 at 0%, 70.0 at 10%, 57.5 at 100%.

### Reasoning or thinking modes

- Not tested in the sections read. The paper cites "Lost in the Noise: How Reasoning Models Fail with Contextual Distractors" (arXiv 2601.07226).

### Mitigations tried and their effect

- Inference time attention temperature below 1 (tau = 0.9) on Llama-3.1-8B-Instruct: accuracy fell at every hard proportion (Figure 6).
- Filtering hard distractors: from 131K down to 47K tokens, removing hard or random passages gave the same gains. Filtering hard passages helped more only once the hard share neared zero (3% at 27K).
- The authors state: "we have not yet identified an effective mitigation strategy."

### Mechanistic claims

- Hard distractor tokens have a much smaller logit margin below the gold passage than easy tokens. On retrieval heads of Llama-3.1-8B-Instruct, the margin to easy distractors is about 7 to 10 and to hard distractors about 2 to 3, mean gap 5.83. Each hard token then adds about 340 times more to the softmax denominator than an easy token.
- Retrieval heads: the top heads (about 1% to 2%) by pre-softmax logit from query tokens to the gold passage, selected on 50 samples, measured on 150.
- The gap shrinks from 8.0 at 1% hard to 4.1 at 90% hard.

### Released artifacts

- None found. The HTML has no code or data link for this work.

### Open questions stated by the authors

- An effective mitigation.
- Generalisation beyond multi-document QA (summarisation, code, multi-turn dialogue).

### Evidence grade

PARTIAL. Read: abstract, sections 3, 5, 6, 7, 8, the first table of Appendix A, Appendix C. Not read: introduction, related work, section 4 (theory), the rest of Appendix A, Appendix B.

## Paper 7: Controlled Memory Interference in Continual LLM Agents, arXiv 2608.07622

### Citation

- Title: Controlled Memory Interference in Continual LLM Agents.
- Authors: Ao Ding, Hongzong Li, Shiqin Tang, Li Zhang, Liang Chen, Xuyang Chen, Zi Liang.
- Venue: arXiv preprint, 2026. No venue listed.
- arXiv id: 2608.07622. Only version v1, dated 2026-08-07.

### Task and benchmark specification

- Setting: an agent with an external memory store and a retriever. This is retrieval level interference, not in-context attention.
- Each case has a query, an expected decision, and a memory collection split into target memories, competing memories (superseded states, transient exceptions, same slot conflicts, misleading updates), and other memories.
- Three behaviour dimensions: plasticity (adopt a valid persistent update), stability (keep an established state when newer evidence is transient), poisoning resistance (reject an invalid update).
- Three protocols: memory load L0 to L6. Relationship composition with matched neutral, unrelated, repeated history, and same slot conflict additions at counts 0, 2, 4, 8. Fixed L3 authority probe P0 to P3 (noise, recent transient mention, explicit persistent update, authoritative correction).
- Sources: TravelPlanner in MemoryArena, PERMA, LongMemEval, HorizonBench, MEME Tracking. TravelPlanner study: 50 base scenarios, 1,400 growth instances, two composition protocols of 800 instances, 1,000 poisoning instances.
- Retrieval: BM25 (k1 = 1.5, b = 0.75), dense retrieval with Qwen3-Embedding-8B, fixed fusion, RRF. Top 3 memories.
- Metrics: plasticity, stability, Target Recall@3, MRR, Preference Ordering, answer success.
- Generation prompts are in Appendix D.3 (not read).

### Models tested and headline numbers

- Answer model: DeepSeek-v4-pro, temperature 0, reasoning disabled. Poisoning study: DeepSeek-v4-flash.
- TravelPlanner noise growth: plasticity 0.510 at L0 and 0.455 at L6.
- Composition at count 8 with lexical retrieval (Table 1), plasticity against a 0.84 baseline: matched neutral 0.84, unrelated 0.82, repeated history 0.50, same slot conflict 0.34.

### Reasoning or thinking modes

- Reasoning was disabled for all answer generation. No comparison.

### Mitigations tried and their effect

- A random forest selector (300 trees) picks the lexical to dense fusion weight. Training it with CMI generated same slot cases raised noisy update answer success (Table 4): TravelPlanner 0.386 to 0.686, PERMA 0.386 to 0.829, LongMemEval 0.886 to 1.000, HorizonBench 0.514 to 0.929, MEME Tracking 0.314 to 0.929.
- The selector does not transfer across embedding backends without retraining.
- No change to the LLM.

### Mechanistic claims

- Interference can block target exposure at retrieval or disrupt use after retrieval. Lexical and dense retrieval show different pathways.
- Poisoning depends more on update authority cues than on recency.

### Released artifacts

- Code: https://github.com/Dawn-Buendia/Controlled-Memory-Interference-in-Continual-LLM-Agents (link in the HTML, repository not opened, license not checked).

### Open questions stated by the authors

- "Preference ordering and competitor exclusion remain open challenges for stronger memory-state reconciliation."
- Zero shot cross domain transfer of the selector.

### Evidence grade

PARTIAL. Read: abstract, sections 3.1, 3.2, 4, parts of 5.1, 5.4, 6, the start of Appendix L. Not read: introduction, related work, sections 5.2 and 5.3, Appendices A to K.

## Paper 8: Cache Engineering for Agents (AKV), arXiv 2610.28577

### Citation

- Title: Cache Engineering for Agents.
- Authors: Tianyu Fu, Ruoxi Wang, Zihan Min, Zhongqi Ren, Zhihao Li, Tengxuan Liu, Shengxu Mu, Hanling Zhang, Yu Wang (Tsinghua University, CUHK).
- Venue: arXiv preprint, 2026. No venue listed.
- arXiv id: 2610.28577. Only version v1, dated 2026-10-20.

### Task and benchmark specification

- Not a PI benchmark. Agent benchmarks: BrowseComp-Plus, xbench-DeepSearch-2510, GAIA text only validation subset (103 tasks), SWE-bench Verified.
- Metrics: success rate, mean per task API cost in US cents, latency, throughput.
- Three cache operations:
  - append: standard cache update.
  - drop(P): removes the KV states at logical token indices P from later attention. Retained states and their position encodings stay unchanged.
  - repos: rotates cached keys to new RoPE positions, K <- R(q) R(p)^-1 K, and resets the next position index.
- Main policy: tool response rolling drop, which keeps the KV states of the latest 12 tool responses and all reasoning.

### Models tested and headline numbers

- MiniMax-M2.7, GPT-OSS-120B (high reasoning effort), MiroThinker-1.7-Mini (thinking mode).
- Average score and cost (Table 1): MiniMax-M2.7 Standard 52.4 at 5.0 cents, AKV 53.7 at 4.4. GPT-OSS-120B Standard 51.0 at 11.2, AKV 53.1 at 7.3. MiroThinker-1.7-Mini Standard 44.1 at 20.4, AKV 45.8 at 14.8.
- Abstract: rolling drop improves average scores by 1.7 points and cuts average cost by 27.7% against threshold triggered summarisation.

### Reasoning or thinking modes

- All three models ran with reasoning or thinking on. No on and off comparison.
- Ablation (Table 3): dropping reasoning cache lowers pass rates against dropping tool response cache alone.

### Mitigations tried and their effect

- Not aimed at interference. The paper frames drop as cost and context length control.
- Cache management exposed as a tool (CAT): the agent decides what to drop. 61.0% for GPT-OSS-120B and 64.0% for MiniMax-M2.7 on a 100 task BCP subset, against 52.1% and 56.0% for Standard.
- No fine tuning. The limitations list "training agents under this scheme" as future work.

### Mechanistic claims

- Older tokens receive less attention than recent ones, and tool responses less than assistant messages (Figure 2, 10 BCP requests).
- Cache states are history dependent: the same retained tokens yield different KV states under different drop timings, so prefix matching cannot guarantee correct reuse.
- Retained cache may keep information from dropped history that entered its computation.

### Released artifacts

- Code: https://github.com/thu-nics/AKV (link in abstract, repository not opened, license not checked).

### Open questions stated by the authors

- More cache engineering strategies, training agents under this scheme, serving support for linear attention models.

### Evidence grade

PARTIAL. Read: abstract, sections 3.2 to 3.4, 5.1, 5.2, 5.4, 6. Not read: introduction, related work, section 4 (serving system), 5.3, appendix.

## Evaluation protocol to replicate

Lines marked "Our note" are recommendations from this review, not claims from the sources.

### PI-LLM (Paper 1)

- Data: https://huggingface.co/datasets/Cog2ai/pi-llm-bench (MIT). Config `core`, split `test`, 580 rows. Config `sequential_additional`, split `test`, 160 rows.
- Code for fresh sessions: https://github.com/zhuangziGiantfish/Unable-to-Forget (MIT). Run `python core/run_pi.py --test test1_updates --model <name>`. Word pool: `testing_data/dict_category_double-word_46-400_v1-1.json`.
- Dataset fields:
  - `prompt`: JSON string holding a list with one message, `{"role": "user", "content": ...}`.
  - `answer_formatted`: JSON string mapping each tracked key to its final value. This is the ground truth.
  - `answer_natural_language`: the same answers as lines `The current value of <key> is <value>.`
  - `experiment`: `exp_updates`, `exp_keys`, `exp_valuelength`, or `extra_exp_updates_randomoff`.
  - `session_id`: 1 to 10. `n_updates`, `n_keys`, `value_length`, `randomize_mode` (`on` or `off`), `n_tokens`, `n_chars`, `n_words`.
- Prompt format (exact, from row 0 of `core`):
  - `As my secretary, I need you to carefully read a text stream where the values of multiple keys are being continuously updated.The 46 keys to track include <k1>, <k2>, ..., <k46>. I will ask you to identify the value of each key later.\n\nThe text stream starts on the next line.\n <key>: <value>; <key>: <value>; ... \n\nWhat are the current value of each key (<k1>, <k2>, ..., <k46>) you are tracking? End your response with: 'The current value of <key> is <value>.'`
  - `exp_valuelength` rows append `Provide the exact current value string without modification or breaking it into pieces.`
  - The key list in the instruction and the query uses the same order within a prompt. The order differs across prompts.
- Condition grid, 10 sessions per cell:

| Experiment | Fixed | Varied | Token range (`n_tokens`) |
|---|---|---|---|
| `exp_updates` | 46 keys, value length 1 | updates per key: 2, 3, 4, 6, 8, 12, 17, 24, 34, 48, 68, 97, 139, 197, 281, 400 | 789 to 100,721 |
| `exp_keys` | updates 125 or 350, value length 1 | keys: 2, 3, 4, 5, 6, 8, 11, 15, 20, 26, 35, 46 | 1,253 to 88,206 |
| `exp_valuelength` | 46 keys, updates 4 or 20 | value length: 1, 2, 3, 4, 6, 10, 16, 25, 40 | 1,302 to 102,494 |
| `extra_exp_updates_randomoff` | 46 keys, value length 1 | same 16 update counts | 789 to 100,721 |

- Tokens per update count in `exp_updates` (max over sessions): 2: 810. 3: 1,072. 4: 1,316. 6: 1,820. 8: 2,329. 12: 3,329. 17: 4,608. 24: 6,351. 34: 8,866. 48: 12,402. 68: 17,430. 97: 24,701. 139: 35,277. 197: 49,833. 281: 70,901. 400: 100,721.
- The fixed length tracked keys experiment (`test3_ntracked`) exists only in the code repository.
- Decoding in the repository configs: temperature 0, `max_tokens` 1024. The card quick start sets no decoding parameters and skips rows whose token count (tiktoken `o200k_base`) exceeds a window you set.
- Scoring, from the card code `grade_pi_response`:
  - Strip backslashes and asterisks from the output.
  - Extract pairs with a verbal pattern (`the current|last|final|latest|most recent ... value ... of <key> is <value>`) and with `key: value` lines. Verbal matches override colon matches.
  - Row accuracy = number of ground truth keys whose extracted value equals the ground truth string, divided by the number of ground truth keys.
- Metrics to report:
  - Mean accuracy per update count over the 10 sessions, plotted against log update count (the paper's Figure 3 format).
  - `auc_log1.5` per experiment: weighted mean of row accuracy with weight log base 1.5 of max(`n_updates`, 2). The card names this the primary metric. For `exp_keys` report easy (`n_updates` at most 125) and hard (at least 350). For `exp_valuelength` report easy (at most 4) and hard (at least 20).
  - Total score: mean of the per experiment `auc_log1.5` values.
  - IES from the paper: area under the accuracy curve over log scaled update counts. The paper does not give the integration rule or normalisation.
  - Error breakdown as in the paper's Figure 5: share of answers that equal the final value, an earlier value by position bin (11 bins), a value never shown, or no value.
- Baselines for small open models: the paper tested qwen3-0.6b, 1.7b, 4b, 8b (with and without thinking), llama-3.1-8b, gemini-1.5-flash-8b, and gpt-4.1-nano, but reports them only as curves (Figures 3, 16, 17). No numeric table exists. Re-measure every baseline.
- Reasoning mode: the paper ran thinking variants only up to about 100 updates because thinking output overflowed.
- Our note: the 400 update prompt is about 100.7K tokens. Check each Qwen3 checkpoint's context window before running the top of the grid.
- Our note: the sampled rows of `sequential_additional` showed interleaved keys, which conflicts with the card's description of per key blocks. Inspect the parquet, or generate sequential streams with `test5_updates_randomoff`.
- Our note: the word pool is public. If fine tuning data uses the same 46 keys and 400 values per key, report a held out vocabulary condition as well.

### RI and PI shared stimulus probe (Paper 2)

- No code or data released. Rebuild from the PI-LLM generator: 46 categories, each appearing N times, N in {3, 10, 50, 100, 200, 300}, no key twice in a row, temperature 0.
- Two queries on the same input: first value (RI) and most recent value (PI). The paper does not print its query wording.
- Metrics: accuracy per N. PIES and RIES = integral of accuracy in percentage points over log10(N+1) from N = 3 to 300, maximum about 188.
- Session rule: add sessions until the 95% bootstrap CI (10,000 resamples) is under 0.10 wide twice in a row, minimum 5, cap 50.
- Baselines for small open models (Table 4, RIES then PIES): Llama-3.2-1b 126.1 and 1.0. Llama-3.2-3b 113.6 and 16.8. Llama-3.1-8b 128.3 and 57.1. Llama-3.2-11b 138.4 and 47.6. Nova-micro 126.1 and 76.8. No Qwen3 model under 30B.
- Our note: the RI query measures selectivity. A gate that suppresses every earlier value should lower RIES, so report both scores.

### Rival statement probe (Paper 4)

- Code: https://github.com/yliusci/length-is-cheap (MIT). Data: https://huggingface.co/datasets/yliusci/length-is-cheap (MIT).
- Prompt (raw text, optional BOS, no chat template):
  - `You are a careful assistant. Read the document and answer the question.` then filler with the evidence ` The secret code of the {key} is {val}.` inserted at a relative depth, then `\nQuestion: what is the secret code of the {key}? Answer: The secret code of the {key} is`, scored continuation ` {val}`.
- Keys: the 51 nouns listed under Paper 4. Values: integers 1000 to 9998.
- Competing filler: `The secret code of the {noun} is {NNNN}.` repeated to fill L tokens, nouns drawn from all 51. A statement with the queried key is a rival.
- Conditions per trial: short (no filler), neutral, competing, masked. Explicit distractors with other keys: 1 to 4 in the attention runs.
- Lengths: 512, 1,024, 2,048 for the main fit. 8K to 32K for the long runs. Depth uniform in 0.1 to 0.9.
- Metrics:
  - Loss = `lp_short - lp_long` in nats, where `lp` is the summed teacher forced log probability of the value tokens.
  - Accuracy: every value token is the top 1 prediction.
  - a*: attention from the last query token to the evidence tokens, mean over the 8 (layer, head) pairs with the highest evidence mass in the short context. Use `harness/attn_probe.py`.
  - Rival count per trial: rebuild from the seed with `scripts/rival_split.py`, or use config `c2_nconf_sample`.
  - R squared of loss on log(1/a*) and on log L.
- Baselines for small open models:

| Model | Loss, no rival (nats) | Loss, at least one rival | R squared, log(1/a*) | R squared, log L |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.175 | 1.812 | 0.35 | 0.03 |
| Qwen2.5-1.5B-Instruct | 0.020 | 2.238 | 0.58 | 0.05 |
| SmolLM2-1.7B-Instruct | 0.022 | 1.381 | 0.41 | 0.07 |
| OLMo-2-0425-1B-Instruct | 0.058 | 2.796 | 0.35 | 0.02 |
| Llama-3.2-3B (base) | 0.344 | 3.239 | 0.45 | 0.08 |

- More baselines: accuracy 0.97 to 1.00 without a rival and 0.25 to 0.52 with at least one (L at most 2,048). Qwen2.5-1.5B at 2,048 tokens by rival count 0 to 6 or more: 0.03, 1.37, 2.24, 2.48, 2.71, 3.26, 3.45 nats. Qwen2.5-1.5B with competing filler at 8K to 32K: accuracy 0.11 to 0.21. a* threshold near 0.35 on Qwen2.5-1.5B.
- No Qwen3 baseline exists. The probe docstring says it handles the per head q_norm of Qwen3.
- Our note: rivals fall on both sides of the evidence, and the scored answer is the evidence value in every case. The probe has no "latest value wins" rule. Split trials by whether rivals come before or after the evidence, and add a variant where the evidence is the last statement for its key.
- Our note: a* gives a direct test of the forget gate mechanism. The paper ran no intervention on a*, so a gate that raises a* and lowers loss would supply the missing causal evidence.

### Flip-flop (Paper 9)

- Data: https://huggingface.co/datasets/synthseq/flipflop. 16M FFL(0.8) training sequences, 16K FFL(0.8) test, 160K sparse FFL(0.98), 4K dense FFL(0.1).
- Format: tokens from {w, r, i, 0, 1}, T = 512, alternating instruction and bit. Train on FFL(0.8) with p_w = p_r = 0.1.
- Evaluation sets used in the paper: 1,000 in distribution sequences (26,508 reads), 100,000 sparse sequences (353,875 reads), 3,000 dense sequences (345,781 reads).
- Metric: error rate on the token after each r, reported for the sparse and dense tails. In distribution error is 0 for converged runs.
- From scratch baseline: 6 layers, d = 512, 8 heads, 19M parameters, AdamW lr 3e-4, weight decay 0.1, 10,000 steps, batch 16. Report at least 25 seeds, as the paper does, because variance across seeds is large.
- Reference points: a 1 layer LSTM reaches zero error in 100 of 100 runs. Training on the mixture p_i in {0.9, 0.98, 0.1} gave zero detected errors in 6 of 25 runs.
- Natural language version for pretrained LLMs: write = "Alice turns the light on" or "Alice turns the light off". Ignore = "Bob eats a banana" or "Bob eats an apple". Read prompt "The light is turned ". A random length 16 sequence precedes each prompt as a demonstration. 500 sequences per length. Metric: accuracy of the predicted state against sequence length.
- Baselines for small open models: only a figure for GPT-2, GPT-2-large, GPT-2-xl, a 12B Pythia model, and GPT-NeoX-20B. No numbers in the text. No modern small model baseline.

### SleepGate synthetic task (Paper 3), for direct comparison

- 100 entity keys, 500 values, vocabulary 1024, one entity per episode, depths n in {1, 2, 5, 10, 15, 20, 30}, 200 episodes per depth.
- Metrics: exact match accuracy, stale retrieval rate, slope of accuracy on log n.
- Numbers to beat: SleepGate 99.5% at n = 5, 97.0% at n = 10, 73.5% at n = 15, 30.5% at n = 20, 16.5% at n = 30. No code released, so the task must be rebuilt from the description.

## Remaining sections

PENDING: learned KV eviction, existing cures, threats to novelty, unverified items.
