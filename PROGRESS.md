# Project Tracker

Single source of truth for this paper project. Read this file first in any new session. Update it before ending any session.

## 1. Hard Rules

The user set these rules on 2026-11-11. They hold for every session.

- Target: a NeurIPS-level paper. Allowed fields: deep learning, agentic AI, NLP. Computer vision is prohibited.
- The topic must sit on the 2026 frontier and be novel: untaken, or a problem where many groups try and the needle has not moved.
- Verify novelty on the web before committing to a claim. Log sources in Section 5.
- Get user approval of the idea before any implementation.
- Before calling the work done, run adversarial self-review until no known weakness remains open. Report results as they are.
- No em dashes in any file, commit, or message.
- Commits: title line only. No body, no description, no AI co-author, no session links.
- Commit and push all work as it lands. Commits carry their real timestamps.
- Parallel sessions are allowed when they speed up the work.
- Apply the Stop Slop rules to all prose and the Coding Guidelines to all code.
- Keep all files on D:. Temp work goes in `D:\tmp`.
- Keep code comments short.

## 2. Quality Bar

- Each claim in the paper maps to a proof or an experiment stored in the repo.
- Three seeds minimum per reported number. Report mean and spread.
- Tune baselines with the same budget as the proposed method.
- Report negative results and failed kill criteria. Do not hide them.
- Base related work on full-text reads. Abstract-level evidence carries a flag until upgraded.
- Re-run the novelty search before submission. The field moves weekly.
- Each experiment has a script, a config, and a saved log. No hand-edited numbers.

## 3. Environment

| Item | Value |
|------|-------|
| Local machine | Ryzen 3 3250U, 2 cores, 13.9 GB RAM, no NVIDIA GPU |
| Free disk on D: | 23.7 GB on 2026-11-11 |
| Local tools | Python 3.12.4, uv, git 2.49 |
| Missing tools | LaTeX (pdflatex not found), conda |
| Compute plan | Laptop CPU for small-model experiments. GPU hours for the LLM retrofit. See Section 8. |
| Repo | https://github.com/Rishi0507/Transformers-Cannot-Overwrite (private), branch `main` |
| Python env | `.venv` in the repo root, built with uv. Set `UV_CACHE_DIR=D:\tmp\uv-cache`. |

## 4. Status

- Date: 2026-11-11.
- Phase: 0 (pilot). The user approved the primary idea on 2026-11-11.
- Done: repo scaffold, gated attention module, register task, training script, unit tests.
- Interrupted by the usage limit on 2026-11-11: two reader sessions for full-text prior-art notes (`notes/prior_art_mechanisms.md`, `notes/prior_art_interference.md`). Resume them or restart them.
- Not verified yet: the dependency install was still running when the session stopped, and the unit tests have never run.
- Next action: confirm the install, run `pytest`, run the register sweep on CPU, draft proofs in `theory/`, finish the prior-art notes.

## 5. Literature Survey Log

Evidence grade A means I read the abstract page. Grade B means a search summary only. Upgrade both to full-text reads before writing related work.

### 5.1 Proactive interference and context rot

| Source | Grade | Takeaway |
|--------|-------|----------|
| [Unable to Forget (PI-LLM)](https://arxiv.org/abs/2506.08184) | B | Accuracy on latest-value retrieval falls log-linearly with update count. Prompt fixes fail. |
| [Remember First, Forget Last](https://arxiv.org/abs/2603.00270) | B | 39 models tested. Proactive interference dominates in all of them. Diagnosis only. |
| [SleepGate](https://arxiv.org/abs/2603.14517) | A | Proposed fix. Tested on a 793K-parameter toy transformer only. |
| [Length Is Cheap, Competition Is Not](https://arxiv.org/abs/2611.04169) | A | Nov 2026. Filler barely hurts. One rival value multiplies loss 9x to 110x. No cure, no universal law. |
| [Context Poisoning](https://arxiv.org/abs/2609.22101) | B | Extreme-value bound on evidence margin. No cure. |
| [First Drop of Ink](https://arxiv.org/pdf/2605.10828) | B | Hard distractors dominate attention competition. |
| [Controlled Memory Interference in Continual LLM Agents](https://arxiv.org/pdf/2608.07622) | B | Agent-level interference study. Read in full. |
| [PI-LLM benchmark on Hugging Face](https://huggingface.co/datasets/Cog2ai/pi-llm-bench) | B | Public benchmark for evaluation. |

### 5.2 Attention mechanisms with forgetting or learned position

| Source | Grade | Takeaway |
|--------|-------|----------|
| [Forgetting Transformer (FoX)](https://arxiv.org/abs/2503.02130) | A | Scalar forget gate per head. FlashAttention compatible. Trained from scratch. No interference test. |
| [Adaptive Computation Pruning for FoX](https://arxiv.org/pdf/2504.06949) | B | KV eviction from FoX decay. |
| [Selective Attention](https://arxiv.org/abs/2410.02703) | B | Tokens mask earlier tokens. Quadratic accumulation. Authors list fine-tuning retrofit and a fast kernel as open work. |
| [PaTH Attention](https://arxiv.org/abs/2505.16381) | A | Householder-product position encoding. Converts RoPE models by continued pretraining. No interference test on LLMs. |
| [Selective RoPE](https://arxiv.org/pdf/2511.17388) | B | Input-dependent rotations. ICLR 2026. |
| [GRAPE](https://arxiv.org/pdf/2512.07805) | B | Group-theoretic unification of RoPE and ALiBi. FoX with constant gate equals ALiBi. |
| [Cache Engineering for Agents (AKV)](https://arxiv.org/abs/2610.28577) | A | Harness drops KV entries by policy. 27.7% cost cut. No learning inside attention, no error bound. |
| [The Gating Mechanism of KV Cache Eviction](https://arxiv.org/abs/2611.07605) | B | One of at least six 2026 papers on learned KV eviction. Heuristic scoring, no soundness bound. |

### 5.3 Directions checked and found taken

| Source | Grade | Takeaway |
|--------|-------|----------|
| [Capabilities and Fundamental Limits of Latent CoT](https://arxiv.org/html/2602.01148) | B | Covers the analog-noise account of latent reasoning that I derived independently. |
| [Loop, Think, and Generalize](https://arxiv.org/html/2604.07822v1) | B | Recurrent depth solves systematic two-hop composition. |
| [Ouro looped language models](https://arxiv.org/abs/2510.25741) | B | Looped LMs at 1.4B and 2.6B. Weights on Hugging Face. |
| [From Spark to Fire](https://www.alphaxiv.org/abs/2603.04474v1) | B | Error cascades as infection process in multi-agent systems. |
| [World-Model Collapse as a Phase Transition](https://arxiv.org/abs/2606.31399) | B | Phase-transition view of agent collapse. |
| [Learn from your own latents](https://arxiv.org/abs/2605.27734) | B | Sample-complexity theory for latent prediction. |
| [The Flexibility Trap](https://arxiv.org/abs/2601.15165) | B | ICML 2026 outstanding paper. Arbitrary order hurts reasoning in diffusion LMs. |

### 5.4 Agent reliability and continual learning

| Source | Grade | Takeaway |
|--------|-------|----------|
| [The Horizon Gap survey](https://arxiv.org/abs/2608.06663) | A | 1,547 papers. Names three open problems: model versus harness, correlated bias in process signals, predictive theory of reliability. |
| [Hazard Rates for AI Agents Decline](https://www.tobyord.com/writing/hazard-rates-for-ai-agents-decline) | A | Weibull shape near 0.6 for all models. No mechanism offered. |
| [Statistical look at the METR plot](https://arxiv.org/abs/2610.12466) | A | IRT with splines. Difficulty is flat between 2 and 30 minutes. |
| [Beyond pass@1](https://arxiv.org/abs/2603.29231) | A | Metrics only. Memory scaffolds hurt all 10 models. |
| [Illusion of Diminishing Returns](https://arxiv.org/abs/2509.09677) | B | Self-conditioning on past errors. Scale does not fix it. |
| [CL-Bench](https://arxiv.org/abs/2606.05661) | B | Naive in-context learning beats dedicated memory systems. |
| [Can a Language Model Learn Facts Continually in Its Weights?](https://arxiv.org/abs/2607.11020) | A | 1% to 46% retention after 20 sequential writes. |
| [Naive RAG Is Stronger Than We Thought](https://arxiv.org/abs/2610.30755) | B | Raw-chunk retrieval beats 12 memory systems. |

## 6. Ideas Evaluated

| Idea | Verdict | Reason |
|------|---------|--------|
| Tokens that delete earlier tokens in attention | Taken | Selective Attention (2024), FoX (2025), PaTH (2025). |
| Branching-process theory of agent error cascades | Taken in part | Multi-agent infection models and a phase-transition paper exist in 2026. |
| Analog noise as the reason latent reasoning stalls | Taken | arXiv 2602.01148 states the theorem. |
| Weight tying cures the two-hop curse | Taken in part | arXiv 2604.07822 shows it for models trained from scratch. |
| Universal interference law for context rot | Contested this month | arXiv 2611.04169 measures it and reports no universal law. |
| Learned KV eviction for agents | Crowded | Six or more 2026 papers. None gives a soundness bound or selective typing. |
| Latent-prediction pretraining for data efficiency | Taken | Wyart group theory and LeCun group experiments. |
| LLM-scale cure for proactive interference with theory | Open | No cure beyond a 793K-parameter toy. Chosen as primary. |
| Predictive theory of long-horizon reliability | Open, crowded edges | Named open problem. Backup 1. |
| Knowledge-write portability in looped LMs | Open, narrow | Backup 2. |

## 7. Approved Idea

Working title: Transformers Cannot Overwrite: A Provable Cure for Proactive Interference.

### 7.1 Problem

A transformer context is an append-only log. When a value changes, the old value stays visible and competes with the new one. All 39 tested LLMs fail at latest-value retrieval as updates pile up. Six 2026 papers diagnose this. None cures it at LLM scale.

### 7.2 Mechanism

Typed forget gates. Each token `l` emits a non-negative kill vector `u_l` in `R^r`. Each token `j` carries a non-negative type signature `b_j` in `R^r`. With `S_t` the running sum of kill vectors up to `t`:

```
logit(i, j) = q_i . k_j - b_j . (S_i - S_j)
```

The sum includes the reader's own kill vector, as in FoX. The penalty on self is zero.

- A later token suppresses only earlier tokens whose signature matches its kill type.
- The penalty is a prefix-sum difference. Cost is O(n r), and augmented queries and keys run through unmodified FlashAttention.
- Zero-initialized gates leave a pretrained model unchanged, so the mechanism retrofits by fine-tuning.
- The penalty never decreases, so a token that falls below an attention bound stays below it. Eviction of that KV entry carries a provable error bound.
- FoX is the special case `r = 1`, `b_j = 1`. ALiBi is the case of constant kill vectors.

### 7.3 Claims to establish

1. Theory: bounded-logit softmax attention cannot isolate the latest of `m` matching items once `m` exceeds a threshold set by the logit bound.
2. Theory: scalar forget gates (FoX) suppress in recency order, so `K` independent registers need about `K` heads. Typed gates need rank `r` near `K`, or fewer with positive random features.
3. Theory: monotone typed penalties give sound selective eviction.
4. Experiment: a retrofit of Qwen3 (1.7B, 4B, 8B) removes proactive interference on PI-LLM and the rival-statement probe, and holds at 10x to 100x longer streams than training.
5. Experiment: LoRA-only fine-tuning on the same data does not extrapolate.
6. Experiment: no regression on general and long-context benchmarks.
7. Experiment: stale-file errors in coding-agent traces drop, and selective eviction cuts KV memory at a bounded error.

### 7.4 Known prior art and the gap

FoX, PaTH, Selective Attention, CoPE, stick-breaking attention, Selective RoPE, GRAPE, and SleepGate cover the mechanism family. The open pieces are: an LLM-scale cure, a theory that ties interference to which gates can express a register, selective eviction with a soundness bound, and a head-to-head retrofit comparison.

### 7.5 Honest rating

A strong main-track paper if Phase 0 passes. Spotlight potential if the 8B retrofit result is clean. Impact at the level of the 2017 transformer paper cannot be promised in advance for this or any idea.

## 8. Plan and Kill Criteria

| Phase | Work | Verify |
|-------|------|--------|
| 0. Pilot | Reproduce interference on Qwen3-1.7B. Train small transformers on a K-register task with RoPE, FoX gates, typed gates. Compare LoRA-only against the retrofit. | Kill criteria below. About 10 to 15 GPU-hours. |
| 1. Theory | Write and check the three proofs. | Each proof has a matching synthetic experiment. |
| 2. Retrofit | Scale to 4B and 8B. Full benchmark suite. | Claims 4 to 6, three seeds. |
| 3. Agents | Stale-file task, eviction, agent traces. | Claim 7 against AKV-style and compaction baselines. |
| 4. Paper | Draft, then adversarial review loops. | No open weakness in the review log. |

Kill criteria for Phase 0:

- K1: LoRA-only matches the retrofit within 3 points at 10x updates and 10x length. The architectural claim dies. Pivot to Backup 1.
- K2: typed gates do not beat scalar gates when registers exceed 4x the head count. The mechanism claim dies.
- K3: the retrofit costs more than 1 point on general benchmarks after tuning. Rework the training recipe before scaling.
- K4: thinking-mode inference already removes the failure in open models. Narrow the claim to no-reasoning and long-stream settings, or pivot.

### 8.1 Compute tiers

The user can buy about 2 to 3 H100-hours (stated 2026-11-11). The plan adapts as follows.

| Tier | GPU budget | What the paper can claim |
|------|------------|--------------------------|
| A. Ideal | About 150 H100-hours | Retrofits at 1.7B, 4B, 8B. Three seeds. Four methods compared. Agent traces and eviction. |
| B. Lean | About 40 H100-hours | Retrofits at 1.7B and 4B. Three seeds on the main method, one on baselines. Reduced agent study. |
| C. Current | 2 to 3 H100-hours plus free GPU quota | Theory and small-model results on CPU. Retrofit at 0.6B and 1.7B on free T4-class GPUs. One confirmation run at 4B on the H100. |

Rules for Tier C:

- Debug every GPU script on CPU with a tiny model first. No H100 minute goes to debugging.
- Use free quota (Kaggle gives about 30 GPU-hours per week on T4-class cards) for 0.6B and 1.7B runs.
- Spend the H100 hours last, on the single run that decides the largest-model claim.

## 9. Open Questions for the User

1. Target venue and deadline?
2. Can the user create a Kaggle account and share an API token, so free GPU runs can launch from this laptop?

## 10. Session Log

| Date | Work |
|------|------|
| 2026-11-11 | Checked machine. Ran about 45 web searches and abstract reads across agents, continual learning, latent reasoning, attention mechanisms, context rot, reliability theory. Rejected seven candidate ideas as taken. Proposed the primary idea and two backups. Created this tracker. |
| 2026-11-11 | User approved the primary idea and shared the repo. Built the scaffold: gated attention, register task, training script, tests. Started two reader sessions for full-text prior-art notes. The user asked for commit dates spread back to August. Declined: commits keep real timestamps. |

## 11. Resume Checklist

1. Read Sections 1, 4, and 9 of this file.
2. Run `git status` and `git log --oneline -10` to see what landed last.
3. Re-run a novelty search on the primary idea if more than two weeks have passed since the last log entry.
4. Add a row to Section 10 before the session ends.
5. Commit and push.
