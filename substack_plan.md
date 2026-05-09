# Substack Content Plan

**Author:** Eyor Getachew
**Program:** Tenx / 10 Academy — FDE Cohort
**Mapping:** 2 articles from Day 01, 2 articles from Day 04, 1 synthesis/meta article

---

## Article 1 — Day 01

### Title
**"What Your Model Card Doesn't Say About Caching"**

### Abstract

Most production model documentation describes KV caching as a binary feature: either it is happening or it is not. This article walks through the mechanism precisely — what tensors are stored, at what granularity, at what memory cost — and then draws the distinction that almost every model card misses: within-generation caching (automatic in all backends) is not the same as cross-request prefix caching (infrastructure-dependent, requires explicit configuration). The article closes with a concrete before/after revision of a production model card, showing how to write an efficiency claim that survives technical scrutiny without becoming inaccessible to a non-ML audience.

**Target reader:** ML engineers writing deployment documentation; technical PMs who need to understand what their documentation actually promises.

**Core argument:** The claim "efficient due to caching" is not wrong — it is unverifiable. There is a meaningful difference.

**Key specifics:** O(n²)→O(n) decode-step computation; `past_key_values` tensor shape; the `[confirm deployment setting]` pattern; vLLM `enable_prefix_caching` and Anthropic `cache_control`.

---

## Article 2 — Day 01

### Title
**"The Memory Cost Your LLM Deployment Forgot to Budget For"**

### Abstract

When a team moves from a single-turn to a multi-turn agent, the inference cost model changes in ways that most deployment guides underspecify. This article presents the KV cache memory cost formula — approximately 0.5 MB per token per concurrent request for a 7B model at float16 — and traces through the implications: at 4K context, roughly 2 GB per concurrent request; at 128K context, roughly 64 GB, which exceeds the model weights. The article explains why this is not a theoretical edge case but a practical constraint that becomes binding under realistic concurrency at production scale. It ends with the questions an engineer should answer before making a long-context deployment decision: how many concurrent users, at what average context length, with what VRAM budget?

**Target reader:** ML engineers planning LLM deployments; infrastructure engineers who will be asked to provision compute for a model they did not train.

**Core argument:** The memory cost of the KV cache is the variable most commonly absent from deployment planning. It is also the variable most likely to cause a production incident at scale.

**Key specifics:** The formula: `2 × layers × heads × head_dim × seq_len × dtype_bytes`; the Kwon et al. (SOSP 2023) empirical data; the PagedAttention solution and when it matters.

---

## Article 3 — Day 04

### Title
**"Your p=0.0001 Is Lying to You"**

### Abstract

In Week 10 of a prior build, a bootstrap test over 10,000 resamples returned p=0.0001 for a +33.67pp improvement. The result was presented as statistical confirmation. In Week 11, the correct bootstrap returned p=0.1953 for a +6.8pp improvement. The Week 11 result was called "underpowered." This article explains why that framing is backwards — and why the Week 10 p-value is less trustworthy, not more impressive, than the Week 11 result. The mechanism: the Week 10 bootstrap resampled from two independent task pools (different tasks for dev and held-out), setting the covariance term to zero by construction and producing an artificially precise estimate. The Week 11 bootstrap resampled per-task deltas from the same 44 tasks, correctly recovering the covariance. More resamples do not fix a structurally mismatched comparison.

**Target reader:** ML engineers who run bootstrap tests on LLM eval results; anyone who has ever cited a p-value from a before/after comparison.

**Core argument:** The number of bootstrap resamples controls precision of the estimate, not validity of the comparison. Structural correctness is a prerequisite, not a parameter.

**Key specifics:** Var(A − B) = Var(A) + Var(B) − 2·Cov(A,B); the before/after vs. A/B distinction; the code difference between unpaired and paired resampling; Dror et al. ACL 2018.

---

## Article 4 — Day 04

### Title
**"How I Corrupted My Own Ablation Study (And How to Not)"**

### Abstract

Ablation studies are supposed to isolate the effect of a single change. This article describes a specific failure mode that makes them unreliable without obvious signs of error: evaluating the baseline and the improved system on different task sets, then bootstrapping the difference. The article walks through the exact mechanics of how this produces misleading p-values, what the correct experimental design looks like (freeze the task set before training begins, evaluate all variants on exactly that set, version-control the task set alongside the model checkpoint), and what the honest reporting of an underpowered result looks like. The Week 11 CI [−6.8, +20.4] pp crossing zero is the example: not a failed experiment, but a correctly instrumented one that measured real uncertainty.

**Target reader:** ML engineers and researchers running ablations on fine-tuned or prompted LLMs; anyone who has reported an improvement from an ablation without a paired comparison.

**Core argument:** The correct response to an underpowered ablation is not "run more resamples" — it is "design the next evaluation correctly." An honest non-result is more valuable than a confident wrong one.

**Key specifics:** Eval set versioning; Biderman et al. 2024 on infrastructure instability as the dominant source of spurious improvements; the `bootstrap_stats.py` docstring as a practical pattern for making the pairing precondition explicit.

---

## Article 5 — Synthesis / Meta-learning

### Title
**"Two Gaps, One Pattern: What Four Sessions of Mechanistic Interrogation Actually Teach You"**

### Abstract

*[Note: This article covers only the two completed sessions honestly. It acknowledges the two uncompleted sessions explicitly and argues that the pattern is visible from two data points — which is itself a defensible claim.]*

A two-session sprint to close knowledge gaps in prior deployed work revealed something unexpected: both gaps had the same structure. In the KV cache session, a claim that sounded precise ("efficient due to caching") turned out to depend on an unverified infrastructure condition. In the bootstrap session, a result that looked confident (p=0.0001) turned out to depend on an unverified structural assumption (that both systems had seen the same tasks). The pattern: claims that survive review because they are directionally correct, but fail under mechanistic interrogation because they omit a critical conditional. This article reflects on why this class of gap is so common in deployed AI systems, what the knowledge gap formulation methodology does that makes it effective at finding them, and what I would have investigated in the two sessions I did not complete — framed as open questions rather than completed work.

**Target reader:** Engineers participating in structured learning programs; senior engineers mentoring newer colleagues on how to identify gaps in their own work; anyone building documentation practices for deployed AI systems.

**Core argument:** The most dangerous technical claims are not the ones that are wrong — those get caught in review. They are the ones that are *conditionally correct but stated unconditionally*. A methodology that forces mechanistic accountability catches this class reliably.

**Honest framing:** This article is written from two sessions. The pattern I describe may be more fully visible across four sessions. I did not complete all four sessions, and I am not going to pretend I did.
