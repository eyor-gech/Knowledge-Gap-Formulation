# Canonical List — Knowledge Gap Formulation Cohort

**Author:** Eyor Getachew
**Scope:** Resources drawn from or directly applicable to Day 01 (KV Cache Mechanics) and Day 04 (Paired Bootstrap for LLM Evaluation). Inferred additions are marked explicitly.

---

## Papers

### Day 01 — Inference Efficiency

**Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 2017.**
https://arxiv.org/abs/1706.03762

- **Why it matters:** The primary source for the scaled dot-product attention formula (`Attention(Q,K,V) = softmax(QK^T/√d_k) · V`), the multi-head projection structure (W_Q, W_K, W_V), and the O(n²) complexity of self-attention. Everything in the KV cache discussion traces back here.
- **Where it was useful:** Day 01 explainer — the O(n²)→O(n) argument requires understanding what the `Q @ K.T` term actually computes and why it scales quadratically.
- **When to use it:** Any time you are making a claim about transformer attention computation. If you cannot cite the formula, you are not in a position to make a precise efficiency claim.

---

**Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.**
https://arxiv.org/abs/2309.06180

- **Why it matters:** The production treatment of KV cache memory management. Establishes the formula for KV cache size as a function of model hyperparameters and sequence length. Provides empirical data on GPU memory fragmentation from KV cache under realistic serving workloads. Introduces PagedAttention as the solution, implemented in vLLM.
- **Where it was useful:** Day 01 — the memory cost formula (2 × layers × heads × head_dim × seq_len × dtype_bytes) comes directly from this paper. The "at 128K context, KV cache exceeds model weights" claim is grounded here.
- **When to use it:** When making infrastructure capacity claims about LLM serving. When evaluating whether prefix caching will help or whether memory is the binding constraint. When presenting to an infrastructure or ML platform team.

---

### Day 04 — Evaluation Statistics

**Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.**

- **Why it matters:** The foundational text. The paired bootstrap is derived from first principles in Chapter 9 as the statistically correct estimator when data is naturally paired. The derivation of Var(A − B) = Var(A) + Var(B) − 2·Cov(A,B) and its implications for paired vs. unpaired resampling are worked out here, not in later references that cite it.
- **Where it was useful:** Day 04 — the entire mechanistic explanation of why the Week 10 bootstrap was misleading traces back to this variance decomposition.
- **When to use it:** When building or reviewing any bootstrap significance test. Read Chapter 9 before claiming that your bootstrap result is "paired." The pairing requirement is structural, not procedural.

---

**Dror, R. et al. (2018). "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing." ACL 2018.**
https://aclanthology.org/P18-1128/

- **Why it matters:** The canonical NLP-specific treatment of significance testing. Section 4 covers paired bootstrap explicitly in the context of NLP model comparison. Distinguishes bootstrap CIs from bootstrap significance tests. Explicitly warns against before/after comparisons with different test sets.
- **Where it was useful:** Day 04 — grounded the claim that the "paired" label requires both systems to score the same examples, not just the same benchmark version or domain.
- **When to use it:** Any time you are running a bootstrap to compare two NLP or LLM systems. Read Section 4 before writing up significance results. If your experimental setup does not satisfy the pairing requirement described here, report descriptive statistics instead.

---

**Bouthillier, X. et al. (2021). "Accounting for Variance in Machine Learning Benchmarks." MLSys 2021.**
https://proceedings.mlsys.org/paper_files/paper/2021/hash/cfecdb276f634854f3ef915e2e980c31-Abstract.html

- **Why it matters:** Empirically demonstrates — on real benchmarks, not toy examples — that evaluation variance (which examples are in the test set) dominates performance variance (which model you trained) for most standard ML comparisons. The paper puts numbers on the claim that small held-out sets are insufficient for strong statistical claims.
- **Where it was useful:** Day 04 — provides the empirical backing for why the Week 11 CI [−6.8, +20.4] crossing zero at n=44 is not a malfunction but a correct measurement of the uncertainty regime.
- **When to use it:** When someone on your team claims that a small held-out benchmark result is statistically significant. The paper gives you quantitative language to push back. Also useful when designing new eval sets — it tells you roughly how large your held-out set needs to be for your effect size.

---

**Biderman, S. et al. (2024). "Lessons from the Trenches on Reproducible Evaluation of Language Models." arXiv 2405.13163.**
https://arxiv.org/abs/2405.13163

- **Why it matters:** Written by the maintainers of the Eleuther AI lm-eval-harness. Documents the same class of errors that appeared in the Week 10 vs. Week 11 comparison: task set drift, evaluation infrastructure changes between runs, random seed variation. Argues that infrastructure stability is as important as model quality for reproducible eval results.
- **Where it was useful:** Day 04 — the argument that version-controlling your eval set alongside your model checkpoint is a first-order engineering requirement, not a nice-to-have.
- **When to use it:** Before deploying any evaluation pipeline to production. When an eval result shows unexpected improvement or regression — the first hypothesis should be infrastructure change, not model change. When reviewing someone else's fine-tuning results.

---

## Blogs

**[Inferred Recommendation] Horace He (2022). "Making Deep Learning Go Brrrr From First Principles." Horace.io.**

- **Why it matters:** Accessible treatment of GPU memory bottlenecks, compute vs. memory-bound operations, and why attention is memory-bound at decode time. Provides the mental model for understanding why the KV cache matters more at long sequences.
- **Where it would be useful:** Day 01 adjacent — the "latency vs. memory" trade-off section of the KV cache explainer.
- **When an FDE should use it:** When explaining inference performance trade-offs to an engineering team that has not worked with GPU-bound ML systems.

---

**[Inferred Recommendation] Anthropic (2024). "Prompt Caching." Anthropic Documentation.**
https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

- **Why it matters:** The operational reference for `cache_control` — the specific mechanism named in the Day 01 grounding commit. Explains which content types are cacheable, the 5-minute TTL, and the cost reduction model.
- **Where it would be useful:** Day 01 — directly relevant to the "confirm deployment setting" placeholder in the revised model card.
- **When an FDE should use it:** When deploying a multi-turn agent on Anthropic's API and making efficiency claims in product documentation.

---

**[Inferred Recommendation] vLLM Team (2024). "Automatic Prefix Caching." vLLM Documentation.**
https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html

- **Why it matters:** The vLLM-specific reference for `enable_prefix_caching=True`. Explains the LRU eviction policy, block-level granularity, and the conditions under which prefix caching provides measurable TTFT reduction.
- **Where it would be useful:** Day 01 — the companion reference to the Anthropic prompt caching doc for self-hosted deployments.
- **When an FDE should use it:** When deploying vLLM in a multi-tenant setting and evaluating whether prefix caching is worth the memory trade-off.

---

## Repositories

**HuggingFace Transformers — `past_key_values` and Cache Classes**
https://github.com/huggingface/transformers

- **Why it matters:** The canonical implementation of KV cache in the Python ML ecosystem. The `past_key_values` structure, `DynamicCache`, `StaticCache`, and `QuantizedCache` are the primary reference points for anyone reasoning about KV cache mechanics at the code level.
- **Where it was useful:** Day 01 — the `model(input_ids=..., use_cache=True).past_key_values` code examples in the explainer are directly from this implementation.
- **When an FDE should use it:** Any time you need to trace what the KV cache actually stores in a HuggingFace-based deployment. The `generation_utils.py` file and the `modeling_llama.py` attention implementation are the two most relevant entry points.

---

**[Inferred Recommendation] EleutherAI / lm-evaluation-harness**
https://github.com/EleutherAI/lm-evaluation-harness

- **Why it matters:** The de facto standard for reproducible LLM evaluation. Supports task versioning, seeded sampling, and standardized metrics. Referenced in Biderman et al. (2024). The eval infrastructure the Day 04 grounding commit implicitly calls for.
- **Where it would be useful:** Day 04 — the recommendation to "version-control the task set alongside the model checkpoint" is most practically implemented using this harness.
- **When an FDE should use it:** Any time you need to establish a reproducible benchmark for model comparison. The harness enforces the structural requirements (same tasks, same preprocessing, same seeds) that the Day 04 explainer argues are necessary for valid paired bootstrap tests.

---

## Frameworks

**vLLM**
https://github.com/vllm-project/vllm

- **Why it matters:** The production inference framework for serving transformer models with KV cache management, PagedAttention, and prefix caching. The specific deployment target named in the Day 01 grounding commit.
- **Where it was useful:** Day 01 — `enable_prefix_caching=True` is the operationalization of the cross-request prefix caching claim.
- **When an FDE should use it:** For any production LLM deployment where multi-turn efficiency, high concurrency, or long-context serving is a requirement.

---

**[Inferred Recommendation] NumPy random generator (`numpy.random.default_rng`)**

- **Why it matters:** The correct tool for reproducible bootstrap implementations. The `rng.choice(..., replace=True)` pattern with a fixed seed is what separates a reproducible bootstrap from a result that cannot be regenerated. The Day 04 code demonstration uses this correctly.
- **When an FDE should use it:** In any bootstrap implementation. Always pass `seed=42` (or document the seed used). Never use `np.random.choice` without a seeded generator in evaluation code.

---

## Evaluation Methods

**Paired Bootstrap Significance Test**

- **What it is:** Resample task indices, compute per-task deltas for system A vs. system B on the same task set, build a CI from the resampled mean-delta distribution.
- **Why it matters:** The structurally correct test for comparing two LLM system configurations. Recovers the covariance term Cov(A,B) that the unpaired bootstrap discards.
- **Grounded in:** Day 04 — `bootstrap_stats.py` is the correct implementation; the Day 04 explainer documents its preconditions.
- **When to use it:** When you have evaluated two system variants on exactly the same held-out task set. Not applicable when comparing before/after results from different task sets.

---

**pass@1 (Single-sample pass rate)**

- **What it is:** Fraction of tasks where the system passes on the first attempt. Used as the primary metric in both Week 10 (53.33%, 85%) and Week 11 (90.9% vs 84.1%) evaluations.
- **Why it matters:** A high-variance metric at small n. A 1-task shift in a 20-task held-out set changes pass@1 by 5pp. This is exactly the regime where the bootstrap CI is informative — and where the unpaired bootstrap produces misleadingly tight CIs.
- **When to use it:** As a primary metric when task completion is binary. Always report with CI, not just point estimate.

---

**[Inferred Recommendation] pass@k (Multi-sample pass rate)**

- **What it is:** Generate k completions per task; count the task correct if any pass. Reduces per-task variance by making each verdict a majority decision.
- **Why it matters:** Directly reduces the evaluation variance that Bouthillier et al. identify as the dominant noise source at small n. Useful when you cannot increase the number of tasks.
- **When an FDE should use it:** When your task count is fixed and you need to reduce evaluation noise. Most relevant for functional correctness benchmarks (code generation, structured output, tool-use verification).

---

## Deployment Patterns

**The `[confirm deployment setting]` pattern**

- **What it is:** A placeholder embedded in documentation that makes an infrastructure-dependent claim conditional on a verifiable check. Used in the Day 01 grounding commit to flag that cross-request prefix caching requires explicit configuration.
- **Why it matters:** Prevents a conditional claim from being treated as an unconditional fact when the document is reused by a downstream team. Creates an audit trail.
- **When to use it:** Any time a performance, cost, or capability claim depends on a deployment configuration that may not be universal across environments.

---

**[Inferred Recommendation] Eval set version control alongside model checkpoints**

- **What it is:** Committing the exact task set (task IDs, prompts, expected outputs, preprocessing config) to version control alongside the model checkpoint that was evaluated against it.
- **Why it matters:** The Day 04 analysis demonstrates that changing the task set between evaluations introduces confounds the bootstrap cannot account for. This pattern prevents that class of error structurally.
- **When an FDE should use it:** Always, for any evaluation that will be used to support a deployment decision. The overhead is low; the protection against spurious improvement claims is high.

---

## Observability / Debugging Tools

**[Inferred Recommendation] Time-to-First-Token (TTFT) measurement**

- **What it is:** Wall-clock latency from the end of the request to the first streamed token. Determined primarily by prefill time, which scales O(n²) with sequence length.
- **Why it matters:** The most direct observable metric for whether cross-request prefix caching is working. If prefix caching is enabled and the prefix is cached, TTFT should be lower on the second and subsequent turns of a multi-turn conversation.
- **Where it would be useful:** Day 01 — one of the two residual questions left open in the sign-off was "can I measure the TTFT difference with and without prefix caching?" This is the measurement.
- **When an FDE should use it:** As the primary latency diagnostic in multi-turn agent deployments. A regression in TTFT is often the first observable symptom of a KV cache configuration problem.

---

**[Inferred Recommendation] GPU memory profiling (`nvidia-smi`, `torch.cuda.memory_stats`)**

- **What it is:** Tools for measuring GPU memory allocation, fragmentation, and peak usage during inference.
- **Why it matters:** The Day 01 memory cost formula (~0.5 MB/token for a 7B model) provides a theoretical estimate. Profiling tools provide the empirical measurement. The gap between them reveals whether PagedAttention or quantization is needed.
- **When an FDE should use it:** When deploying at scale or when VRAM constraints are the limiting factor on serving capacity. Before claiming that a given model can support a given context length at a given concurrency level.

---

*All items without the `[Inferred Recommendation]` marker were directly cited in Day 01 or Day 04 source files.*
