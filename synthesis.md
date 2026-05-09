# Synthesis — Knowledge Gap Formulation Cohort

**Author:** Eyor Getachew
**Program:** Tenx / 10 Academy — Forward-Deployed Engineering Cohort
**Completed sessions:** Day 01, Day 04
**Partial completion acknowledged:** Days 02 and 03 were not completed. This synthesis is built honestly from two sessions, not four.

---

## What This Program Was Asking Me to Do

The structure of this program was not "learn concepts and write about them." It was closer to: identify a specific claim you have made or repeated in deployed work, trace the claim back to its load-bearing mechanism, close the gap between the surface claim and its underlying justification, and produce an artifact that makes the closure auditable. That is a harder task than it looks. Most of the difficulty is in the first step — finding a claim that feels solid enough to have passed review but weak enough that it cannot withstand mechanistic interrogation.

Both sessions I completed followed this pattern. In each case, the starting point was a real artifact from prior work. In each case, the gap I found was not about missing information but about missing precision: I had the right answer at too low a resolution.

---

## The Two Gaps I Actually Closed

### Day 01: The KV Cache

The claim I had written — and repeated — was: "multi-turn conversations are efficient due to caching." It appeared in a model card and a CFO memo. It survived review because it is directionally correct. It did not survive the question: *what exactly is cached, what computation is avoided, and when does this hold?*

The mechanistic answer requires three things I had not previously articulated together. First, what is stored: the key and value projection tensors (`X @ W_K` and `X @ W_V`) for every processed token, per attention layer, per head, with shape `(batch_size, num_heads, seq_len, head_dim)` — not embeddings, not attention weights, not final hidden states. Second, what is avoided: on each decode step, only the new token's K and V projections are computed; the n prior ones are retrieved from cache, reducing per-step projection work from O(n) to O(1) and making the full attention operation O(n) instead of O(n²). Third, the conditional: within-generation KV caching is automatic in all inference backends; cross-request prefix caching — the mechanism that makes multi-turn *across API calls* efficient — requires explicit infrastructure (vLLM `enable_prefix_caching=True`, Anthropic prompt caching with `cache_control`). The original claim was only defensible if that third condition held, and whether it held was a verifiable deployment question, not an assumption.

The grounding commit replaced the vague efficiency claim with a formula for memory cost (approximately 0.5 MB per token per concurrent request for a 7B model at float16), a precise description of what is cached and what is not, and a `[confirm deployment setting]` placeholder that prevents the claim from being restated without verification. The change is small in word count and large in accountability.

### Day 04: The Bootstrap Pairing Problem

The claim from Week 10 of the prior cohort work was: "Bootstrap statistical test: 10,000 resamples, p=0.0001. The system demonstrated statistically significant improvement from the development baseline (53.33%) to the held-out evaluation (85%)." That claim was never challenged during submission.

The problem is structural. The Week 10 bootstrap compared dev performance (n=30 tasks) against held-out performance (n=20 different tasks). These are not the same tasks. The bootstrap resampled from two independent pools. The variance of the resulting difference includes task-selection uncertainty from both pools, with no cancellation — because Var(A − B) = Var(A) + Var(B) − 2·Cov(A,B), and when A and B see different tasks, Cov(A,B) is zero by construction.

The Week 11 bootstrap was structurally correct: the ORPO-trained model and the GPT-4o-mini baseline both scored every one of the 44 held-out tasks, and the bootstrap resampled per-task deltas. The resulting CI [−6.8, +20.4] pp crossed zero and the p-value was 0.1953. That was called "underpowered." It should have been called "honest."

The gap I closed was understanding why the Week 10 p-value of 0.0001 is not more impressive than the Week 11 p-value of 0.1953 — it is less trustworthy. More resamples do not fix a structurally wrong comparison. The grounding commit removed the false precision from the Week 10 Act V summary and documented the pairing precondition in `bootstrap_stats.py`.

---

## Engineering Judgment Developed

**On documentation:** A technical claim that cannot survive the question "what mechanism produces this outcome?" should not appear in a model card, a memo to a CFO, or a presentation to a hiring committee. The bar is not "is this true?" but "is this auditable?" The KV cache revision demonstrates this: the original claim was true in the right conditions; the revision makes those conditions explicit and checkable.

**On statistical framing:** There is a meaningful difference between "evidence sufficient to act" and "evidence sufficient to publish a statistical claim." In production, you often need to ship on incomplete evidence — that is normal. What is not normal, and not defensible, is claiming statistical confirmation when the test setup was structurally mismatched. The correct response to the Week 10 bootstrap is not to run more resamples; it is to re-run the comparison with both systems on the same task set, or to report the delta descriptively and lean on convergent qualitative evidence from the ablation variants.

**On scope discipline:** Both sessions explicitly excluded adjacent topics that could have expanded indefinitely. The KV cache explainer excluded GQA, PagedAttention, and speculative decoding. The bootstrap explainer excluded permutation tests, Bayesian alternatives, and power analysis formulas. This is not laziness — it is precision. An explainer that attempts to cover everything typically clarifies nothing.

---

## Most Surprising Insight

The most surprising finding across both sessions was that the two gaps are structurally identical, despite appearing in completely different domains. In both cases: a claim that sounds precise is actually conditional on an infrastructure or design choice that was never verified. For the KV cache, the condition is whether prefix caching is enabled across API calls. For the bootstrap, the condition is whether both system states were evaluated on the same task set. In both cases, the claim is plausible enough to pass review, and wrong enough to mislead a downstream decision-maker. The program's methodology — force a mechanistic account, trace it to a grounding artifact — catches this class of error reliably.

---

## Demonstrated Learnings

- KV cache mechanics: what tensors are stored, what computation is avoided, how memory cost scales with sequence length and concurrency
- Paired vs. unpaired bootstrap resampling: the variance mechanics, the covariance term, the production implication for eval set design
- How to translate an infrastructure concept into auditable documentation language (the `[confirm deployment setting]` pattern)
- How to distinguish a real claim from a claim that relies on an unverified conditional

---

## Inferred Learnings

*[These are themes consistent with the work but not directly evidenced by session artifacts. Labeled explicitly as inferred.]*

- The program likely covered prompt engineering rigor and tool-call reliability patterns on Days 02 and 03. Based on the Day 04 context (ORPO fine-tuning of a judge model, multi-turn agent evaluation), I would infer those sessions addressed retrieval quality, context management, and structured output reliability — gaps I have but did not close in this cohort.
- The multi-system evaluation setup (TenaciousBench, ORPO-trained Qwen2.5-7B judge) suggests experience with fine-tuning for evaluation tasks, which is a non-trivial capability not fully explicated in the available artifacts.

---

## Future Work Areas

- Properly close the two questions explicitly left open in Day 01: (1) does the current inference deployment have prefix caching enabled, measurable by TTFT comparison; (2) at what concurrent user count does KV cache memory become the binding capacity constraint?
- Run the correct statistical comparison for Week 10: evaluate both the pre-improvement and post-improvement system states on the same held-out task set, then run the paired bootstrap on per-task deltas.
- Complete or revisit Days 02 and 03. The gaps from those sessions remain open.

---

## Patterns Worth Reusing

**The `[confirm deployment setting]` placeholder.** When documentation makes a claim that depends on infrastructure configuration, embed a check-step directly in the document. This forces verification at the point of reuse rather than relying on the reader to remember the conditional.

**The morning call interrogation protocol.** The Day 04 morning call is a model of question sharpening: a vague question ("why different p-values?") is pushed through successive interrogation rounds until it reaches a mechanism-centered, artifact-grounded form that can be resolved in a single explainer. The trace of that sharpening — four draft questions with explicit annotations of each failure — is itself a useful artifact.

**The before/after grounding commit.** Making the gap closure auditable by showing the exact text that existed before and the exact text that replaces it, with a section explaining what changed and why, creates a portable record of the engineering decision.

---

## Canonical Reading List

See [canonical_list.md](canonical_list.md) for the full annotated list. Core texts:

- Vaswani et al. (2017) — the source of the attention equations and O(n²) complexity analysis
- Kwon et al. (2023) — PagedAttention; the production treatment of KV cache memory management
- Efron & Tibshirani (1993) — the foundational bootstrap text; Chapter 9 on paired resampling
- Dror et al., ACL 2018 — the canonical NLP significance testing reference; the paired bootstrap section applies directly
- Biderman et al. (2024) — operational LLM evaluation; eval infrastructure instability as the dominant source of spurious "improvements"

---

## Canonical Tooling Stack

- **HuggingFace Transformers** — `past_key_values`, `use_cache=True`, `DynamicCache`/`StaticCache`
- **vLLM** — `enable_prefix_caching=True` for cross-request KV reuse
- **Anthropic API** — `cache_control` for prompt caching
- **NumPy** — `rng.choice(..., replace=True)` for bootstrap implementation
- **lm-eval-harness** (Eleuther AI) — the evaluation framework referenced in Biderman et al.; the correct substrate for reproducible LLM benchmarking
- **Python dataclasses / JSON versioning** — for eval set version control alongside model checkpoints

---

*Total sessions completed: 2 of 4. This synthesis is complete and honest for those two sessions.*
