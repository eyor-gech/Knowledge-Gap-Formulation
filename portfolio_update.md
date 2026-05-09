# Portfolio Update — Forward-Deployed Engineering

**Eyor Getachew** |
**Program:** Tenx / 10 Academy — FDE Cohort, Week 12
**Submission scope:** Two completed sessions (Day 01, Day 04) from a four-session knowledge gap formulation sprint.

---

## Summary

This sprint required identifying real gaps in prior deployed work, tracing each gap to its underlying mechanism, and producing an auditable closure. Two sessions were completed. Both resulted in grounding commits that improved the technical defensibility of production artifacts.

---

## Project 1 — KV Cache Documentation Audit

**Session:** Day 01 | **Gap domain:** Inference infrastructure + model documentation

### What the gap was

A production model card and an executive memo both stated that multi-turn conversations were "efficient due to caching." That claim survived review because it is directionally correct. It could not survive the question: *what is cached, what computation is avoided, and when does this hold?*

### What was produced

A grounding commit that replaced the vague efficiency claim with:

- A precise description of what is stored: key and value projection tensors (`X @ W_K`, `X @ W_V`) for all processed tokens, per layer, per head — not embeddings, not attention weights
- The computational consequence: O(n²)→O(n) per decode step, with an enumeration of the 65,536 matrix multiplications eliminated per token in a 32-layer model at 2048 tokens
- A concrete memory cost formula: ~0.5 MB per token per concurrent request for a 7B model at float16
- A critical conditional: within-generation caching is automatic; cross-request prefix caching is infrastructure-dependent and requires explicit configuration (`vLLM enable_prefix_caching=True` or Anthropic `cache_control`)
- A `[confirm deployment setting]` placeholder to prevent the claim from being restated without verification

### Engineering judgment demonstrated

Identifying that the original claim was not *wrong* but was *unverifiable* — and that the distinction matters for production documentation. The revision is auditable: an infrastructure engineer can check the deployment setting, run a TTFT comparison, and either confirm or falsify the claim. The original could not be falsified.

### Technical stack

HuggingFace Transformers (`past_key_values`, `use_cache=True`), vLLM, Anthropic API prompt caching, PyTorch attention mechanics

### Canonical sources

Vaswani et al. 2017 (attention formula, O(n²) complexity), Kwon et al. 2023 / SOSP (memory cost formula, PagedAttention)

---

## Project 2 — Bootstrap Evaluation Audit

**Session:** Day 04 | **Gap domain:** LLM evaluation statistics

### What the gap was

Week 10 of the cohort's prior build reported p=0.0001 from a bootstrap test comparing dev performance (53.33%, n=30 tasks) to held-out performance (85%, n=20 different tasks). The claim was that the improvement was statistically significant. Week 11 ran a bootstrap on the same 44 tasks and got p=0.1953 for a +6.8pp lift. The Week 11 result was called "underpowered."

The difference is not power. It is structure.

### What was produced

A grounding commit that:

- Revised the Week 10 Act V summary to remove the false statistical claim, replacing it with a correctly framed descriptive delta and an explicit statement of the structural limitation
- Added a docstring to `bootstrap_stats.py` that makes the pairing precondition explicit: `scores_a[i]` and `scores_b[i]` must be scores for the *same task i*, not scores from independent task pools
- Documented the correct engineering decision: "evidence sufficient to act" is not the same as "evidence sufficient to publish a statistical claim"

The explainer demonstrates the variance mechanism: Var(A − B) = Var(A) + Var(B) − 2·Cov(A,B). When both systems see the same tasks, the covariance is real and large. When they see different tasks, it is zero by construction. The Week 10 p-value of 0.0001 reflects artificially low within-pool variance, not a reliable system improvement signal.

### Engineering judgment demonstrated

The ability to look at an impressively small p-value and ask "is this bootstrap comparing the right things?" rather than accepting it. The ability to distinguish a structural problem (wrong test design) from a statistical problem (insufficient power). The ability to reframe a failed significance test as honest, informative evidence rather than a deficiency.

### Technical stack

NumPy (`numpy.random.default_rng`, bootstrap resampling), ORPO fine-tuning (Qwen2.5-7B judge), GPT-4o-mini baseline evaluation, JSON-versioned eval artifacts

### Canonical sources

Efron & Tibshirani 1993 (paired bootstrap mechanics), Dror et al. ACL 2018 (NLP significance testing), Bouthillier et al. MLSys 2021 (evaluation variance), Biderman et al. 2024 (LLM eval reproducibility)

---

## Operational Lessons

1. **Documentation debt is a production risk.** A claim that cannot be audited is a claim that will eventually be misused. Both projects involved replacing vague-but-plausible claims with precise, conditional, verifiable ones.

2. **Statistical tooling does not validate experimental design.** 10,000 bootstrap resamples produce a precise estimate of the wrong thing if the comparison is structurally mismatched. Rigor in execution does not substitute for rigor in design.

3. **Scope discipline is a deliverable.** Both explainers explicitly excluded adjacent topics (GQA, PagedAttention, permutation tests, power analysis) to maintain sharpness. This is not a limitation — it is a judgment call that makes the core claim clearer.

---

## Completion Scope

Two of four sessions completed. The work presented here represents two grounding commits, two mechanism-level explainers, two Twitter threads, and two annotated source sets. Days 02 and 03 were not completed and are not represented in this portfolio.
