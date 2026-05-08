# Final Question — Week 12 Day 04

## Topic
Evaluation and Statistics for Modern AI Systems

## The Question

**When comparing two LLM system configurations using bootstrap resampling, why does evaluating both systems on the *same* held-out task set (pairing) matter — and what specific statistical failure mode occurs when you treat a before/after comparison across two *different* task samples as if it were a paired test?**

---

## Why This Question

In Week 10 Act V, a bootstrap test reported p=0.0001 over 10,000 resamples, claiming a +33.67pp improvement from dev performance (53.33%, n=30 tasks) to held-out performance (85%, n=20 tasks). The two samples involved *different task sets* — dev and held-out — evaluated at different points in the development timeline.

In Week 11, the ablation bootstrap correctly evaluated the ORPO-trained model and the GPT-4o-mini baseline on the *same* 44 held-out tasks, and reported p=0.1953 — a statistically honest non-result at n=44.

The delta in reported p-values is not explained by sample size alone. It is explained by the structural difference between a paired and an unpaired comparison. Understanding this mechanism is the difference between correctly interpreting "the system improved" and incorrectly claiming statistical certainty about that improvement.

---

## Sharpening Trail

| Version | Question | Problem |
|---------|----------|---------|
| Draft 1 | "Why did I get different p-values in Week 10 vs Week 11?" | Descriptive; asks about outcomes not mechanisms |
| Draft 2 | "What does a bootstrap p-value measure when comparing LLM systems?" | Too broad — covers many bootstrap variants |
| Draft 3 | "Is the Week 10 bootstrap valid given it compared dev vs held-out?" | "Valid" is vague; doesn't name the failure mode |
| **Final** | As stated above | Mechanism-centered, grounded in both artifacts, identifies the exact failure mode, resolvable in one explainer |

---

## What a Satisfying Answer Looks Like

- Names the mechanism that makes pairing effective (variance reduction via shared noise cancellation)
- Explains what the unpaired bootstrap actually estimates vs. what you think it estimates
- Quantifies how much the CI widens when you move from paired to unpaired
- Identifies the engineering decision that changes: how you structure future ablation comparisons
- Connects to the Week 11 `bootstrap_stats.py` as the correct implementation

## Scope

**In scope:** Paired bootstrap mechanics, unpaired bootstrap behavior, the statistical assumption mismatch, and its practical consequence on reported p-values in LLM evals.

**Out of scope:** Permutation tests, Bayesian alternatives, multi-comparison correction, power analysis in general.
