# Grounding Commit — Week 12 Day 04

## What Changes in the Week 10 Artifact

**File:** `week10_final/results/act5_evidence_graph.json`  
**File:** `week10_final/reports/report_week10.md` (Act V section)

---

## Before

**Act V summary (current):**

> "Bootstrap statistical test: 10,000 resamples, p=0.0001. The system demonstrated statistically significant improvement from the development baseline (53.33%) to the held-out evaluation (85%), Δ=+33.67pp, 95% CI [70%, 100%]."

**Implicit claim:** The bootstrap p-value confirms that the improvement is unlikely to be due to chance.

---

## After

**Act V summary (revised):**

> "Dev evaluation (Act I, n=30 tasks) returned 53.33% pass@1. Held-out evaluation (Act IV, n=20 different tasks) returned 85% pass@1. The observed delta is +33.67pp. However, because the dev and held-out task sets are different, this comparison is structurally unpaired — we cannot run a paired bootstrap across two disjoint task sets. The reported p=0.0001 was computed as if the two samples were comparable, which underestimates the variance introduced by using different task samples under different development conditions.

> The correct claim is: pass rates were 53.33% on dev and 85% on held-out. These task sets differ in timing, composition, and development exposure. The improvement direction is consistent across ablation variants (confidence_aware 0.76 > binary_threshold 0.70 > no_confidence 0.66), which provides convergent qualitative evidence. A paired statistical test would require evaluating both the pre-improvement and post-improvement system states on the same task set, which was not done in this evaluation design."

---

## Why This Change

**Before:** The p-value was framed as confirming a causal claim ("the system improved") when it only describes a descriptive comparison of two independently-drawn pass rates. The bootstrap cannot close the logical gap between "dev performance was 53.33%" and "held-out performance was 85%" because too many things changed simultaneously: task set composition, development history, and potential overfitting to dev patterns.

**After:** The same data is presented honestly. The +33.67pp delta is real and worth reporting. The convergent ablation evidence (all three confidence variants showed improvement in the same direction on held-out) is meaningful. What's removed is the false precision of p=0.0001 as statistical confirmation.

---

## How This Changes the Engineering Decision

**Old decision:** "We have p=0.0001, so the system is provably better. Ship it."

**New decision:** "We have consistent directional evidence across three ablation variants, a large observed delta, and a held-out evaluation that wasn't used during development. That's evidence worth acting on. We should not claim statistical significance because we didn't run the right test. In future evaluation designs, we will evaluate baseline and final-system states on the same task set before claiming significance."

This is a better engineering decision because it correctly separates "evidence sufficient to act" from "evidence sufficient to publish a statistical claim." In production, you often need to act on incomplete evidence. That's fine — as long as you're honest about what you've measured and what you haven't.

---

## What to Add to bootstrap_stats.py (Week 11 — Already Correct, Annotate)

The Week 11 implementation is correct. Add a docstring to `bootstrap_stats.py` that makes the pairing requirement explicit:

```python
def paired_bootstrap_test(scores_a: list[float], scores_b: list[float],
                          n_resamples: int = 10_000, seed: int = 42) -> dict:
    """
    Paired bootstrap significance test for LLM evaluation.

    Requires: scores_a[i] and scores_b[i] must be scores for the SAME task i
    under system A and system B respectively. If the two lists come from
    different task sets, this function is computing the wrong thing.

    Returns p-value for H1: mean(B) > mean(A) (one-sided).
    """
```

---

## Rubric Impact

| Artifact | Before | After |
|----------|--------|-------|
| Week 10 Act V | Claims p=0.0001 as confirmation of improvement | Correctly frames delta as descriptive, notes structural limitation |
| bootstrap_stats.py | Correct implementation, no documentation of pairing requirement | Documented precondition makes the assumption explicit |
| Model card | Silent on this distinction | Could note that ablation CIs reflect paired bootstrap on the same 44 tasks |
