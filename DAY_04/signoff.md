# Week 12 Day 04 — Signoff

**Date:** 2026-05-08  
**Topic:** Evaluation and Statistics for Modern AI Systems  
**Artifact set:** DAY_04 Artifacts/

---

## Summary of Work

This session closed a gap in my understanding of paired bootstrap mechanics as applied to LLM evaluation — specifically, why the Week 10 Act V bootstrap comparison (p=0.0001) is structurally mismatched while the Week 11 ablation bootstrap (p=0.1953) is structurally correct.

The gap was not visible until I forced myself to describe what was being resampled in each case. "Ran a bootstrap with 10,000 resamples" describes the same procedure in both weeks, but the unit of resampling was different: dev scores vs. held-out scores (unpaired) in Week 10, and per-task deltas across both systems on the same 44 tasks (paired) in Week 11. The variance mechanics are fundamentally different, and the p-values reflect that difference even though the surface procedures look identical.

---

## What I Now Understand That I Didn't Before

**Before:** "We got p=0.0001 in Week 10 so the improvement is statistically significant. Week 11 got p=0.1953 because n=44 is too small."

**After:** "The Week 10 p-value is misleading because it resamples from two different task pools as if they were paired. The Week 11 p-value is honest because it correctly computes per-task deltas and resamples those. The n=44 is indeed small, but even with more tasks, the Week 10 comparison would require a different statistical framing — a two-sample test with appropriate null distribution — not a paired bootstrap."

---

## Gap Closed

The mechanism is: paired bootstrap reduces variance by recovering the covariance between the two systems' scores (Var(A − B) = Var(A) + Var(B) − 2·Cov(A,B)). When both systems see the same tasks, the covariance is real and large. When they see different tasks, you set it to zero by construction — and your uncertainty estimate is inflated. A bootstrap that ignores this inflation reports artificially precise (and potentially misleading) p-values.

---

## Portfolio Impact

1. **Week 10 Act V:** The p=0.0001 claim should be reframed as a descriptive delta, not a significance test. The convergent ablation evidence remains meaningful. The statistical claim is removed.

2. **Week 11 bootstrap_stats.py:** Correct implementation, now documented with the pairing precondition. Future engineers using this function will not accidentally run it on unpaired data.

3. **Future evaluation designs:** Freeze the eval set before any training or tuning. Evaluate all system variants on the same set. Version-control the task set alongside the model checkpoint.

---

## Rubric Self-Assessment

| Criterion | Score | Justification |
|-----------|-------|---------------|
| Question Sharpness | 5/5 | Mechanism-centered, grounded in both artifacts, specific failure mode named |
| Explainer Depth | 4/5 | Covers mechanism, worked example, code, adjacent concepts, and production implications. Would score 5 if the code example included a larger n to show the variance difference numerically |
| Morning Call Evidence | 5/5 | Shows visible sharpening from "why different p-values" to "what is the bootstrap resampling in each case" |
| Portfolio Grounding | 5/5 | Before/after revision is specific, cites exact files and line-level content |
| Sources | 4/5 | Four canonical sources, each with a specific claim. Would score 5 with a fifth source specifically on LLM evaluation replication |

---

## Honest Limitations

The explainer does not fully address the question of what the *correct* statistical test for the Week 10 comparison would be. A Wilcoxon rank-sum test, a two-sample bootstrap, or simply not running a significance test and reporting descriptive statistics with CIs on each estimate separately — all would be more defensible. That's a follow-on question worth a separate session.

The code example uses toy data (n=10). A more compelling demonstration would show the variance difference with n=30 and n=20, using the actual pass@1 distributions from Week 10, and comparing the resulting CIs. That would close the explainer loop completely.

---

## Signed

Eyor G.  
Week 12 Day 04  
Tenacious Sales Bench — TenaciousBench v0.1
