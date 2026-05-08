# Sources — Week 12 Day 04

## Primary Sources

### 1. Efron & Tibshirani (1993) — *An Introduction to the Bootstrap*

**Why canonical:** The foundational text. The paired bootstrap is not a technique layered onto the standard bootstrap — it is derived from first principles in Chapter 9 as the correct estimator when your data is naturally paired. The math on variance reduction through pairing is worked out explicitly. Every applied bootstrap tutorial eventually traces back to this.

**What it clarifies:** Why Var(A − B) = Var(A) + Var(B) − 2·Cov(A,B) is load-bearing. In the paired case, Cov(A,B) is positive and real. In the unpaired case, you've set it to zero by construction. The paired bootstrap recovers the covariance by resampling (A_i, B_i) pairs together.

**Misconception it corrects:** That more bootstrap resamples = more reliable conclusions. Resamples control the precision of the bootstrap estimate of the resampled distribution. They do not fix a structurally wrong comparison. 10,000 resamples of a mismatched unpaired comparison produce a very precisely wrong p-value.

---

### 2. Dror et al. (2018) — "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing" (ACL 2018)

**Why canonical:** The most-cited practical treatment of significance testing in NLP. Section 4 covers paired bootstrap specifically in the context of NLP model comparison. The paper distinguishes between bootstrap CIs and bootstrap significance tests, and explicitly warns against the before/after comparison pattern.

**What it clarifies:** The paired bootstrap test for NLP should compare two systems on the same test set, with per-example outcomes as the unit of resampling. If you don't have matched examples across systems, you are running a two-sample test and should be using the appropriate two-sample null distribution instead.

**Misconception it corrects:** That "paired" just means "same domain" or "same benchmark version." Pairing is a strict structural requirement: each example must produce an outcome under both systems, and those outcomes must be resampled as a unit.

---

### 3. Bouthillier et al. (2021) — "Accounting for Variance in Machine Learning Benchmarks" (MLSys 2021)

**Why canonical:** Empirically demonstrates — on a large set of standard benchmarks — that evaluation variance (variance due to which examples are in the test set) dominates performance variance (variance due to the model) for most standard ML comparisons. This is not a theoretical warning. It is measured on real benchmark splits.

**What it clarifies:** At n=44, a +6.8pp lift is within the range of task-selection noise for many benchmark types. The honest interpretation is that you cannot distinguish a real improvement from favorable task sampling at this n. The Week 11 result (p=0.1953) is consistent with this finding, not anomalous.

**Misconception it corrects:** That a 44-task held-out set is sufficient to make strong statistical claims about model superiority. It's sufficient to measure direction and rough magnitude, not to confirm statistical significance for small effect sizes.

---

### 4. Biderman et al. (2024) — "Lessons from the Trenches on Reproducible Evaluation of Language Models" (arXiv 2405.13163)

**Why canonical:** Written by the practitioners who maintain the Eleuther AI Language Model Evaluation Harness (lm-eval-harness). This is operational, not theoretical. They document the same class of errors — task set drift, evaluation infrastructure changes between runs, random seed variation — that corrupt before/after comparisons in practice.

**What it clarifies:** How version control of your eval set is as important as version control of your model. If you change task phrasing, ordering, or preprocessing between your baseline run and your fine-tuned run, you've introduced confounds that no statistical test can remove.

**Misconception it corrects:** That infrastructure stability is a secondary concern to model quality. In production eval pipelines, infrastructure instability is frequently the largest source of reported "improvement" in model comparisons.

---

## Portfolio Artifacts Cited

- `week10_final/results/act5_evidence_graph.json` — the source of p=0.0001, Δ=+33.67pp
- `ablations/bootstrap_stats.py` — the correct paired bootstrap implementation
- `ablations/ablation_results.json` — Delta A/B/C with CIs
- `week10_final/results/act4_heldout_summary.json` — held-out 85% pass@1

---

## Not Cited (Excluded)

- **Wilcoxon signed-rank test papers** — An alternative to paired bootstrap for ordinal data, but not the implementation used in the portfolio work.
- **Bayesian A/B testing literature** — A valid alternative framework, but out of scope for this explainer. The question is specifically about the paired bootstrap, not about choosing between frequentist and Bayesian approaches.
- **Power analysis formulas** — The n=44 power question is adjacent but separate. The explainer scopes to why pairing matters, not how to compute minimum sample sizes.
