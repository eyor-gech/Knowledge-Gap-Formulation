# Why Pairing Matters in Bootstrap Resampling for LLM Evaluation

*Week 12 Explainer — Evaluation and Statistics for Modern AI Systems*

---

## Why This Question Mattered in My Work

In Week 10, the final evidence synthesis reported p=0.0001 from a bootstrap comparison of two pass@1 numbers: 53.33% from a 30-task dev evaluation, and 85% from a 20-task held-out evaluation. Those are not the same tasks. They were evaluated at different stages of the build, with different task distributions. The bootstrap was run, 10,000 resamples were drawn, and an impressively small p-value came back.

In Week 11, the ablation bootstrap was set up correctly — both the ORPO-trained model and the GPT-4o-mini baseline were scored on the same 44 held-out tasks, and the per-task deltas were bootstrapped. That returned p=0.1953 for a +6.8pp lift. The Week 11 result was labeled "underpowered." The Week 10 result was not questioned.

The difference between those two setups is not primarily a function of n=20 vs. n=44. It's a function of what the bootstrap was actually resampling in each case. Understanding why reveals one of the most common evaluation mistakes in applied ML: treating an unpaired before/after comparison as if it were a paired significance test.

---

## The Load-Bearing Mechanism

Start with what a bootstrap confidence interval actually does. You have some observed statistic — say, the difference in pass@1 between two systems. To estimate sampling uncertainty, you resample your data with replacement many times, recompute the statistic on each resample, and look at the distribution. The width of that distribution tells you how sensitive your estimate is to which specific examples were included.

The key variable is: *what gets resampled together.*

**Paired bootstrap.** System A and System B are both evaluated on tasks 1 through n. For each task, you have a pair (scoreA, scoreB). The statistic you care about is the mean of (scoreA_i − scoreB_i) across tasks. In the bootstrap, you resample task *indices* — which means each resample pulls a set of paired (scoreA, scoreB) values. The per-pair delta captures the same tasks for both systems. The variance of the resampled mean-delta reflects only how much the system difference varies across tasks.

**Unpaired bootstrap.** You have two pools: n₁ scores for System A (from one task set) and n₂ scores for System B (from a different task set). You resample each pool independently. The statistic is the difference in resampled means. The variance of that difference now includes:
1. How much System A's performance varies across its task sample
2. How much System B's performance varies across its task sample
3. And — the part that matters — no cancellation between them

In the paired case, if a task is hard, both systems struggle on it. That shared difficulty is captured in the pair and cancels out when you compute the delta. Variance in Var(A − B) = Var(A) + Var(B) − 2·Cov(A,B). The covariance term is large when both systems see the same tasks. When they see different tasks, the covariance is zero by construction, and your variance estimate is inflated.

Now consider what happens to the p-value when variance is underestimated. If you're running a bootstrap significance test (the fraction of resampled null differences that exceed the observed difference), a lower variance means tighter resampled distributions, which means fewer null resamples exceed the observed effect, which means a smaller p-value. The p=0.0001 in Week 10 is consistent with a variance underestimation problem: the unpaired comparison gave the bootstrap artificially tight distributions because the two task sets happened to have consistent pass rates within each pool.

---

## Worked Example

Suppose System A (dev) is evaluated on 5 tasks: [1, 1, 0, 1, 1] → mean = 0.80  
System B (held-out) is evaluated on 5 different tasks: [1, 1, 1, 1, 0] → mean = 0.80

The observed difference is 0. Run unpaired bootstrap 5 times:

| Resample | Pool A mean | Pool B mean | Diff |
|----------|-------------|-------------|------|
| 1 | 0.80 | 0.60 | +0.20 |
| 2 | 0.60 | 0.80 | −0.20 |
| 3 | 1.00 | 0.80 | +0.20 |
| 4 | 0.80 | 1.00 | −0.20 |
| 5 | 0.60 | 0.60 | 0.00 |

The variance of the difference across resamples is the sum of each pool's variance.

Now suppose both systems are evaluated on the *same* 5 tasks: A = [1,1,0,1,1], B = [1,1,1,1,0].  
Per-task deltas: [0, 0, −1, 0, +1] → mean delta = 0.

| Resample | Drawn deltas | Mean |
|----------|-------------|------|
| 1 | [0, 0, −1, 0, +1] | 0.00 |
| 2 | [0, +1, +1, 0, 0] | +0.40 |
| 3 | [−1, 0, 0, −1, 0] | −0.40 |
| 4 | [0, 0, 0, 0, 0] | 0.00 |
| 5 | [+1, −1, 0, +1, 0] | +0.20 |

Same two systems, same pass rates, same observed delta — but the paired bootstrap has lower variance because within-task noise cancels. Both approaches are technically "running a bootstrap," but they estimate different things.

---

## Code Demonstration

```python
import numpy as np

rng = np.random.default_rng(42)
n_boot = 10_000

# Unpaired: two independent score arrays (different tasks)
scores_a = np.array([1,1,0,1,1,1,0,1,1,0])  # dev set, system A
scores_b = np.array([1,1,1,1,0,1,1,1,0,1])  # held-out, system B (different tasks)

obs_diff = scores_b.mean() - scores_a.mean()

# Unpaired bootstrap
boot_diffs_unpaired = []
for _ in range(n_boot):
    ra = rng.choice(scores_a, len(scores_a), replace=True).mean()
    rb = rng.choice(scores_b, len(scores_b), replace=True).mean()
    boot_diffs_unpaired.append(rb - ra)

# Paired: same tasks, two score arrays
deltas = scores_b - scores_a  # per-task difference

boot_diffs_paired = []
for _ in range(n_boot):
    rd = rng.choice(deltas, len(deltas), replace=True).mean()
    boot_diffs_paired.append(rd)

print(f"Observed diff:          {obs_diff:.3f}")
print(f"Unpaired CI:  [{np.percentile(boot_diffs_unpaired,2.5):.3f}, {np.percentile(boot_diffs_unpaired,97.5):.3f}]")
print(f"Paired CI:    [{np.percentile(boot_diffs_paired,2.5):.3f}, {np.percentile(boot_diffs_paired,97.5):.3f}]")
```

In practice, the paired CI will be narrower (smaller uncertainty, tighter bounds) because the covariance term absorbs within-task difficulty variance. The unpaired CI will be wider — and any p-value derived from it will be more conservative, which is actually *correct*. The unpaired bootstrap doesn't underestimate uncertainty — it correctly includes the task-selection uncertainty that you failed to control for by not holding the task set constant.

The Week 10 case runs the opposite direction: the unpaired bootstrap appeared to give an extremely small p-value because the within-pool variance happened to be low (both dev and held-out were consistent within themselves). That's not evidence of a reliable system improvement — it's evidence that both task samples had low internal variance.

---

## Adjacent Concepts

**Permutation test.** An alternative to bootstrap for significance testing. Permute which scores are assigned to which system, rather than resampling with replacement. Shares the same pairing requirement — you need paired examples to construct the null distribution correctly.

**pass@k.** A related evaluation metric where you generate k samples per prompt and count a task correct if any of them pass. Relevant here because pass@k with k>1 reduces evaluation variance by making the per-task verdict more stable. Think of it as a within-task bootstrap. It's complementary to the bootstrap CI question but operates at the task level rather than the aggregate level.

**Task-level vs. aggregate evaluation.** The pairing question forces you to think about the right level of analysis. If you care about system comparison, pair at the task level. If you care about absolute capability, aggregate across a fixed task set.

---

## Common Failure Modes

**Evaluating development vs. deployment on different task sets and bootstrapping the difference.** This is exactly the Week 10 failure mode. Before/after is not the same as A/B.

**Using the bootstrap to "confirm" an observed improvement when the improvement was discovered on the same data.** If you tuned your system on dev and then ran the bootstrap on dev, you have a selection bias problem the bootstrap can't fix.

**Treating a large bootstrap resample count (10,000) as validation of the result.** The number of resamples controls the precision of the bootstrap estimate — not the validity of what the bootstrap is estimating. If the structural setup is wrong, 100,000 resamples make it more precisely wrong.

**Using the 95% CI to claim significance without checking if it crosses zero.** The Week 11 CI [−6.8, +20.4] crosses zero. This is the honest signal. Reporting "90.9% vs 84.1%" without that CI is omitting the uncertainty.

---

## Production Implications

Every time you compare two evaluation results and claim improvement, ask: did both systems see exactly the same tasks? If not, you don't have a paired comparison — and any significance test you run on it is making an additional assumption that the two task samples are exchangeable. That assumption is rarely justified in practice, because task sets are not random samples from a fixed distribution. They reflect decisions about difficulty, domain coverage, adversarial intensity, and recency.

In production eval pipelines, this means: freeze your eval set before training begins, evaluate all candidate systems on that exact set, and version-control the task set alongside the model checkpoint. Anything else introduces confounds the bootstrap cannot account for.

---

## Sources and Further Reading

1. **Efron & Tibshirani (1993), *An Introduction to the Bootstrap*.** Chapter 9 covers the variance of bootstrap estimates and the paired case. The paired-vs-unpaired distinction is treated as fundamental, not a footnote.

2. **Dror et al. (2018), "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing," ACL 2018.** The canonical NLP-specific treatment. Section 4 covers paired bootstrap explicitly and warns against the before/after pattern I describe here.

3. **Bouthillier et al. (2021), "Accounting for Variance in Machine Learning Benchmarks," MLSys 2021.** Empirical demonstration of how small-n benchmarks mislead: evaluation variance dominates performance variance in most standard ML comparisons.

4. **Biderman et al. (2024), "Lessons from the Trenches on Reproducible Evaluation of Language Models," arXiv 2405.13163.** The most recent and practically grounded treatment of LLM evaluation failure modes, including task-set stability and contamination.
