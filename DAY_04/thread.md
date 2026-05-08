# Thread — Paired Bootstrap in LLM Evaluation

*4 posts. Each carries one load-bearing insight.*

---

## Post 1

We reported p=0.0001 in our Week 10 eval. Week 11 reported p=0.1953 for a +6.8pp lift on our trained judge.

We called Week 11 "underpowered." We should have asked a different question: did both evals actually run the same test?

They didn't. And the difference explains almost everything.

---

## Post 2

A bootstrap works by resampling your data and asking: how much does my statistic vary across samples?

If you compare System A on 30 dev tasks vs. System B on 20 different held-out tasks, you're resampling two independent pools. The variance of the difference includes task-selection uncertainty for both pools. No task appears in both systems. Nothing cancels.

If you compare System A and System B on the *same* 44 tasks, you compute per-task deltas and resample those. Tasks where both systems fail cancel out. Tasks where both succeed cancel out. You're left with variance over the *actual system difference* — not the system difference plus sampling noise from picking different tasks.

In math: Var(A − B) = Var(A) + Var(B) − 2·Cov(A,B)

When A and B see the same tasks, Cov(A,B) is real and large. When they see different tasks, it's zero by construction. You're not getting a tighter estimate in the unpaired case. You're getting a biased one.

---

## Post 3

Why did Week 10 get p=0.0001 then?

Because the two task pools (dev and held-out) each happened to have consistent internal pass rates. The bootstrap drew from each pool, saw low within-pool variance, and concluded there was very little uncertainty in the difference. That's true — there's little uncertainty about the difference in those two specific samples. There's *enormous* uncertainty about whether that difference reflects a real system improvement, because we never evaluated both system states on the same tasks.

The p=0.0001 is not wrong about the math. It's answering the wrong question.

The correct Week 10 claim: "Pass rates were 53.33% on dev and 85% on held-out. These task sets differ in timing, difficulty, and development context. We cannot attribute the +33.67pp difference entirely to system improvement without controlling for task-set variance."

That's less exciting. It's also defensible.

---

## Post 4

The fix is boring and structural: before you start training or tuning, freeze an eval set. Evaluate every candidate system — baseline, fine-tuned, prompt-only — on that exact set, in the same order, with the same random seeds for any stochastic components.

Then run the paired bootstrap: resample task indices, compute per-task deltas, build the CI from the resampled mean-delta distribution.

The Week 11 `bootstrap_stats.py` got this right. Both Delta A (ORPO) and the GPT-4o-mini baseline scored every one of the 44 held-out tasks. The resulting CI [−6.8, +20.4] pp crosses zero. That's not a failure of the experiment — that's the experiment working correctly. At n=44 and +6.8pp true lift (if real), you don't have enough power to reject the null. Saying so is more honest than reporting p=0.0001 on a structurally mismatched comparison.

---

## Post 5

Three things to double-check before reporting any improvement:

1. Did both systems see the exact same tasks? If not, you have an observational comparison, not a paired significance test.

2. Does your CI cross zero? If it does, report it. A CI that crosses zero is not a malfunction. It's the correct interval.

3. What is your bootstrap actually resampling — task pairs, or independent pools? The code difference is one line. The interpretive difference is everything.

Full write-up of the mechanism, with code and worked example: [link to explainer.md]
