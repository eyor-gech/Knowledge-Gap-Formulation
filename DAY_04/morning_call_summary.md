# Morning Call Summary — Week 12 Day 04

**Topic:** Evaluation and Statistics for Modern AI Systems  
**Session format:** Self-paired interrogation  
**Duration:** ~35 minutes

---

## Background: What I Built

**Week 10 — Tenacious B2B Conversion Engine**

A five-act evaluation of a multi-channel outreach agent. The core statistical claim was in Act V: a bootstrap test over 10,000 resamples returned p=0.0001 for a +33.67pp improvement. Dev pass@1 was 53.33% (n=30 tasks), held-out was 85% (n=20 tasks).

**Week 11 — TenaciousBench v0.1**

An ORPO-fine-tuned Qwen2.5-7B judge model evaluated on 44 held-out benchmark tasks. The ablation bootstrap (also 10,000 resamples) returned p=0.1953 for a +6.8pp lift over the GPT-4o-mini baseline. CI: [−6.8, +20.4] pp.

---

## Initial Question (9:12 AM)

> "Why did I get p=0.0001 in Week 10 and p=0.1953 in Week 11? Is the Week 11 result just underpowered?"

**Interrogation response:** That framing assumes the two p-values are measuring the same thing. Are they? What did the Week 10 bootstrap actually compare?

**Pause.** The Week 10 bootstrap compared dev performance (53.33%, 30 tasks) against held-out performance (85%, 20 tasks). Those are different task sets. In Week 11, the bootstrap compared two systems — ORPO and baseline — on the *same* 44 tasks.

That's not a power difference. That's a structural difference.

---

## Interrogation Round 1 (9:18 AM)

**Q: What mechanism are you actually asking about?**

Initial answer: "Something about sample size and how confident I can be in the improvement."

**Push:** No. You said two different task sets vs. one task set. What does the bootstrap resample in each case? What is the unit of resampling?

Working through it: In the unpaired case (Week 10), you resample from two independent pools — 30 dev tasks and 20 held-out tasks separately. The difference you compute is between two independently resampled means. In the paired case (Week 11), you resample *pairs* — for each resampled task, you compute the per-task delta (ORPO score minus baseline score), then average the deltas.

**The mechanism:** In the paired case, you're averaging within-task variance, which cancels out shared noise. In the unpaired case, you're summing across-sample variance, which includes task-selection variance on top of system variance.

---

## Interrogation Round 2 (9:26 AM)

**Q: Is this a statistics question disguised as an evaluation question?**

Yes — but it has direct evaluation consequences. The Week 10 p=0.0001 is implausibly low for n=20. At n=20, with a +33.67pp observed difference, you can get very small p-values *if* you treat the bootstrap as if both samples came from the same population. But they didn't. They came from a development-phase vs. post-training-phase evaluation, evaluated on different tasks. The variance in that comparison includes task-selection variance that the bootstrap isn't accounting for correctly.

**Q: What exact engineering decision changes if you understand this?**

How you set up ablation comparisons going forward: both system variants must see the exact same tasks, in the exact same order, with the same random seeds. Evaluating on separate task splits and bootstrapping the difference is structurally wrong if you're claiming a paired test.

---

## Interrogation Round 3 (9:31 AM)

**Q: Could this be answered by Wikipedia?**

Not fully. Wikipedia explains paired vs. unpaired tests in the parametric case (t-test). The bootstrap analogue — why pairing reduces variance in the resampled distribution of the mean difference — requires working through what the bootstrap actually resamples and why the covariance term disappears in the unpaired case.

**Q: Is this actually three questions pretending to be one?**

Potentially. Let me scope: the core question is about the variance mechanics of paired vs. unpaired bootstrap. The contamination test question (whether the two task sets are truly from different distributions) and the power analysis question (how many tasks you need) are adjacent but separable.

**Scoped final question:** Why does pairing matter in bootstrap resampling for LLM eval — specifically, what failure mode occurs when you compare two different task samples (acting like a paired test) rather than two systems on the same tasks?

---

## Final Question (9:35 AM)

> When comparing two LLM system configurations using bootstrap resampling, why does evaluating both systems on the *same* held-out task set (pairing) matter — and what specific statistical failure mode occurs when you treat a before/after comparison across two *different* task samples as if it were a paired test?

---

## What Changed

| Time | State | Trigger |
|------|-------|---------|
| 9:12 | Thought this was about power / sample size | Noticed both bootstraps used 10,000 resamples |
| 9:18 | Realized the two bootstraps compare different structures | Asked what the unit of resampling is |
| 9:26 | Named the mechanism: variance reduction via pairing | Traced through what "resample" means in each case |
| 9:31 | Scoped out adjacent questions | Checked it wasn't actually three questions |
| 9:35 | Final question locked | Grounded in both Week 10 and Week 11 artifacts |

---

## Rubric Check

- [x] Specific — names the exact structural difference (same vs. different task sets)
- [x] Mechanism-centered — asks about variance mechanics, not just outcomes
- [x] Grounded in portfolio artifact — Week 10 Act V and Week 11 `bootstrap_stats.py`
- [x] Generalizable — applies to any LLM ablation comparison
- [x] Resolvable in one explainer — the mechanism fits in 800 words
