# README Addition — Knowledge Gap Formulation

> This file contains the section to be appended to (or used as the basis for) the repository README.

---

## Repository Overview

This repository documents a knowledge gap formulation sprint completed as part of the Tenx / 10 Academy Forward-Deployed Engineering cohort. The program's methodology asks engineers to identify gaps in their own deployed work — not gaps in general knowledge, but specific claims or decisions in real artifacts that cannot survive mechanistic interrogation — and produce auditable closures. Each session follows a fixed structure: a morning call that sharpens a vague question into a mechanism-centered one, a technical explainer that resolves it, a grounding commit that applies the resolution to the original artifact, and a sign-off that distinguishes what was closed from what remains open.

The two sessions documented here produced grounding commits in transformer inference infrastructure and LLM evaluation statistics. The work is technical, specific, and honestly scoped.

---

## Final Submission Summary

### Completed Sessions

| Session | Topic | Gap Closed |
|---------|-------|------------|
| Day 01 | KV Cache Mechanics | Replaced vague caching claim in production model card and CFO memo with mechanistic, auditable description of what is stored, what computation is avoided, and what the memory cost formula is |
| Day 04 | Paired Bootstrap for LLM Evaluation | Identified structural mismatch in Week 10 bootstrap (unpaired, different task sets), corrected Week 10 Act V claim, documented pairing precondition in `bootstrap_stats.py` |

### Completion Scope

**Days completed:** 2 of 4 (Day 01, Day 04)
**Days not completed:** Day 02, Day 03

Days 02 and 03 are not represented in this repository. No artifacts were fabricated to fill those gaps. The synthesis, canonical list, and portfolio materials reflect only the two sessions that were completed.

---

## Publication Slots

### Blog Posts (Substack)

*Five articles mapped to completed sessions and synthesis content. Links to be added upon publication.*

| # | Title | Source | Link |
|---|-------|--------|------|
| 1 | "What Your Model Card Doesn't Say About Caching" | Day 01 | [coming soon] |
| 2 | "The Memory Cost Your LLM Deployment Forgot to Budget For" | Day 01 | [coming soon] |
| 3 | "Your p=0.0001 Is Lying to You" | Day 04 | [coming soon] |
| 4 | "How I Corrupted My Own Ablation Study (And How to Not)" | Day 04 | [coming soon] |
| 5 | "Two Gaps, One Pattern: On Making Technical Claims Auditable" | Synthesis | [coming soon] |

### Twitter/X Threads

*Five technical threads. Links to be added upon publication.*

| # | Thread Hook | Source | Link |
|---|-------------|--------|------|
| 1 | "My model card says 'efficient due to caching.' Here's what that actually means..." | Day 01 | [coming soon] |
| 2 | "At 128K context, your KV cache weighs more than your model weights. A thread on the numbers." | Day 01 | [coming soon] |
| 3 | "We reported p=0.0001 in Week 10 and p=0.1953 in Week 11. We called Week 11 underpowered. We were wrong." | Day 04 | [coming soon] |
| 4 | "3 things to check before reporting any LLM improvement as statistically significant." | Day 04 | [coming soon] |
| 5 | "Two sessions, two gaps, one structural pattern. What knowledge gap formulation actually taught me." | Synthesis | [coming soon] |

---

## Ethical Structure for Remaining Publication Slots

With two completed sessions out of four, five publication slots need to be filled without fabricating work that did not happen. The following approach is both honest and professionally credible:

**Articles 1 and 2** cover Day 01 directly — the KV cache mechanism and the memory cost implications. These are grounded in completed work and can be written with full technical specificity.

**Articles 3 and 4** cover Day 04 directly — the bootstrap pairing problem and the production eval design implications. Same grounding.

**Article 5** is a synthesis/meta-learning post that explicitly acknowledges the partial completion, reflects on what the program's methodology revealed about the two completed sessions, and proposes what would have been investigated in Days 02 and 03 based on the patterns established. Framing: "what two sessions of this methodology taught me, and what I'd investigate next." This is honest, self-aware, and demonstrates the kind of reflective technical communication that senior engineering roles require. It does not claim completed work that did not happen.

The five Twitter threads follow the same mapping. The synthesis thread stands on its own as a "two gaps, one pattern" observation — which is a genuinely interesting frame and does not require fabricating the missing sessions.

---

## Repository Structure

```
Knowledge-Gap-Formulation/
├── DAY_01/
│   ├── grounding_commit.md    # Before/after for model_card.md and cfo_memo.md
│   ├── explainer.md           # KV cache mechanics, O(n²)→O(n), memory cost
│   ├── sources.md             # Vaswani 2017, HuggingFace docs, Kwon 2023
│   ├── thread.md              # 6-tweet thread on KV cache
│   ├── evening_call_summary.md
│   ├── signoff.md
│   └── question.md
├── DAY_04/
│   ├── grounding_commit.md    # Before/after for Week 10 Act V and bootstrap_stats.py
│   ├── explainer.md           # Paired vs. unpaired bootstrap, variance mechanics
│   ├── sources.md             # Efron 1993, Dror 2018, Bouthillier 2021, Biderman 2024
│   ├── thread.md              # 5-post thread on bootstrap pairing
│   ├── morning_call_summary.md
│   ├── signoff.md
│   └── Final_Question.md
├── synthesis.md               # Cross-session synthesis (~1500 words)
├── canonical_list.md          # Annotated resource list
├── portfolio_update.md        # Hiring-manager-facing summary
└── README_addition.md         # This file
```
