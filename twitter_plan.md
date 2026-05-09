# Twitter/X Thread Plan

**Author:** Eyor Getachew
**Source sessions:** Day 01, Day 04, Synthesis

---

## Thread 1 — Day 01

**Hook:**
> My model card says multi-turn conversations are "efficient due to caching."
> I couldn't say what was cached, what computation was skipped, or when the claim breaks down.
> Here's the mechanism. 🧵

**Structure (7 tweets):**

1. **The claim vs. the question.** "Efficient due to caching" passes review. But which tensors? What computation? Under what conditions? Most documentation doesn't say — including mine.

2. **What attention actually computes.** Every layer computes `Attention(Q,K,V) = softmax(Q @ K.T / √d_k) @ V`. The `Q @ K.T` term is O(n²). At step 1001 of a conversation, without optimization, you recompute K and V for all 1000 prior tokens. Every single step.

3. **What the cache stores.** After prefill, the K and V projections for every processed token are stored as `past_key_values`: a tuple of `(key, value)` tensors per layer, shape `(batch, heads, seq_len, head_dim)`. Not embeddings. Not attention weights. Just `X @ W_K` and `X @ W_V`.

4. **What is avoided.** On each decode step, only the new token's K and V are computed. All prior projections are retrieved. O(n²) → O(n) per decode step. For a 32-layer model at 2048 tokens: 65,536 matrix multiplications eliminated per token generated.

5. **The memory cost.** KV cache ≈ 0.5 MB × seq_len for a 7B model at float16. At 4K tokens: ~2 GB per concurrent request. At 128K tokens: ~64 GB — comparable to the model weights. The latency win is real. The memory cost is also real.

6. **The critical conditional.** Within-generation caching is automatic. Cross-request prefix caching — the mechanism that makes multi-turn *across API calls* efficient — requires explicit infrastructure: vLLM `enable_prefix_caching=True` or Anthropic `cache_control`. If you haven't confirmed this setting, your efficiency claim is unverified.

7. **The fix.** Replaced "efficient due to caching" with a formula, a conditional, and a `[confirm deployment setting]` placeholder. The new claim can be falsified. That's the standard.

**CTA:**
> Full explainer with code and memory formula: [link to DAY_01/explainer.md]
> If you've written "caching" in a model card and can't say what the cache stores — this one's for you.

---

## Thread 2 — Day 01

**Hook:**
> At 128K context, your KV cache uses more GPU memory than your model weights.
> Most deployment guides don't mention this.
> A thread on the numbers, and what to do about them. 🧵

**Structure (6 tweets):**

1. **The formula.** KV cache memory = `2 × num_layers × num_heads × head_dim × seq_len × dtype_bytes`. For a 7B model (32 layers, 32 heads, head_dim=128, float16): ~0.5 MB per token per concurrent request.

2. **At short context.** 4K tokens: ~2 GB per request. 10 concurrent users: 20 GB just for KV cache, before you count the model weights (~14 GB at float16). That's 34 GB. Not a problem on an A100. Getting tight on a 40 GB card.

3. **At long context.** 128K tokens: ~64 GB per request. One concurrent user saturates an A100. This is not a theoretical edge case — it's why long-context inference is expensive and why PagedAttention exists.

4. **What PagedAttention solves.** vLLM's PagedAttention treats KV cache memory like OS virtual memory: allocates in fixed-size pages, shares pages across requests with identical prefixes, evicts on demand. The Kwon et al. (SOSP 2023) paper measures 2–4× throughput improvement under realistic serving loads.

5. **The deployment question to answer first.** Before making any per-turn efficiency claim: (1) what is the expected average context length? (2) How many concurrent users? (3) What is the VRAM budget? (4) Is prefix caching enabled? The answers determine whether the efficiency claim holds.

6. **The honest model card version.** "Memory cost is approximately 0.5 MB per token per concurrent user at this model size, which becomes material at context lengths above 8K tokens or under high concurrency." That's auditable. "Efficient due to caching" is not.

**CTA:**
> The full derivation and production implications: [link to DAY_01/explainer.md]
> The memory cost formula is the thing most deployment guides omit. Don't be the team that discovers it in production.

---

## Thread 3 — Day 04

**Hook:**
> We got p=0.0001 in Week 10. We called it statistically significant.
> Week 11 returned p=0.1953 and we called it "underpowered."
> We had the logic exactly backwards.
> 🧵

**Structure (5 tweets):**

1. **The setup.** Week 10: dev pass@1 was 53.33% (n=30 tasks). Held-out was 85% (n=20 different tasks). Bootstrap: 10,000 resamples, p=0.0001. Week 11: ORPO-trained judge vs. GPT-4o-mini baseline, same 44 held-out tasks. Bootstrap: 10,000 resamples, p=0.1953, CI [−6.8, +20.4] pp.

2. **The structural difference.** Week 10 resampled from two independent pools. Week 11 resampled per-task deltas. That is not a power difference. It is a structural difference in what the bootstrap is estimating. Var(A − B) = Var(A) + Var(B) − 2·Cov(A,B). When A and B see different tasks, the covariance is zero by construction. When they see the same tasks, it cancels within-task difficulty noise.

3. **Why p=0.0001 is suspicious, not impressive.** Both task pools happened to have low internal variance. The bootstrap saw consistent dev scores and consistent held-out scores, and concluded there was very little uncertainty in the difference. There was. The bootstrap just wasn't measuring it — because no task appeared in both samples.

4. **The correct claim.** "Pass rates were 53.33% on dev and 85% on held-out. These task sets differ in timing, composition, and development exposure. We cannot attribute the +33.67pp difference entirely to system improvement without controlling for task-set variance. Convergent ablation evidence across three variants supports the improvement direction."

5. **The fix.** Freeze your eval set before training begins. Evaluate every system variant on that exact set. Run the bootstrap on per-task deltas. If your CI crosses zero, report it — that is the correct interval, not a malfunction.

**CTA:**
> Mechanism, worked example, and code in the full explainer: [link to DAY_04/explainer.md]
> The bootstrap_stats.py docstring that prevents this in the future: [link to DAY_04/grounding_commit.md]

---

## Thread 4 — Day 04

**Hook:**
> 3 things to check before reporting any LLM improvement as statistically significant.
> Got burned on #1. Writing this so you don't have to. 🧵

**Structure (6 tweets):**

1. **Check 1: Did both systems see the exact same tasks?** If not, you have an observational comparison, not a paired significance test. A bootstrap will run. It will return a p-value. That p-value will be answering the wrong question.

2. **What "same tasks" means in practice.** Same task IDs. Same prompt text. Same preprocessing. Same random seeds for any stochastic components. Same order if order matters. Changing any of these introduces a confound the bootstrap cannot account for. (See: Biderman et al. 2024, Lessons from the Trenches.)

3. **Check 2: Does your CI cross zero?** If it does, report it. A CI that crosses zero is not a failed result — it is the correct interval for that n and that effect size. Calling it "underpowered" and moving on loses the information. Report the CI, state the minimum n required to detect the observed effect, and use it to design the next evaluation.

4. **Check 3: What is your bootstrap actually resampling?** Open the code. Find the line that calls `rng.choice(...)`. Is it resampling task indices (and computing per-task deltas)? Or is it resampling from two independent pools separately? The code difference is one loop. The interpretive difference is whether you have a paired or unpaired comparison.

5. **The structural requirement, once more.** Paired bootstrap: resample task *pairs* (index i → scoreA_i, scoreB_i). Mean over the delta distribution. This is the correct estimator. Unpaired bootstrap: resample from pool A, resample from pool B, take the difference of means. This is a two-sample test, and its variance correctly includes task-selection uncertainty. It just needs to be labeled as such.

6. **The production implication.** In a production eval pipeline: freeze your task set before training begins, version-control it alongside the model checkpoint, evaluate all candidates on that exact set. This is the boring structural work that makes statistical claims defensible. It costs one design decision up front.

**CTA:**
> Full explainer with variance math and worked examples: [link to DAY_04/explainer.md]
> If you're about to ship an improvement claim, run these three checks first.

---

## Thread 5 — Synthesis

**Hook:**
> Two sessions. Two completely different domains.
> Both gaps had the same structure.
> Here's what I learned from a program that makes you mechanistically interrogate your own deployed work. 🧵

**Structure (8 tweets):**

1. **The methodology.** Not "learn a topic and write about it." Find a claim in your own deployed work that sounds right. Force a mechanistic account. Trace it to its grounding. Produce an artifact that makes the closure auditable. Do this for four sessions. I did it for two.

2. **Gap 1: KV cache.** My model card said "efficient due to caching." Correct in the right conditions. Unverifiable as written. The gap: I couldn't name what was cached, what computation was avoided, or when the claim breaks down. The fix: a formula, a conditional, and a placeholder that forces future verification.

3. **Gap 2: Bootstrap statistics.** My Week 10 eval reported p=0.0001 and called it confirmation of improvement. The bootstrap compared dev performance and held-out performance on different task sets — structurally unpaired, covariance zero by construction. The Week 11 result was not underpowered. The Week 10 result was incorrectly framed.

4. **The pattern.** Both gaps had the same structure: a claim that is directionally correct but omits a critical conditional. "Efficient due to caching" omits whether prefix caching is configured. "p=0.0001 confirms improvement" omits whether the bootstrap comparison was paired. Both pass review because they are not wrong. Both fail under mechanistic interrogation.

5. **Why this class of gap is common.** Because review processes optimize for "is this plausible?" not "is this auditable?" A mechanistic account requires knowing what a claim *depends on*, not just whether it *sounds right*. Most documentation and most statistical reporting does not include that dependency.

6. **The patterns worth keeping.** (1) The `[confirm deployment setting]` placeholder — embeds the check at the point of reuse. (2) The morning call sharpening protocol — forces a vague question into a mechanism-centered one before you write anything. (3) The before/after grounding commit — makes the closure auditable by showing exactly what changed and why.

7. **Honest scope.** I completed two of four sessions. I don't know what Days 02 and 03 would have revealed. The pattern I'm describing is visible from two data points. It might generalize. It might not. Both points are real.

8. **What I'd look for in the next two.** Context management and retrieval quality (inferred from the multi-turn agent context). Structured output reliability and tool-call failure patterns (inferred from the evaluation infrastructure). Two specific claims I've made in prior work that I haven't interrogated yet.

**CTA:**
> The session artifacts are in this repo: [link to GitHub]
> The synthesis: [link to synthesis.md]
> If you're doing something similar — the morning call format is the most transferable piece.
