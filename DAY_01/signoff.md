# Day 1 Sign-Off

## Gap Status: CLOSED

---

## What I Can Now State That I Could Not Before

**Before:** "Multi-turn conversations are efficient due to caching."
This was a consequence claim with no mechanistic support. I could not say what was cached, why caching mattered computationally, or under what conditions the claim holds.

**After:**

1. **What is cached:** The key and value projection tensors — `X @ W_K` and `X @ W_V` — for every previously processed token, stored per layer, per attention head, with shape `(batch_size, num_heads, seq_len, head_dim)` in HuggingFace's `past_key_values` structure.

2. **What computation is avoided:** On each decode step, K and V projections for all n previous tokens are retrieved from cache rather than recomputed. This reduces per-step projection work from O(n) to O(1) (one new token only), and makes the overall attention operation O(n) per generated token instead of O(n²).

3. **What it costs:** KV cache memory grows linearly with sequence length. For a 7B model at float16, the cost is approximately 0.5 MB per token per concurrent request. At 128K context, this exceeds the model weights in memory.

4. **The critical conditional for my system:** Within-generation KV caching is automatic. Cross-request prefix caching — the only form that makes multi-turn efficiency claims defensible — requires explicit infrastructure (vLLM prefix caching or equivalent). Whether my current deployment has this enabled is a verifiable question, not an assumption.

---

## Residual Questions (Not Gaps — Logged for Future Work)

- Does my inference API deployment have prefix caching enabled, and can I measure the TTFT difference with and without it?
- At what concurrent user count does KV cache memory become the binding constraint on my serving capacity?
- What is the cache hit rate for my system's most common conversation patterns (system prompt reuse, tool definitions)?

These are scoped for a future day's investigation, not this one.
