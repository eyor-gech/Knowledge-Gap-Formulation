# Morning Call Summary — Day 1

## Topic: KV Cache Mechanics in Multi-Turn Inference

---

**Initial framing (vague):** "I said multi-turn conversations are efficient because of caching, but I don't fully know what that means mechanically."

**After interrogation:** The word "caching" in my model card refers specifically to the key-value (KV) cache — a structure that stores the K and V projection tensors for each previously processed token, per attention head, per layer — not a generic HTTP or application-level cache. The critical interrogation point was *mechanism vs. consequence*: the consequence (faster token generation) is true, but the mechanism (avoiding recomputation of K and V for all previous tokens on each decode step) had never been stated, and therefore could not be defended when challenged. The gap is not a matter of vague language — it is a structural absence of the causal chain from tensor storage to latency reduction. Today's work must establish that chain explicitly, including the prefill vs. decode distinction, the asymmetry between O(n²) full-sequence attention and O(n) cached-decode attention, and the memory cost that scales with sequence length and model depth.
