# Knowledge Gap Question — Day 1

## Topic: Inference-Time Mechanics / KV Cache

---

## The Question

**When my multi-turn agent system processes a conversation, which specific tensors are reused across decoding steps, what computation is avoided because of that reuse, and under what conditions does the memory cost of maintaining those tensors outweigh the latency savings?**

---

## Grounding in My System

My inference API serves responses to a multi-turn, tool-using agent system. In my model card and CFO memo, I claim:

> "Multi-turn conversations are efficient due to caching."

This claim is currently mechanistically undefended. I cannot answer the following directly:

1. Which tensors constitute "the cache", is it activations, embeddings, attention weights, or something else?
2. What computation is literally skipped when those tensors exist?
3. At what sequence length does the memory overhead of the KV cache become a material infrastructure cost?
4. Does my current inference setup actually leverage prefix caching across turns, or does it recompute from scratch on each API call?

Without answers to these questions, my efficiency claim is a hypothesis, not a defensible engineering statement.
