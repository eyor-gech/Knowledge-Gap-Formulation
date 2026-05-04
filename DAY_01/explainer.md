# How the KV Cache Actually Works: A Mechanism-Level Explainer

**Published context:** This explainer was written to close a specific knowledge gap in the documentation of a production multi-turn agent system, where the claim "multi-turn conversations are efficient due to caching" lacked mechanistic support.

---

## The Question Being Answered

When a transformer generates the next token in a multi-turn conversation, which tensors are reused from previous steps, what computation is avoided because of that reuse, and how does the cost of storing those tensors scale with sequence length?

---

## Background: What Attention Actually Computes

Every transformer layer performs multi-head self-attention. For an input matrix X (shape: `[seq_len, d_model]`), three projection matrices are learned: W_Q, W_K, W_V. The layer computes:

```
Q = X @ W_Q    # shape: [seq_len, d_k]
K = X @ W_K    # shape: [seq_len, d_k]
V = X @ W_V    # shape: [seq_len, d_v]

Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

The critical observation: `Q @ K.T` produces a matrix of shape `[seq_len, seq_len]`. Computing it requires O(n²) operations where n is the sequence length. Every token attends to every previous token.

---

## Prefill vs. Decode: Two Distinct Phases

Autoregressive generation has two operationally different phases.

**Prefill:** The entire prompt (or conversation history) is processed in one forward pass. All token positions exist simultaneously, so the full O(n²) attention matrix is computed. This is heavily parallelizable on GPU — all n tokens can be projected to Q, K, V in a single batched matrix multiply.

**Decode:** Tokens are generated one at a time. At step t, only one new token exists, but it must attend to all n previous tokens. Without optimization, this requires recomputing K and V for all n previous tokens on every single decode step — even though those tokens have not changed.

This is the exact waste that the KV cache eliminates.

---

## What the KV Cache Stores

After the prefill pass, every layer has already computed K and V for all prompt tokens. Instead of discarding these tensors, the runtime stores them:

```python
# HuggingFace Transformers: past_key_values structure
# Tuple of length num_layers
# Each element: (key_states, value_states)
# Shape of each: (batch_size, num_heads, seq_len, head_dim)

past_key_values = model(input_ids=prompt_ids, use_cache=True).past_key_values
```

On the next decode step, only the new token's Q, K, V are computed. Then:

```python
# Decode step: only new_token_id is in input_ids
new_output = model(
    input_ids=new_token_ids,       # shape: [batch_size, 1]
    past_key_values=past_key_values,
    use_cache=True
)

# Internally, per layer:
new_key   = new_token @ W_K        # shape: [batch_size, num_heads, 1, head_dim]
new_value = new_token @ W_V        # shape: [batch_size, num_heads, 1, head_dim]

# Cache is extended, not recomputed:
key_states   = torch.cat([cached_key,   new_key],   dim=2)
value_states = torch.cat([cached_value, new_value], dim=2)

# Attention is now computed with full history, but K and V came from cache:
attn_output = softmax(new_query @ key_states.T / sqrt(d_k)) @ value_states
```

The result: O(n²) → O(n) per decode step. The new token's query attends to n cached keys in O(n), rather than recomputing n key vectors from scratch.

---

## Concrete Demonstration

The following pseudo-experiment illustrates the compute asymmetry. It measures the FLOPs for a single attention head, naive vs. cached:

```python
import torch
import time

d_k = 64
num_heads = 32
batch = 1

def attention_naive(seq_len):
    # Simulate full recomputation of K and V on each step
    X = torch.randn(batch, seq_len, d_k * num_heads)
    W_K = torch.randn(d_k * num_heads, d_k)
    W_V = torch.randn(d_k * num_heads, d_k)
    W_Q = torch.randn(d_k * num_heads, d_k)

    Q = X[:, -1:, :] @ W_Q          # only last token's query
    K = X @ W_K                      # recompute ALL keys
    V = X @ W_V                      # recompute ALL values
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    return torch.softmax(scores, dim=-1) @ V

def attention_cached(new_token, cached_K, cached_V):
    # New token only; K and V are retrieved from cache
    new_K = new_token @ W_K_single
    new_V = new_token @ W_V_single
    K = torch.cat([cached_K, new_K], dim=1)
    V = torch.cat([cached_V, new_V], dim=1)
    Q = new_token @ W_Q_single
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    return torch.softmax(scores, dim=-1) @ V

# At seq_len=2048, naive recomputes 2048 * d_model projections per step.
# Cached recomputes 1. The ratio is 2048x in projection ops alone.
```

At 2048 tokens, the naive approach recomputes 2048 key and value projections per decode step. The cached approach computes exactly 1. For a 32-layer model, that is 32 × 2048 = 65,536 unnecessary matrix multiplications per token — eliminated entirely by the cache.

---

## The Tensor Reuse Explained

In HuggingFace Transformers, `past_key_values` is a tuple of `(key, value)` pairs — one per layer. After prefill, its shape per layer is:

```
(batch_size, num_heads, prompt_len, head_dim)
```

After each decode step, it grows by 1 along the `seq_len` dimension. The tensors being reused are the **intermediate projections of all previous tokens**, not the embeddings, not the attention weights, and not the final output representations.

This is a precise statement: the cache stores `X @ W_K` and `X @ W_V` for every token that has been processed, preventing those multiplications from being repeated.

---

## Trade-offs: Latency vs. Memory

### Latency benefit
- Time-to-first-token (TTFT) is determined by prefill time, which scales O(n²) and cannot be avoided.
- Time-per-output-token (TPOT) in the decode phase scales O(n) per step instead of O(n²) because K and V projections are cached.
- At n=4096, this is a ~4096× reduction in projection work per decode step.

### Memory cost
The KV cache memory footprint per request is:

```
cache_bytes = 2 × num_layers × num_heads × head_dim × seq_len × dtype_bytes
```

For a typical 7B-parameter model (32 layers, 32 heads, head_dim=128, float16):
```
= 2 × 32 × 32 × 128 × seq_len × 2 bytes
= 524,288 × seq_len bytes
≈ 0.5 MB per token
```

At a context length of 4096 tokens: ~2 GB per concurrent request. At 128K tokens (common in modern deployments): ~64 GB — comparable to the model weights themselves.

This is why KV cache management (eviction, paging, quantization) is an active infrastructure problem, not a solved one.

---

## Application to Multi-Turn Agent Systems

In a multi-turn, tool-using agent:
- Each turn appends to the conversation history.
- Without prefix caching, each API call reprocesses the full conversation history during prefill (O(n²)).
- With prefix caching (e.g., vLLM's `prefix_caching=True`, Anthropic's prompt caching with `cache_control`), the KV tensors for the static prefix (system prompt, prior turns) are stored server-side and reused across requests.

The distinction matters: a naive inference API may perform KV caching *within* a single forward pass (always true for decode) but *not* across API calls (requires explicit prefix cache infrastructure). A claim that "multi-turn is efficient due to caching" is only defensible if the deployment is confirmed to reuse KV state across turns, not just within a single generation.

---

## Adjacent Concepts (Briefly)

**Context window:** The maximum sequence length the model supports is bounded in part by KV cache memory. Longer context windows require proportionally more VRAM per concurrent request.

**Streaming:** Token streaming (returning tokens as they are generated) does not change KV cache mechanics — it is a transport-layer feature. The cache still grows incrementally during decode.

**Prefill chunking:** Some serving frameworks split long prompts into chunks to manage GPU memory, processing them sequentially. This changes TTFT characteristics but does not change which tensors are cached.

---

## What This Explainer Does Not Cover

- Grouped-query attention (GQA) and multi-query attention (MQA), which reduce cache size by sharing K/V heads across query heads.
- KV cache quantization (e.g., INT8 keys/values) as a memory reduction technique.
- PagedAttention (vLLM's memory virtualization for KV cache).
- Speculative decoding, which modifies the generate-verify loop but not the cache structure.

These are real and important topics. They are excluded here to preserve the sharpness of this explainer's scope: the fundamental mechanics of what is stored, what is avoided, and what it costs.
