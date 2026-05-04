# Sources — Day 1

## Canonical References

---

### 1. Vaswani et al. (2017) — "Attention Is All You Need"

**Citation:**
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

**URL:** https://arxiv.org/abs/1706.03762

**What it establishes relevant to this gap:**
- The scaled dot-product attention formula: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V` (Section 3.2.1)
- The multi-head attention structure and the per-head projection matrices W_Q, W_K, W_V (Section 3.2.2)
- The O(n²) complexity of self-attention with respect to sequence length (Section 4, Table 1)

This paper is the direct source for the attention equations and the computational complexity analysis used in the explainer.

---

### 2. HuggingFace Transformers — `past_key_values` and KV Cache Documentation

**Primary reference:**
HuggingFace. (2024). *Transformers documentation: Generation with KV cache*. https://huggingface.co/docs/transformers/main/en/kv_cache

**Supporting reference:**
HuggingFace. (2024). *Transformers documentation: Cache classes for efficient generation*. https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.Cache

**What it establishes relevant to this gap:**
- The `use_cache=True` parameter and `past_key_values` return structure in `model.generate()` and `model.forward()`
- The shape contract for cached tensors: `(batch_size, num_heads, seq_len, head_dim)`
- The `DynamicCache`, `StaticCache`, and `QuantizedCache` implementations and their memory trade-offs
- The `enable_prefix_caching` flag in vLLM and equivalent constructs for cross-request reuse

---

### 3. Kwon et al. (2023) — "Efficient Memory Management for Large Language Model Serving with PagedAttention"

**Citation:**
Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*.

**URL:** https://arxiv.org/abs/2309.06180

**What it establishes relevant to this gap:**
- Quantitative analysis of KV cache memory fragmentation and per-request overhead in production serving (Section 2)
- The formula for KV cache size as a function of model hyperparameters and sequence length (Section 2.1)
- Empirical data on the fraction of GPU memory consumed by KV cache vs. model weights under realistic workloads

This paper is cited specifically for the memory cost analysis and the infrastructure implications section of the explainer.
