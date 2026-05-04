# Grounding Commit — Day 1

## System: Inference API / Model Card Documentation

---

## Commit Description

**Title:** `docs: replace shallow caching claim with mechanistic KV cache explanation`

**Scope:** Model card (model_card.md) and CFO memo (cfo_memo.md) — the two documents that contained the unsupported efficiency claim.

---

## Before

```markdown
<!-- model_card.md — BEFORE -->

## Efficiency

Multi-turn conversations are efficient due to caching. Repeated prompts
and conversation history do not incur full reprocessing costs.
```

```markdown
<!-- cfo_memo.md — BEFORE -->

The system handles long conversations efficiently through built-in caching,
reducing compute costs at scale.
```

---

## After

```markdown
<!-- model_card.md — AFTER -->

## Inference Efficiency: KV Cache

During token generation, the transformer stores key and value projection
tensors (`X @ W_K`, `X @ W_V`) for all previously processed tokens in a
structure called the KV cache (exposed as `past_key_values` in HuggingFace
Transformers). On each subsequent decode step, only the new token's K and V
projections are computed; all prior projections are retrieved from cache.
This reduces per-step attention work from O(n²) to O(n) in the sequence length.

**Within-generation caching** is automatic for all inference backends.

**Cross-request prefix caching** — which makes multi-turn conversations
efficient by reusing KV state across API calls — requires explicit
infrastructure support. This deployment uses [vLLM with
`enable_prefix_caching=True` / Anthropic prompt caching with `cache_control`].
Confirm this setting before citing per-turn efficiency claims in production
documentation.

**Memory cost:** KV cache storage scales as:
`2 × num_layers × num_heads × head_dim × seq_len × dtype_bytes`
For this model (32 layers, 32 heads, head_dim=128, float16):
approximately 0.5 MB per token per concurrent request.
```

```markdown
<!-- cfo_memo.md — AFTER -->

The system reduces compute costs in multi-turn conversations through KV
caching: the server stores intermediate attention tensors (keys and values)
from prior turns and reuses them rather than reprocessing the full
conversation history on each response. This avoids the quadratic compute
cost of full-sequence attention on every turn.

This efficiency is conditional on prefix caching being enabled in the
inference infrastructure (currently: [confirm deployment setting]).
Memory cost is approximately 0.5 MB per token per concurrent user at
this model size, which becomes material at context lengths above 8K tokens
or under high concurrency.
```

---

## What Changed and Why

The original documentation made an efficiency claim that would not survive scrutiny from an infrastructure engineer or a technical auditor. The revised versions:

1. Name the specific mechanism (KV cache, `past_key_values`, key/value projections).
2. State the computational consequence precisely (O(n²) → O(n) per decode step).
3. Distinguish within-generation caching (always present) from cross-request prefix caching (infrastructure-dependent).
4. Give a concrete memory cost formula so the claim is auditable.
5. Include a `[confirm deployment setting]` placeholder to prevent the claim from being restated without verification.
