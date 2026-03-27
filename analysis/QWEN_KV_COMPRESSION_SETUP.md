# Qwen3.5-4B KV Cache Compression: Setup Design Document

**Date:** 2026-03-27
**Status:** Draft — awaiting red team review
**Prior art:** GPT-2 combined compression experiment (entropy-turboquant-combined repo)

---

## 0. Executive Summary

We want to port the proven GPT-2 combined KV cache compression (entropy-adaptive eviction + TurboQuant quantization) to Qwen3.5-4B. The GPT-2 experiment achieved BLEU 0.596 at 8x compression (2x eviction + 4-bit quant) with PPL ratio 1.08. Qwen3.5-4B is architecturally different: hybrid attention/SSM with only 8/32 full-attention layers, GQA (4:1), and head_dim=256. This document designs the experiment to validate whether those results transfer.

**Recommended approach:** HuggingFace-first (Approach A/C hybrid) — run the experiment in transformers with monkey-patched attention, then port to llama.cpp if results are positive.

---

## 1. Integration Approach

### Options Evaluated

| Approach | Pros | Cons |
|----------|------|------|
| **A: HuggingFace only** | Direct port from GPT-2 experiment; full control over attention internals; GPU available for bf16 inference | ~8 GB VRAM for bf16; not production-ready |
| **B: llama.cpp native** | Production-ready; GGUF model already works; actual deployment path | Requires C++ KV cache modifications; much harder to iterate; no attention weight access without patches |
| **C: Hybrid (HF experiment -> llama.cpp production)** | Best of both: fast iteration in Python, production path in C++ | Two implementations to maintain |

### Recommendation: Approach C (HF-first, llama.cpp later)

**Rationale:**

1. **We already have the HF experiment pattern.** The GPT-2 `combined_compression.py` monkey-patches `ALL_ATTENTION_FUNCTIONS["eager"]`. Qwen3.5 uses the same transformers infrastructure — the calibration script (`calibrate_qwen35.py`) already loads the model via `Qwen3_5ForCausalLM.from_pretrained()` with `attn_implementation="eager"` and extracts attention weights.

2. **GPU is available.** The calibration script ran successfully on CUDA with bf16. The model fits in ~8 GB VRAM at bf16, which the system has (the llama.cpp config uses `-ngl 99` for full GPU offload).

3. **We need to prove the concept before C++ work.** If the hybrid architecture limits compression gains to only the 8 full-attention layers, the total memory savings may not justify a llama.cpp implementation. We need numbers first.

4. **llama.cpp KV cache is accessible but non-trivial.** The engine (`engine.py`) runs llama-cli as a subprocess — there's no hook into KV cache internals. Modifying llama.cpp's `llama_kv_cache` struct requires C++ changes and recompilation. That's Phase 2 work, contingent on Phase 1 results.

### Implementation Plan

```
Phase 1: HuggingFace Experiment (this document)
  - Load Qwen3.5-4B via transformers (bf16, CUDA)
  - Monkey-patch attention for the 8 full-attention layers only
  - Run eviction + quantization experiments
  - Measure perplexity, generation quality, BLEU

Phase 2 (conditional): llama.cpp Integration
  - Only if Phase 1 shows >4x compression with <10% quality loss
  - Modify llama.cpp KV cache management for selective layer compression
  - Integrate with existing hone-tools pipeline
```

---

## 2. Qwen3.5-Specific Adaptations

### 2.1 Only 8/32 Layers Have KV Cache

**Architecture:** 24 layers use Gated Delta Net (linear attention, no traditional KV cache). Only layers [3, 7, 11, 15, 19, 23, 27, 31] have standard multi-head attention with KV cache.

**Implications for compression ratio:**

The "compressible" KV cache is only 25% (8/32) of what a standard transformer would have. The other 75% of layers use a fixed-size recurrent state (delta net), which is already O(1) in sequence length.

Effective compression ratio formula:

```
Total KV memory = (8 layers * KV_per_layer) + (24 layers * delta_net_state)
                = 8 * (n_kv_heads * 2 * seq_len * head_dim * dtype_bytes) + fixed_SSM_state

Compressed KV memory = 8 * (n_kv_heads * 2 * compressed_tokens * compressed_bits / 8) + fixed_SSM_state
```

If we achieve 8x compression on the 8 attention layers (matching GPT-2), the **system-level** compression of the attention KV cache is still 8x — but the **total model memory** savings are smaller because the delta net states are incompressible by this method. The KV cache is still the dominant memory consumer at long contexts because it scales with seq_len while delta net state is fixed.

**Critical insight:** At long contexts (>4K tokens), the 8 attention layers' KV cache dominates memory because it grows linearly with sequence length, while the 24 delta net layers have constant-size state. This makes compression of just these 8 layers **more** valuable than it first appears — at 32K context, the attention KV cache is the overwhelming majority of runtime memory.

### 2.2 GQA: 16 Query Heads, 4 KV Heads

**Problem:** Entropy is measured per **query head** (16 per layer, 128 total across 8 layers). But the KV cache stores per **KV head** (4 per layer, 32 total). Each KV head serves 4 query heads.

**Eviction impact:** When deciding which KV entries to evict for a given KV head, we need to consider the attention patterns of all 4 query heads that share it. A KV entry that's irrelevant to 3 query heads but critical to 1 should be kept.

**Proposed adaptation for eviction:**
- For each KV head group (4 query heads sharing 1 KV head), compute eviction based on the **union** of important positions across all 4 query heads
- Alternatively: use the **max attention weight** across the 4 query heads for each position, then apply top-k eviction
- This is more conservative than the GPT-2 approach (where each head has its own KV) but safer

**Proposed adaptation for quantization:**
- Entropy-informed bit allocation should aggregate per KV head group
- For KV head `k` serving query heads `[4k, 4k+1, 4k+2, 4k+3]`: use the **mean** entropy of the 4 query heads
- If the 4 query heads have very different entropies (high variance within a group), prefer higher bit allocation (conservative)

**Entropy data from calibration (per KV head group):**

| Layer | KV Head 0 (Q0-3) | KV Head 1 (Q4-7) | KV Head 2 (Q8-11) | KV Head 3 (Q12-15) |
|-------|-------------------|-------------------|---------------------|---------------------|
| L3    | 3.54 (std 0.20)   | 3.86 (std 0.19)   | 3.90 (std 0.15)     | 3.32 (std 0.71)     |
| L7    | 3.17 (std 0.32)   | 2.54 (std 0.69)   | 3.54 (std 0.21)     | 3.42 (std 0.26)     |
| L11   | 3.23 (std 0.17)   | 3.17 (std 0.57)   | 3.14 (std 0.54)     | 3.38 (std 0.15)     |
| L15   | 3.53 (std 0.48)   | 3.35 (std 0.12)   | 3.28 (std 0.10)     | 2.87 (std 0.78)     |
| L19   | 3.24 (std 0.28)   | 2.79 (std 0.24)   | 3.51 (std 0.47)     | 2.80 (std 0.58)     |
| L23   | 3.49 (std 0.18)   | 3.40 (std 0.03)   | 3.42 (std 0.02)     | 3.29 (std 0.19)     |
| L27   | 3.11 (std 0.27)   | 3.13 (std 0.06)   | 3.22 (std 0.03)     | 2.84 (std 0.31)     |
| L31   | 3.81 (std 0.01)   | 3.69 (std 0.14)   | 2.30 (std 1.05)     | 2.87 (std 0.69)     |

**Observations:**
- Most KV head groups have moderate-to-high entropy (2.3-3.9 range)
- L31 KV Head 2 (Q8-11) has the highest intra-group variance (std 1.05) — contains both very low entropy heads (L31_H9=0.31, L31_H10=1.63) and moderate heads (L31_H8=1.65, L31_H11=3.62). This KV head needs careful handling.
- L31_H9 (entropy 0.31) is the only "sink" head in the entire model. It and L31_H12 (entropy 0.62) are the only focused/sink heads. Their KV head group (KV Head 2, containing L31_H8-H11) needs to preserve the sink pattern — eviction must not remove the sink position.

### 2.3 head_dim=256 vs GPT-2's 64

**TurboQuant implications:**

1. **Rotation matrix is 256x256 (vs 64x64).** QR decomposition of a 256x256 Gaussian matrix is still fast (~ms) but the matrix occupies 256 KB (float32). With 4 KV heads x 8 layers x 2 (K+V) = 64 rotation matrices total, that's ~16 MB of rotation matrix overhead. Trivial compared to KV cache but worth noting.

2. **Lloyd-Max operates on more coordinates.** Each vector has 256 scalar values to quantize independently. The rotation decorrelates them, so each coordinate should still be approximately N(0, sigma^2). The larger dimension actually **helps** — more coordinates means the law of large numbers kicks in harder, making the Gaussian assumption for sigma estimation more accurate.

3. **QJL residual correction scales linearly.** With dim=256, the QJL projection matrix is 256 x m_projections. The default m_projections=64 from GPT-2 may need to increase proportionally. **Proposed: m_projections=128** (maintaining dim/m ratio of ~2:1, vs GPT-2's ~1:1). This is a hyperparameter to sweep.

4. **Memory per vector is 4x larger.** Each K or V vector is 256 x 2 bytes (bf16) = 512 bytes, vs GPT-2's 64 x 4 bytes (float32) = 256 bytes. The 4-bit quantization reduces this to 256 x 0.5 = 128 bytes per vector. Compression gain per vector is proportionally the same.

### 2.4 Double Quantization: Model Q4_K_M + KV Cache Quantization

**The GGUF model is already Q4_K_M quantized (weights).** The HuggingFace experiment loads the **full-precision model** from HuggingFace Hub (bf16), so there is NO double quantization in the experiment.

However, for Phase 2 (llama.cpp), the model weights are Q4_K_M while we'd also quantize the KV cache. This creates a chain:

```
Full precision weights -> Q4_K_M weights -> bf16 KV cache -> 4-bit KV cache
```

**Risks:**
- Attention scores computed from Q4_K_M queries/keys already have quantization noise
- Further quantizing the KV cache adds a second noise source
- These noise sources may compound nonlinearly in softmax (amplification of small differences)

**Mitigation:** The HF experiment will measure quality degradation from KV compression alone (bf16 weights). If we proceed to llama.cpp, we'll need a separate experiment measuring the compound effect. We can approximate this in HF by loading a GPTQ or AWQ 4-bit version of the model.

### 2.5 Gated Delta Net State Compression

The 24 linear attention layers use Gated Delta Net, which maintains a recurrent state instead of a KV cache. This state has fixed size regardless of sequence length.

**Can it be compressed?**

- The delta net state is a matrix (typically head_dim x head_dim or similar), not a growing sequence of KV vectors
- It's updated incrementally, not appended to
- Standard eviction doesn't apply (there's no "which tokens to keep" question)
- Quantization could theoretically apply (quantize the state matrix)

**Recommendation: Out of scope for Phase 1.** The delta net state is already compact (fixed-size). The attention KV cache is the scaling bottleneck. Focus there first. If delta net state compression is needed, it's a separate research question requiring different techniques (low-rank approximation, not eviction/TurboQuant).

---

## 3. Memory Budget Analysis

### 3.1 KV Cache Size Formula

For the 8 full-attention layers:

```
KV_bytes = n_attn_layers * n_kv_heads * 2 * seq_len * head_dim * dtype_bytes
         = 8            * 4             * 2 * seq_len * 256      * 2 (bf16)
         = 32,768 * seq_len bytes
         = 32 KB per token of context
```

### 3.2 KV Cache at Various Context Lengths

| Context Length | KV Cache (bf16) | KV Cache (4-bit quant) | KV Cache (2x evict + 4-bit) | Savings |
|---------------|-----------------|----------------------|---------------------------|---------|
| 512 tokens    | 16 MB           | 4 MB                 | 2 MB                      | 14 MB   |
| 2,048 tokens  | 64 MB           | 16 MB                | 8 MB                      | 56 MB   |
| 4,096 tokens  | 128 MB          | 32 MB                | 16 MB                     | 112 MB  |
| 8,192 tokens  | 256 MB          | 64 MB                | 32 MB                     | 224 MB  |
| 32,768 tokens | 1,024 MB        | 256 MB               | 128 MB                    | 896 MB  |
| 131,072 tokens| 4,096 MB        | 1,024 MB             | 512 MB                    | 3,584 MB|
| 262,144 tokens| 8,192 MB        | 2,048 MB             | 1,024 MB                  | 7,168 MB|

### 3.3 Total Model Memory Budget

| Component | Size | Notes |
|-----------|------|-------|
| Model weights (bf16 HF) | ~8 GB | Full precision for experiment |
| Model weights (Q4_K_M GGUF) | ~2.5 GB | Production via llama.cpp |
| Delta net state (24 layers) | ~small, fixed | O(1) in seq_len, likely <100 MB |
| KV cache (8 layers, 2K ctx) | 64 MB | The compression target |
| KV cache (8 layers, 32K ctx) | 1,024 MB | Where compression really matters |

**Key insight:** At 32K context, the KV cache exceeds the Q4_K_M model weights. At 262K (max context), KV cache is 8 GB — dwarfing everything else. This is exactly where compression pays off most.

### 3.4 Compression Achievability

From GPT-2 results, mapping to Qwen3.5-4B:

| Config | Compression | GPT-2 BLEU | GPT-2 PPL Ratio | Expected Qwen3.5 |
|--------|-------------|------------|-----------------|-------------------|
| 4-bit quant only | 4x | 0.669 | 1.05 | Should transfer — Lloyd-Max is architecture-agnostic |
| 2x evict only | 2x | 0.776 | 1.03 | Uncertain — GQA eviction is harder, but 75% mixed/diffuse heads suggest tolerance |
| 2x evict + 4-bit | 8x | 0.596 | 1.08 | Target configuration — conservative estimate BLEU >0.4 |
| 3x evict + 4-bit | 12x | 0.464 | N/A | Aggressive — may degrade more with GQA |
| 2-bit quant only | 8x | 0.308 | 1.29 | Too aggressive for GPT-2, may work better with 256-dim rotation |

---

## 4. Experiment Design

### 4.1 Model Loading

```python
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

model_id = "Qwen/Qwen3.5-4B"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

model = Qwen3_5ForCausalLM.from_pretrained(
    model_id,
    config=config.text_config,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="eager",   # Required for attention weight access
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model.eval()
```

This mirrors the existing `calibrate_qwen35.py` loading code exactly.

### 4.2 Monkey-Patch Strategy

**Key difference from GPT-2:** We must only patch the 8 full-attention layers and skip the 24 linear-attention layers.

```python
# The attention mechanism in Qwen3.5 uses transformers' ALL_ATTENTION_FUNCTIONS
# We need to identify the correct module path for Qwen3.5's attention

FULL_ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]

def patched_qwen_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """Modified attention with eviction/quantization for Qwen3.5."""
    # Determine which layer this is
    layer_idx = _get_layer_idx(module)  # Need to identify layer from module

    # Only compress full-attention layers
    if layer_idx not in FULL_ATTENTION_LAYERS:
        # Pass through unmodified (shouldn't happen if patched correctly)
        return original_attention_forward(module, query, key, value, attention_mask, **kwargs)

    # GQA-aware compression
    # key, value shapes: (batch, n_kv_heads=4, seq_len, head_dim=256)
    # query shape: (batch, n_heads=16, seq_len, head_dim=256)

    if state['active'] and state['mode'] in ('quantization', 'combined'):
        key, value = _quantize_kv_layer_gqa(key, value, layer_idx)

    # Standard scaled dot-product attention
    # ... (compute attention weights)

    if state['active'] and state['mode'] in ('eviction', 'combined'):
        # GQA-aware eviction: consider all 4 query heads per KV head
        eviction_mask = _gqa_aware_eviction(attn_weights, keep_ratio, layer_idx)
        # ... apply mask
```

**Investigation needed:** Before implementation, we need to confirm:
1. The exact module path for Qwen3.5's attention function dispatch
2. Whether the model uses `ALL_ATTENTION_FUNCTIONS["eager"]` like GPT-2 or a different mechanism
3. How to determine layer_idx from the module reference (likely via `module.layer_idx` attribute)

### 4.3 GQA-Aware Eviction Function

```python
def _gqa_aware_eviction(attn_weights, keep_ratio, layer_idx, head_entropies):
    """
    Eviction that respects GQA structure.
    attn_weights: (batch, n_heads=16, seq_len, seq_len)
    Returns: mask (batch, n_heads=16, seq_len, seq_len)
    """
    n_kv_heads = 4
    gqa_ratio = 4  # 4 query heads per KV head

    mask = torch.ones_like(attn_weights)

    for kv_h in range(n_kv_heads):
        q_heads = range(kv_h * gqa_ratio, (kv_h + 1) * gqa_ratio)

        for pos in range(seq_len):
            context_len = pos + 1
            # Aggregate importance across all query heads sharing this KV head
            # Use max attention weight across query heads (conservative)
            max_attn = torch.zeros(context_len)
            for q_h in q_heads:
                max_attn = torch.max(max_attn, attn_weights[0, q_h, pos, :context_len])

            # Entropy-adaptive keep ratio for this KV head group
            adj_ratio = _entropy_adjusted_ratio(keep_ratio, layer_idx, kv_h, head_entropies)

            k = max(1, int(context_len * adj_ratio))
            if context_len > k:
                _, keep_idx = max_attn.topk(k)
                evict_mask = torch.zeros(context_len)
                evict_mask[keep_idx] = 1.0
                # Apply same eviction to all query heads in this group
                for q_h in q_heads:
                    mask[0, q_h, pos, :context_len] = evict_mask

    return mask
```

### 4.4 TurboQuant Adaptation for head_dim=256

```python
# Pre-compute rotation matrices for each KV head (not query head)
# 4 KV heads * 8 layers * 2 (K, V) = 64 rotation matrices

for layer_idx in FULL_ATTENTION_LAYERS:
    for kv_h in range(4):
        for n_bits in [2, 3, 4]:
            seed = GLOBAL_SEED + layer_idx * 100 + kv_h * 10 + n_bits
            tq_cache[(layer_idx, kv_h, n_bits)] = TurboQuantVectorQuantizer(
                dim=256,        # head_dim (4x GPT-2)
                n_bits=n_bits,
                m_projections=128,  # Scaled up from GPT-2's 64
                seed=seed,
                use_qjl=True,
            )
```

**Note:** Pre-computation of 64 rotation matrices at dim=256 takes ~1-2 seconds. Each QR decomposition of a 256x256 matrix is ~0.5ms on CPU.

### 4.5 Evaluation Data

**Option A: WikiText-2 (matching GPT-2 experiment)**
- Pro: Direct comparison with GPT-2 results
- Con: Qwen3.5 is a chat model; WikiText may not reflect its strengths

**Option B: Hone-tool tasks**
- Pro: Practical quality validation on the actual deployment tasks
- Con: Hone-tools use llama-cli subprocess, not HF model directly

**Option C: Both**
- WikiText-2 for perplexity and BLEU comparison with GPT-2
- A set of representative prompts for generation quality (manual inspection)

**Recommendation: Option C.** Use WikiText-2 as the primary quantitative benchmark for comparability, but also run 10-20 diverse prompts (coding, reasoning, factual QA) through the HF model to assess practical quality.

### 4.6 Compression Configurations to Test

**Phase 1: Individual components (8 configs)**

| # | Mode | Eviction | Bits | Expected Ratio |
|---|------|----------|------|----------------|
| 1 | Baseline | None | 16 (bf16) | 1x |
| 2 | Eviction only | 2x (keep 50%) | 16 | 2x |
| 3 | Eviction only | 3x (keep 33%) | 16 | 3x |
| 4 | Quant only | None | 4 | 4x |
| 5 | Quant only | None | 3 | 5.3x |
| 6 | Quant only | None | 2 | 8x |
| 7 | Combined | 2x | 4 | 8x |
| 8 | Combined | 3x | 4 | 12x |

**Phase 2: Entropy-informed (4 configs)**

| # | Mode | Details | Expected Ratio |
|---|------|---------|----------------|
| 9 | Entropy quant | Per-KV-head bits: 2-4 range, high_fewer | ~4-5x |
| 10 | Entropy quant | Per-KV-head bits: 2-4 range, high_more | ~4-5x |
| 11 | Combined + entropy quant | 2x evict + entropy-informed bits | ~8-10x |
| 12 | Combined + entropy quant | 2x evict + entropy-informed bits + GQA-aware eviction | ~8-10x |

**Phase 3: QJL ablation (2 configs)**

| # | Mode | QJL | Purpose |
|---|------|-----|---------|
| 13 | 4-bit quant, no QJL | Off | Measure QJL contribution at dim=256 |
| 14 | 4-bit quant, QJL m=256 | m=256 | Test if more projections help at higher dim |

### 4.7 Metrics

1. **Perplexity** on WikiText-2 validation (30 sequences, 128 tokens each — matching GPT-2 setup)
2. **Perplexity ratio** (compressed / baseline) — the primary quality metric
3. **BLEU-4** between baseline and compressed generation (50-token continuations)
4. **ROUGE-L** between baseline and compressed generation
5. **Token match rate** (position-by-position exact match)
6. **Generation quality** (manual inspection of 10-20 diverse prompts)
7. **Wall-clock time** (compression overhead per forward pass)
8. **Effective compression ratio** (accounting for overhead like codebooks, norms)

### 4.8 Estimated Runtime

| Step | Time Estimate |
|------|---------------|
| Model loading (bf16 CUDA) | ~30-60s |
| Entropy calibration (already done) | 0s (load from JSON) |
| TurboQuant pre-computation (64 rotation matrices) | ~2s |
| Baseline perplexity (30 seqs) | ~2-3 min |
| Each compression config perplexity (30 seqs) | ~3-5 min |
| Each generation experiment (20 prompts x 50 tokens) | ~5-10 min |
| **Total for 14 configs** | **~90-150 min** |

This is comparable to the GPT-2 experiment runtime (45-60 min on CPU). GPU should offset the larger model size.

---

## 5. Risk Assessment

### 5.1 High-Probability Risks

**R1: GQA eviction is fundamentally harder than per-head eviction.**
- **Impact:** Eviction quality may be significantly worse than GPT-2
- **Probability:** High
- **Mitigation:** The conservative union-based eviction strategy. Also: test quantization-only first (no eviction) since that's GQA-agnostic.
- **Detection:** Compare eviction-only BLEU/PPL against GPT-2 eviction-only results

**R2: Monkey-patching Qwen3.5 attention is more complex than GPT-2.**
- **Impact:** Experiment may not work at all, or may not intercept the right code path
- **Probability:** Medium-high (Qwen3.5 is a newer, less-documented architecture)
- **Mitigation:** Study the `transformers` Qwen3.5 attention implementation before coding. The calibration script already accesses `output_attentions=True` successfully, confirming the eager attention path works.
- **Detection:** Verify that compressed outputs differ from baseline (if patch isn't applied, they'd be identical)

**R3: bf16 precision affects TurboQuant quality.**
- **Impact:** GPT-2 experiment used float32; bf16 has less precision for rotation and sigma estimation
- **Probability:** Medium
- **Mitigation:** Cast to float32 for the rotation + quantization step, then cast back. This adds minimal overhead.
- **Detection:** Compare sigma estimates and reconstruction error in bf16 vs float32

### 5.2 Medium-Probability Risks

**R4: Only 8 layers of compression may not yield meaningful system-level savings at short context.**
- **Impact:** Results may be unimpressive at 2K context (only 64 MB -> 8 MB, while model is 8 GB)
- **Probability:** Medium (depends on use case)
- **Mitigation:** Frame results for long-context scenarios (32K+) where KV cache dominates. The hone-tools currently use ctx_size=2048, but the model supports 262K.
- **Detection:** Compute savings at multiple context lengths (done in Section 3.2)

**R5: Entropy distribution is narrow (CV=0.19) — entropy-informed allocation may not help much.**
- **Impact:** Per-head bit allocation may not differ meaningfully from uniform allocation
- **Probability:** Medium
- **Mitigation:** The GPT-2 had wider entropy spread. If Qwen3.5's heads are more homogeneous, uniform 4-bit quantization may be the best approach (simpler is fine).
- **Detection:** Compare entropy-informed vs uniform quantization metrics. If <2% difference, entropy-informed is not worth the complexity.

**R6: VRAM insufficiency.**
- **Impact:** Cannot run experiment if model + KV cache + overhead > available VRAM
- **Probability:** Low-medium (bf16 model is ~8 GB; most consumer GPUs have 8-24 GB)
- **Mitigation:** Use gradient checkpointing or reduce sequence length. Or use CPU fallback (slower but works). The existing calibration ran on CUDA, so the GPU has enough for 128-token sequences.

### 5.3 Low-Probability Risks

**R7: Qwen3.5's attention implementation doesn't use the standard `ALL_ATTENTION_FUNCTIONS` dispatch.**
- **Impact:** Monkey-patch approach fails entirely
- **Mitigation:** Read the source code before implementing. Worst case, override the forward() method of each attention module directly.

**R8: Gated Delta Net layers interact with attention layers in ways that make compression of attention layers more harmful.**
- **Impact:** Quality degradation is worse than expected because delta net layers depend on attention layer outputs in unexpected ways
- **Probability:** Low (transformers are generally layer-independent in forward pass)
- **Mitigation:** Test progressive compression (compress 1 layer, 2 layers, ... 8 layers) to identify if specific layers are more sensitive.

### 5.4 Key Assumptions

1. **Entropy calibration at 128 tokens transfers to longer contexts.** The calibration used short sequences. Attention patterns change at longer contexts. May need to recalibrate at target context length.

2. **Gaussian assumption for rotated vectors holds at dim=256.** The rotation decorrelation should make this better than at dim=64, but Qwen3.5's bf16 KV values may have different distributional properties than GPT-2's float32.

3. **Perplexity on WikiText-2 is a meaningful metric for a chat model.** Qwen3.5 is instruction-tuned; WikiText perplexity may not correlate with chat quality. Generation quality assessment is essential.

4. **The HuggingFace model and GGUF model produce equivalent outputs.** The Q4_K_M quantization means the GGUF model's baseline is already degraded from the HF bf16 model. Phase 1 results (HF) may be optimistic vs production (GGUF).

### 5.5 Unknown Unknowns

- How does Qwen3.5 handle the sink pattern? Only 1 head (L31_H9) is a sink. If the model relies critically on this single head, any compression of L31's KV head 2 could be disproportionately harmful.
- Does the interleaving of attention and delta net layers create dependencies that make attention KV cache compression riskier than in a pure-attention model?
- What is the delta net state size? We assumed "small, fixed" but haven't measured it. If it's larger than expected, the relative benefit of attention KV compression is smaller.

---

## 6. Success Criteria

The experiment succeeds if we demonstrate ANY of the following for Qwen3.5-4B:

| Criterion | Threshold | GPT-2 Reference |
|-----------|-----------|-----------------|
| 4x compression with PPL ratio < 1.10 | Quant-only 4-bit | GPT-2: 1.05 |
| 8x compression with BLEU > 0.40 | Combined 2x evict + 4-bit | GPT-2: 0.596 |
| Entropy-informed allocation beats uniform | Any statistically significant improvement | GPT-2: confirmed |
| GQA-aware eviction is viable | BLEU > 0.60 at 2x eviction | GPT-2: 0.776 |

The experiment provides **negative** but valuable results if:
- GQA makes eviction impractical (BLEU < 0.30 at 2x) — tells us to focus on quantization only
- Narrow entropy distribution makes per-head allocation useless — tells us uniform quantization is sufficient
- The hybrid architecture limits total savings — tells us Phase 2 (llama.cpp) isn't worth it for short-context use cases

---

## 7. File Structure

```
BitNet/analysis/
  QWEN_KV_COMPRESSION_SETUP.md          # This document
  qwen_kv_compression_experiment.py      # Main experiment script (to be written)
  qwen_attention_patch.py                # Monkey-patch module for Qwen3.5 attention
  results/
    qwen35_compression_results.json      # Raw results
    qwen35_compression_report.md         # Analysis report
    plots/                               # Visualization outputs
```

---

## 8. Open Questions for Red Team

1. Is the GQA union-based eviction strategy too conservative? Should we try per-query-head eviction (which would require duplicating KV entries)?
2. Should we test at longer sequence lengths (512, 1024) even though calibration was at 128? The entropy patterns may shift.
3. Is QJL residual correction worth the implementation complexity for a proof-of-concept? Could skip it and add later.
4. Should we profile the delta net state size to verify our assumption that it's small?
5. The entropy calibration data shows all non-attention layers as 0.0 entropy — is this correct handling, or should those entries be excluded rather than zeroed?
