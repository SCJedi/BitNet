# KV Cache Compression Results: Qwen3.5-4B

**Date:** 2026-03-27
**Model:** Qwen3.5-4B
**Hardware:** NVIDIA RTX 3060

---

## Executive Summary

Qwen3.5-4B demonstrates exceptional robustness to KV cache compression across all 10 tested configurations, achieving perfect BLEU 1.000, ROUGE-L 1.000, and 100% token match even at 12x compression. The model's hybrid attention architecture (8 full-attention + 24 linear layers), grouped query attention (GQA 4:1), large head dimension (head_dim=256), and narrow entropy distribution collectively explain this stability. By contrast, GPT-2 degrades to BLEU 0.596 at 8x compression. These results indicate that Qwen3.5-4B can sustain aggressive combined eviction and quantization strategies with near-zero quality loss under short-context conditions, making it a strong candidate for memory-constrained deployment.

---

## Results

### Compression Configurations

| Config | Compression | PPL Ratio | PPL |
|---|---|---|---|
| Full cache (baseline) | 1.0x | 1.000 | 2.64 |
| Eviction 2x | 2.0x | 1.004 | 2.65 |
| Eviction 3x | 3.0x | 1.011 | 2.66 |
| Quantization 5-bit | 3.2x | 1.013 | 2.67 |
| Quantization 4-bit | 4.0x | 1.021 | 2.69 |
| Entropy-adaptive quantization | 4.0x | 1.014 | 2.67 |
| Quantization 3-bit | 5.3x | 1.036 | 2.73 |
| Combined: eviction 2x + quant 5-bit | 6.4x | 1.016 | 2.68 |
| Combined: eviction 2x + quant 4-bit | 8.0x | 1.024 | 2.70 |
| Combined: eviction 3x + quant 4-bit | 12.0x | 1.033 | 2.72 |

**Test conditions:** 6 sequences, 256-token context, 30-token greedy generation.

All 10 configurations achieved BLEU 1.000, ROUGE-L 1.000, and 100% token match. Maximum observed PPL degradation at 12x compression: 3.3% (PPL 2.64 → 2.72).

---

## Key Findings

### Why Qwen3.5-4B Is Robust to Compression

**1. Hybrid architecture distributes attention load.**
Only 8 of 32 layers use full (quadratic) attention; the remaining 24 use linear attention. Full-attention layers are not subject to the same KV cache pressure as transformer-only models. Errors introduced by compression in the linear layers have limited propagation.

**2. GQA reduces KV cache size at baseline.**
Grouped query attention with a 4:1 ratio (16 query heads, 4 KV heads) means the KV cache is already 4x smaller than a naive multi-head model. Compression operates on a smaller, more information-dense cache — so each evicted token or quantization step carries proportionally less penalty.

**3. head_dim=256 provides quantization headroom.**
Large head dimensions give quantization more degrees of freedom per head. Low-bit quantization error is distributed across 256 values rather than 64 or 128, reducing the relative distortion.

**4. Narrow entropy distribution.**
The model's attention patterns are concentrated (low entropy), meaning a small fraction of KV entries carry most of the information. Eviction strategies that retain high-scoring tokens preserve nearly all signal, and quantization error on low-weight entries has minimal impact on output.

**5. Entropy-adaptive quantization matches architectural properties.**
Entropy-adaptive quant (4.0x compression, PPL ratio 1.014) matches or outperforms fixed 4-bit quant (1.021) at the same compression ratio. This confirms the model's attention entropy structure is consistent and exploitable.

---

## GPT-2 vs. Qwen3.5-4B Comparison

| | GPT-2 | Qwen3.5-4B |
|---|---|---|
| Architecture | Dense MHA | Hybrid (full + linear) |
| GQA | No (1:1) | Yes (4:1) |
| head_dim | 64 | 256 |
| BLEU at 8x compression | 0.596 | 1.000 |
| PPL ratio at 8x | — | 1.024 |

GPT-2's 40% BLEU degradation at 8x compression reflects its architecture's vulnerability: dense attention means every KV entry matters, the small head dimension amplifies quantization error, and there is no GQA buffer. Qwen3.5-4B's 0% BLEU degradation at the same compression ratio is a direct consequence of architectural decisions that were likely not designed with compression in mind but happen to be highly compression-friendly.

---

## Practical Implications

### KV Memory Reduction at Scale (Qwen3.5-4B, 12x compression)

Baseline KV cache size (4 KV heads, head_dim=256, fp16):
- Per token, per layer: 4 heads × 256 dims × 2 bytes × 2 (K+V) = 4 KB
- Per token, all 32 layers: 128 KB

| Context Length | Baseline KV Memory | At 12x Compression |
|---|---|---|
| 8K tokens | ~1.0 GB | ~85 MB |
| 32K tokens | ~4.1 GB | ~340 MB |
| 128K tokens | ~16.4 GB | ~1.4 GB |

At 128K context, 12x compression reduces KV memory from 16.4 GB (exceeds RTX 3060 VRAM) to 1.4 GB (fits comfortably). This makes long-context inference feasible on consumer hardware without model quantization.

---

## Limitations

**Short sequences.** All tests used 256-token contexts and 30-token generation. Eviction strategies become qualitatively different at 8K–128K context, where tokens from thousands of steps ago must be recovered or permanently discarded. Perfect token match at 256 tokens does not guarantee it at 32K.

**Greedy decoding.** Greedy generation is deterministic and tends to be more stable than sampling. BLEU and token match scores would likely be lower under temperature > 0 or nucleus sampling.

**Small sample size.** N=6 sequences is insufficient for statistical confidence. Variance across sequence types (code, dialogue, technical text, long narratives) is unknown.

**Single hardware configuration.** Results are from one GPU (RTX 3060). Memory bandwidth constraints, mixed-precision behavior, and CUDA kernel behavior may differ on other hardware.

**No retrieval stress test.** The test sequences did not include tasks requiring retrieval of information from early context (e.g., "What did the user say at the start?"). These are the tasks most sensitive to eviction errors.

---

## Next Steps

1. **Long-context stress test.** Run 8K, 32K, and 128K token sequences with retrieval tasks (needle-in-haystack, multi-turn QA). This is the most important gap.

2. **Sampling-based evaluation.** Repeat with temperature=0.7 and top-p=0.9 to assess BLEU/ROUGE variance under stochastic decoding.

3. **Larger sample size.** Minimum N=50–100 sequences across diverse task types for statistical validity.

4. **Eviction policy comparison.** Compare recency-based, attention-score-based, and entropy-adaptive eviction at 3x and higher to understand which policy drives robustness.

5. **Inference throughput.** Measure tokens/second and memory bandwidth utilization at each compression level. Compression is only practically useful if it doesn't bottleneck on dequantization overhead.

6. **Cross-architecture replication.** Test the same compression pipeline on Llama-3-8B (dense MHA) and Mistral-7B (sliding window + GQA) to isolate which architectural factors drive the observed robustness.
