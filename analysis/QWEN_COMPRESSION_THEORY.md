# Qwen3.5-4B KV Cache Compression Theory: Entropy-Adaptive Eviction + TurboQuant

**Date:** 2026-03-27
**Status:** Theoretical analysis for hybrid attention-SSM architecture
**Prerequisite reading:** `COMBINED_COMPRESSION_THEORY.md`, `COMBINED_COMPRESSION_RESULTS.md` (GPT-2 results)

---

## 0. Architecture Summary

| Parameter | GPT-2 small | Qwen3.5-4B | Ratio |
|-----------|-------------|-------------|-------|
| Parameters | 124M | 4B | 32x |
| Total layers | 12 | 32 | 2.7x |
| Full attention layers | 12 | 8 (layers 3,7,11,15,19,23,27,31) | 0.67x |
| Query heads per attn layer | 12 | 16 | 1.3x |
| KV heads per attn layer | 12 | 4 (GQA 4:1) | 0.33x |
| head_dim | 64 | 256 | 4x |
| Max context | 1,024 | 262,144 | 256x |
| Weight quantization | FP32 | Q4_K_M (GGUF) | N/A |
| Entropy mean | 2.77 bits | 3.23 bits | 1.17x |
| Entropy range | 0.03 - 5.95 | 0.31 - 4.11 | 13x vs 220x |
| Entropy std | ~1.8 bits | 0.60 bits | 0.33x |
| Sink heads | 71/144 (49%) | 1/32 (3%) | N/A |

The critical architectural differences:
1. **Hybrid**: 24/32 layers use Gated Delta Net (linear attention) — no traditional KV cache
2. **GQA**: 4 KV heads serve 16 query heads per attention layer
3. **head_dim = 256**: 4x larger vectors than GPT-2
4. **Narrow entropy**: CV = 0.19 (Qwen) vs ~0.65 (GPT-2)

---

## 1. Effective Compression Ratio with Hybrid Architecture

### 1.1 The Real KV Cache Footprint

Qwen3.5-4B has 32 layers, but only 8 use traditional softmax attention with a standard KV cache. The other 24 layers use Gated Delta Net, a linear attention variant that maintains a fixed-size recurrent state rather than a growing token-by-token cache.

**Gated Delta Net state size:** The recurrent state is a matrix of shape $(d_k, d_v)$ per head — it does NOT grow with sequence length. For Qwen3.5-4B with head_dim = 256, this is $256 \times 256 = 65,536$ elements per head, regardless of whether the sequence is 100 or 100,000 tokens long.

**Full attention KV cache size:** For sequence length $n$, each full attention layer stores $n$ key vectors and $n$ value vectors, each of dimension $d = 256$, for each of $n_{kv} = 4$ KV heads. Per layer:

$$M_{\text{attn}}^{(\ell)} = 2 \cdot n_{kv} \cdot n \cdot d \cdot b_0 = 2 \cdot 4 \cdot n \cdot 256 \cdot b_0 = 2048 \cdot n \cdot b_0$$

**Linear attention state size:** Per layer:

$$M_{\text{linear}}^{(\ell)} = n_{kv} \cdot d^2 \cdot b_0 = 4 \cdot 256^2 \cdot b_0 = 262{,}144 \cdot b_0$$

This is constant in $n$. At $n = 256$ tokens, the linear state equals one attention layer's KV cache. At $n = 4096$, the attention cache is 16x larger per layer.

### 1.2 Total Memory Breakdown

Total KV-related memory:

$$M_{\text{total}} = \underbrace{8 \cdot 2048 \cdot n \cdot b_0}_{\text{8 attention layers}} + \underbrace{24 \cdot 262{,}144 \cdot b_0}_{\text{24 linear layers (constant)}}$$

$$M_{\text{total}} = b_0 \left(16{,}384 \cdot n + 6{,}291{,}456\right)$$

The fraction of total memory in the compressible attention KV cache:

$$f_{\text{attn}}(n) = \frac{16{,}384 \cdot n}{16{,}384 \cdot n + 6{,}291{,}456} = \frac{n}{n + 384}$$

| Sequence length $n$ | $f_{\text{attn}}$ | Attention KV cache fraction |
|---------------------|--------------------|-----------------------------|
| 128 | 0.25 | 25% |
| 384 | 0.50 | 50% |
| 1,024 | 0.73 | 73% |
| 4,096 | 0.91 | 91% |
| 16,384 | 0.98 | 98% |
| 262,144 (max) | 0.999 | 99.9% |

**Key insight:** At short sequences ($n < 384$), most memory is in the linear attention states, which are incompressible by our methods. At long sequences ($n > 4096$), almost all memory is in the attention KV cache, and compression is highly effective.

### 1.3 Effective Overall Compression Ratio

If we apply combined compression ratio $R_c = R_e \times R_q$ to the 8 attention layers and leave the 24 linear layers untouched:

$$M_{\text{compressed}} = \frac{8 \cdot 2048 \cdot n \cdot b_0}{R_c} + 24 \cdot 262{,}144 \cdot b_0$$

The effective overall compression ratio is:

$$R_{\text{eff}}(n, R_c) = \frac{M_{\text{total}}}{M_{\text{compressed}}} = \frac{n + 384}{\frac{n}{R_c} + 384}$$

| $R_c$ | $n = 128$ | $n = 1{,}024$ | $n = 4{,}096$ | $n = 16{,}384$ | $n = 262{,}144$ |
|--------|-----------|---------------|---------------|-----------------|-------------------|
| 2x | 1.14x | 1.37x | 1.65x | 1.88x | 1.998x |
| 4x | 1.21x | 1.59x | 2.47x | 3.37x | 3.995x |
| 8x | 1.24x | 1.69x | 3.06x | 5.54x | 7.982x |
| 16x | 1.25x | 1.74x | 3.42x | 7.35x | 15.93x |

**The hybrid architecture severely limits compression at short sequences but has negligible impact at long sequences.** Since Qwen3.5-4B's primary use case is long-context (262K max), the effective ratio approaches the raw ratio $R_c$ for practical deployments.

### 1.4 The Crossover Length

For a target effective compression $R_{\text{target}}$, the minimum sequence length where this is achievable:

$$n_{\min} = \frac{384 \cdot R_{\text{target}} \cdot (R_c - 1)}{R_c - R_{\text{target}}}$$

For $R_c = 8$ (our sweet spot from GPT-2):
- $R_{\text{target}} = 2$: $n_{\min} = 896$
- $R_{\text{target}} = 4$: $n_{\min} = 2{,}688$
- $R_{\text{target}} = 6$: $n_{\min} = 8{,}064$

**Practical implication:** For sequences longer than ~3K tokens, we can achieve 4x+ effective compression. For Qwen3.5-4B's intended long-context use cases (16K-256K), the hybrid architecture overhead is negligible.

---

## 2. GQA Impact on Entropy-Adaptive Eviction

### 2.1 The Constraint Problem

In GPT-2, each attention head has its own independent KV cache. Eviction decisions are per-head: if head $h$ has entropy $H(h)$, we set its retention budget $r_h$ proportional to $H(h)$ and evict tokens independently.

In Qwen3.5-4B with GQA (4:1 ratio), 4 query heads share each KV head. The KV cache stores 4 KV head entries per attention layer, but 16 query heads read from them. Specifically, query heads $\{4g, 4g+1, 4g+2, 4g+3\}$ share KV head $g$ for $g \in \{0,1,2,3\}$.

**The eviction decision for KV head $g$ must satisfy ALL 4 query heads simultaneously.** If we evict a token from KV head $g$, all 4 query heads in group $g$ lose access to that token.

### 2.2 Query Group Entropy Profiles

Let the 4 query heads sharing KV head $g$ have entropies $H_0^{(g)}, H_1^{(g)}, H_2^{(g)}, H_3^{(g)}$. The entropy config reports per-query-head entropies (16 heads labeled H0-H15 per layer). Grouping these into KV head groups of 4:

**Layer 3 (first attention layer):**

| KV head | Query heads | Entropies | Min | Max | Mean |
|---------|-------------|-----------|-----|-----|------|
| 0 | H0-H3 | 3.35, 3.54, 3.82, 3.43 | 3.35 | 3.82 | 3.54 |
| 1 | H4-H7 | 3.65, 4.11, 3.88, 3.80 | 3.65 | 4.11 | 3.86 |
| 2 | H8-H11 | 4.02, 3.79, 3.70, 3.76 | 3.70 | 4.02 | 3.82 |
| 3 | H12-H15 | 4.02, 2.78, 4.01, 2.48 | 2.48 | 4.02 | 3.32 |

**Layer 31 (last attention layer):**

| KV head | Query heads | Entropies | Min | Max | Mean |
|---------|-------------|-----------|-----|-----|------|
| 0 | H0-H3 | 3.79, 3.81, 3.82, 3.81 | 3.79 | 3.82 | 3.81 |
| 1 | H4-H7 | 3.89, 3.61, 3.49, 3.77 | 3.49 | 3.89 | 3.69 |
| 2 | H8-H11 | 1.65, 0.31, 1.63, 3.62 | 0.31 | 3.62 | 1.80 |
| 3 | H12-H15 | 0.62, 3.63, 2.79, 3.44 | 0.62 | 3.63 | 2.62 |

### 2.3 The Bottleneck Principle: Use Minimum Entropy

**Claim:** The optimal eviction budget for KV head $g$ should be determined by the **minimum** entropy across its query group:

$$r_g = f\left(\min_{j \in \text{group}(g)} H_j^{(g)}\right)$$

**Proof sketch:** The output error from eviction for the entire group is:

$$\mathcal{L}_{\text{group}(g)} = \sum_{j=0}^{3} \mathcal{L}_j^{(e)}(r_g)$$

Each query head's eviction error $\mathcal{L}_j^{(e)}(r_g)$ depends on how many of *its* high-attention tokens survive the shared eviction decision. A low-entropy (focused) query head concentrates attention on few tokens. If those tokens are evicted because the shared budget $r_g$ is too small, the focused head suffers catastrophic error — it loses its primary information source.

A high-entropy (diffuse) query head distributes attention broadly. Even if some tokens are evicted, the remaining tokens still carry substantial information. The error degrades gracefully.

Formally, the eviction error for head $j$ with entropy $H_j$ at retention ratio $r$ scales approximately as:

$$\mathcal{L}_j^{(e)}(r) \propto \max\left(0, 1 - r \cdot 2^{H_j}\right)^{\beta}$$

The argument $1 - r \cdot 2^{H_j}$ measures the fraction of the "effective support" (approximately $2^{H_j}$ tokens) that is lost. For a head with low $H_j$ (small effective support), even a moderate $r$ retains the full support. For a head with high $H_j$, the effective support is large and eviction begins to bite at higher $r$.

The head with the **lowest** $H_j$ has the smallest support and is the first to lose critical tokens as $r$ decreases. It is the bottleneck. Optimizing for the mean or max entropy would under-provision the focused head, leading to disproportionate error.

**Quantitative example from Layer 31, KV head 2:**

- Query heads: H8 ($H=1.65$), H9 ($H=0.31$), H10 ($H=1.63$), H11 ($H=3.62$)
- Head H9 has entropy 0.31 — it concentrates attention on ~$2^{0.31} \approx 1.24$ tokens
- At 50% eviction ($r = 0.5$), head H9's primary token has ~50% chance of being evicted
- Heads H8, H10 would need ~$2^{1.65} \approx 3.1$ tokens retained
- Head H11 would need ~$2^{3.62} \approx 12.3$ tokens retained

If we used mean entropy ($\bar{H} = 1.80$), the budget would be sized for ~3.5 tokens — adequate for H8 and H10, catastrophic for H9, wasteful for H11. Using min entropy ($H = 0.31$) ensures H9's dominant token is retained.

### 2.4 GQA Reduces the Effective Head Count for Eviction

With independent eviction per KV head group, we have effectively:

$$H_{\text{eff}} = n_{\text{attn\_layers}} \times n_{kv} = 8 \times 4 = 32 \text{ KV head groups}$$

vs. GPT-2's 144 independent heads. Fewer independent eviction decisions means:

1. **Less granularity** in entropy-adaptive allocation — the min-entropy bottleneck smooths out diversity within groups
2. **Higher average budget per group** — the bottleneck query head forces conservative retention
3. **Reduced benefit from adaptive eviction** — fewer distinct entropy levels to exploit

The within-group entropy spread quantifies the cost:

**Mean within-group entropy range across all 8 attention layers:**

Layer 3: ranges = [0.47, 0.46, 0.32, 1.54] → mean range 0.70
Layer 7: ranges = [0.60, 1.97, 0.50, 0.31] → mean range 0.84
Layer 11: ranges = [0.41, 0.83, 1.49, 0.16] → mean range 0.72
Layer 15: ranges = [1.16, 0.24, 0.01, 0.55] → mean range 0.49
Layer 19: ranges = [0.31, 0.44, 1.08, 1.23] → mean range 0.77
Layer 23: ranges = [0.71, 0.04, 0.04, 0.35] → mean range 0.29
Layer 27: ranges = [0.66, 0.07, 0.04, 0.69] → mean range 0.37
Layer 31: ranges = [0.03, 0.40, 3.31, 3.01] → mean range 1.69

**Overall mean within-group range: ~0.73 bits.** This entropy is "wasted" — it exists in the query heads but cannot be exploited because the KV cache is shared. In GPT-2, this diversity was fully exploitable.

### 2.5 Effective Entropy Distribution After GQA Bottleneck

Applying the min-entropy rule to each KV group, the effective entropy distribution for the 32 KV cache entries becomes:

| KV group | Min entropy | vs. Group mean |
|----------|-------------|----------------|
| L3_KV0 | 3.35 | 3.54 |
| L3_KV1 | 3.65 | 3.86 |
| L3_KV2 | 3.70 | 3.82 |
| L3_KV3 | 2.48 | 3.32 |
| L7_KV0 | 2.84 | 3.17 |
| L7_KV1 | 1.88 | 2.93 |
| L7_KV2 | 3.35 | 3.54 |
| L7_KV3 | 3.12 | 3.43 |
| L11_KV0 | 2.72 | 3.17 |
| L11_KV1 | 2.53 | 3.17 |
| L11_KV2 | 2.41 | 2.90 |
| L11_KV3 | 3.32 | 3.47 |
| L15_KV0 | 2.87 | 3.53 |
| L15_KV1 | 3.35 | 3.50 |
| L15_KV2 | 3.21 | 3.28 |
| L15_KV3 | 1.78 | 3.17 |
| L19_KV0 | 2.87 | 3.24 |
| L19_KV1 | 2.43 | 2.80 |
| L19_KV2 | 2.28 | 3.36 |
| L19_KV3 | 2.23 | 3.10 |
| L23_KV0 | 3.14 | 3.49 |
| L23_KV1 | 3.37 | 3.40 |
| L23_KV2 | 3.40 | 3.42 |
| L23_KV3 | 3.01 | 3.29 |
| L27_KV0 | 2.82 | 3.11 |
| L27_KV1 | 3.09 | 3.15 |
| L27_KV2 | 3.19 | 3.22 |
| L27_KV3 | 2.56 | 2.84 |
| L31_KV0 | 3.79 | 3.81 |
| L31_KV1 | 3.49 | 3.69 |
| L31_KV2 | 0.31 | 1.80 |
| L31_KV3 | 0.62 | 2.62 |

**Effective entropy statistics after GQA bottleneck (32 entries):**
- Mean: 2.81 bits (down from 3.23 raw)
- Min: 0.31 bits (L31_KV2, driven by the single sink head L31_H9)
- Max: 3.79 bits (L31_KV0)
- Std: ~0.72 bits (slightly higher than raw 0.60 due to the min-rule amplifying low outliers)
- Range: 0.31 - 3.79 = 3.48 bits

The bottleneck principle shifts the distribution downward and creates a slightly wider effective spread. The one true sink head (L31_H9, $H=0.31$) constrains its entire KV group (L31_KV2).

---

## 3. TurboQuant with head_dim = 256

### 3.1 Random Rotation and CLT Convergence

TurboQuant applies a random orthogonal rotation $R \in \mathbb{R}^{d \times d}$ to each value (or key) vector before scalar quantization. Each coordinate of the rotated vector $\tilde{v} = Rv$ is:

$$\tilde{v}_j = \sum_{i=1}^{d} R_{ji} v_i$$

By the Central Limit Theorem, for large $d$, $\tilde{v}_j$ converges to a Gaussian distribution (assuming the original coordinates have finite variance and are not pathologically correlated). The Berry-Esseen bound gives the rate of convergence:

$$\sup_x \left|P\left(\frac{\tilde{v}_j - \mu}{\sigma} \leq x\right) - \Phi(x)\right| \leq \frac{C \cdot \rho}{\sigma^3 \cdot \sqrt{d}}$$

where $\rho = \mathbb{E}[|v_i - \mu|^3]$ is the third absolute moment and $C \leq 0.4748$ (Shevtsova's constant).

**For head_dim = 64 (GPT-2):** CLT residual $\propto 1/\sqrt{64} = 0.125$
**For head_dim = 256 (Qwen):** CLT residual $\propto 1/\sqrt{256} = 0.0625$

The Gaussianity of the rotated coordinates improves by a factor of 2. This means Lloyd-Max quantization levels optimized for a Gaussian distribution are **closer to optimal** for Qwen3.5-4B than for GPT-2.

### 3.2 Lloyd-Max Quantization: MSE Scaling with Dimension

Lloyd-Max scalar quantization is applied independently to each coordinate of the rotated vector. The per-coordinate MSE at $b$ bits for a Gaussian source is:

$$\text{MSE}_{\text{scalar}}(b) = \sigma^2 \cdot c(b)$$

where $c(b)$ is the Lloyd-Max distortion coefficient (e.g., $c(1) = 0.3634$, $c(2) = 0.09497$, $c(3) = 0.03454$, $c(4) = 0.009497$).

The per-vector MSE is the sum over all $d$ coordinates:

$$\text{MSE}_{\text{vector}}(b, d) = d \cdot \sigma^2 \cdot c(b)$$

This scales **linearly** with $d$. For Qwen ($d = 256$) vs. GPT-2 ($d = 64$):

$$\frac{\text{MSE}_{\text{Qwen}}}{\text{MSE}_{\text{GPT2}}} = \frac{256}{64} = 4$$

**However**, the *relative* error (MSE divided by vector norm squared) is dimension-independent:

$$\frac{\text{MSE}_{\text{vector}}}{\|v\|^2} = \frac{d \cdot \sigma^2 \cdot c(b)}{d \cdot \sigma^2} = c(b)$$

The relative error depends only on the bit-width, not the dimension. This means TurboQuant's cosine similarity (which measures relative error) is **equally good** at $d = 256$ as at $d = 64$.

### 3.3 The Attention Output Error: Where Dimension Helps

Recall from the GPT-2 theory that the attention output error from quantization is:

$$\mathbb{E}[\|\epsilon_h\|^2] = \sigma_q^2 \cdot d \cdot 2^{-H_2(h)}$$

where $\sigma_q^2$ is the per-element MSE. The output is also a $d$-dimensional vector. The *relative* output error is:

$$\frac{\mathbb{E}[\|\epsilon_h\|^2]}{\mathbb{E}[\|o_h\|^2]} = \frac{\sigma_q^2 \cdot d \cdot 2^{-H_2(h)}}{\sigma_v^2 \cdot d} = \frac{\sigma_q^2}{\sigma_v^2} \cdot 2^{-H_2(h)} = c(b) \cdot 2^{-H_2(h)}$$

The dimension $d$ cancels in the relative error. **TurboQuant at a given bit-width produces the same relative output error regardless of head_dim.**

However, there is a second-order effect: the random rotation's CLT guarantee is tighter at $d = 256$, so $\sigma_q^2$ is closer to the optimal Lloyd-Max value for a true Gaussian. This provides a small additional quality benefit.

### 3.4 QJL Residual Correction: Dimension Scaling

The Quantized Johnson-Lindenstrauss (QJL) projection preserves inner products. For a random projection from $\mathbb{R}^d$ to $\mathbb{R}^m$, the JL guarantee states:

$$P\left[\left|\frac{\|Sx\|^2}{\|x\|^2} - 1\right| > \epsilon\right] \leq 2 \exp\left(-\frac{m(\epsilon^2 - \epsilon^3)}{4}\right)$$

The required projection dimension $m$ for a given $\epsilon$ and failure probability $\delta$ is:

$$m \geq \frac{4 \ln(2/\delta)}{\epsilon^2 - \epsilon^3}$$

**This bound is independent of the original dimension $d$.** The JL projection dimension depends only on the desired accuracy and confidence, not on the ambient dimension. This means QJL correction works identically at $d = 256$ and $d = 64$.

However, the total memory for QJL correction is:

$$M_{\text{QJL}} = n \cdot m \cdot b_{\text{QJL}}$$

where $n$ is the number of vectors (tokens) and $b_{\text{QJL}}$ is the bits per projected coordinate. Since $m$ is independent of $d$, the QJL memory overhead as a **fraction** of the total KV cache is:

$$\frac{M_{\text{QJL}}}{M_{\text{KV}}} = \frac{m \cdot b_{\text{QJL}}}{d \cdot b_{\text{main}}}$$

At $d = 256$, this fraction is 4x smaller than at $d = 64$. **QJL residual correction is relatively cheaper at larger head_dim.**

### 3.5 Rate-Distortion Perspective

The Shannon rate-distortion function for an i.i.d. Gaussian source of dimension $d$ at rate $R$ bits per element gives:

$$D(R) = \sigma^2 \cdot 2^{-2R}$$

This is per-element and dimension-independent. The total distortion for a $d$-dimensional vector at $R$ bits per element and $Rd$ total bits is $d \cdot \sigma^2 \cdot 2^{-2R}$ — linear in $d$ at fixed rate.

If instead we measure total bits per vector (budget $B = Rd$ bits), then at $d=256$, each element gets $B/256$ bits vs. $B/64$ at GPT-2's dimension. **At fixed total bits per vector, larger $d$ is worse** — more elements share the same budget. But at fixed bits per element (the natural parameterization for scalar quantization), dimension doesn't matter.

**Bottom line:** TurboQuant operates at fixed bits per element, so head_dim = 256 is neither fundamentally better nor worse than head_dim = 64 for the quantization itself. The secondary effects (better CLT, cheaper QJL overhead) provide small advantages to the larger dimension.

### 3.6 Summary: head_dim = 256 vs. 64

| Aspect | Effect of $d = 256$ vs $d = 64$ | Magnitude |
|--------|----------------------------------|-----------|
| CLT convergence | Better (2x tighter) | Small: ~2% better Gaussianity |
| Lloyd-Max relative error | Same | Neutral |
| Attention output relative error | Same | Neutral |
| QJL overhead as fraction of KV | 4x lower | Moderate: cheaper residual correction |
| Absolute vector MSE | 4x higher | Irrelevant (relative error unchanged) |
| Quantization memory savings | Same rate | Neutral |

**Verdict: head_dim = 256 is marginally BETTER for TurboQuant.** The improvements are second-order (better CLT, cheaper QJL), but there are no disadvantages. The core quantization quality at a given bit-rate is dimension-independent.

---

## 4. Entropy Distribution Analysis

### 4.1 Qwen vs. GPT-2 Entropy Structure

**GPT-2 (144 heads):** Bimodal distribution.
- 49% sink heads ($H < 1.0$): concentrated near 0
- 17% diffuse heads ($H > 4.0$): concentrated near 5-6
- 34% moderate/mixed: spread across 1-4
- Coefficient of variation: ~0.65
- Entropy range: 5.92 bits (0.03 to 5.95)

**Qwen3.5-4B (128 heads across 8 attention layers, reported per query head):** Unimodal distribution.
- Only 1 sink head (L31_H9, $H = 0.31$) — 0.8% of heads
- Only 3 low-entropy heads ($H < 2.0$): L7_H4, L15_H13, L31_H8, L31_H9, L31_H10, L31_H12 — ~5%
- 5 diffuse heads ($H > 4.0$): L3_H5, L3_H8, L3_H12, L3_H14, L15_H0 — ~4%
- ~91% of heads cluster in [2.2, 4.1] — a 1.9-bit range
- Coefficient of variation: 0.19
- Entropy range: 3.80 bits (0.31 to 4.11)

### 4.2 Impact on Entropy-Adaptive Eviction

The benefit of entropy-adaptive eviction comes from differential allocation: giving high-entropy heads more tokens and low-entropy heads fewer. The savings come from the low-entropy heads that need very few tokens.

**GPT-2:** 49% of heads are sinks needing ~1-2 tokens. This frees massive budget for the 17% diffuse heads. The reallocation is dramatic — a 200:1 ratio between max and min budgets is optimal.

**Qwen3.5-4B (after GQA bottleneck, 32 KV groups):** Only 2/32 KV groups have min-entropy below 1.0 (L31_KV2 at 0.31, L31_KV3 at 0.62). The other 30 groups have min-entropy between 1.88 and 3.79.

The budget freed by the 2 low-entropy groups:
- L31_KV2 needs ~$2^{0.31} \approx 1.2$ tokens → saves $n - 1$ tokens per position
- L31_KV3 needs ~$2^{0.62} \approx 1.5$ tokens → saves $n - 2$ tokens per position
- The remaining 30 groups need $2^{1.88}$ to $2^{3.79}$ = 3.7 to 13.8 tokens minimum

**Expected benefit reduction:** Define the entropy-adaptive gain as the ratio of uniform-budget error to adaptive-budget error at the same total memory. For GPT-2, this was large because reallocating from 49% sinks provided massive savings. For Qwen:

The optimal budget for KV group $g$ is proportional to $2^{H_g}$ (retain enough tokens to cover the effective support). The uniform budget is $\bar{r} = n / R_e$ for all groups. The entropy-adaptive budget is:

$$r_g^* = \frac{2^{H_g}}{\sum_{g'} 2^{H_{g'}}} \cdot \frac{32 \cdot n}{R_e}$$

For GPT-2, $\text{Var}[2^{H_h}]$ was enormous due to bimodality (some heads with $2^0 = 1$, others with $2^6 = 64$). For Qwen, $\text{Var}[2^{H_g}]$ is much smaller because most $H_g$ cluster in [2.2, 3.8], giving $2^{H_g} \in [4.6, 13.9]$ — only a 3:1 range.

**Estimated adaptive gain for Qwen:** The Kullback-Leibler divergence reduction from entropy-adaptive allocation depends on the variance of the optimal budget. For a rough estimate, the improvement factor is approximately:

$$G_{\text{adaptive}} \approx 1 + \text{CV}^2(2^{H_g})$$

where CV is the coefficient of variation of $2^{H_g}$.

For GPT-2: $\text{CV}(2^H) \approx 2.5$ → $G \approx 7.25$ (substantial)
For Qwen: $\text{CV}(2^H) \approx 0.6$ (estimated from range 4.6-13.9 with most values in 6-10) → $G \approx 1.36$

**Entropy-adaptive eviction provides roughly 36% quality improvement over uniform eviction for Qwen, vs. ~600% for GPT-2.** The narrower entropy distribution limits the gains.

### 4.3 Impact on Entropy-Informed Bit Allocation

From the GPT-2 theory, the optimal bit allocation is:

$$b_h^* = \bar{b} + \frac{\bar{H}_2 - H_2(h)}{2}$$

The range of reallocation is $(\max H_2 - \min H_2) / 2$. For GPT-2, this was $(5.95 - 0.03)/2 = 2.96$ bits — nearly a 3-bit spread around the mean.

For Qwen (effective, after GQA min-rule): $(\max H - \min H)/2 = (3.79 - 0.31)/2 = 1.74$ bits. But this range is driven almost entirely by two outlier KV groups. For the central 90% of groups ($H \in [1.88, 3.79]$), the range is $(3.79 - 1.88)/2 = 0.96$ bits — less than 1 bit of reallocation.

**Expected BLEU improvement from adaptive vs. uniform bit allocation:**

The total error with optimal allocation vs. uniform allocation at mean rate $\bar{b}$:

$$\frac{\mathcal{L}_{\text{uniform}}}{\mathcal{L}_{\text{adaptive}}} = \frac{\sum_g 2^{-2\bar{b} - H_g}}{\sum_g 2^{-2b_g^* - H_g}}$$

With optimal $b_g^* = \bar{b} + (\bar{H} - H_g)/2$:

$$2^{-2b_g^* - H_g} = 2^{-2\bar{b} - \bar{H}}$$

This is constant across groups. So:

$$\frac{\mathcal{L}_{\text{uniform}}}{\mathcal{L}_{\text{adaptive}}} = \frac{\sum_g 2^{-H_g}}{32 \cdot 2^{-\bar{H}}} = \frac{\sum_g 2^{-H_g}}{32 \cdot 2^{-\bar{H}}}$$

By Jensen's inequality (since $2^{-x}$ is convex), $\frac{1}{32}\sum_g 2^{-H_g} \geq 2^{-\bar{H}}$, so this ratio is $\geq 1$ (adaptive always wins or ties).

Computing numerically with our 32 KV groups:

$$\sum_g 2^{-H_g} \approx 2^{-0.31} + 2^{-0.62} + \sum_{\text{other 30}} 2^{-H_g}$$

The two outliers contribute $2^{-0.31} + 2^{-0.62} = 0.807 + 0.650 = 1.457$.
The 30 central groups with mean $H \approx 3.0$ contribute approximately $30 \times 2^{-3.0} = 30 \times 0.125 = 3.75$.
Total: $\sum_g 2^{-H_g} \approx 5.21$.
Mean: $\bar{H} \approx 2.81$, so $32 \times 2^{-2.81} = 32 \times 0.142 = 4.55$.

$$\frac{\mathcal{L}_{\text{uniform}}}{\mathcal{L}_{\text{adaptive}}} \approx \frac{5.21}{4.55} = 1.145$$

**Adaptive bit allocation provides ~14.5% error reduction for Qwen.** This is modest compared to GPT-2 where the bimodal distribution created much larger gains. However, 14.5% is still meaningful — it corresponds roughly to a 0.2-bit average precision improvement "for free."

### 4.4 The L31 Anomaly

Layer 31 (the final attention layer) contains the only entropy outliers in the entire model:
- L31_H9: $H = 0.31$ (the sole sink head)
- L31_H12: $H = 0.62$ (strongly focused)
- L31_H8: $H = 1.65$
- L31_H10: $H = 1.63$

These 4 heads are in KV groups 2 and 3 of layer 31. The remaining 124 query heads all have entropy > 1.78.

**Hypothesis:** The last attention layer performs a specialized role — perhaps sharp retrieval of specific tokens for the final prediction. The other 7 attention layers all have homogeneous, moderate entropy. This layer structure is consistent with Qwen3.5-4B's hybrid design: the Gated Delta Net layers handle broad context aggregation (normally done by high-entropy attention heads), leaving the few attention layers to handle precise, content-addressable retrieval (normally moderate entropy) — except the very last layer which also does sharp lookback.

**Implication for compression:** Layer 31 KV groups 2 and 3 should receive special treatment: high retention budgets (for the sink/focused query heads) and high bit allocation (to preserve the precise retrieval). The other 28 KV groups can be treated uniformly with moderate settings.

---

## 5. Double Quantization: Q4_K_M Weights + Quantized KV Cache

### 5.1 The Noise Chain

The Qwen3.5-4B GGUF model uses Q4_K_M weight quantization (4-bit with k-quant mixed precision). The KV values and keys are computed as:

$$K = X W_K + b_K, \quad V = X W_V + b_V$$

where $W_K, W_V$ are the (quantized) projection weights and $X$ is the input hidden state. With weight quantization error $\Delta W$:

$$\hat{K} = X(W_K + \Delta W_K) + b_K = K + X \Delta W_K$$

The KV vectors already contain weight-quantization noise $\eta = X \Delta W_K$ before any KV cache compression is applied.

### 5.2 Variance Analysis

The total variance of a KV vector element is:

$$\text{Var}[\hat{K}_{ij}] = \text{Var}[K_{ij}] + \text{Var}[\eta_{ij}] + 2\text{Cov}[K_{ij}, \eta_{ij}]$$

For Q4_K_M, the weight quantization error is approximately:
- Per-weight MSE: $\sigma_W^2 \approx c_W \cdot \bar{W}^2$ where $c_W$ depends on the quantization scheme
- The Q4_K_M scheme uses mixed 4-bit and 6-bit blocks, with typical reported MSE roughly 0.1-0.5% of weight variance

The covariance term $\text{Cov}[K, \eta]$ is generally small because $\eta = X\Delta W$ and $\Delta W$ is approximately uncorrelated with $W$ for symmetric quantization.

**Does weight quantization increase KV variance?**

$$\text{Var}[\hat{K}] = \text{Var}[K] + \text{Var}[\eta] \geq \text{Var}[K]$$

Yes, always. Weight quantization adds noise variance. The question is whether the increase is large enough to matter.

For Q4_K_M with ~0.3% weight MSE relative to weight variance, and assuming $\|X\|^2$ is moderate:

$$\frac{\text{Var}[\eta]}{\text{Var}[K]} \approx \frac{\text{Var}[X\Delta W]}{\text{Var}[XW]} \approx \frac{\sigma_{\Delta W}^2}{\sigma_W^2} \approx 0.003$$

The KV variance increases by ~0.3%. This is negligible.

### 5.3 The Noise Floor

When we further quantize the KV cache to $b$ bits per element, we add quantization noise $\sigma_q^2 = c(b) \cdot \text{Var}[\hat{K}]$. The total noise in the final KV representation is:

$$\sigma_{\text{total}}^2 = \underbrace{\sigma_{\eta}^2}_{\text{weight quant}} + \underbrace{\sigma_q^2}_{\text{KV quant}} = \sigma_{\eta}^2 + c(b) \cdot (\sigma_K^2 + \sigma_{\eta}^2)$$

For KV quantization to be the dominant error source:

$$c(b) \cdot \sigma_K^2 \gg \sigma_{\eta}^2$$

At 4-bit KV quantization: $c(4) \approx 0.0095$. With $\sigma_{\eta}^2 / \sigma_K^2 \approx 0.003$:

$$c(4) = 0.0095 \gg 0.003 = \sigma_{\eta}^2 / \sigma_K^2$$

At 4-bit, KV quantization error (0.95%) is about 3x the weight quantization error (0.3%). KV quantization dominates.

At 3-bit: $c(3) \approx 0.035$, which is 12x the weight error. Still dominated by KV quantization.

**Is there a noise floor below which further KV quantization is pointless?**

The noise floor is at:

$$c(b_{\text{floor}}) \approx \frac{\sigma_{\eta}^2}{\sigma_K^2} \approx 0.003$$

This corresponds to approximately $b_{\text{floor}} \approx 4.2$ bits (interpolating the Lloyd-Max table). Below 4.2 bits, KV quantization dominates. Above 4.2 bits, the weight quantization noise is already the bottleneck, and higher KV precision provides diminishing returns.

**This means 4-bit KV quantization is near-optimal for a Q4_K_M model.** Going to 5 or 6-bit KV would provide minimal improvement because the weight quantization noise is already comparable. Going to 3-bit or 2-bit KV would degrade quality significantly beyond what weight quantization already introduces.

### 5.4 Optimal Bit Allocation with Pre-Existing Weight Noise

The optimal bit allocation formula must account for the noise floor:

$$\mathcal{L}_h^{(q)}(b_h) = \left(c(b_h) + \frac{\sigma_{\eta}^2}{\sigma_K^2}\right) \cdot d \cdot 2^{-H_2(h)}$$

Minimizing total error subject to $\sum_h b_h = B$ now gives:

$$b_h^* = \bar{b} + \frac{\bar{H}_2 - H_2(h)}{2} \cdot \frac{c(b_h)}{c(b_h) + \sigma_{\eta}^2/\sigma_K^2}$$

The correction factor $c(b_h)/(c(b_h) + \sigma_{\eta}^2/\sigma_K^2)$ dampens the reallocation when the noise floor is significant relative to quantization error. At $b = 4$ bits, this factor is $0.0095 / (0.0095 + 0.003) = 0.76$.

**Weight quantization reduces the optimal bit reallocation range by ~24%.** The entropy-informed allocation is still directionally correct (give focused heads more bits) but the magnitude of reallocation should be more conservative — roughly $0.76 \times 0.5 = 0.38$ bits per unit of entropy difference instead of 0.5 bits.

---

## 6. Predicted Performance

### 6.1 Memory Model for Qwen3.5-4B

Per attention layer, the KV cache stores $n_{kv} = 4$ heads of dimension $d = 256$ for both keys and values. At baseline precision $b_0$ bits per element:

$$M_{\text{attn-layer}} = 2 \times 4 \times n \times 256 \times b_0 = 2048 \cdot n \cdot b_0 \text{ bits}$$

For 8 attention layers total:

$$M_{\text{attn-total}} = 16{,}384 \cdot n \cdot b_0 \text{ bits}$$

At $b_0 = 16$ bits and $n = 4096$ tokens:

$$M_{\text{attn-total}} = 16{,}384 \times 4{,}096 \times 16 = 1.074 \text{ Gbit} = 128 \text{ MB}$$

(Plus ~6.3 Mbit = 0.75 MB for the 24 linear attention states — negligible at this sequence length.)

### 6.2 Predicted Compression-Quality Tradeoff

Based on the GPT-2 empirical results, adapted for Qwen's architecture:

**Adjustments from GPT-2 baseline:**
1. **GQA bottleneck** reduces adaptive eviction benefit: ~36% improvement vs ~600% for GPT-2
2. **Narrow entropy** limits differential bit allocation: ~14.5% error reduction vs ~50%+ for GPT-2
3. **Q4_K_M noise floor** caps the benefit of high KV precision at ~4.2 bits
4. **head_dim = 256** provides marginal TurboQuant improvement (~2% better CLT)
5. **Only 8/32 layers compressible** but negligible at long sequences ($n > 4K$)

**Predicted configurations (assuming $n \geq 4096$ for negligible hybrid overhead):**

| Configuration | $R_e$ | Bits | $R_c$ | $R_{\text{eff}}$ | Predicted BLEU range | Confidence |
|---------------|--------|------|--------|-------------------|---------------------|------------|
| Conservative | 2x | 4-bit | 8x | ~7.3x | 0.55 - 0.70 | High |
| Sweet spot | 2x | 4-bit + adaptive | 8x | ~7.3x | 0.60 - 0.75 | Medium |
| Aggressive | 3x | 4-bit | 12x | ~10x | 0.40 - 0.55 | Medium |
| Maximum | 3x | 3-bit | 16x | ~12x | 0.20 - 0.35 | Low |
| Extreme | 4x | 3-bit | 21x | ~15x | 0.10 - 0.25 | Low |

### 6.3 Optimal Eviction + Quantization Split

The GPT-2 analysis found the optimal split at $R_e \approx 2\text{-}3\times$ eviction + 3-5 bit quantization. For Qwen:

**The noise floor shifts the optimum toward lower KV bit-rates:**

Since Q4_K_M weights already inject noise equivalent to ~4.2-bit KV quantization, going beyond 4-bit KV precision (5, 6, or 8-bit) provides diminishing returns. This means the quantization axis "maxes out" at 4-bit, and any additional compression should come from eviction.

**The narrower entropy distribution weakens adaptive eviction:**

With less entropy diversity, the eviction budget reallocation provides smaller gains. Uniform eviction performs closer to adaptive eviction than in GPT-2.

**Predicted optimal split for Qwen at various total compression targets:**

| Target $R_c$ | Optimal $R_e$ | Optimal bits | Rationale |
|---------------|----------------|--------------|-----------|
| 4x | 1x (no eviction) | 4-bit | Below noise floor — just quantize |
| 8x | 2x | 4-bit | GPT-2 sweet spot, confirmed by noise floor |
| 12x | 3x | 4-bit | Push eviction, hold precision |
| 16x | 4x | 4-bit | Keep hitting eviction, 4-bit floor |
| 24x | 3x | 2-bit | Diminishing returns on eviction, must cut bits |

**The Q4_K_M noise floor makes 4-bit KV a natural anchor point.** The optimal strategy for Qwen is: fix KV at 4-bit, tune eviction ratio for the desired compression level.

### 6.4 Expected Improvement from Entropy-Informed Bit Allocation

From Section 4.3, the theoretical error reduction is 14.5%. Given the noise floor dampening factor of 0.76 from Section 5.4:

$$\text{Effective gain} = 1 + (0.145 - 1) \times 0.76 \approx 11\% \text{ error reduction}$$

In BLEU terms, this translates to approximately:
- At 8x compression (BLEU ~0.60): adaptive allocation → BLEU ~0.63 (+0.03)
- At 12x compression (BLEU ~0.45): adaptive allocation → BLEU ~0.48 (+0.03)

**The gain is real but modest.** It may not justify the implementation complexity of variable-rate quantization. A simpler approach — just giving the 2 anomalous L31 KV groups fixed 6-bit allocation and everything else 4-bit — would capture most of the benefit.

### 6.5 Synergy Prediction: Stronger or Weaker Than GPT-2?

The super-additive synergy in GPT-2 arose from two mechanisms:
1. **Eviction removes noisy tokens** — tokens with low attention contribute more quantization noise per unit of information than they carry
2. **Eviction concentrates the value distribution** — fewer, more coherent vectors quantize better (Lloyd-Max calibration improves)

For Qwen, both mechanisms still apply but are modulated:

**Mechanism 1 (noise removal) — WEAKER:**
With only 1 sink head (vs 49% in GPT-2), there are far fewer "obvious waste" tokens to remove. Most Qwen attention heads already attend moderately broadly, so evicted tokens are less clearly "noise."

**Mechanism 2 (distribution concentration) — SIMILAR:**
The post-eviction value distribution is more concentrated regardless of entropy profile. This mechanism depends on the value vector statistics, not entropy structure.

**Additional factor: Q4_K_M noise floor — WEAKENS synergy:**
The weight quantization noise is always present. Eviction removes tokens but doesn't reduce weight noise in the surviving vectors. The irreducible noise floor limits how much eviction can "clean up" the quantization.

**Prediction: The super-additive synergy will be WEAKER for Qwen than for GPT-2.**

Quantitatively, define the synergy factor:

$$S = \frac{\Delta Q_e + \Delta Q_q}{\Delta Q_{combined}}$$

where $\Delta Q$ is quality loss from baseline. For GPT-2 at 8x compression, the data gives:

$$S_{\text{GPT2}} = \frac{(1 - 0.811) + (1 - 0.115)}{1 - 0.596} = \frac{0.189 + 0.885}{0.404} = 2.66$$

(Combined loss is 2.66x smaller than the sum of individual losses.)

For Qwen, predicting at 8x compression (2x eviction + 4-bit):
- Eviction 2x alone: BLEU ~0.82 (similar to GPT-2 since the mechanism is similar)
- Quant 2-bit (for 8x pure quant): BLEU ~0.15 (slightly better due to head_dim CLT)
- Combined 2x+4bit: BLEU ~0.62

$$S_{\text{Qwen}} \approx \frac{(1 - 0.82) + (1 - 0.15)}{1 - 0.62} = \frac{0.18 + 0.85}{0.38} = 2.71$$

**Wait — the synergy factor is actually similar.** The weaker entropy-adaptive benefit is offset by the stronger eviction-before-quantization cleanup at 4-bit (which is near the noise floor, so removing noisy tokens matters more). The prediction is:

$$\boxed{S_{\text{Qwen}} \approx 2.0 \text{ to } 3.0}$$

Comparable to GPT-2's $S = 2.66$, but with wider uncertainty due to the untested architecture.

### 6.6 Best Achievable Compression with BLEU > 0.5

From the predicted performance table, the constraint BLEU > 0.5 limits us to:

| Configuration | $R_c$ | $R_{\text{eff}}$ ($n=4096$) | Predicted BLEU |
|---------------|--------|-------------------------------|----------------|
| 2x eviction + 4-bit | 8x | 7.3x | 0.55 - 0.70 |
| 2x eviction + 4-bit adaptive | 8x | 7.3x | 0.60 - 0.75 |
| 3x eviction + 4-bit | 12x | 10x | 0.40 - 0.55 (borderline) |

**Best compression ratio achieving BLEU > 0.5 with high confidence: 8x (2x eviction + 4-bit quantization).**

With entropy-adaptive bit allocation and some optimism on the eviction front, 10-12x may be achievable while maintaining BLEU > 0.5 — but this requires experimental validation.

At long sequences ($n = 64K$), the effective ratio approaches the raw ratio, so 8x raw gives ~8x effective.

---

## 7. Summary of Key Theoretical Results

| # | Result | Section | Qwen-specific implication |
|---|--------|---------|---------------------------|
| 1 | Hybrid architecture limits compression at short sequences | 1.2-1.4 | At $n < 384$, effective compression is < 2x regardless of technique. At $n > 4K$, negligible. |
| 2 | GQA eviction uses min-entropy bottleneck across query group | 2.3 | 32 effective eviction units (not 128). Budget set by the most focused query head per group. |
| 3 | Entropy-adaptive gain is ~36% for Qwen vs ~600% for GPT-2 | 4.2 | Narrow entropy distribution (CV=0.19) limits adaptive allocation benefit. |
| 4 | head_dim=256 is marginally better for TurboQuant | 3.6 | 2x tighter CLT, 4x cheaper QJL overhead. Core relative error unchanged. |
| 5 | Q4_K_M creates noise floor at ~4.2 bits KV | 5.3 | 4-bit KV is the natural operating point. Higher precision wastes bits fighting weight noise. |
| 6 | Noise floor dampens optimal bit reallocation by ~24% | 5.4 | Use 0.38 bits per entropy-bit, not 0.5. |
| 7 | Adaptive bit allocation gives ~11% effective error reduction | 6.4 | Modest but real. Simplify: 6-bit for L31 KV2/KV3, 4-bit everywhere else. |
| 8 | Optimal strategy: fix 4-bit KV, vary eviction | 6.3 | Noise floor anchors quantization; eviction is the free axis. |
| 9 | Super-additive synergy comparable to GPT-2 ($S \approx 2.0\text{-}3.0$) | 6.5 | Mechanism shifts: less entropy-driven, more noise-removal-driven. |
| 10 | Best BLEU>0.5 compression: 8x (2x eviction + 4-bit) | 6.6 | Same sweet spot as GPT-2, anchored by the noise floor. |

---

## 8. Experimental Priorities

### Tier 1: Must-Run (validate core theory)

1. **Baseline KV cache profiling.** Measure actual KV vector statistics (mean, variance, distribution shape) from the 8 attention layers under Q4_K_M inference. Verify the noise floor estimate of ~0.3% relative variance.

2. **Combined 2x eviction + 4-bit at $n = 4096$.** The predicted sweet spot. Compare against pure eviction 2x and pure quantization 4-bit independently to measure the synergy factor $S$.

3. **Layer 31 ablation.** Test compression with layer 31 KV groups 2-3 exempted (full precision, no eviction) vs. compressed uniformly. This validates the L31 anomaly hypothesis.

### Tier 2: Optimization (if Tier 1 validates)

4. **GQA eviction rule comparison.** Test min-entropy vs. mean-entropy vs. max-entropy for setting group eviction budgets. The theory predicts min is optimal.

5. **Adaptive bit allocation: simple scheme.** 6-bit for L31 KV2/KV3, 4-bit for rest. Compare against uniform 4-bit. Expected gain: ~5-8% of the 11% theoretical maximum (since most of the non-uniformity benefit comes from the 2 outlier groups).

6. **Eviction ratio sweep.** Test $R_e \in \{1.5, 2, 2.5, 3, 4\}$ at fixed 4-bit quantization to locate the quality cliff for eviction.

### Tier 3: Advanced (research value)

7. **Sequence length scaling.** Test at $n \in \{512, 2048, 8192, 32768\}$ to measure how the hybrid architecture crossover affects real-world compression.

8. **3-bit quantization viability.** Test whether head_dim=256's CLT benefit meaningfully shifts the 4-bit→3-bit quality cliff observed in GPT-2.

---

## Appendix A: Full Entropy Data by KV Group

Each KV group consists of 4 query heads. Entropies are Shannon entropy of the attention distribution, averaged over calibration positions.

### Layer 3 (full attention)
| KV Group | Q-Heads | Entropies | Min | Max | Mean |
|----------|---------|-----------|-----|-----|------|
| KV0 | H0-H3 | 3.35, 3.54, 3.82, 3.43 | 3.35 | 3.82 | 3.54 |
| KV1 | H4-H7 | 3.65, 4.11, 3.88, 3.80 | 3.65 | 4.11 | 3.86 |
| KV2 | H8-H11 | 4.02, 3.79, 3.70, 3.76 | 3.70 | 4.02 | 3.82 |
| KV3 | H12-H15 | 4.02, 2.78, 4.01, 2.48 | 2.48 | 4.02 | 3.32 |

### Layer 7
| KV Group | Q-Heads | Entropies | Min | Max | Mean |
|----------|---------|-----------|-----|-----|------|
| KV0 | H0-H3 | 3.60, 2.84, 3.21, 3.05 | 2.84 | 3.60 | 3.17 |
| KV1 | H4-H7 | 1.88, 1.99, 3.40, 2.87 | 1.88 | 3.40 | 2.54 |
| KV2 | H8-H11 | 3.35, 3.85, 3.42, 3.54 | 3.35 | 3.85 | 3.54 |
| KV3 | H12-H15 | 3.73, 3.44, 3.39, 3.12 | 3.12 | 3.73 | 3.42 |

### Layer 11
| KV Group | Q-Heads | Entropies | Min | Max | Mean |
|----------|---------|-----------|-----|-----|------|
| KV0 | H0-H3 | 3.02, 3.18, 3.28, 3.43 | 3.02 | 3.43 | 3.23 |
| KV1 | H4-H7 | 3.91, 3.08, 2.53, 3.26 | 2.53 | 3.91 | 3.19 |
| KV2 | H8-H11 | 2.72, 3.90, 2.41, 2.53 | 2.41 | 3.90 | 2.89 |
| KV3 | H12-H15 | 3.57, 3.32, 3.15, 3.48 | 3.15 | 3.57 | 3.38 |

### Layer 15
| KV Group | Q-Heads | Entropies | Min | Max | Mean |
|----------|---------|-----------|-----|-----|------|
| KV0 | H0-H3 | 4.02, 2.87, 3.48, 3.74 | 2.87 | 4.02 | 3.53 |
| KV1 | H4-H7 | 3.53, 3.55, 3.58, 3.35 | 3.35 | 3.58 | 3.50 |
| KV2 | H8-H11 | 3.22, 3.34, 3.21, 3.36 | 3.21 | 3.36 | 3.28 |
| KV3 | H12-H15 | 3.08, 1.78, 2.98, 3.63 | 1.78 | 3.63 | 2.87 |

### Layer 19
| KV Group | Q-Heads | Entropies | Min | Max | Mean |
|----------|---------|-----------|-----|-----|------|
| KV0 | H0-H3 | 3.42, 3.48, 2.87, 3.19 | 2.87 | 3.48 | 3.24 |
| KV1 | H4-H7 | 3.10, 2.43, 2.74, 2.88 | 2.43 | 3.10 | 2.79 |
| KV2 | H8-H11 | 3.79, 3.58, 3.25, 3.43 | 3.25 | 3.79 | 3.51 |
| KV3 | H12-H15 | 2.28, 3.51, 2.23, 3.19 | 2.23 | 3.51 | 2.80 |

### Layer 23
| KV Group | Q-Heads | Entropies | Min | Max | Mean |
|----------|---------|-----------|-----|-----|------|
| KV0 | H0-H3 | 3.57, 3.40, 3.14, 3.85 | 3.14 | 3.85 | 3.49 |
| KV1 | H4-H7 | 3.43, 3.37, 3.41, 3.39 | 3.37 | 3.43 | 3.40 |
| KV2 | H8-H11 | 3.44, 3.40, 3.41, 3.46 | 3.40 | 3.46 | 3.43 |
| KV3 | H12-H15 | 3.36, 3.39, 3.41, 3.01 | 3.01 | 3.41 | 3.29 |

### Layer 27
| KV Group | Q-Heads | Entropies | Min | Max | Mean |
|----------|---------|-----------|-----|-----|------|
| KV0 | H0-H3 | 2.95, 2.82, 3.47, 3.19 | 2.82 | 3.47 | 3.11 |
| KV1 | H4-H7 | 3.10, 3.16, 3.09, 3.16 | 3.09 | 3.16 | 3.13 |
| KV2 | H8-H11 | 3.19, 3.25, 3.22, 3.23 | 3.19 | 3.25 | 3.22 |
| KV3 | H12-H15 | 2.90, 2.60, 3.29, 2.56 | 2.56 | 3.29 | 2.84 |

### Layer 31
| KV Group | Q-Heads | Entropies | Min | Max | Mean |
|----------|---------|-----------|-----|-----|------|
| KV0 | H0-H3 | 3.79, 3.81, 3.82, 3.81 | 3.79 | 3.82 | 3.81 |
| KV1 | H4-H7 | 3.89, 3.61, 3.49, 3.77 | 3.49 | 3.89 | 3.69 |
| KV2 | H8-H11 | 1.65, 0.31, 1.63, 3.62 | **0.31** | 3.62 | 1.80 |
| KV3 | H12-H15 | 0.62, 3.63, 2.79, 3.44 | **0.62** | 3.63 | 2.62 |

---

## Appendix B: Notation Reference

| Symbol | Meaning |
|--------|---------|
| $n$ | Sequence length (number of tokens in KV cache) |
| $d$ | Head dimension (256 for Qwen3.5-4B) |
| $n_{kv}$ | Number of KV heads per attention layer (4) |
| $n_{q}$ | Number of query heads per attention layer (16) |
| $R_e$ | Eviction compression ratio |
| $R_q$ | Quantization compression ratio ($b_0 / b$) |
| $R_c$ | Combined compression ratio ($R_e \times R_q$) |
| $R_{\text{eff}}$ | Effective overall ratio including linear attention layers |
| $H(h)$, $H_2(h)$ | Shannon / Renyi-2 entropy of head $h$'s attention distribution |
| $b_0$ | Baseline bits per element (16) |
| $b_h$ | Allocated bits per element for head $h$ |
| $c(b)$ | Lloyd-Max distortion coefficient at $b$ bits for Gaussian source |
| $\sigma_q^2$ | KV quantization MSE per element |
| $\sigma_{\eta}^2$ | Weight quantization noise variance in KV vectors |
| $f_{\text{attn}}(n)$ | Fraction of total KV-related memory in attention layers |
| $S$ | Synergy factor: (sum of individual losses) / (combined loss) |
| $G_{\text{adaptive}}$ | Gain from entropy-adaptive vs. uniform allocation |
