# Red Team Review: Qwen3.5-4B KV Cache Compression Theory

**Date:** 2026-03-27
**Reviewed document:** `QWEN_COMPRESSION_THEORY.md`
**Supporting documents:** `COMBINED_COMPRESSION_THEORY.md`, `COMBINED_COMPRESSION_RESULTS.md`, `entropy_config.json`
**Status:** Adversarial mathematical review

---

## 1. Errors Found

### ERROR 1 (Critical): R_eff Table Values Are Wrong

**Section 1.3**, the table of effective compression ratios is internally inconsistent with the formula presented immediately above it.

The formula is:

$$R_{\text{eff}}(n, R_c) = \frac{n + 384}{\frac{n}{R_c} + 384}$$

This formula is correct and consistent with the memory breakdown in Section 1.2 (verified: 16384 * 384 = 6,291,456). However, the table values were computed with a **different, unknown formula**. Every interior cell is wrong. Examples:

| R_c | n | Paper claims | Formula gives | Error |
|-----|---|-------------|---------------|-------|
| 8 | 4,096 | 3.06x | **5.00x** | -1.94x |
| 16 | 4,096 | 3.42x | **7.00x** | -3.58x |
| 8 | 16,384 | 5.54x | **6.90x** | -1.36x |
| 16 | 16,384 | 7.35x | **11.91x** | -4.56x |
| 4 | 1,024 | 1.59x | **2.20x** | -0.61x |

Only the boundary cases (n=128 and n=262,144) are approximately correct. The errors systematically **understate** the effective compression ratio, making the analysis overly pessimistic about compression achievability.

**Impact:** The conclusions in Sections 1.4 and 6.6 that reference this table (crossover lengths, achievable compression at various sequence lengths) are based on incorrect numbers. The actual effective compression is substantially BETTER than reported. For example, at n=4096 with R_c=8, the effective ratio is 5.0x, not 3.06x.

### ERROR 2 (Moderate): Linear Attention State Memory Off by 16x

**Section 6.1** states: "Plus ~6.3 Mbit = 0.75 MB for the 24 linear attention states -- negligible at this sequence length."

The actual linear attention state memory (from the paper's own formula in Section 1.1):
- Per layer: 4 * 256^2 * 16 = 4,194,304 bits = 4.19 Mbit
- 24 layers: 100,663,296 bits = **100.7 Mbit = 12.0 MB**

The claim of 6.3 Mbit is **16x too small**. This error does not propagate to the main formulas (which correctly use the constant 6,291,456 = 24 * 262,144 before multiplying by b_0), but is misleading in the text.

### ERROR 3 (Moderate): Within-Group Entropy Ranges Are Mostly Wrong

**Section 2.4** tabulates within-group entropy ranges per layer. Multiple values are incorrect when verified against the actual entropy data:

| Layer | KV Group | Paper range | Actual range |
|-------|----------|-------------|--------------|
| 7 | KV0 | 0.60 | **0.77** |
| 7 | KV1 | 1.97 | **1.52** |
| 7 | KV3 | 0.31 | **0.61** |
| 11 | KV1 | 0.83 | **1.38** |
| 11 | KV3 | 0.16 | **0.43** |
| 15 | KV2 | 0.01 | **0.14** |
| 15 | KV3 | 0.55 | **1.85** |
| 19 | KV0 | 0.31 | **0.61** |
| 19 | KV1 | 0.44 | **0.67** |
| 19 | KV2 | 1.08 | **0.55** |

The overall mean within-group range is **0.81 bits**, not 0.73 as claimed. The qualitative conclusion (within-group entropy is "wasted" diversity) still holds, but the quantitative details are unreliable.

### ERROR 4 (Minor): Section 4.1 Self-Contradictory Head Count

The text states "Only 3 low-entropy heads ($H < 2.0$)" but then lists **6 head names**: L7_H4, L15_H13, L31_H8, L31_H9, L31_H10, L31_H12. The actual count from the entropy data is **7** (the listed 6 plus L7_H5 at H=1.985). The "~5%" claim is approximately correct for 7/128 = 5.5%.

### ERROR 5 (Minor): Formula Typo in Section 6.4

The paper writes:

$$\text{Effective gain} = 1 + (0.145 - 1) \times 0.76 \approx 11\%$$

As written, this evaluates to 1 + (-0.855)(0.76) = 0.35, which is 65% error INCREASE, not 11% reduction. The intended formula is:

$$G_{\text{eff}} = 1 + (G - 1) \times \text{damping} = 1 + 0.145 \times 0.76 = 1.110$$

The conclusion (~11% reduction) is correct despite the formula being wrong.

### ERROR 6 (Moderate): Adaptive Bit Allocation Gain Is Understated

**Section 4.3** computes $\mathcal{L}_{\text{uniform}} / \mathcal{L}_{\text{adaptive}} \approx 1.145$ (14.5% error reduction).

The actual computation with exact min-entropy values from the entropy config gives:
- $\sum_g 2^{-H_g} = 5.513$ (paper approximates as 5.21)
- $32 \times 2^{-\bar{H}} = 4.529$ (paper says 4.55)
- **Ratio = 1.217** (21.7% error reduction, not 14.5%)

The discrepancy comes from the paper's rough approximation of the central 30 groups as having mean H=3.0 (actual: 2.978) and using $30 \times 2^{-3.0} = 3.75$ when the actual sum is 4.059. The convexity of $2^{-x}$ means the average of $2^{-H}$ is always above $2^{-\bar{H}}$, and the paper's linear approximation understates this effect.

### ERROR 7 (Minor): Noise Floor Bit-Width Calculation

**Section 5.3** claims the noise floor corresponds to $b_{\text{floor}} \approx 4.2$ bits. Interpolating the Lloyd-Max distortion coefficients (c(4)=0.009497, c(3)=0.03454) using the rate-distortion scaling $c(b) \propto 2^{-2b}$ gives $b_{\text{floor}} \approx 4.8$ bits for $c(b)=0.003$.

The conclusion that "4-bit KV is near the noise floor" is actually **strengthened** by this correction: 4-bit is further below the floor than claimed, meaning KV quantization noise dominates more strongly over weight quantization noise at 4 bits. The qualitative recommendation (4-bit as natural operating point) is unaffected, but it means going to 5-bit KV might still provide meaningful improvement, contradicting the paper's claim of diminishing returns above 4 bits.

---

## 2. Questionable Assumptions

### ASSUMPTION 1: Min-Entropy Bottleneck for GQA

**Section 2.3** argues the eviction budget for a GQA group should be set by the minimum entropy across its query heads. The "proof sketch" is heuristic, not rigorous.

**Challenge:** The minimum-entropy head has the smallest effective support, so it suffers most from eviction. This is correct *if all heads contribute equally to the final output*. But they may not. In the transformer output, query heads' contributions are combined through the output projection matrix $W_O$. If the low-entropy head has a small $\|W_O^{(j)}\|$ (small contribution to the residual stream), its errors may be negligible even under aggressive eviction.

The correct formulation should minimize **weighted** group error:

$$r_g^* = \arg\min_{r} \sum_j w_j \cdot \mathcal{L}_j^{(e)}(r)$$

where $w_j$ reflects head $j$'s contribution to the output. Without measuring $W_O$ column norms, we cannot know whether min-entropy is truly the bottleneck.

**Testable prediction:** If the sole sink head L31_H9 has a small output projection norm, the mean-entropy rule may outperform min-entropy for its KV group. The experimental priority in Section 8 (Tier 2, item 4) correctly identifies this as needing empirical validation.

### ASSUMPTION 2: The Gated Delta Net State Contributes Zero to Compressible Cache

**Section 1.1** treats linear attention states as fixed-size and incompressible. This deserves scrutiny.

The Gated Delta Net maintains a recurrent state matrix of shape $(d_k, d_v)$ per head. While the state does not grow with sequence length, it IS an information bottleneck -- it summarizes the entire prefix. Questions:

1. **Can the state itself be quantized?** If the state tolerates 4-bit quantization (as an outer product accumulator, it may have low effective rank), this provides additional compression that the theory ignores entirely.
2. **Does the state's fixed size mean information is already being "evicted" implicitly?** The linear attention layers are performing their own form of lossy compression by maintaining a fixed-size state for an unbounded context. Our KV eviction on the full-attention layers is doing the same thing but explicitly. The theory should acknowledge this parallel.
3. **At short sequences, the linear state is a significant fraction of memory** (25% at n=128, 50% at n=384 per the corrected formula). If these states can be compressed even modestly, it meaningfully improves short-sequence effective compression.

### ASSUMPTION 3: Q4_K_M Noise Is Uniform Across Layers

**Section 5.2** assumes ~0.3% relative MSE from weight quantization uniformly. Q4_K_M is a **mixed-precision** scheme: it assigns 4-bit to most weight blocks but 6-bit to blocks with higher variance. The actual noise varies by layer:

- Early layers (computing initial K,V projections from embeddings) may have different weight distributions than later layers.
- The 8 full-attention layers may have systematically different weight quantization error than the 24 linear attention layers, since their $W_K, W_V$ matrices serve different functions.
- Layer 31, which the paper identifies as anomalous in entropy, may also have anomalous weight quantization error.

The noise floor of 4.2 (or 4.8 corrected) bits is a **model-wide average**. Individual layers could have floors ranging from 3.5 to 6+ bits. This matters for per-layer bit allocation.

### ASSUMPTION 4: Shannon Entropy Is the Right Metric

The paper uses Shannon entropy throughout, but the original GPT-2 theory correctly identifies **Renyi entropy of order 2** ($H_2$) as the quantity that directly governs quantization error (Section 2.2 of the GPT-2 theory). The Qwen paper conflates these:

- Section 0 reports Shannon entropy (from the calibration)
- Section 2.3 uses Shannon entropy for the bottleneck analysis
- Section 4.3 uses Shannon entropy in the $b_h^*$ formula

But the bit allocation formula was derived for $H_2$, not Shannon entropy $H_1$. For distributions that are not maximally concentrated, $H_2 \leq H_1$, and the gap depends on the shape of the attention distribution. Using $H_1$ where $H_2$ is needed **overestimates** how well diffuse heads suppress quantization noise, leading to over-aggressive bit reallocation away from them.

**Severity:** For the tightly clustered Qwen entropy distribution (most heads around 3.0-3.8), $H_1 - H_2$ is likely 0.2-0.5 bits per head. This is small relative to the entropy values but could shift the optimal bit allocation meaningfully.

### ASSUMPTION 5: GPT-2 Eviction Quality Curves Transfer to Qwen

**Section 6.2** adapts GPT-2's BLEU results to predict Qwen performance. The eviction quality-vs-ratio curve from GPT-2 (fitted as $\alpha(R_e-1)^\beta$) was measured on:
- A 124M parameter model with 12 layers of full attention
- 128-token sequences on WikiText-2
- 144 independent heads with bimodal entropy

Applying this curve to Qwen3.5-4B (4B parameters, 8 attention layers out of 32, 256K max context, GQA, unimodal entropy) is a major extrapolation. The qualitative shape may transfer, but the quantitative parameters ($\alpha$, $\beta$) almost certainly differ.

---

## 3. Missing Analysis

### MISSING 1: RoPE and Token Eviction Interaction

Qwen3.5-4B uses RoPE (Rotary Position Embeddings). RoPE encodes absolute position information into the key vectors via rotation. When tokens are evicted from the KV cache:

1. The surviving keys still carry their original position-encoded rotations.
2. The query vector is rotated by its absolute position.
3. The relative position information ($q_m \cdot k_n$ encodes $m - n$ via the rotation angle difference) is preserved for surviving tokens.

**However:** When the attention distribution is renormalized after eviction, the weighted sum of value vectors may shift. If evicted tokens were positionally systematic (e.g., "evict the oldest tokens"), the RoPE-induced position bias in the surviving set is non-uniform. The theory does not analyze whether position-aware eviction strategies are needed, or whether the entropy-adaptive strategy already implicitly handles this.

This is less critical than for absolute position embeddings (where eviction would create gaps in the position sequence), but it warrants at least a brief analysis or acknowledgment.

### MISSING 2: Error Propagation Through the Hybrid Architecture

The 32 layers alternate: 3 linear attention + 1 full attention, repeated 8 times. The output of each full attention layer feeds into the next 3 linear attention layers before reaching the next full attention layer.

When we compress the KV cache in a full attention layer, the error propagates:
1. Through the residual connection of that layer
2. Through 3 subsequent linear attention layers (which may amplify or dampen the error)
3. Into the next full attention layer's K, V projections

The paper treats each attention layer's compression error independently. But errors compound across layers. In particular:
- Are later attention layers more sensitive to errors (since they receive compounded noise from earlier layers)?
- Does the Gated Delta Net's recurrent state accumulate errors from previous full attention layers?

### MISSING 3: Causal Mask Edge Cases With Eviction

When tokens are evicted from the KV cache, the causal mask must be updated consistently. Edge cases:
- If token $t$ is evicted but token $t+1$ (which depended on $t$ during training) is retained, the attention pattern over the retained tokens is different from what the model learned.
- The softmax renormalization after eviction changes the effective temperature of attention. This is equivalent to an implicit temperature scaling that the theory does not model.

### MISSING 4: Interaction Between Weight Quantization and Eviction Decisions

Eviction decisions are based on attention weights, which are computed from $q \cdot k^T$ where both $q$ and $k$ are produced by quantized weight matrices. The attention weights used for eviction are therefore **noisy**. The paper analyzes the noise floor for KV values but not for the eviction decision quality itself.

If weight quantization causes attention weight errors of 5-10% (plausible at Q4_K_M), the "wrong" tokens may be evicted, especially for moderate-entropy heads where multiple tokens have similar attention weights. The eviction error under Q4_K_M may be worse than the paper's GPT-2-derived curves suggest, since GPT-2 used FP32 weights.

### MISSING 5: Calibration Validity at Long Sequences

The entropy calibration used `n_sample_positions = 4` at presumably moderate context lengths. But Qwen3.5-4B supports 262K tokens. The attention entropy at position 200K may be very different from position 200:
- More tokens in the KV cache = higher potential entropy
- Attention patterns may shift as context grows (e.g., sliding window effects)
- The recurrent state from linear attention layers carries a lossy summary of the long prefix, potentially changing what the full attention layers need to look up

A calibration done at short/moderate lengths may not represent the operating regime where compression matters most.

---

## 4. What's Solid

### SOLID 1: Multiplicative Memory Compression (Section 1.2)

The core identity $R_c = R_e \times R_q$ is algebraically trivial and correct. The formula for effective compression ratio $R_{\text{eff}}$ is correctly derived (even though the table applying it is wrong).

### SOLID 2: The Interaction Term Analysis (from GPT-2 theory, Section 1.3)

The decomposition $O_{eq} - O = (\tilde{A} - A)V + A(\hat{V} - V) + (\tilde{A} - A)(\hat{V} - V)$ is a standard algebraic identity, correctly applied. The Frobenius norm bound on the interaction term is a direct application of sub-multiplicativity. The $O(\epsilon_e \cdot \epsilon_q)$ characterization is correct.

### SOLID 3: Quantization Error Scaling with Head Entropy (from GPT-2 theory, Section 2.2)

The derivation $\mathbb{E}[\|\epsilon_h\|^2] = \sigma_q^2 \cdot d \cdot 2^{-H_2(h)}$ is mathematically correct given the i.i.d. quantization error assumption. The Renyi-2 connection through the collision probability is standard.

### SOLID 4: Dimension Independence of Relative Error (Section 3.2-3.3)

The analysis that TurboQuant's relative quantization error is dimension-independent at fixed bits-per-element is correct. The cancellation of $d$ in the relative output error formula is verified.

### SOLID 5: CLT Convergence Improvement (Section 3.1)

The Berry-Esseen bound scaling as $1/\sqrt{d}$ is standard, and the 2x improvement from $d=64$ to $d=256$ follows directly. The claim of "~2% better Gaussianity" is reasonable for the magnitude of improvement from a 0.125 to 0.0625 bound on the CDF deviation.

### SOLID 6: QJL Overhead Scaling (Section 3.4)

The JL dimension independence from ambient dimension is a well-known result. The $M_{\text{QJL}}/M_{\text{KV}} \propto 1/d$ scaling is correct, and the 4x overhead reduction at $d=256$ vs $d=64$ follows directly.

### SOLID 7: Noise Floor Concept (Section 5.3)

The analysis that weight quantization creates an irreducible noise floor is qualitatively correct and well-reasoned. The specific bit-level is off (4.8 not 4.2, see Error 7), but the framework and the conclusion that 4-bit KV is near-optimal for Q4_K_M models is sound.

### SOLID 8: The L31 Anomaly Observation (Section 4.4)

The identification of layer 31 as uniquely containing the model's only low-entropy heads is a direct observation from the data, not a derived claim. The hypothesis that the last attention layer performs specialized retrieval is reasonable and testable.

---

## 5. Impact on Experiment Design

### 5.1 Fix the R_eff Table Before Using It for Experiment Planning

The corrected table shows significantly better effective compression at medium sequence lengths. At n=4096, R_c=8 gives R_eff=5.0x (not 3.06x). This means the experiments can be **more optimistic** about achievable compression. The minimum sequence lengths for target compression levels are lower than the paper states.

### 5.2 Test Min vs Mean vs Weighted-Mean Entropy for GQA Groups

The min-entropy bottleneck is the paper's most novel claim and its most questionable assumption. The Tier 2 experiment (item 4) correctly plans this comparison. Additionally, test a **weighted-min** using $W_O$ column norms if accessible from the GGUF model. If the sole sink head (L31_H9) has a small output projection, the min-entropy rule wastes budget on protecting a low-impact head.

### 5.3 Measure Actual Weight Quantization Noise Per Layer

The noise floor analysis (Section 5.3) relies on a model-wide average. Before the experiment, run a simple diagnostic: compute the KV vectors once with the Q4_K_M model and once with the FP16 model (or use a calibration dataset), and measure the per-layer relative MSE in KV values. This validates the 0.3% assumption and reveals whether per-layer noise floors differ enough to matter.

### 5.4 Run Calibration at Multiple Sequence Lengths

The current entropy calibration used 4 sample positions. For the long-context regime where compression matters most, recalibrate at n=4096, n=16384, and n=65536. The entropy profile may change, and the optimal eviction budgets may shift.

### 5.5 The BLEU > 0.5 Prediction Is Reasonable But Barely

GPT-2 at 8x combined achieved BLEU 0.596 -- barely above 0.5. The paper predicts Qwen at 8x combined will achieve 0.55-0.70. Given:
- Weaker adaptive eviction benefit (narrower entropy)
- Pre-existing weight quantization noise
- Untested GQA eviction dynamics

the lower bound of 0.55 is optimistic. BLEU 0.45-0.65 would be a more honest prediction range. The experiment should plan for the possibility that 8x combined falls below 0.5, and have 4x and 6x fallback configurations ready.

### 5.6 Add a 5-bit KV Quantization Test

The corrected noise floor (~4.8 bits, not 4.2) suggests that 5-bit KV may still provide meaningful improvement over 4-bit for Q4_K_M models. Add a 5-bit configuration to distinguish between "4-bit is the floor" (paper's claim) and "4-bit is below the floor with room to improve."

### 5.7 Test Linear Attention State Compression

The paper dismisses linear attention states as fixed and incompressible. At n=4096, these states are 12 MB (not 0.75 MB as the paper claims). Even modest compression (e.g., low-rank approximation or quantization of the recurrent state) could meaningfully improve short-to-medium sequence compression. At minimum, measure the effective rank of the Gated Delta Net state matrices.

### 5.8 Correct the Adaptive Gain Estimate

The actual $\mathcal{L}_{\text{uniform}} / \mathcal{L}_{\text{adaptive}}$ ratio is 1.217 (21.7% error reduction), not 1.145 (14.5%). After noise floor dampening, the effective gain is approximately 16.5%, not 11%. This makes per-head bit allocation more worthwhile than the paper concludes. The "simple scheme" (6-bit for L31 KV2/KV3, 4-bit elsewhere) should still be tested first, but a full adaptive scheme deserves more consideration than the paper suggests.
