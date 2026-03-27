# Red Team Report: Qwen3.5-4B KV Cache Compression Setup

**Date:** 2026-03-27
**Reviewed document:** `analysis/QWEN_KV_COMPRESSION_SETUP.md`
**Status:** Review complete

---

## 1. Critical Issues (Must Fix Before Implementing)

### C1. The GPT-2 monkey-patch pattern will NOT work for Qwen3.5

**The design's core assumption is wrong.** The setup document says (Section 4.2):

> "The GPT-2 `combined_compression.py` monkey-patches `ALL_ATTENTION_FUNCTIONS["eager"]`."

The GPT-2 experiment patches `ALL_ATTENTION_FUNCTIONS["eager"]` as a dict entry (line 534 of `combined_compression.py`):
```python
gpt2_module.ALL_ATTENTION_FUNCTIONS["eager"] = patched_eager_attention_forward
```

**But Qwen3.5 does not dispatch "eager" through the global dict.** Inspecting the actual transformers code:

- `ALL_ATTENTION_FUNCTIONS` is an `AttentionInterface` (dict subclass).
- Its `get_interface()` method (line 4833 of `modeling_utils.py`) returns `super().get(attn_implementation, default)`.
- The `_global_mapping` does NOT contain an `"eager"` key -- it only maps `"flash_attention_2"`, `"flash_attention_3"`, `"flex_attention"`, `"sdpa"`, and paged variants.
- When `attn_implementation="eager"`, `get_interface` returns the **`default` parameter**, which is the **model-local** `eager_attention_forward` function imported at module scope.

In Qwen3.5's `Qwen3_5Attention.forward()` (line 770-771):
```python
attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)
```

The `eager_attention_forward` here is the one defined in `modeling_qwen3_5.py` at line 688. Since `"eager"` is not in the global dict, the default is returned -- the local function reference. **Patching `ALL_ATTENTION_FUNCTIONS["eager"]` will have zero effect on Qwen3.5.**

**Fix:** You must either:
1. Directly replace the `eager_attention_forward` function in the `transformers.models.qwen3_5.modeling_qwen3_5` module namespace, OR
2. Replace each `Qwen3_5Attention` layer's bound `forward` method individually, OR
3. Register your patched function under `"eager"` in `ALL_ATTENTION_FUNCTIONS` AND ensure it's found before the default. Actually, `dict.get("eager", default)` will return the dict value if you set it, so `ALL_ATTENTION_FUNCTIONS["eager"] = patched_fn` WOULD work -- the `super().get()` call checks the dict first. Let me recheck...

**Correction after re-analysis:** `ALL_ATTENTION_FUNCTIONS` inherits from `dict`. `super().get("eager", default)` on a dict WILL return the dict value if `"eager"` is set as a key, even though `"eager"` is not in `_global_mapping`. So `ALL_ATTENTION_FUNCTIONS["eager"] = patched_fn` SHOULD work IF the patched function is set BEFORE the model's forward pass. The GPT-2 approach of setting it globally may work after all.

**However, there is still a critical signature difference.** The GPT-2 eager function:
```python
def eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
```

The Qwen3.5 eager function:
```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs)
```

Qwen3.5's function receives `scaling` as a positional argument and `dropout`. The GPT-2 experiment's patched function does NOT accept `scaling` -- it reads `module.scale_attn_weights` and uses `value.size(-1) ** 0.5` directly. **A direct port of the GPT-2 patched function will crash on Qwen3.5 because `scaling` will be captured by `**kwargs` or cause a TypeError.**

**Also:** Qwen3.5 calls `repeat_kv()` to expand KV heads from 4 to 16 INSIDE the eager attention function (line 698-699). The GPT-2 version does NOT do this because GPT-2 has no GQA. The patched function must either:
- Accept pre-GQA-expanded K,V (shape `[batch, 4, seq, 256]`) and call `repeat_kv` itself, or
- Work with already-expanded K,V (shape `[batch, 16, seq, 256]`).

Currently the design's pseudocode (Section 4.2) comments say `key, value shapes: (batch, n_kv_heads=4, seq_len, head_dim=256)`, which is correct for what's passed into the eager function. But the eviction pseudocode (Section 4.3) operates on `attn_weights: (batch, n_heads=16, seq_len, seq_len)`, which means it works on the post-repeat_kv outputs. The quantization must happen BEFORE repeat_kv on the 4-head tensors, and eviction AFTER repeat_kv on the 16-head attention weights. The interaction here is tricky and underdocumented.

**Required fix:** Write the patched function with the Qwen3.5 signature (`scaling`, `dropout`), handle the `repeat_kv` call, and test that the patch actually intercepts calls.

### C2. Eviction applied to attention weights, not the KV cache itself

The design (Section 4.3) applies eviction as a **mask on attention weights** after softmax. This matches the GPT-2 experiment. But this approach has a fundamental limitation: **it does not actually reduce memory**.

Masking attention weights to zero does not remove KV entries from memory. The full KV cache still exists. The "compression" measured by BLEU/PPL tells you whether eviction-based sparsity preserves quality, but the actual memory savings require **physically removing** entries from the KV cache tensors.

For a proof-of-concept this is acceptable (the design acknowledges this implicitly), but it should be made explicit. The PPL/BLEU measurements from mask-based eviction represent an **upper bound** on quality -- physically evicting entries and not being able to look them up again during generation will likely be worse, because:
- During prefill (processing the whole prompt), the model can "see" all tokens even with masking (they're computed then masked).
- During generation, physically evicted tokens are truly gone.

**Fix:** Add a note that mask-based eviction overstates quality. Consider also implementing physical eviction (removing rows from K,V tensors) for at least one config to measure the gap.

### C3. Generation function reprocesses entire sequence each step -- no KV cache reuse

The GPT-2 experiment's `generate_with_compression()` function (line 542-580) calls `model(generated)` with the FULL sequence at every step. It does NOT use `past_key_values`. This means:
1. Each generation step recomputes all KV from scratch (no cache reuse).
2. The "compression" is applied fresh each step to the full recomputation.
3. This does NOT test how compression interacts with incremental KV cache building during actual generation.

For Qwen3.5 with its hybrid cache (`Qwen3_5DynamicCache` managing both attention KV and delta net recurrent states), this is an even bigger problem. Real generation uses `past_key_values` to avoid redundant computation. The experiment's approach of recomputing everything each step:
- Is O(n^2 * T) instead of O(n * T) in compute
- Does not test the production-relevant scenario (compressing a growing cache)
- May give misleadingly good results (the model sees full uncompressed context at every step before compression is applied)

**Fix:** For perplexity evaluation (single forward pass), this is fine. For generation, implement at least one config that uses `past_key_values` with compression applied to the cache itself.

---

## 2. Important Concerns (Should Address, Could Cause Problems)

### I1. The entropy table in Section 2.2 does not match the actual calibration data

The setup document presents an entropy table "per KV head group" in Section 2.2 with values like `L3 KV Head 0 (Q0-3): 3.54 (std 0.20)`. Let me verify against the actual entropy_config.json:

- L3_H0=3.35, L3_H1=3.54, L3_H2=3.82, L3_H3=3.43
- Mean = 3.535, Std = 0.17

The table says `3.54 (std 0.20)`. The mean is 3.535, rounded to 3.54 -- close enough. Std is 0.17 vs 0.20 -- a discrepancy. This is minor, but the table should be generated from data, not approximated.

More importantly, the document says: `L31_H9 (entropy 0.31) is the only "sink" head in the entire model. It and L31_H12 (entropy 0.62) are the only focused/sink heads.`

From the JSON: L31_H9=0.314 (sink), L31_H12=0.622 (focused). This is correct. But L31_H8=1.653 and L31_H10=1.633 are also fairly low (moderate range). The claim that "L31 KV Head 2 (Q8-11) needs careful handling" is correct, but the design should note that Q8 and Q10 are also low-entropy (moderate), not just Q9.

### I2. Delta net state is NOT "small, fixed, <100 MB" -- it's potentially significant

The document assumes delta net state is "likely <100 MB" (Section 3.3). From the actual config:

- `linear_num_value_heads` = 32 (default for 4B)
- `linear_value_head_dim` = 128
- `linear_key_head_dim` = 128
- Recurrent state shape per layer: `(batch, num_v_heads, head_k_dim, head_v_dim)` = `(1, 32, 128, 128)`
- Bytes per layer: `32 * 128 * 128 * 2` (bf16) = **1,048,576 bytes = 1 MB**
- 24 layers: **24 MB** for recurrent states
- Plus conv states: `(batch, conv_dim, conv_kernel_size)` where `conv_dim = key_dim*2 + value_dim = 128*16*2 + 128*32 = 4096+4096 = 8192`. At kernel_size=4: `8192 * 4 * 2` = 65,536 bytes per layer. 24 layers: ~1.5 MB.
- **Total: ~25.5 MB**

So the "<100 MB" estimate is correct but loose. The actual value is ~25 MB. This matters for the "at what context length does KV cache dominate" analysis:
- At 512 tokens: KV cache is 16 MB, delta net state is 25 MB. **Delta net state actually exceeds the KV cache.** The document's "critical insight" that KV cache dominates at >4K tokens needs updating. The crossover point is around 800 tokens (25 MB / 32 KB per token ~= 800 tokens).
- At 2K tokens: KV=64 MB, delta=25 MB -- KV dominates 2.5:1.
- At 32K: KV=1024 MB, delta=25 MB -- KV dominates 40:1.

The conclusions are still directionally correct, but the numbers should be precise.

### I3. bf16 quantization concern is underweighted

The design mentions (R3) that GPT-2 used float32 while Qwen3.5 uses bf16, and suggests casting to float32 for rotation/quantization. This is more important than stated:

- bf16 has only 7 bits of mantissa (vs float32's 23). The rotation matrix Q is 256x256 in float32, but if KV vectors are bf16, you get mixed-precision matmul.
- The `sigma` estimation from `rotated.std()` on bf16 data will be less accurate.
- Lloyd-Max `quantize()` calls `np.digitize()` on numpy arrays. The `.numpy()` call in the GPT-2 code (line 273) only works on CPU tensors. For CUDA bf16 tensors, you need `.float().cpu().numpy()`. This is not just a performance suggestion -- it's a correctness requirement.

**Fix:** The TurboQuant pipeline must explicitly convert bf16 CUDA tensors to float32 CPU before numpy operations. Budget the memory for these temporary copies (at 256 dim, seq_len tokens, the peak copy is small -- seq_len * 256 * 4 bytes).

### I4. `output_attentions=True` returns 8 or 32 tensors -- behavior is ambiguous

The calibration script (line 147-154) handles three cases for the number of attention outputs. The setup document assumes the patched attention function will always be called for the right layers. But during the experiment, if using `output_attentions=True` for monitoring:

- Does Qwen3.5 return 32 attention tensors (with None for linear layers) or 8 (only for attention layers)?
- The calibration script handles both, but the experiment code may need similar disambiguation.

This is worth verifying early, not discovering during the experiment run.

### I5. Eviction pseudocode has an O(n^2 * h) problem for long contexts

The GQA-aware eviction function (Section 4.3) has triply-nested loops: `for kv_h` (4), `for pos` (seq_len), `for q_h` (4). Inside the pos loop, it calls `topk`. Total complexity: O(4 * seq_len * 4 * seq_len) = O(16 * seq_len^2).

At 2K context, this is 16 * 4M = 64M operations per layer, times 8 layers = 512M operations. On GPU this might be fast, but the Python-level for loops will be extremely slow. The GPT-2 experiment could get away with this at 128 tokens because 128^2 = 16K. At 2K tokens, it's 4M -- 250x slower.

**Fix:** Vectorize the eviction computation. The inner loops over `q_heads` and `pos` can be replaced with tensor operations (torch.max across the head dimension, then topk along the last dimension). This is not just an optimization -- at target context lengths, the Python loops will make the experiment impractically slow.

### I6. Quantization before attention computes repeat_kv on quantized KV

In the proposed approach (Section 4.2), quantization happens on the raw KV tensors (4 heads) BEFORE `repeat_kv` expands them to 16 heads. This is correct and efficient (quantize once, expand after). But the design's pseudocode shows quantization INSIDE the patched eager attention, which means it happens after K,V are already passed in.

Looking at Qwen3.5's eager attention (line 698):
```python
key_states = repeat_kv(key, module.num_key_value_groups)
value_states = repeat_kv(value, module.num_key_value_groups)
```

So `key` and `value` arrive at the eager function with shape `(batch, 4, seq, 256)`. Good -- you CAN quantize them before repeat_kv. But you MUST do the quantization before the repeat_kv call, not after. Make this ordering explicit in the implementation.

### I7. The "layer_counter" mechanism from GPT-2 won't work for Qwen3.5

The GPT-2 experiment tracks which layer is being processed via `state['layer_counter']`, incrementing it with each attention forward call. Since GPT-2 has 12 layers all with attention, the counter goes 0..11.

For Qwen3.5, only 8 of 32 layers have attention. If the counter increments each time the patched function fires, it goes 0..7. But the actual layer indices are [3, 7, 11, 15, 19, 23, 27, 31]. The entropy data is keyed by actual layer index (e.g., `L3_H0`, `L7_H4`). If the code uses counter=0 to look up entropy for "the first attention layer" but the entropy data is stored under layer_idx=3, there's a mismatch.

**Fix:** Use `module.layer_idx` attribute (available on `Qwen3_5Attention` at line 718) instead of a counter. Or maintain a counter-to-actual-layer mapping. The design mentions `module.layer_idx` as "likely" available -- confirm this and make it the primary mechanism.

---

## 3. Minor Issues (Nice to Fix, Won't Block Progress)

### M1. Memory calculation for rotation matrices is slightly off

Section 2.3 says: "Each matrix occupies 256 KB (float32)." A 256x256 float32 matrix is 256 * 256 * 4 = 262,144 bytes = 256 KB. Correct. But it says 64 rotation matrices total at 16 MB. 64 * 256 KB = 16 MB. Also correct. No issue here -- just verified.

### M2. QJL m_projections=128 rationale is backwards

Section 2.3: "Proposed: m_projections=128 (maintaining dim/m ratio of ~2:1, vs GPT-2's ~1:1)."

GPT-2: dim=64, m=64, ratio = 1:1. Proposed: dim=256, m=128, ratio = 2:1. The design says this maintains the ratio, but 2:1 != 1:1. If the goal is maintaining the ratio, m should be 256. If the goal is a compromise, 128 is fine, but the rationale text is misleading.

### M3. Memory per vector comparison is confused

Section 2.3 item 4: "Each K or V vector is 256 x 2 bytes (bf16) = 512 bytes, vs GPT-2's 64 x 4 bytes (float32) = 256 bytes."

This compares bf16 Qwen3.5 (512 bytes) with float32 GPT-2 (256 bytes), making it look like Qwen3.5 vectors are only 2x larger. In equivalent precision (both float32), Qwen3.5 vectors would be 1024 bytes vs 256 = 4x. The comparison should note this is an apples-to-oranges comparison.

### M4. Config 5 compression ratio

Config 5 (3-bit quant only) claims "5.3x" compression. For bf16 (16-bit) to 3-bit: 16/3 = 5.33x. This is correct only if we ignore codebook and QJL overhead. With QJL (m=128, 1 bit each plus float32 norms), the effective ratio is lower. Not a real issue for the experiment, but the "expected ratio" should say "theoretical" or "pre-overhead."

### M5. The design should specify the transformers version

Qwen3.5 support in HuggingFace transformers is relatively recent. The `ALL_ATTENTION_FUNCTIONS` as a class-based registry (with `get_interface`) is a newer API. The installed version (`C:\Users\ericl\AppData\Roaming\Python\Python314\site-packages\transformers\`) should be pinned in the experiment to avoid future breakage.

---

## 4. What's Good (Things the Design Got Right)

### G1. Architecture understanding is solid
The layer types, attention layer indices, GQA ratio, and head dimensions all match the actual model config. The entropy calibration data is correctly structured and mapped.

### G2. The phased approach is smart
HuggingFace-first with conditional llama.cpp port is the right call. Proving the concept in Python before committing to C++ work is high-value.

### G3. GQA eviction strategy is well-reasoned
The union-based (max attention weight) eviction across query heads sharing a KV head is the conservative-correct approach. The document correctly identifies that this is harder than per-head eviction and plans for it.

### G4. Risk assessment is thorough
The risk section (Section 5) covers the right concerns, especially R1 (GQA difficulty), R3 (bf16 precision), R5 (narrow entropy range), and R8 (delta net interaction). The "unknown unknowns" section shows good intellectual honesty.

### G5. Success criteria are well-calibrated
Having both positive and negative result criteria (Section 6) prevents the "must get good numbers" trap. Knowing that GQA makes eviction impractical is a valuable negative result.

### G6. Double quantization awareness
Section 2.4's analysis of the Q4_K_M weight quantization + KV cache quantization interaction is an important consideration that many designs would miss.

### G7. Correct identification of the sink head
L31_H9 (entropy 0.31) is correctly identified as a sink head, and the design correctly warns about protecting it during eviction of KV head group 2.

---

## 5. Recommended Changes (Specific, Actionable)

### R1. Write and test the monkey-patch before anything else

Before writing the full experiment, create a minimal test:
```python
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_module

original_fn = qwen_module.eager_attention_forward

call_count = 0
def test_patch(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    global call_count
    call_count += 1
    return original_fn(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

# Method 1: Module-level replacement
qwen_module.eager_attention_forward = test_patch

# Method 2: Global registry (may also work)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
ALL_ATTENTION_FUNCTIONS["eager"] = test_patch
```

Run a single forward pass and verify `call_count == 8`. If neither works, fall back to per-layer `forward` method replacement.

### R2. Compute delta net state size explicitly

Add to the experiment startup:
```python
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True).text_config
n_v_heads = config.linear_num_value_heads  # 32
k_dim = config.linear_key_head_dim  # 128
v_dim = config.linear_value_head_dim  # 128
recurrent_state_per_layer = n_v_heads * k_dim * v_dim * 2  # bf16
print(f"Delta net recurrent state: {recurrent_state_per_layer / 1024:.0f} KB/layer, {24 * recurrent_state_per_layer / 1024**2:.1f} MB total")
```

### R3. Vectorize eviction -- eliminate Python loops over positions

Replace the triple-nested Python loop with vectorized PyTorch ops:
```python
# For each KV head group: max attention across query heads
# attn_weights: (batch, 16, seq, seq)
attn_reshaped = attn_weights.view(batch, 4, 4, seq, seq)  # (batch, kv_heads, qh_per_kv, seq, seq)
max_attn = attn_reshaped.max(dim=2).values  # (batch, 4, seq, seq)
# Then topk on the last dim for each position
```

This eliminates the O(seq_len) Python loop and makes 32K-token experiments feasible.

### R4. Use `module.layer_idx` instead of a counter

The Qwen3.5 attention module has `self.layer_idx` set in `__init__` (line 718). Use this directly in the patched function:
```python
def patched_fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    layer_idx = module.layer_idx  # Actual layer index (3, 7, 11, ...)
```

### R5. Add a physical eviction test

For at least one config, implement actual KV cache row removal instead of attention weight masking. Compare BLEU/PPL against the mask-based result to quantify the gap. This tells you how optimistic the mask-based numbers are.

### R6. Test at 512+ token context, not just 128

The calibration was done at 128 tokens. Attention patterns at 128 tokens are very different from 2K or 32K tokens. Add at least a 512-token evaluation (and ideally 2048) even if it's slower. The entropy calibration data from 128 tokens may not be representative of longer contexts.

### R7. Pin the transformers version

Add `transformers==X.Y.Z` to the experiment's requirements. The `ALL_ATTENTION_FUNCTIONS` API and `Qwen3_5` module structure could change between versions.

### R8. Handle bf16-to-numpy conversion explicitly

The TurboQuant code from GPT-2 calls `.numpy()` on tensors. For CUDA bf16 tensors:
```python
# This will fail:
rotated.numpy()  # RuntimeError: can't convert cuda:0 device type tensor to numpy

# Required:
rotated.float().cpu().numpy()
```

Make this a documented requirement in the TurboQuant adaptation section.
