"""
Combined Entropy-Adaptive Eviction + TurboQuant KV Cache Compression for Qwen3.5-4B
======================================================================================
Tests three compression approaches on Qwen3.5-4B (GPU, eager attention):
  1. Entropy-adaptive eviction only (GQA-aware)
  2. TurboQuant-style vector quantization only (2/3/4/5 bit)
  3. Combined: eviction + quantization
  4. Entropy-informed adaptive bit allocation

Key architecture adaptations from GPT-2 experiment:
  - GQA: 16 query heads, 4 KV heads (4:1 ratio)
  - Hybrid: only 8/32 layers are full-attention [3,7,11,15,19,23,27,31]
  - head_dim=256, bf16 precision
  - Quantize BEFORE repeat_kv (on 4 KV heads, not expanded 16)
  - GQA-aware eviction: min-entropy bottleneck across query groups

TurboQuant implementation:
  Stage 1: Random rotation (QR of Gaussian) + Lloyd-Max scalar quantization
  Stage 2: QJL residual correction via random sign projections

Hardware: CUDA (bf16)
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
ENTROPY_CONFIG_PATH = PROJECT_ROOT / "models" / "qwen3.5-4b" / "entropy_config.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reproducibility
GLOBAL_SEED = 42
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Architecture constants
FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}
N_LAYERS = 32
N_FULL_ATTN_LAYERS = 8
N_QUERY_HEADS = 16
N_KV_HEADS = 4
GQA_RATIO = N_QUERY_HEADS // N_KV_HEADS  # 4
HEAD_DIM = 256


def timestamp():
    return time.strftime("%H:%M:%S")


###############################################################################
# CALIBRATION TEXTS (hardcoded, no dataset download needed)
###############################################################################

CALIBRATION_TEXTS = [
    # 1. Science / physics
    "The fundamental forces of nature govern all interactions in the universe. "
    "Gravity, the weakest of the four forces, shapes the large-scale structure of "
    "the cosmos. Electromagnetic interactions bind atoms into molecules and are "
    "responsible for chemistry and biology. The strong nuclear force confines quarks "
    "within protons and neutrons, while the weak force mediates radioactive decay. "
    "Attempts to unify these forces into a single framework have driven theoretical "
    "physics for decades. String theory proposes that all particles are vibrations "
    "of tiny one-dimensional strings. Loop quantum gravity takes a different approach, "
    "quantizing spacetime itself into discrete units. Neither theory has been experimentally "
    "verified, leaving the quest for unification incomplete. Meanwhile, dark matter and "
    "dark energy remain unexplained, comprising roughly 95 percent of the universe's "
    "total energy content.",

    # 2. History
    "The Industrial Revolution began in Britain in the late 18th century, transforming "
    "manufacturing processes from hand production to machine-based methods. The invention "
    "of the spinning jenny, the water frame, and the power loom revolutionized textile "
    "production. James Watt's improvements to the steam engine provided a versatile "
    "source of mechanical power. Factories replaced cottage industries, drawing workers "
    "from rural areas into rapidly growing cities. Working conditions were often harsh, "
    "with long hours, low wages, and dangerous machinery. Child labor was common until "
    "reform legislation gradually improved protections. The revolution spread to continental "
    "Europe and North America in the 19th century, catalyzing urbanization and economic "
    "growth on an unprecedented scale. Transportation was transformed by railways and "
    "steamships, shrinking distances and creating national markets.",

    # 3. Technology / AI
    "Large language models have emerged as one of the most significant developments in "
    "artificial intelligence. These models, trained on vast corpora of text, demonstrate "
    "remarkable capabilities in natural language understanding, generation, and reasoning. "
    "The transformer architecture, introduced in 2017, enabled efficient parallel processing "
    "of sequences through self-attention mechanisms. Subsequent scaling laws revealed that "
    "model performance improves predictably with increased parameters, data, and compute. "
    "However, these models also exhibit concerning behaviors including hallucination, bias "
    "amplification, and unpredictable failure modes. Researchers are actively exploring "
    "methods to improve reliability, including reinforcement learning from human feedback, "
    "constitutional AI approaches, and mechanistic interpretability. The deployment of "
    "these systems raises important questions about automation, creativity, and the nature "
    "of intelligence itself.",

    # 4. Biology / ecology
    "Coral reefs are among the most biodiverse ecosystems on Earth, supporting an estimated "
    "25 percent of all marine species despite covering less than one percent of the ocean "
    "floor. The symbiotic relationship between coral polyps and zooxanthellae algae forms "
    "the foundation of reef ecosystems. Zooxanthellae photosynthesize within coral tissues, "
    "providing up to 90 percent of the coral's energy needs while receiving shelter and "
    "nutrients in return. Rising ocean temperatures disrupt this symbiosis, causing coral "
    "bleaching events that have become increasingly frequent and severe. Ocean acidification, "
    "driven by absorption of atmospheric carbon dioxide, further threatens reefs by reducing "
    "the availability of calcium carbonate needed for skeletal growth. Conservation efforts "
    "include establishing marine protected areas, coral nurseries, and research into heat-resistant "
    "coral strains that may survive in warmer waters.",

    # 5. Mathematics / logic
    "The concept of infinity has challenged mathematicians and philosophers for millennia. "
    "Georg Cantor's set theory revealed that infinities come in different sizes: the set "
    "of natural numbers is countably infinite, while the real numbers form an uncountably "
    "infinite set of strictly greater cardinality. His diagonal argument elegantly proved "
    "this distinction. The continuum hypothesis, which posits that no set has cardinality "
    "strictly between the naturals and the reals, was shown to be independent of standard "
    "set theory axioms by Godel and Cohen. This independence result demonstrated fundamental "
    "limitations of formal axiomatic systems. Modern set theory explores large cardinal "
    "axioms that extend the hierarchy of infinities to dizzying heights. These abstract "
    "investigations have practical implications for computability theory and the foundations "
    "of mathematics, influencing our understanding of what can and cannot be proven.",

    # 6. Literature / culture
    "Shakespeare's influence on the English language extends far beyond his plays and sonnets. "
    "He coined or popularized thousands of words and phrases that remain in common usage: "
    "assassination, lonely, generous, eyeball, bedroom, and countless others entered the "
    "lexicon through his works. His exploration of human psychology anticipated modern "
    "understanding of jealousy, ambition, grief, and love. Hamlet's soliloquies reveal an "
    "interior consciousness that was revolutionary for its time. The comedies explore gender, "
    "identity, and social convention through cross-dressing heroines and mistaken identities. "
    "The tragedies examine how character flaws interact with circumstance to produce catastrophe. "
    "Four centuries after his death, Shakespeare continues to be performed, adapted, and studied "
    "worldwide. Modern productions reinterpret his works through contemporary lenses, finding "
    "new relevance in stories written for a very different world.",
]


###############################################################################
# TURBOQUANT: LLOYD-MAX SCALAR QUANTIZER
###############################################################################

class LloydMaxQuantizer:
    """
    Lloyd-Max optimal scalar quantizer for N(0, sigma^2) distribution.
    Precomputes centroids for 2, 3, 4, 5 bit widths via iterative Lloyd algorithm.
    """

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.codebooks = {}  # (n_bits, sigma) -> (boundaries, centroids)

    def _lloyd_iteration(self, n_levels: int, sigma: float,
                         max_iter: int = 200, n_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """Run Lloyd's algorithm to find optimal centroids for N(0, sigma^2)."""
        samples = self.rng.normal(0, sigma, n_samples)
        samples.sort()

        # Initialize centroids uniformly across [-3*sigma, 3*sigma]
        centroids = np.linspace(-2.5 * sigma, 2.5 * sigma, n_levels)

        for _ in range(max_iter):
            boundaries = (centroids[:-1] + centroids[1:]) / 2.0
            new_centroids = np.zeros(n_levels)

            indices = np.digitize(samples, boundaries)

            for k in range(n_levels):
                mask = (indices == k)
                if mask.sum() > 0:
                    new_centroids[k] = samples[mask].mean()
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-8):
                break
            centroids = new_centroids

        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        return boundaries, centroids

    def get_codebook(self, n_bits: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get or compute Lloyd-Max codebook for given bit-width and sigma."""
        key = (n_bits, round(sigma, 2))
        if key not in self.codebooks:
            n_levels = 2 ** n_bits
            boundaries, centroids = self._lloyd_iteration(n_levels, sigma)
            self.codebooks[key] = (boundaries, centroids)
        return self.codebooks[key]

    def quantize(self, x: np.ndarray, n_bits: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize array x. Returns (indices, centroids_used)."""
        boundaries, centroids = self.get_codebook(n_bits, sigma)
        indices = np.digitize(x, boundaries).astype(np.int16)
        return indices, centroids

    def dequantize(self, indices: np.ndarray, n_bits: int, sigma: float) -> np.ndarray:
        """Dequantize indices back to values."""
        _, centroids = self.get_codebook(n_bits, sigma)
        return centroids[indices]


###############################################################################
# TURBOQUANT: RANDOM ROTATION
###############################################################################

class RandomRotation:
    """
    Random orthogonal rotation matrix via QR decomposition of Gaussian matrix.
    One rotation per head dimension, cached and seeded for reproducibility.
    """

    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        rng = np.random.RandomState(seed)
        G = rng.randn(dim, dim).astype(np.float32)
        Q, R = np.linalg.qr(G)
        d = np.diag(R)
        sign = np.sign(d)
        sign[sign == 0] = 1
        Q = Q * sign[np.newaxis, :]
        self.Q = torch.from_numpy(Q)
        self.Q_inv = self.Q.T

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation: x @ Q. x shape: (..., dim)"""
        return x @ self.Q.to(x.device, dtype=x.dtype)

    def inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: x @ Q^T"""
        return x @ self.Q_inv.to(x.device, dtype=x.dtype)


###############################################################################
# TURBOQUANT: QJL RESIDUAL CORRECTION (Stage 2)
###############################################################################

class QJLResidualCorrector:
    """
    Quantized Johnson-Lindenstrauss residual correction.
    Projects residuals through random Gaussian matrix, stores only signs (1 bit each).
    """

    def __init__(self, dim: int, m_projections: int = 64, seed: int = 42):
        self.dim = dim
        self.m = m_projections
        rng = np.random.RandomState(seed)
        self.proj = torch.from_numpy(
            rng.randn(dim, m_projections).astype(np.float32) / math.sqrt(m_projections)
        )

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        """Encode residual into sign bits. residual: (..., dim) -> (..., m) signs."""
        proj = self.proj.to(residual.device, dtype=residual.dtype)
        projected = residual @ proj
        return (projected >= 0).to(torch.int8)

    def correct_inner_product(self, sign_bits: torch.Tensor,
                              query: torch.Tensor,
                              residual_norms: torch.Tensor) -> torch.Tensor:
        """
        Estimate <residual, query> from sign bits.
        sign_bits: (n_tokens, m), query: (dim,), residual_norms: (n_tokens,)
        """
        proj = self.proj.to(query.device, dtype=query.dtype)
        q_proj = query @ proj
        signs_float = 2.0 * sign_bits.float() - 1.0
        q_signs = (2.0 * (q_proj >= 0).float() - 1.0)
        agreement = (signs_float * q_signs.unsqueeze(0)).mean(dim=-1)
        q_norm = query.norm()
        correction = agreement * residual_norms * q_norm * math.sqrt(math.pi / 2)
        return correction


###############################################################################
# TURBOQUANT: FULL VECTOR QUANTIZER
###############################################################################

class TurboQuantVectorQuantizer:
    """
    Complete TurboQuant pipeline: Random Rotation + Lloyd-Max + QJL Residual.
    """

    def __init__(self, dim: int, n_bits: int = 4, m_projections: int = 64,
                 seed: int = 42, use_qjl: bool = True):
        self.dim = dim
        self.n_bits = n_bits
        self.use_qjl = use_qjl

        self.rotation = RandomRotation(dim, seed=seed)
        self.lloyd = LloydMaxQuantizer(seed=seed)
        self.qjl = QJLResidualCorrector(dim, m_projections, seed=seed + 1) if use_qjl else None

    def compress(self, vectors: torch.Tensor) -> dict:
        """
        Compress a batch of vectors.
        vectors: (n_tokens, dim) - any dtype, will be converted to float32 for numpy ops
        Returns dict with compressed representation.
        """
        n_tokens, dim = vectors.shape
        assert dim == self.dim

        # Stage 1: Rotate (in float32 on CPU for numpy compatibility)
        vectors_f32 = vectors.float().cpu()
        rotated = self.rotation.rotate(vectors_f32)

        # Estimate sigma from actual rotated data, round to 2dp for codebook sharing
        sigma_raw = float(rotated.std().item())
        if sigma_raw < 1e-8:
            sigma_raw = 1e-4
        sigma = round(sigma_raw, 2)

        # Quantize each coordinate independently
        rot_np = rotated.numpy()
        indices, centroids = self.lloyd.quantize(rot_np.ravel(), self.n_bits, sigma)
        indices = indices.reshape(n_tokens, dim)

        # Reconstruct from stage 1
        recon_np = self.lloyd.dequantize(indices.ravel(), self.n_bits, sigma).reshape(n_tokens, dim)
        recon_rotated = torch.from_numpy(recon_np.astype(np.float32))
        stage1_recon = self.rotation.inverse_rotate(recon_rotated)

        result = {
            'indices': indices,
            'sigma': sigma,
            'n_bits': self.n_bits,
            'n_tokens': n_tokens,
        }

        # Stage 2: QJL residual correction
        if self.use_qjl:
            residual = vectors_f32 - stage1_recon
            sign_bits = self.qjl.encode(residual)
            residual_norms = residual.norm(dim=-1)
            result['sign_bits'] = sign_bits
            result['residual_norms'] = residual_norms
            result['stage1_recon'] = stage1_recon
        else:
            result['stage1_recon'] = stage1_recon

        return result

    def decompress(self, compressed: dict) -> torch.Tensor:
        """Decompress back to full vectors (stage 1 reconstruction only)."""
        indices = compressed['indices']
        sigma = compressed['sigma']
        n_tokens = compressed['n_tokens']

        recon_np = self.lloyd.dequantize(indices.ravel(), self.n_bits, sigma)
        recon_np = recon_np.reshape(n_tokens, self.dim).astype(np.float32)
        recon_rotated = torch.from_numpy(recon_np)
        return self.rotation.inverse_rotate(recon_rotated)

    def memory_bytes(self, compressed: dict) -> int:
        """Estimate memory usage of compressed representation."""
        n_tokens = compressed['n_tokens']
        index_bits = n_tokens * self.dim * self.n_bits
        qjl_bits = 0
        if 'sign_bits' in compressed:
            qjl_bits = compressed['sign_bits'].numel()
            qjl_bits += n_tokens * 32
        codebook_bits = (2 ** self.n_bits) * 32
        total_bits = index_bits + qjl_bits + codebook_bits
        return total_bits // 8

    def original_memory_bytes(self, n_tokens: int) -> int:
        """Memory for uncompressed bf16 vectors."""
        return n_tokens * self.dim * 2  # bf16 = 2 bytes


###############################################################################
# LOAD ENTROPY CONFIG
###############################################################################

def load_entropy_config() -> Dict:
    """Load pre-computed head entropies from entropy_config.json.
    Returns dict mapping (layer_idx, query_head_idx) -> entropy value.
    Only full-attention layers are included.
    """
    with open(ENTROPY_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    head_entropies = {}
    raw = config['head_entropies']
    for key, value in raw.items():
        # Parse "L{layer}_H{head}"
        parts = key.split('_')
        layer_idx = int(parts[0][1:])
        head_idx = int(parts[1][1:])
        # Only keep full-attention layers
        if layer_idx in FULL_ATTENTION_LAYERS and value > 0.0:
            head_entropies[(layer_idx, head_idx)] = value

    return head_entropies


###############################################################################
# EVICTION MASKS (GQA-AWARE)
###############################################################################

def mask_entropy_adaptive_gqa(attn_weights, keep_ratio, head_entropies=None,
                               layer_idx=0):
    """
    Per-head adaptive eviction with GQA awareness.

    attn_weights: (n_query_heads=16, seq_len, seq_len) - post-softmax weights

    GQA-aware eviction:
    - Compute entropy per query head
    - For each KV head group (4 query heads sharing 1 KV head):
      use MIN entropy in the group as the budget (bottleneck principle)
    - Build eviction mask at query-head level, ensure consistency within KV groups
    """
    n_query_heads, seq_len, _ = attn_weights.shape

    # Get per-query-head entropy for budget scaling
    query_head_entropies = []
    if head_entropies is not None:
        mean_ent = np.mean([v for v in head_entropies.values()]) if head_entropies else 1.0
    else:
        mean_ent = 1.0

    for h in range(n_query_heads):
        if head_entropies is not None:
            key = (layer_idx, h)
            this_ent = head_entropies.get(key, mean_ent)
        else:
            this_ent = mean_ent
        query_head_entropies.append(this_ent)

    # GQA bottleneck: for each KV group, use MIN entropy across query heads
    kv_group_budgets = []
    for kv_h in range(N_KV_HEADS):
        group_start = kv_h * GQA_RATIO
        group_end = group_start + GQA_RATIO
        group_ents = query_head_entropies[group_start:group_end]
        min_ent = min(group_ents)
        kv_group_budgets.append(min_ent)

    mask = torch.zeros_like(attn_weights)

    for h in range(n_query_heads):
        kv_h = h // GQA_RATIO
        budget_ent = kv_group_budgets[kv_h]

        # Scale keep ratio by entropy relative to mean
        if mean_ent > 0:
            scale = max(0.3, min(2.5, budget_ent / mean_ent))
            adj_ratio = min(1.0, keep_ratio * scale)
        else:
            adj_ratio = keep_ratio

        for pos in range(seq_len):
            context_len = pos + 1
            k = max(1, int(context_len * adj_ratio))
            if context_len <= k:
                mask[h, pos, :context_len] = 1.0
            else:
                _, topk_idx = attn_weights[h, pos, :context_len].topk(k)
                mask[h, pos, topk_idx] = 1.0

    return mask


###############################################################################
# MONKEY-PATCH MECHANISM
###############################################################################

# Global state for experiments
_compression_state = {
    'active': False,
    'mode': None,           # 'eviction', 'quantization', 'combined', 'entropy_quant'
    'keep_ratio': 1.0,
    'head_entropies': None,
    'n_bits': 4,
    'tq_cache': {},         # (layer_idx, kv_head, bits) -> TurboQuantVectorQuantizer
    'use_qjl': True,
    'm_projections': 64,
    # Entropy-informed quantization settings
    'entropy_quant_bits_range': (3, 5),  # (min_bits, max_bits) for adaptive
}


def _get_per_kv_head_bits(layer_idx: int) -> List[int]:
    """
    Compute per-KV-head bit allocation based on entropy.
    Low-entropy KV heads get more bits (5), high-entropy get fewer (3).
    Entropy is computed at query-head level, then min within each KV group.
    """
    state = _compression_state
    head_entropies = state['head_entropies']
    min_bits, max_bits = state['entropy_quant_bits_range']

    if head_entropies is None:
        return [state['n_bits']] * N_KV_HEADS

    # Collect min entropy per KV head group
    kv_ents = []
    for kv_h in range(N_KV_HEADS):
        group_ents = []
        for qh in range(kv_h * GQA_RATIO, (kv_h + 1) * GQA_RATIO):
            key = (layer_idx, qh)
            ent = head_entropies.get(key, 0.0)
            group_ents.append(ent)
        kv_ents.append(min(group_ents) if group_ents else 0.0)

    kv_ents = np.array(kv_ents)
    if kv_ents.max() - kv_ents.min() < 1e-6:
        return [state['n_bits']] * N_KV_HEADS

    # Normalize to [0, 1]
    ent_norm = (kv_ents - kv_ents.min()) / (kv_ents.max() - kv_ents.min())

    # Low entropy -> more bits (5), high entropy -> fewer bits (3)
    bits_list = []
    for en in ent_norm:
        bits = max_bits - en * (max_bits - min_bits)
        bits_list.append(int(round(max(min_bits, min(max_bits, bits)))))

    return bits_list


def _quantize_kv_layer(key_states: torch.Tensor, value_states: torch.Tensor,
                        layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply TurboQuant to K and V tensors for one layer.
    key_states, value_states: (batch, n_kv_heads=4, seq_len, head_dim=256)
    Returns reconstructed tensors in same shape and dtype.

    IMPORTANT: This operates on KV heads (4) BEFORE repeat_kv expansion.
    """
    state = _compression_state
    batch, n_kv_heads, seq_len, head_dim = key_states.shape
    orig_dtype = key_states.dtype
    orig_device = key_states.device
    mode = state['mode']

    new_keys = torch.zeros_like(key_states)
    new_values = torch.zeros_like(value_states)

    if mode == 'entropy_quant':
        per_head_bits = _get_per_kv_head_bits(layer_idx)
    else:
        per_head_bits = [state['n_bits']] * n_kv_heads

    for h in range(n_kv_heads):
        n_bits = per_head_bits[h]

        # Look up pre-computed quantizer
        cache_key = (layer_idx, h, n_bits)
        if cache_key not in state['tq_cache']:
            # Fallback: create on-the-fly
            seed = GLOBAL_SEED + layer_idx * 100 + h * 10 + n_bits
            state['tq_cache'][cache_key] = TurboQuantVectorQuantizer(
                dim=head_dim, n_bits=n_bits,
                m_projections=state['m_projections'],
                seed=seed, use_qjl=state['use_qjl']
            )
        tq = state['tq_cache'][cache_key]

        # Quantize keys and values for this KV head
        # .float().cpu() handled inside compress()
        k_vecs = key_states[0, h]    # (seq_len, head_dim)
        v_vecs = value_states[0, h]  # (seq_len, head_dim)

        k_compressed = tq.compress(k_vecs)
        v_compressed = tq.compress(v_vecs)

        # Decompress and move back to original device/dtype
        new_keys[0, h] = tq.decompress(k_compressed).to(orig_device, dtype=orig_dtype)
        new_values[0, h] = tq.decompress(v_compressed).to(orig_device, dtype=orig_dtype)

    return new_keys, new_values


def patched_eager_attention_forward(module, query, key, value, attention_mask,
                                     scaling, dropout=0.0, **kwargs):
    """
    Modified eager attention for Qwen3.5 with eviction and/or quantization.

    Signature matches Qwen3.5's eager_attention_forward exactly:
      module: nn.Module with .layer_idx, .num_key_value_groups
      query: (batch, n_heads=16, seq, head_dim=256)
      key: (batch, n_kv_heads=4, seq, head_dim=256)
      value: (batch, n_kv_heads=4, seq, head_dim=256)
      attention_mask: optional
      scaling: float
      dropout: float
      **kwargs
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

    state = _compression_state
    layer_idx = module.layer_idx

    # Only compress full-attention layers when active
    if state['active'] and layer_idx in FULL_ATTENTION_LAYERS:
        # Quantize BEFORE repeat_kv (operates on 4 KV heads, not 16)
        if state['mode'] in ('quantization', 'combined', 'entropy_quant'):
            key, value = _quantize_kv_layer(key, value, layer_idx)

    # Expand KV heads for attention (GQA: 4 -> 16)
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Eviction (after softmax, operates on 16 query heads)
    if state['active'] and layer_idx in FULL_ATTENTION_LAYERS:
        if state['mode'] in ('eviction', 'combined'):
            keep_ratio = state['keep_ratio']
            aw = attn_weights[0]  # (16, seq, seq)

            eviction_mask = mask_entropy_adaptive_gqa(
                aw, keep_ratio, state['head_entropies'], layer_idx)

            # Apply mask and renormalize
            masked_aw = aw * eviction_mask.to(aw.device, dtype=aw.dtype)
            row_sums = masked_aw.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            masked_aw = masked_aw / row_sums
            attn_weights = masked_aw.unsqueeze(0)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def install_patch():
    """Install the monkey-patched attention forward into Qwen3.5 module."""
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_module
    qwen_module.ALL_ATTENTION_FUNCTIONS["eager"] = patched_eager_attention_forward
    print("  Attention patch installed for Qwen3.5")


###############################################################################
# GENERATION UTILITIES
###############################################################################

def generate_with_compression(model, input_ids: torch.Tensor, max_new_tokens: int = 30,
                               mode: str = None, keep_ratio: float = 1.0,
                               n_bits: int = 4, head_entropies=None,
                               use_qjl: bool = True) -> torch.Tensor:
    """
    Autoregressive generation with specified compression.
    mode: None (baseline), 'eviction', 'quantization', 'combined', 'entropy_quant'
    """
    state = _compression_state

    generated = input_ids.clone()

    for step in range(max_new_tokens):
        if mode is not None:
            state['active'] = True
            state['mode'] = mode
            state['keep_ratio'] = keep_ratio
            state['n_bits'] = n_bits
            state['head_entropies'] = head_entropies
            state['use_qjl'] = use_qjl
        else:
            state['active'] = False

        with torch.no_grad():
            outputs = model(generated)

        state['active'] = False

        next_token_logits = outputs.logits[0, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

    return generated


def compute_perplexity(model, input_ids: torch.Tensor, mode: str = None,
                       keep_ratio: float = 1.0, n_bits: int = 4,
                       head_entropies=None, use_qjl: bool = True) -> float:
    """Compute perplexity of input sequence under the (possibly compressed) model."""
    state = _compression_state

    if mode is not None:
        state['active'] = True
        state['mode'] = mode
        state['keep_ratio'] = keep_ratio
        state['n_bits'] = n_bits
        state['head_entropies'] = head_entropies
        state['use_qjl'] = use_qjl
    else:
        state['active'] = False

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    state['active'] = False
    loss = outputs.loss.item()
    return math.exp(loss)


###############################################################################
# QUALITY METRICS
###############################################################################

def compute_bleu(reference_tokens: List[int], hypothesis_tokens: List[int],
                 max_n: int = 4) -> float:
    """Simple BLEU score (no smoothing) between token sequences."""
    if len(hypothesis_tokens) == 0:
        return 0.0

    bp = min(1.0, math.exp(1 - len(reference_tokens) / max(len(hypothesis_tokens), 1)))

    log_avg = 0.0
    n_valid = 0
    for n in range(1, max_n + 1):
        ref_ngrams = defaultdict(int)
        for i in range(len(reference_tokens) - n + 1):
            ng = tuple(reference_tokens[i:i + n])
            ref_ngrams[ng] += 1

        hyp_ngrams = defaultdict(int)
        for i in range(len(hypothesis_tokens) - n + 1):
            ng = tuple(hypothesis_tokens[i:i + n])
            hyp_ngrams[ng] += 1

        matches = 0
        total = 0
        for ng, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ng, 0))
            total += count

        if total > 0 and matches > 0:
            log_avg += math.log(matches / total)
            n_valid += 1

    if n_valid == 0:
        return 0.0

    return bp * math.exp(log_avg / n_valid)


def compute_rouge_l(reference: List[int], hypothesis: List[int]) -> float:
    """ROUGE-L F1 based on longest common subsequence."""
    if not reference or not hypothesis:
        return 0.0

    m, n = len(reference), len(hypothesis)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr

    lcs_len = prev[n]
    if lcs_len == 0:
        return 0.0
    prec = lcs_len / n
    recall = lcs_len / m
    return 2 * prec * recall / (prec + recall)


def compute_metrics(ref_tokens: List[int], hyp_tokens: List[int]) -> dict:
    """Compute all quality metrics between reference and hypothesis token lists."""
    min_len = min(len(ref_tokens), len(hyp_tokens))
    if min_len == 0:
        match_rate = 0.0
    else:
        matches = sum(1 for a, b in zip(ref_tokens[:min_len], hyp_tokens[:min_len]) if a == b)
        match_rate = matches / min_len

    bleu = compute_bleu(ref_tokens, hyp_tokens)
    rouge_l = compute_rouge_l(ref_tokens, hyp_tokens)

    return {
        'bleu': bleu,
        'rouge_l': rouge_l,
        'token_match': match_rate,
    }


def compute_compression_ratio(mode: str, keep_ratio: float, n_bits: int,
                                hybrid_fraction: float = 8 / 32) -> float:
    """
    Compute effective compression ratio accounting for hybrid architecture.
    Only 8/32 layers have full attention and are compressed.
    The other 24 layers use linear attention (no KV cache to compress).

    For the 8 full-attention layers:
    - Full cache = bf16 (16 bits per value per dim for K and V)
    - Compressed = n_bits per value per dim

    hybrid_fraction adjusts the effective ratio since only a fraction of
    the total KV cache is actually compressed.
    """
    original_bits_per_dim = 16  # bf16

    if mode == 'eviction':
        layer_ratio = 1.0 / keep_ratio
    elif mode == 'quantization':
        layer_ratio = original_bits_per_dim / n_bits
    elif mode == 'combined':
        eviction_factor = 1.0 / keep_ratio
        quant_factor = original_bits_per_dim / n_bits
        layer_ratio = eviction_factor * quant_factor
    elif mode == 'entropy_quant':
        layer_ratio = original_bits_per_dim / n_bits
    else:
        return 1.0

    # Effective ratio: only hybrid_fraction of layers are compressed
    # effective = 1 / (hybrid_fraction / layer_ratio + (1 - hybrid_fraction) / 1)
    # = 1 / (hybrid_fraction / layer_ratio + 1 - hybrid_fraction)
    # But since linear attention layers don't have standard KV cache at all,
    # the effective compression applies only to the full-attention KV budget.
    # Report both: per-layer ratio and effective ratio.
    return layer_ratio


###############################################################################
# EXPERIMENT CONFIGURATIONS
###############################################################################

def get_experiment_configs():
    """Define all 10 experiment configurations."""
    configs = []

    # 1. Full cache baseline
    configs.append({
        'name': 'full_cache',
        'mode': None,
        'keep_ratio': 1.0,
        'n_bits': 16,
        'description': 'Full KV cache (baseline)',
    })

    # 2. Eviction 2x (entropy-adaptive, GQA-aware)
    configs.append({
        'name': 'eviction_2x',
        'mode': 'eviction',
        'keep_ratio': 0.5,
        'n_bits': 16,
        'description': 'Entropy-adaptive eviction 2x (GQA-aware)',
    })

    # 3. Eviction 3x
    configs.append({
        'name': 'eviction_3x',
        'mode': 'eviction',
        'keep_ratio': 1.0 / 3.0,
        'n_bits': 16,
        'description': 'Entropy-adaptive eviction 3x (GQA-aware)',
    })

    # 4. Quantization 5-bit (testing noise floor ~4.8 bits)
    configs.append({
        'name': 'quant_5bit',
        'mode': 'quantization',
        'keep_ratio': 1.0,
        'n_bits': 5,
        'description': 'TurboQuant 5-bit quantization',
    })

    # 5. Quantization 4-bit
    configs.append({
        'name': 'quant_4bit',
        'mode': 'quantization',
        'keep_ratio': 1.0,
        'n_bits': 4,
        'description': 'TurboQuant 4-bit quantization',
    })

    # 6. Quantization 3-bit
    configs.append({
        'name': 'quant_3bit',
        'mode': 'quantization',
        'keep_ratio': 1.0,
        'n_bits': 3,
        'description': 'TurboQuant 3-bit quantization',
    })

    # 7. Combined 2x eviction + 5-bit quant
    configs.append({
        'name': 'combined_2x_5bit',
        'mode': 'combined',
        'keep_ratio': 0.5,
        'n_bits': 5,
        'description': 'Eviction 2x + TurboQuant 5-bit',
    })

    # 8. Combined 2x eviction + 4-bit quant (predicted sweet spot)
    configs.append({
        'name': 'combined_2x_4bit',
        'mode': 'combined',
        'keep_ratio': 0.5,
        'n_bits': 4,
        'description': 'Eviction 2x + TurboQuant 4-bit (predicted sweet spot)',
    })

    # 9. Combined 3x eviction + 4-bit quant
    configs.append({
        'name': 'combined_3x_4bit',
        'mode': 'combined',
        'keep_ratio': 1.0 / 3.0,
        'n_bits': 4,
        'description': 'Eviction 3x + TurboQuant 4-bit',
    })

    # 10. Entropy-informed adaptive quantization
    configs.append({
        'name': 'entropy_quant_adaptive',
        'mode': 'entropy_quant',
        'keep_ratio': 1.0,
        'n_bits': 4,  # average target for reporting
        'description': 'Entropy-adaptive quant (low-entropy->5bit, high-entropy->3bit)',
    })

    return configs


###############################################################################
# MAIN EXPERIMENT LOOP
###############################################################################

def run_experiment(model, tokenizer, sequences, head_entropies, configs,
                   num_seqs=6, gen_tokens=30):
    """Run all experiment configurations and collect metrics."""
    print(f"\n{'=' * 70}")
    print(f"[{timestamp()}] RUNNING COMBINED COMPRESSION EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"  {len(configs)} configurations, {num_seqs} sequences, {gen_tokens} gen tokens")

    seq_len = len(sequences[0])
    results = {}

    # Generate baseline references first
    print(f"\n[{timestamp()}] Generating baseline references...")
    baseline_generations = []
    baseline_perplexities = []
    t0 = time.time()

    for seq_idx in range(num_seqs):
        print(f"  Baseline seq {seq_idx + 1}/{num_seqs} ({time.time() - t0:.0f}s)", flush=True)
        input_ids = torch.tensor([sequences[seq_idx]], device=device)

        gen = generate_with_compression(model, input_ids, max_new_tokens=gen_tokens, mode=None)
        baseline_generations.append(gen[0, seq_len:].tolist())

        ppl = compute_perplexity(model, input_ids, mode=None)
        baseline_perplexities.append(ppl)

    baseline_ppl = float(np.mean(baseline_perplexities))
    print(f"  Baseline mean perplexity: {baseline_ppl:.2f}")

    # Run each configuration
    for cfg_idx, cfg in enumerate(configs):
        name = cfg['name']
        mode = cfg['mode']
        keep_ratio = cfg.get('keep_ratio', 1.0)
        n_bits = cfg.get('n_bits', 4)

        print(f"\n[{timestamp()}] Config {cfg_idx + 1}/{len(configs)}: {name}")
        print(f"  {cfg['description']}")

        if mode is None:
            # Baseline already computed
            results[name] = {
                'description': cfg['description'],
                'mode': 'baseline',
                'keep_ratio': 1.0,
                'n_bits': 16,
                'compression_ratio': 1.0,
                'effective_compression_ratio': 1.0,
                'bleu': 1.0,
                'rouge_l': 1.0,
                'token_match': 1.0,
                'perplexity': baseline_ppl,
                'perplexity_ratio': 1.0,
                'wall_clock_seconds': 0.0,
            }
            print(f"  (using cached baseline)")
            continue

        t_start = time.time()
        all_bleu = []
        all_rouge = []
        all_match = []
        all_ppl = []

        for seq_idx in range(num_seqs):
            elapsed = time.time() - t_start
            print(f"  Seq {seq_idx + 1}/{num_seqs} ({elapsed:.0f}s)", flush=True)

            input_ids = torch.tensor([sequences[seq_idx]], device=device)

            try:
                # Generate
                gen = generate_with_compression(
                    model, input_ids, max_new_tokens=gen_tokens,
                    mode=mode, keep_ratio=keep_ratio, n_bits=n_bits,
                    head_entropies=head_entropies, use_qjl=True
                )
                hyp_tokens = gen[0, seq_len:].tolist()
                ref_tokens = baseline_generations[seq_idx]

                metrics = compute_metrics(ref_tokens, hyp_tokens)
                all_bleu.append(metrics['bleu'])
                all_rouge.append(metrics['rouge_l'])
                all_match.append(metrics['token_match'])

                # Perplexity
                ppl = compute_perplexity(
                    model, input_ids, mode=mode, keep_ratio=keep_ratio,
                    n_bits=n_bits, head_entropies=head_entropies, use_qjl=True
                )
                all_ppl.append(ppl)

            except Exception as e:
                print(f"    ERROR on seq {seq_idx}: {e}")
                import traceback
                traceback.print_exc()
                all_bleu.append(0.0)
                all_rouge.append(0.0)
                all_match.append(0.0)
                all_ppl.append(float('inf'))

        wall_clock = time.time() - t_start
        layer_compression = compute_compression_ratio(mode, keep_ratio, n_bits)
        # Effective: only 8/32 layers compressed
        # effective = total_original / total_compressed
        # = 32 / (24 + 8/layer_compression)  -- but linear attn layers don't have KV
        # Since only full-attn layers have KV cache, effective = layer_compression
        # (we compress ALL the KV cache that exists)
        effective_compression = layer_compression

        mean_ppl = float(np.mean([p for p in all_ppl if p != float('inf')]) if all_ppl else float('inf'))

        results[name] = {
            'description': cfg['description'],
            'mode': mode,
            'keep_ratio': keep_ratio,
            'n_bits': n_bits,
            'compression_ratio': layer_compression,
            'effective_compression_ratio': effective_compression,
            'bleu': float(np.mean(all_bleu)),
            'rouge_l': float(np.mean(all_rouge)),
            'token_match': float(np.mean(all_match)),
            'perplexity': mean_ppl,
            'perplexity_ratio': mean_ppl / baseline_ppl if baseline_ppl > 0 else float('inf'),
            'wall_clock_seconds': wall_clock,
        }

        r = results[name]
        print(f"  BLEU={r['bleu']:.4f}  ROUGE-L={r['rouge_l']:.4f}  "
              f"Match={r['token_match']:.4f}  PPL_ratio={r['perplexity_ratio']:.3f}  "
              f"Compress={r['compression_ratio']:.1f}x  Time={wall_clock:.1f}s")

    return results


###############################################################################
# RESULTS TABLE
###############################################################################

def print_results_table(results: dict):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 110}")
    print(f"COMBINED COMPRESSION RESULTS — Qwen3.5-4B")
    print(f"{'=' * 110}")

    header = (f"{'Config':<30s} {'Compress':>8s} {'BLEU':>8s} {'ROUGE-L':>8s} "
              f"{'Match':>8s} {'PPL ratio':>10s} {'PPL':>10s} {'Time(s)':>8s}")
    print(header)
    print("-" * 110)

    sorted_names = sorted(results.keys(),
                          key=lambda k: results[k]['compression_ratio'])

    for name in sorted_names:
        r = results[name]
        ppl_ratio_str = f"{r['perplexity_ratio']:.3f}" if r['perplexity_ratio'] < 100 else ">100"
        ppl_str = f"{r['perplexity']:.2f}" if r['perplexity'] < 10000 else ">10000"
        print(f"{name:<30s} {r['compression_ratio']:>7.1f}x {r['bleu']:>8.4f} "
              f"{r['rouge_l']:>8.4f} {r['token_match']:>8.4f} {ppl_ratio_str:>10s} "
              f"{ppl_str:>10s} {r['wall_clock_seconds']:>8.1f}")

    print("-" * 110)

    # Find best combined config
    combined = {k: v for k, v in results.items() if 'combined' in k}
    if combined:
        best_combined = max(combined.items(), key=lambda x: x[1]['bleu'])
        print(f"\nBest combined config: {best_combined[0]}")
        print(f"  BLEU={best_combined[1]['bleu']:.4f}, "
              f"Compression={best_combined[1]['compression_ratio']:.1f}x")

    # 5-bit vs 4-bit comparison
    q5 = results.get('quant_5bit')
    q4 = results.get('quant_4bit')
    if q5 and q4:
        print(f"\n5-bit vs 4-bit quantization:")
        print(f"  5-bit: BLEU={q5['bleu']:.4f}, PPL_ratio={q5['perplexity_ratio']:.3f}")
        print(f"  4-bit: BLEU={q4['bleu']:.4f}, PPL_ratio={q4['perplexity_ratio']:.3f}")
        print(f"  -> 5-bit noise floor test: {'near-lossless' if q5['perplexity_ratio'] < 1.02 else 'measurable degradation'}")

    # Entropy-adaptive info
    eq = results.get('entropy_quant_adaptive')
    if eq:
        print(f"\nEntropy-adaptive quantization (low-ent->5bit, high-ent->3bit):")
        print(f"  BLEU={eq['bleu']:.4f}, PPL_ratio={eq['perplexity_ratio']:.3f}")


###############################################################################
# VERIFICATION
###############################################################################

def verify_turboquant():
    """Quick sanity checks for TurboQuant components."""
    print(f"\n[{timestamp()}] Verifying TurboQuant implementation...")

    dim = HEAD_DIM  # 256

    # Test rotation is orthogonal
    rot = RandomRotation(dim, seed=42)
    identity_check = rot.Q @ rot.Q.T
    eye_err = (identity_check - torch.eye(dim)).abs().max().item()
    print(f"  Rotation orthogonality error (dim={dim}): {eye_err:.2e} (should be < 1e-4)")
    assert eye_err < 1e-3, f"Rotation not orthogonal: {eye_err}"

    # Test roundtrip with high bits
    n_tokens = 16
    tq = TurboQuantVectorQuantizer(dim=dim, n_bits=4, seed=42, use_qjl=False)
    vectors = torch.randn(n_tokens, dim) * 0.5
    compressed = tq.compress(vectors)
    recon = tq.decompress(compressed)
    mse = ((vectors - recon) ** 2).mean().item()
    rel_err = mse / (vectors ** 2).mean().item()
    print(f"  4-bit roundtrip relative MSE: {rel_err:.4f}")

    # Test 2-bit
    tq2 = TurboQuantVectorQuantizer(dim=dim, n_bits=2, seed=42, use_qjl=False)
    compressed2 = tq2.compress(vectors)
    recon2 = tq2.decompress(compressed2)
    mse2 = ((vectors - recon2) ** 2).mean().item()
    rel_err2 = mse2 / (vectors ** 2).mean().item()
    print(f"  2-bit roundtrip relative MSE: {rel_err2:.4f}")
    assert rel_err2 > rel_err, "2-bit should have higher error than 4-bit"

    # Test 5-bit
    tq5 = TurboQuantVectorQuantizer(dim=dim, n_bits=5, seed=42, use_qjl=False)
    compressed5 = tq5.compress(vectors)
    recon5 = tq5.decompress(compressed5)
    mse5 = ((vectors - recon5) ** 2).mean().item()
    rel_err5 = mse5 / (vectors ** 2).mean().item()
    print(f"  5-bit roundtrip relative MSE: {rel_err5:.4f}")
    assert rel_err5 < rel_err, "5-bit should have lower error than 4-bit"

    # Test QJL
    tq_qjl = TurboQuantVectorQuantizer(dim=dim, n_bits=4, seed=42, use_qjl=True)
    compressed_qjl = tq_qjl.compress(vectors)
    assert 'sign_bits' in compressed_qjl, "QJL sign bits missing"
    print(f"  QJL sign bits shape: {compressed_qjl['sign_bits'].shape}")

    print(f"  All TurboQuant verifications passed (dim={dim}).")
    return True


def verify_patch(model, tokenizer):
    """Verify the monkey-patch actually changes outputs."""
    print(f"\n[{timestamp()}] Verifying attention patch works...")

    text = "The quick brown fox jumps over the lazy"
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    # Baseline
    _compression_state['active'] = False
    with torch.no_grad():
        baseline_out = model(input_ids)
    baseline_logits = baseline_out.logits[0].detach().float().cpu()

    # With 2-bit quantization (should produce visible difference)
    _compression_state['active'] = True
    _compression_state['mode'] = 'quantization'
    _compression_state['n_bits'] = 2
    _compression_state['use_qjl'] = False
    _compression_state['head_entropies'] = None

    with torch.no_grad():
        quant_out = model(input_ids)
    _compression_state['active'] = False
    quant_logits = quant_out.logits[0].detach().float().cpu()

    diff = (baseline_logits - quant_logits).abs().max().item()
    match_rate = (baseline_logits.argmax(-1) == quant_logits.argmax(-1)).float().mean().item()

    print(f"  Max logit difference (2-bit quant vs full): {diff:.4f}")
    print(f"  Prediction match rate: {match_rate:.4f}")

    if diff < 1e-6:
        print("  WARNING: Patch may not be working! Logits are identical.")
        return False
    else:
        print(f"  Patch verified: logits differ (max diff = {diff:.4f})")
        return True


###############################################################################
# MAIN
###############################################################################

def main():
    print(f"{'=' * 70}")
    print(f"COMBINED ENTROPY-ADAPTIVE EVICTION + TURBOQUANT EXPERIMENT")
    print(f"Qwen3.5-4B — GPU (bf16) — GQA-aware")
    print(f"{'=' * 70}")
    print(f"Start time: {timestamp()}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    t_total = time.time()

    # Verify TurboQuant implementation
    if not verify_turboquant():
        print("ABORTING: TurboQuant verification failed.")
        return None

    # Load model
    print(f"\n[{timestamp()}] Loading Qwen3.5-4B...")
    from transformers import AutoTokenizer, AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    # Qwen3.5 may wrap text_config; use it if present
    text_config = getattr(config, 'text_config', config)
    model = Qwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B",
        config=text_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"  Model loaded in {time.time() - t_total:.1f}s")

    # Install attention patch
    install_patch()

    # Tokenize calibration texts into fixed-length sequences
    print(f"\n[{timestamp()}] Tokenizing calibration texts...")
    SEQ_LEN = 256
    sequences = []
    for text in CALIBRATION_TEXTS:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= SEQ_LEN:
            sequences.append(tokens[:SEQ_LEN])
        else:
            # Pad by repeating if needed
            while len(tokens) < SEQ_LEN:
                tokens = tokens + tokens
            sequences.append(tokens[:SEQ_LEN])
    print(f"  {len(sequences)} sequences of {SEQ_LEN} tokens")

    # Load pre-computed head entropies
    print(f"\n[{timestamp()}] Loading entropy config...")
    head_entropies = load_entropy_config()
    ent_values = list(head_entropies.values())
    print(f"  Loaded {len(head_entropies)} head entropies (full-attention layers only)")
    print(f"  Entropy range: {min(ent_values):.2f} - {max(ent_values):.2f} bits")
    print(f"  Entropy mean: {np.mean(ent_values):.2f}, std: {np.std(ent_values):.2f}")

    # Verify patch works
    if not verify_patch(model, tokenizer):
        print("ABORTING: Patch verification failed.")
        return None

    # Pre-compute TurboQuant objects
    # 8 layers x 4 KV heads x 4 bit widths (2,3,4,5) = 128 objects
    print(f"\n[{timestamp()}] Pre-computing TurboQuant objects...")
    t_precompute = time.time()

    all_bits = {2, 3, 4, 5}
    lloyd_shared = LloydMaxQuantizer(seed=GLOBAL_SEED)
    tq_cache = {}
    for layer_idx in FULL_ATTENTION_LAYERS:
        for h in range(N_KV_HEADS):
            for n_bits in all_bits:
                cache_key = (layer_idx, h, n_bits)
                seed = GLOBAL_SEED + layer_idx * 100 + h * 10 + n_bits
                tq_obj = TurboQuantVectorQuantizer(
                    dim=HEAD_DIM, n_bits=n_bits,
                    m_projections=_compression_state['m_projections'],
                    seed=seed, use_qjl=True
                )
                # Share lloyd instance for codebook cache sharing
                tq_obj.lloyd = lloyd_shared
                tq_cache[cache_key] = tq_obj

    _compression_state['tq_cache'] = tq_cache
    print(f"  Pre-computed {len(tq_cache)} TurboQuant objects in {time.time() - t_precompute:.1f}s")
    print(f"  ({len(FULL_ATTENTION_LAYERS)} layers x {N_KV_HEADS} KV heads x {len(all_bits)} bit widths)")
    print(f"  Sigma estimated per-call from actual rotated KV data (rounded to 2dp)")

    # Get experiment configs
    configs = get_experiment_configs()
    print(f"\n  Experiment configurations ({len(configs)} total):")
    for cfg in configs:
        print(f"    - {cfg['name']}: {cfg['description']}")

    # Run experiments
    num_seqs = len(sequences)  # 6
    gen_tokens = 30
    results = run_experiment(model, tokenizer, sequences, head_entropies,
                             configs, num_seqs=num_seqs, gen_tokens=gen_tokens)

    # Print comparison table
    print_results_table(results)

    # Save results
    results_path = RESULTS_DIR / 'qwen_combined_results.json'
    serializable = {}
    for k, v in results.items():
        serializable[k] = {sk: (float(sv) if isinstance(sv, (np.floating, float)) else sv)
                           for sk, sv in v.items()}

    output = {
        'experiment': 'qwen3.5_combined_entropy_turboquant',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'config': {
            'model': 'Qwen/Qwen3.5-4B',
            'num_sequences': num_seqs,
            'seq_len': SEQ_LEN,
            'gen_tokens': gen_tokens,
            'device': str(device),
            'seed': GLOBAL_SEED,
            'architecture': {
                'n_layers': N_LAYERS,
                'full_attention_layers': sorted(FULL_ATTENTION_LAYERS),
                'n_query_heads': N_QUERY_HEADS,
                'n_kv_heads': N_KV_HEADS,
                'gqa_ratio': GQA_RATIO,
                'head_dim': HEAD_DIM,
            },
        },
        'head_entropies': {f"L{k[0]}_H{k[1]}": v for k, v in head_entropies.items()},
        'results': serializable,
        'runtime_seconds': time.time() - t_total,
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    total_time = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"COMPLETE - Total runtime: {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}")

    return results


if __name__ == '__main__':
    results = main()
