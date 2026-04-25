import torch
import torch.nn.functional as F

from src.models.vocalmorph_v2 import (
    AttentiveStatsPooling,
    ConformerBlock,
    CrossAttentionBlock,
    LearnedFeatureNormalizer,
    PositiveLinear,
)


def test_positive_linear_effective_weights_are_positive():
    layer = PositiveLinear(5, 3)
    y = layer(torch.randn(2, 5))
    assert y.shape == (2, 3)
    assert torch.all(F.softplus(layer.weight_raw) > 0)


def test_conformer_block_shape_stability():
    block = ConformerBlock(d_model=32, n_heads=4, dropout=0.1)
    x = torch.randn(2, 7, 32)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    y = block(x, padding_mask=mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_attentive_pooling_weights_are_normalized():
    pool = AttentiveStatsPooling(in_dim=16, hidden_dim=12)
    x = torch.randn(3, 9, 16)
    mask = torch.zeros(3, 9, dtype=torch.bool)
    mask[0, -3:] = True
    pooled, weights = pool(x, padding_mask=mask)
    assert pooled.shape == (3, 32)
    assert weights.shape == (3, 9)
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-5)


def test_cross_attention_block_returns_attention_maps():
    block = CrossAttentionBlock(dim=32, n_heads=4, dropout=0.1)
    query = torch.randn(2, 6, 32)
    context = torch.randn(2, 3, 32)
    query_mask = torch.zeros(2, 6, dtype=torch.bool)
    context_mask = torch.zeros(2, 3, dtype=torch.bool)
    out, attn = block(
        query,
        context,
        query_mask=query_mask,
        context_mask=context_mask,
        return_attention=True,
    )
    assert out.shape == query.shape
    assert attn.shape == (2, 4, 6, 3)
    assert torch.isfinite(out).all()
    assert torch.isfinite(attn).all()


def test_learned_feature_normalizer_respects_padding_mask():
    normalizer = LearnedFeatureNormalizer(input_dim=6, eps=1e-5)
    x = torch.randn(2, 5, 6)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, -2:] = True
    y = normalizer(x, padding_mask=mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.equal(y[0, -2:], torch.zeros_like(y[0, -2:]))
