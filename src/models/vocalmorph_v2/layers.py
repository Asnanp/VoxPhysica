"""Reusable neural network layers for VocalMorph V2."""

from __future__ import annotations

import math
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _sanitize_padding_mask, _validate_class_labels


class MCDropout(nn.Dropout):
    """Dropout layer with explicit MC-sampling control."""

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)
        self.mc_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(
            x, self.p, training=(self.training or self.mc_enabled), inplace=False
        )


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float):
        ctx.lambda_grl = float(lambda_grl)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_init: float = 0.0):
        super().__init__()
        self.lambda_grl = float(lambda_init)

    def set_lambda(self, value: float) -> None:
        self.lambda_grl = float(value)

    def forward(
        self, x: torch.Tensor, lambda_override: Optional[float] = None
    ) -> torch.Tensor:
        lam = self.lambda_grl if lambda_override is None else float(lambda_override)
        return GradientReversalFunction.apply(x, lam)


class DropPath(nn.Module):
    """Per-sample stochastic depth for residual branches."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= float(drop_prob) < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x * random_tensor / keep_prob


class LayerScale(nn.Module):
    """Learned residual rescaling for deep transformer-style blocks."""

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class PositiveLinear(nn.Module):
    """Monotonic linear layer with positive effective weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.softplus(self.weight_raw), self.bias)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding, AMP-safe and device-safe."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.register_buffer("pe", self._build_table(self.max_len), persistent=False)

    def _build_table(self, length: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(
                f"x must have shape (B, T, {self.d_model}), got {tuple(x.shape)}"
            )
        time_steps = x.size(1)
        if time_steps > self.pe.size(1):
            self.pe = self._build_table(max(time_steps, self.max_len)).to(
                device=x.device
            )
        pos = self.pe[:, :time_steps].to(device=x.device, dtype=x.dtype)
        return x + pos


class RelativePositionBias(nn.Module):
    """Head-specific learnable relative position bias."""

    def __init__(self, n_heads: int, max_distance: int = 128):
        super().__init__()
        self.n_heads = int(n_heads)
        self.max_distance = int(max_distance)
        self.bias_table = nn.Embedding(2 * self.max_distance + 1, self.n_heads)
        nn.init.zeros_(self.bias_table.weight)

    def forward(
        self, q_len: int, k_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device)
        k_pos = torch.arange(k_len, device=device)
        relative = (k_pos.unsqueeze(0) - q_pos.unsqueeze(1)).clamp(
            -self.max_distance, self.max_distance
        )
        relative = relative + self.max_distance
        bias = self.bias_table(relative)
        return bias.permute(2, 0, 1).to(dtype=dtype)


class AdvancedMultiHeadAttention(nn.Module):
    """Attention with learnable temperature and relative position bias."""

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_rel_pos_bias: bool = True,
        rel_pos_max_distance: int = 128,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.dim // self.n_heads
        self.base_scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.zeros(self.n_heads))
        self.rel_pos_bias = (
            RelativePositionBias(
                n_heads=self.n_heads, max_distance=rel_pos_max_distance
            )
            if use_rel_pos_bias
            else None
        )

    def _project(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        x = proj(x)
        x = x.view(batch_size, time_steps, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Any:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key, and value must all have shape (B, T, D)")
        if query.size(0) != key.size(0) or query.size(0) != value.size(0):
            raise ValueError("query, key, and value batch dimensions must match")
        if (
            query.size(-1) != self.dim
            or key.size(-1) != self.dim
            or value.size(-1) != self.dim
        ):
            raise ValueError(f"All attention inputs must have feature dim {self.dim}")

        batch_size, q_len, _ = query.shape
        _, k_len, _ = key.shape
        q_mask = (
            None
            if query_mask is None
            else _sanitize_padding_mask(
                query_mask, expected_shape=(batch_size, q_len), name="query_mask"
            )
        )
        kv_mask = (
            None
            if key_padding_mask is None
            else _sanitize_padding_mask(
                key_padding_mask,
                expected_shape=(batch_size, k_len),
                name="key_padding_mask",
            )
        )

        q = self._project(query, self.q_proj)
        k = self._project(key, self.k_proj)
        v = self._project(value, self.v_proj)

        scale = self.base_scale * self.logit_scale.exp().clamp(max=4.0).view(
            1, self.n_heads, 1, 1
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if self.rel_pos_bias is not None:
            scores = scores + self.rel_pos_bias(
                q_len, k_len, device=query.device, dtype=scores.dtype
            ).unsqueeze(0)
        if kv_mask is not None:
            scores = scores.masked_fill(
                kv_mask.unsqueeze(1).unsqueeze(2), -torch.finfo(scores.dtype).max
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.dim)
        out = self.out_dropout(self.out_proj(out))

        if q_mask is not None:
            out = out.masked_fill(q_mask.unsqueeze(-1), 0.0)
            attn = attn.masked_fill(q_mask.unsqueeze(1).unsqueeze(-1), 0.0)

        if return_attention:
            return out, attn
        return out


class AttentiveStatsPooling(nn.Module):
    """Context-conditioned attentive mean/std pooling over the temporal dimension."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, eps: float = 1e-5):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.input_norm = nn.LayerNorm(in_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.gate_proj = nn.Linear(in_dim * 2, in_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.logit_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T, D), got {tuple(x.shape)}")

        if padding_mask is not None:
            if padding_mask.ndim != 2:
                raise ValueError(
                    f"padding_mask must have shape (B, T), got {tuple(padding_mask.shape)}"
                )
            if padding_mask.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "padding_mask shape must match the first two dims of x: "
                    f"x={tuple(x.shape)}, padding_mask={tuple(padding_mask.shape)}"
                )
            padding_mask = _sanitize_padding_mask(padding_mask)

        x_norm = self.input_norm(x)
        if padding_mask is None:
            context = x_norm.mean(dim=1)
        else:
            valid = (~padding_mask).to(dtype=x.dtype).unsqueeze(-1)
            context = (x_norm * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        context_expanded = context.unsqueeze(1).expand(-1, x.size(1), -1)
        joint = torch.cat([x_norm, context_expanded], dim=-1)
        gated_values = x * torch.sigmoid(self.gate_proj(joint))
        logits = self.score_proj(self.context_proj(joint)).squeeze(-1)
        logits = logits * self.logit_scale.exp().clamp(max=4.0)
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask, -torch.finfo(logits.dtype).max)

        weights = torch.softmax(logits, dim=1)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask, 0.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=self.eps)

        weights_expanded = weights.unsqueeze(-1)
        mean = (gated_values * weights_expanded).sum(dim=1)
        second_moment = (gated_values.pow(2) * weights_expanded).sum(dim=1)
        std = torch.sqrt((second_moment - mean.pow(2)).clamp(min=self.eps))

        pooled = torch.cat([mean, std], dim=-1)
        if pooled.shape != (x.size(0), x.size(-1) * 2):
            raise ValueError(
                f"Expected pooled shape {(x.size(0), x.size(-1) * 2)}, got {tuple(pooled.shape)}"
            )
        return pooled, weights


class ConditionalLayerNorm(nn.Module):
    def __init__(self, dim: int, n_domains: int = 2, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Embedding(n_domains, dim)
        self.beta = nn.Embedding(n_domains, dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, domain: Optional[torch.Tensor]) -> torch.Tensor:
        x_norm = F.layer_norm(x, (self.dim,), eps=self.eps)
        if domain is None:
            return x_norm
        _validate_class_labels(domain, "domain", self.gamma.num_embeddings)
        domain_idx = domain.to(device=x.device, dtype=torch.long)
        gamma = self.gamma(domain_idx)
        beta = self.beta(domain_idx)
        return x_norm * gamma + beta


class SqueezeExcitationVector(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, dim // reduction)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.net(x))
        return x * gate


class ConformerConvModule(nn.Module):
    """Conformer convolution branch working on `(B, T, C)` tensors."""

    def __init__(self, d_model: int = 256, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.norm = nn.GroupNorm(1, d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = x.transpose(1, 2)
        if padding_mask is not None:
            y = y.masked_fill(padding_mask.unsqueeze(1), 0.0)
        y = self.pointwise_in(y)
        y = F.glu(y, dim=1)
        y = self.depthwise(y)
        y = self.norm(y)
        y = F.silu(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        if padding_mask is not None:
            y = y.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return y


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.05,
        layer_scale_init: float = 1e-4,
        rel_pos_max_distance: int = 128,
    ):
        super().__init__()
        ff_dim = d_model * ff_mult
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm_mha = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(
            d_model=d_model, kernel_size=31, dropout=dropout
        )
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        x_norm = self.norm_mha(x)
        attn_out, _ = self.mha(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.conv(self.norm_conv(x), padding_mask=padding_mask)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        x = self.final_norm(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention from acoustic tokens into physics tokens."""

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.05,
        layer_scale_init: float = 1e-4,
        rel_pos_max_distance: int = 64,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Any:
        q = self.query_norm(query)
        k = self.context_norm(context)
        if return_attention:
            attn_out, attn_weights = self.attn(
                q,
                k,
                k,
                key_padding_mask=context_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attn_out, _ = self.attn(
                q,
                k,
                k,
                key_padding_mask=context_mask,
                need_weights=False,
            )
            attn_weights = None
        x = query + attn_out
        x = x + self.ff(self.ff_norm(x))
        x = self.out_norm(x)
        if query_mask is not None:
            x = x.masked_fill(query_mask.unsqueeze(-1), 0.0)
        if return_attention:
            return x, attn_weights
        return x


class CrossAttentionFusion(nn.Module):
    """Single cross-attention block for fusing acoustic and physics tokens."""

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.05,
        layer_scale_init: float = 1e-4,
        rel_pos_max_distance: int = 64,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    drop_path=drop_path,
                    layer_scale_init=layer_scale_init,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
            ]
        )

    def forward(
        self,
        acoustic_tokens: torch.Tensor,
        physics_tokens: torch.Tensor,
        acoustic_mask: Optional[torch.Tensor] = None,
        physics_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Any:
        x = acoustic_tokens
        attn_maps: List[torch.Tensor] = []
        for block in self.layers:
            if return_attention:
                x, attn = block(
                    x,
                    physics_tokens,
                    query_mask=acoustic_mask,
                    context_mask=physics_mask,
                    return_attention=True,
                )
                attn_maps.append(attn)
            else:
                x = block(
                    x,
                    physics_tokens,
                    query_mask=acoustic_mask,
                    context_mask=physics_mask,
                )
        if return_attention:
            return x, attn_maps
        return x


__all__ = [
    "AdvancedMultiHeadAttention",
    "AttentiveStatsPooling",
    "ConditionalLayerNorm",
    "ConformerBlock",
    "ConformerConvModule",
    "CrossAttentionBlock",
    "DropPath",
    "GradientReversalFunction",
    "GradientReversalLayer",
    "LayerScale",
    "MCDropout",
    "PositiveLinear",
    "RelativePositionBias",
    "SinusoidalPositionalEncoding",
    "SqueezeExcitationVector",
    "CrossAttentionFusion",
]
