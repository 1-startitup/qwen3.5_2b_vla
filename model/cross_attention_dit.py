"""
Diffusion Transformer (DiT) with cross-attention for flow matching action prediction.
Adapted from StarVLA's cross_attention_dit.py for Qwen3.5 VLA integration.

The DiT takes encoded noisy actions and cross-attends to VLM hidden states
to predict the velocity field for flow matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Norm conditioned on timestep embedding."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.scale_shift(timestep_emb).chunk(2, dim=-1)
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mult, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        H = self.num_heads

        q = self.to_q(x).reshape(B, N, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(B, -1, H, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, -1, H, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if context_mask is not None:
            attn = attn.masked_fill(~context_mask[:, None, None, :].bool(), float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = CrossAttention(hidden_dim, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x)


class DiTBlock(nn.Module):
    """
    Single DiT block: AdaLN -> SelfAttn -> AdaLN -> CrossAttn -> AdaLN -> FFN
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = AdaLayerNorm(hidden_dim)
        self.self_attn = SelfAttention(hidden_dim, num_heads, dropout)

        self.norm2 = AdaLayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)

        self.norm3 = AdaLayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, mult=4, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        timestep_emb: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x, timestep_emb))
        x = x + self.cross_attn(self.norm2(x, timestep_emb), context, context_mask)
        x = x + self.ffn(self.norm3(x, timestep_emb))
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for flow matching action prediction.
    """

    PRESETS = {
        "DiT-S": {"hidden_dim": 384, "num_heads": 6, "num_layers": 6},
        "DiT-B": {"hidden_dim": 768, "num_heads": 12, "num_layers": 12},
        "DiT-L": {"hidden_dim": 1024, "num_heads": 16, "num_layers": 16},
    }

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        timestep_emb: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_proj(timestep_emb)

        for block in self.blocks:
            x = block(x, context, t_emb, context_mask)

        return self.final_norm(x)
