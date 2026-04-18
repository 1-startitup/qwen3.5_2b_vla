"""
Action Encoder for flow matching VLA.
Encodes noisy actions + sinusoidal timestep embeddings into latent representations
for the DiT cross-attention blocks.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal timestep embedding (same as used in DDPM / DiT).

    When timesteps are in [0, 1] (flow matching convention), set
    ``timestep_scale`` to a large value (e.g. 1000) so that the
    sinusoidal frequencies can distinguish nearby timesteps.
    """

    def __init__(self, dim: int, max_period: int = 10000, timestep_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.timestep_scale = timestep_scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = (timesteps[:, None].float() * self.timestep_scale) * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding.to(timesteps.dtype)


class ActionEncoder(nn.Module):
    """
    Encodes (noisy_action_chunk, timestep) -> action_embeddings.

    Architecture:
        action_proj: Linear(action_dim -> hidden_dim)
        time_enc:    SinusoidalPE(hidden_dim) -> MLP(hidden_dim -> hidden_dim)
        fusion:      Concat [action_emb, time_emb] -> Linear(2*hidden_dim -> hidden_dim) -> SiLU
    """

    def __init__(self, action_dim: int, hidden_dim: int, timestep_scale: float = 1000.0):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        self.time_enc = nn.Sequential(
            SinusoidalPositionalEncoding(hidden_dim, timestep_scale=timestep_scale),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self, noisy_actions: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, T, action_dim) noisy action chunk
            timesteps:     (B,) diffusion timestep in [0, 1]
        Returns:
            action_emb: (B, T, hidden_dim)
        """
        _, T, _ = noisy_actions.shape
        act_emb = self.action_proj(noisy_actions)
        time_emb = self.time_enc(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(-1, T, -1)
        return self.fusion(torch.cat([act_emb, time_emb], dim=-1))
