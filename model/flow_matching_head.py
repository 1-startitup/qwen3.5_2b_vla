"""
Flow Matching Action Head for VLA.
Implements continuous normalizing flow (CNF) via flow matching for action prediction.

Training: learns velocity field v(x_t, t) via MSE loss between predicted and true velocity.
Inference: Euler integration from Gaussian noise x_0 to clean actions x_1 over N steps.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_encoder import ActionEncoder
from .cross_attention_dit import DiT


@dataclass
class FlowMatchingConfig:
    """Configuration for the single-stream flow matching action head."""

    head_type: str = "single_fm"
    dit_preset: str = "DiT-B"
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.0

    action_dim: int = 7
    action_horizon: int = 8

    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    num_inference_steps: int = 4
    timestep_buckets: int = 1000

    vlm_hidden_dim: int = 2048
    state_dim: int = 8

    @classmethod
    def from_preset(cls, preset: str = "DiT-B", **overrides):
        dit_cfg = DiT.PRESETS[preset]
        return cls(
            dit_preset=preset,
            hidden_dim=dit_cfg["hidden_dim"],
            num_heads=dit_cfg["num_heads"],
            num_layers=dit_cfg["num_layers"],
            **overrides,
        )


class FlowMatchingActionHead(nn.Module):
    """Single-stream flow matching action prediction head."""

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        self.action_encoder = ActionEncoder(
            config.action_dim, H,
            timestep_scale=float(getattr(config, 'timestep_buckets', 1000)),
        )

        self.vlm_proj = nn.Sequential(
            nn.Linear(config.vlm_hidden_dim, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

        if config.state_dim > 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(config.state_dim, H),
                nn.SiLU(),
                nn.Linear(H, H),
            )
        else:
            self.state_encoder = None

        self.pos_embed = nn.Parameter(
            torch.randn(1, config.action_horizon, H) * 0.02
        )

        self.dit = DiT(
            hidden_dim=H,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        self.velocity_decoder = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, config.action_dim),
        )

        self.beta_dist = torch.distributions.Beta(
            config.beta_alpha, config.beta_beta
        )

    def _build_context(
        self,
        vlm_hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        context = self.vlm_proj(vlm_hidden_states)

        if self.state_encoder is not None and state is not None:
            state_emb = self.state_encoder(state).unsqueeze(1)
            context = torch.cat([state_emb, context], dim=1)
            if attention_mask is not None:
                ones = torch.ones(
                    attention_mask.shape[0],
                    1,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([ones, attention_mask], dim=1)

        return context, attention_mask

    def forward(
        self,
        vlm_hidden_states: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, A = actions.shape
        device = actions.device
        dtype = actions.dtype

        t = self.beta_dist.sample((B,)).to(device=device, dtype=dtype)
        noise = torch.randn(B, T, A, device=device, dtype=dtype)

        t_expand = t[:, None, None]
        noisy_actions = (1.0 - t_expand) * noise + t_expand * actions
        true_velocity = actions - noise

        action_emb = self.action_encoder(noisy_actions, t)
        action_emb = action_emb + self.pos_embed[:, :T, :]

        context, mask = self._build_context(vlm_hidden_states, state, attention_mask)
        time_emb = self.action_encoder.time_enc(t)

        dit_out = self.dit(action_emb, context, time_emb, mask)
        predicted_velocity = self.velocity_decoder(dit_out)
        loss = F.mse_loss(predicted_velocity, true_velocity)

        return {"loss": loss, "predicted_velocity": predicted_velocity}

    @torch.no_grad()
    def predict_action(
        self,
        vlm_hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        deterministic_seed: Optional[int] = None,
    ) -> torch.Tensor:
        B = vlm_hidden_states.shape[0]
        T = self.config.action_horizon
        device = vlm_hidden_states.device
        dtype = vlm_hidden_states.dtype

        context, mask = self._build_context(vlm_hidden_states, state, attention_mask)

        if deterministic_seed is not None:
            gen = torch.Generator(device=device).manual_seed(deterministic_seed)
            x = torch.randn(
                B,
                T,
                self.config.action_dim,
                device=device,
                dtype=dtype,
                generator=gen,
            )
        else:
            x = torch.randn(B, T, self.config.action_dim, device=device, dtype=dtype)

        steps = num_steps if num_steps is not None else self.config.num_inference_steps
        dt = 1.0 / steps

        for i in range(steps):
            t_val = i * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)

            action_emb = self.action_encoder(x, t) + self.pos_embed[:, :T, :]
            time_emb = self.action_encoder.time_enc(t)

            dit_out = self.dit(action_emb, context, time_emb, mask)
            velocity = self.velocity_decoder(dit_out)
            x = x + velocity * dt

        return x.clamp(-1.0, 1.0)
