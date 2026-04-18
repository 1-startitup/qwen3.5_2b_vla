"""
Layer-wise flow matching action head.

Each DiT block cross-attends to exactly ONE VLM transformer layer output.
No grouping, no mean aggregation — 1-to-1 mapping from VLM layers to DiT blocks.

If ``condition_layer_indices`` is provided it must have exactly ``num_layers``
entries (one per DiT block).  Otherwise indices are evenly spaced across the
available VLM layers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_encoder import ActionEncoder
from .cross_attention_dit import DiTBlock


@dataclass
class LayerwiseFlowMatchingConfig:
    """Configuration for the layerwise flow matching head."""

    head_type: str = "layerwise_fm"
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 12
    dropout: float = 0.0

    action_dim: int = 7
    action_horizon: int = 8

    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    num_inference_steps: int = 10
    timestep_buckets: int = 1000

    vlm_hidden_dim: int = 2048
    state_dim: int = 8
    num_future_tokens: int = 0

    # Exactly num_layers indices selecting which VLM layers to tap.
    # If None, auto-select evenly spaced layers.
    condition_layer_indices: Optional[List[int]] = None

    # --- kept for yaml backward-compat but no longer used ---
    num_condition_layers: int = 12
    condition_slot_aggregation: str = "mean"


class LayerwiseFlowMatchingActionHead(nn.Module):
    """
    Layerwise flow matching action head.

    Each of the ``num_layers`` DiT blocks receives cross-attention context
    from exactly one VLM hidden-state layer, projected through a per-block
    ``vlm_proj`` MLP.  No grouping or aggregation.
    """

    def __init__(self, config: LayerwiseFlowMatchingConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # --- encoders ---
        self.action_encoder = ActionEncoder(
            config.action_dim, H, timestep_scale=float(config.timestep_buckets)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

        # --- per-block VLM projection (one per DiT block) ---
        self.vlm_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.vlm_hidden_dim, H),
                    nn.SiLU(),
                    nn.Linear(H, H),
                )
                for _ in range(config.num_layers)
            ]
        )

        # --- DiT backbone ---
        self.blocks = nn.ModuleList(
            [DiTBlock(H, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(H)

        # --- state encoder ---
        self.state_encoder = (
            nn.Sequential(
                nn.Linear(config.state_dim, H),
                nn.SiLU(),
                nn.Linear(H, H),
            )
            if config.state_dim > 0
            else None
        )

        # --- optional future / register tokens ---
        self.future_tokens = (
            nn.Embedding(config.num_future_tokens, H)
            if config.num_future_tokens > 0
            else None
        )
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, config.action_horizon, H) * 0.02
        )

        # --- velocity decoder ---
        self.velocity_decoder = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, config.action_dim),
        )

        self.beta_dist = torch.distributions.Beta(config.beta_alpha, config.beta_beta)

    # ------------------------------------------------------------------
    #  Layer selection: 1-to-1 mapping, no grouping
    # ------------------------------------------------------------------

    def _resolve_layer_indices(self, num_available_layers: int) -> List[int]:
        """Return exactly ``num_layers`` VLM layer indices (one per DiT block).

        If ``condition_layer_indices`` is set in config it is used directly
        (must have length == num_layers).  Otherwise we pick ``num_layers``
        evenly spaced indices from [0, num_available_layers).
        """
        n_blocks = self.config.num_layers

        if self.config.condition_layer_indices is not None:
            indices = [int(i) for i in self.config.condition_layer_indices]
            if len(indices) != n_blocks:
                raise ValueError(
                    f"condition_layer_indices has {len(indices)} entries but "
                    f"num_layers={n_blocks}; they must match (1-to-1 mapping)."
                )
        else:
            # Evenly spaced, always including the last layer.
            indices = (
                torch.linspace(0, num_available_layers - 1, steps=n_blocks)
                .round()
                .to(torch.long)
                .tolist()
            )

        # Validate bounds.
        for idx in indices:
            if idx < 0 or idx >= num_available_layers:
                raise IndexError(
                    f"Condition layer index {idx} out of range for "
                    f"{num_available_layers} available VLM layers."
                )
        return indices

    def _select_hidden_states(
        self, hidden_states_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Pick one hidden state per DiT block — returns a flat list."""
        if not hidden_states_list:
            raise ValueError("Expected non-empty hidden_states_list.")
        indices = self._resolve_layer_indices(len(hidden_states_list))
        return [hidden_states_list[i] for i in indices]

    # ------------------------------------------------------------------
    #  Query construction
    # ------------------------------------------------------------------

    def _build_query(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T, _ = noisy_actions.shape
        action_emb = self.action_encoder(noisy_actions, timesteps)
        action_emb = action_emb + self.action_pos_embed[:, :T, :]

        parts: list[torch.Tensor] = []
        if self.state_encoder is not None and state is not None:
            parts.append(self.state_encoder(state).unsqueeze(1))
        if self.future_tokens is not None:
            parts.append(
                self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
            )
        parts.append(action_emb)

        return torch.cat(parts, dim=1)

    def _decode_action_velocity(
        self, hidden: torch.Tensor, action_steps: int
    ) -> torch.Tensor:
        pred = self.velocity_decoder(self.final_norm(hidden))
        return pred[:, -action_steps:, :]

    # ------------------------------------------------------------------
    #  Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        vlm_hidden_states_list: List[torch.Tensor],
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, A = actions.shape
        device = actions.device
        dtype = actions.dtype

        per_block_context = self._select_hidden_states(vlm_hidden_states_list)

        t = self.beta_dist.sample((B,)).to(device=device, dtype=dtype)
        noise = torch.randn(B, T, A, device=device, dtype=dtype)
        noisy_actions = (1.0 - t[:, None, None]) * noise + t[:, None, None] * actions
        true_velocity = actions - noise

        query = self._build_query(noisy_actions, t, state)
        time_emb = self.time_encoder(self.action_encoder.time_enc(t))

        for block_idx, block in enumerate(self.blocks):
            context = self.vlm_proj[block_idx](per_block_context[block_idx])
            query = block(query, context, time_emb, attention_mask)

        predicted_velocity = self._decode_action_velocity(query, T)
        loss = F.mse_loss(predicted_velocity, true_velocity)
        return {"loss": loss, "predicted_velocity": predicted_velocity}

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(
        self,
        vlm_hidden_states_list: List[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        deterministic_seed: Optional[int] = None,
    ) -> torch.Tensor:
        per_block_context = self._select_hidden_states(vlm_hidden_states_list)

        B = per_block_context[-1].shape[0]
        T = self.config.action_horizon
        device = per_block_context[-1].device
        dtype = per_block_context[-1].dtype

        steps = num_steps if num_steps is not None else self.config.num_inference_steps

        if deterministic_seed is not None:
            gen = torch.Generator(device=device).manual_seed(deterministic_seed)
            x = torch.randn(B, T, self.config.action_dim, device=device, dtype=dtype, generator=gen)
        else:
            x = torch.randn(B, T, self.config.action_dim, device=device, dtype=dtype)

        # Pre-project all VLM contexts (reused across Euler steps).
        projected_contexts = [
            self.vlm_proj[i](per_block_context[i]) for i in range(self.config.num_layers)
        ]

        dt = 1.0 / steps
        for step in range(steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            query = self._build_query(x, t, state)
            time_emb = self.time_encoder(self.action_encoder.time_enc(t))

            for block_idx, block in enumerate(self.blocks):
                query = block(query, projected_contexts[block_idx], time_emb, attention_mask)

            velocity = self._decode_action_velocity(query, T)
            x = x + velocity * dt

        return x.clamp(-1.0, 1.0)
