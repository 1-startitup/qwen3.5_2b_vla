"""
Layer-wise flow matching action head (v3).

v3 keeps the successful v2 layerwise cross-attention backbone, while adding:
1. Motion / gripper split: diffusion for 6D motion, BCE classification for gripper.
2. Short-term history tokens for closed-loop correction.
3. Explicit text-conditioning token plus an instruction consistency loss.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_encoder import ActionEncoder
from .cross_attention_dit import DiTBlock


@dataclass
class LayerwiseFlowMatchingV3Config:
    """Configuration for the v3 layerwise flow matching head."""

    head_type: str = "layerwise_fm_v3"
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

    history_len: int = 6
    history_dim: int = 16
    gripper_loss_weight: float = 0.25
    consistency_loss_weight: float = 0.01
    consistency_margin: float = 0.10
    use_text_conditioning: bool = True
    use_state_prompt: bool = True
    state_prompt_bins: int = 256
    state_prompt_history_len: int = 0

    condition_layer_indices: Optional[List[int]] = None


class LayerwiseFlowMatchingActionHeadV3(nn.Module):
    """
    v3 action head with explicit text, history, and discrete gripper prediction.
    """

    def __init__(self, config: LayerwiseFlowMatchingV3Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.motion_dim = config.action_dim - 1
        if self.motion_dim <= 0:
            raise ValueError(
                f"action_dim must be at least 2 for motion/gripper split, got {config.action_dim}"
            )

        H = config.hidden_dim

        self.action_encoder = ActionEncoder(
            self.motion_dim, H, timestep_scale=float(config.timestep_buckets)
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, H),
        )

        self.text_proj = (
            nn.Sequential(
                nn.Linear(config.vlm_hidden_dim, H),
                nn.SiLU(),
                nn.Linear(H, H),
            )
            if config.use_text_conditioning
            else None
        )

        self.state_encoder = (
            nn.Sequential(
                nn.Linear(config.state_dim, H),
                nn.SiLU(),
                nn.Linear(H, H),
            )
            if config.state_dim > 0
            else None
        )

        self.history_encoder = (
            nn.Sequential(
                nn.Linear(config.history_dim, H),
                nn.SiLU(),
                nn.Linear(H, H),
            )
            if config.history_len > 0
            else None
        )

        self.future_tokens = (
            nn.Embedding(config.num_future_tokens, H)
            if config.num_future_tokens > 0
            else None
        )
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, config.action_horizon, H) * 0.02
        )

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

        self.blocks = nn.ModuleList(
            [DiTBlock(H, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(H)

        self.velocity_decoder = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, self.motion_dim),
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, 1),
        )

        self.beta_dist = torch.distributions.Beta(config.beta_alpha, config.beta_beta)

    def _resolve_layer_indices(self, num_available_layers: int) -> List[int]:
        n_blocks = self.config.num_layers
        if self.config.condition_layer_indices is not None:
            indices = [int(i) for i in self.config.condition_layer_indices]
            if len(indices) != n_blocks:
                raise ValueError(
                    f"condition_layer_indices has {len(indices)} entries but "
                    f"num_layers={n_blocks}; they must match (1-to-1 mapping)."
                )
        else:
            indices = (
                torch.linspace(0, num_available_layers - 1, steps=n_blocks)
                .round()
                .to(torch.long)
                .tolist()
            )

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
        if not hidden_states_list:
            raise ValueError("Expected non-empty hidden_states_list.")
        indices = self._resolve_layer_indices(len(hidden_states_list))
        return [hidden_states_list[i] for i in indices]

    def _build_query(
        self,
        noisy_motion: torch.Tensor,
        timesteps: torch.Tensor,
        state: Optional[torch.Tensor],
        history: Optional[torch.Tensor],
        text_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T, _ = noisy_motion.shape
        motion_emb = self.action_encoder(noisy_motion, timesteps)
        motion_emb = motion_emb + self.action_pos_embed[:, :T, :]

        parts: List[torch.Tensor] = []
        if self.text_proj is not None and text_embedding is not None:
            parts.append(self.text_proj(text_embedding).unsqueeze(1))
        if self.state_encoder is not None and state is not None:
            parts.append(self.state_encoder(state).unsqueeze(1))
        if self.history_encoder is not None and history is not None and history.shape[1] > 0:
            parts.append(self.history_encoder(history))
        if self.future_tokens is not None:
            parts.append(self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1))
        parts.append(motion_emb)
        return torch.cat(parts, dim=1)

    def _run_blocks(
        self,
        query: torch.Tensor,
        projected_contexts: List[torch.Tensor],
        time_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden = query
        for block_idx, block in enumerate(self.blocks):
            hidden = block(hidden, projected_contexts[block_idx], time_emb, attention_mask)
        return hidden

    def _decode_motion_velocity(self, hidden: torch.Tensor, action_steps: int) -> torch.Tensor:
        pred = self.velocity_decoder(self.final_norm(hidden))
        return pred[:, -action_steps:, :]

    def _decode_gripper_logits(self, hidden: torch.Tensor, action_steps: int) -> torch.Tensor:
        logits = self.gripper_head(self.final_norm(hidden))
        return logits[:, -action_steps:, 0]

    def _build_wrong_text_embedding(
        self, text_embedding: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if text_embedding is None or text_embedding.shape[0] <= 1:
            return None
        return text_embedding.roll(shifts=1, dims=0)

    def forward(
        self,
        vlm_hidden_states_list: List[torch.Tensor],
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = actions.shape
        device = actions.device
        dtype = actions.dtype

        motion_actions = actions[..., : self.motion_dim]
        gripper_targets = (actions[..., self.motion_dim] > 0.0).to(dtype)

        per_block_context = self._select_hidden_states(vlm_hidden_states_list)
        projected_contexts = [
            self.vlm_proj[i](per_block_context[i]) for i in range(self.config.num_layers)
        ]

        t = self.beta_dist.sample((B,)).to(device=device, dtype=dtype)
        noise = torch.randn(B, T, self.motion_dim, device=device, dtype=dtype)
        noisy_motion = (1.0 - t[:, None, None]) * noise + t[:, None, None] * motion_actions
        true_velocity = motion_actions - noise

        query = self._build_query(noisy_motion, t, state, history, text_embedding)
        time_emb = self.time_encoder(self.action_encoder.time_enc(t))
        hidden = self._run_blocks(query, projected_contexts, time_emb, attention_mask)

        predicted_velocity = self._decode_motion_velocity(hidden, T)
        gripper_logits = self._decode_gripper_logits(hidden, T)

        motion_loss = F.mse_loss(predicted_velocity, true_velocity)
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logits, gripper_targets)

        consistency_loss = torch.zeros((), device=device, dtype=dtype)
        wrong_text_embedding = self._build_wrong_text_embedding(text_embedding)
        if (
            wrong_text_embedding is not None
            and self.text_proj is not None
            and self.config.consistency_loss_weight > 0
        ):
            wrong_query = self._build_query(
                noisy_motion,
                t,
                state,
                history,
                wrong_text_embedding,
            )
            wrong_hidden = self._run_blocks(
                wrong_query,
                projected_contexts,
                time_emb,
                attention_mask,
            )
            correct_summary = self.final_norm(hidden)[:, -T:, :].mean(dim=1)
            wrong_summary = self.final_norm(wrong_hidden)[:, -T:, :].mean(dim=1)
            distance = torch.norm(correct_summary - wrong_summary, dim=-1)
            consistency_loss = F.relu(self.config.consistency_margin - distance).mean()

        loss = (
            motion_loss
            + self.config.gripper_loss_weight * gripper_loss
            + self.config.consistency_loss_weight * consistency_loss
        )
        return {
            "loss": loss,
            "motion_loss": motion_loss,
            "gripper_loss": gripper_loss,
            "consistency_loss": consistency_loss,
            "predicted_velocity": predicted_velocity,
            "gripper_logits": gripper_logits,
        }

    @torch.no_grad()
    def predict_action(
        self,
        vlm_hidden_states_list: List[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        deterministic_seed: Optional[int] = None,
    ) -> torch.Tensor:
        per_block_context = self._select_hidden_states(vlm_hidden_states_list)
        projected_contexts = [
            self.vlm_proj[i](per_block_context[i]) for i in range(self.config.num_layers)
        ]

        B = per_block_context[-1].shape[0]
        T = self.config.action_horizon
        device = per_block_context[-1].device
        dtype = per_block_context[-1].dtype
        steps = num_steps if num_steps is not None else self.config.num_inference_steps

        if deterministic_seed is not None:
            gen = torch.Generator(device=device).manual_seed(deterministic_seed)
            motion = torch.randn(B, T, self.motion_dim, device=device, dtype=dtype, generator=gen)
        else:
            motion = torch.randn(B, T, self.motion_dim, device=device, dtype=dtype)

        dt = 1.0 / steps
        for step in range(steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            query = self._build_query(motion, t, state, history, text_embedding)
            time_emb = self.time_encoder(self.action_encoder.time_enc(t))
            hidden = self._run_blocks(query, projected_contexts, time_emb, attention_mask)
            velocity = self._decode_motion_velocity(hidden, T)
            motion = motion + velocity * dt

        final_t = torch.ones((B,), device=device, dtype=dtype)
        final_query = self._build_query(motion, final_t, state, history, text_embedding)
        final_time_emb = self.time_encoder(self.action_encoder.time_enc(final_t))
        final_hidden = self._run_blocks(final_query, projected_contexts, final_time_emb, attention_mask)
        gripper_logits = self._decode_gripper_logits(final_hidden, T)
        gripper = torch.where(gripper_logits > 0.0, 1.0, -1.0).to(dtype).unsqueeze(-1)

        return torch.cat([motion.clamp(-1.0, 1.0), gripper], dim=-1)
