"""
Layer-wise flow matching action head (v4).

v4 extends v3 with closed-loop-oriented modifications:
1. Diffusion motion planner for 6D chunk prediction.
2. Discrete gripper head with phase-aware gating.
3. Explicit phase head (approach / grasp / transport / release).
4. Immediate control head for the executed first action, blended with the chunk plan.
5. Dynamic multi-task loss balancing so auxiliary losses do not dominate motion learning.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_encoder import ActionEncoder
from .cross_attention_dit import DiTBlock


@dataclass
class LayerwiseFlowMatchingV4Config:
    head_type: str = "layerwise_fm_v4"
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
    use_text_conditioning: bool = True
    use_state_prompt: bool = True
    state_prompt_bins: int = 256
    state_prompt_history_len: int = 0
    use_local_action_frame: bool = True

    phase_loss_weight: float = 0.10
    immediate_loss_weight: float = 0.25
    gripper_loss_weight: float = 0.20
    use_dynamic_loss_balance: bool = True
    correction_blend: float = 0.60
    phase_gate_strength: float = 2.0
    grasp_phase_steps: int = 2

    condition_layer_indices: Optional[List[int]] = None


class LayerwiseFlowMatchingActionHeadV4(nn.Module):
    """v4 action head with phase-aware gripper and immediate correction."""

    phase_names = ("approach", "grasp", "transport", "release")

    def __init__(self, config: LayerwiseFlowMatchingV4Config):
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
        self.raw_gripper_head = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, 1),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, 4),
        )
        self.immediate_motion_head = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.SiLU(),
            nn.Linear(H, self.motion_dim),
        )

        if config.use_dynamic_loss_balance:
            self.log_loss_scales = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter("log_loss_scales", None)

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
                    f"Condition layer index {idx} out of range for {num_available_layers} available VLM layers."
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

    def _split_hidden(self, hidden: torch.Tensor, action_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden = self.final_norm(hidden)
        action_hidden = norm_hidden[:, -action_steps:, :]
        prefix_hidden = norm_hidden[:, :-action_steps, :]
        return prefix_hidden, action_hidden

    def _decode_motion_velocity(self, action_hidden: torch.Tensor) -> torch.Tensor:
        return self.velocity_decoder(action_hidden)

    def _decode_phase_logits(self, action_hidden: torch.Tensor) -> torch.Tensor:
        return self.phase_head(action_hidden)

    def _apply_phase_gating(
        self,
        raw_gripper_logits: torch.Tensor,
        phase_logits: torch.Tensor,
    ) -> torch.Tensor:
        phase_probs = phase_logits.softmax(dim=-1)
        close_gate = phase_probs[..., 1] + phase_probs[..., 2]  # grasp + transport
        gate_bias = self.config.phase_gate_strength * (2.0 * close_gate - 1.0)
        return raw_gripper_logits + gate_bias

    def _decode_gripper_logits(self, action_hidden: torch.Tensor, phase_logits: torch.Tensor) -> torch.Tensor:
        raw = self.raw_gripper_head(action_hidden)[..., 0]
        return self._apply_phase_gating(raw, phase_logits)

    def _decode_immediate_motion(
        self,
        prefix_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_hidden.shape[1] > 0:
            prefix_summary = prefix_hidden.mean(dim=1)
        else:
            prefix_summary = action_hidden.mean(dim=1)
        first_action_hidden = action_hidden[:, 0, :]
        immediate_input = torch.cat([prefix_summary, first_action_hidden], dim=-1)
        return self.immediate_motion_head(immediate_input)

    def _derive_phase_targets(
        self,
        gripper_targets: torch.Tensor,
    ) -> torch.Tensor:
        close = gripper_targets > 0.5
        B, T = close.shape
        targets = torch.zeros((B, T), device=gripper_targets.device, dtype=torch.long)
        grasp_steps = max(int(self.config.grasp_phase_steps), 1)

        for b in range(B):
            close_idx = torch.nonzero(close[b], as_tuple=False).flatten()
            if close_idx.numel() == 0:
                continue

            first_close = int(close_idx[0].item())
            last_close = int(close_idx[-1].item())
            grasp_end = min(first_close + grasp_steps - 1, T - 1)

            if first_close > 0:
                targets[b, :first_close] = 0  # approach
            targets[b, first_close : grasp_end + 1] = 1  # grasp
            if grasp_end + 1 <= last_close:
                targets[b, grasp_end + 1 : last_close + 1] = 2  # transport
            if last_close + 1 < T:
                targets[b, last_close + 1 :] = 3  # release

        return targets

    def _combine_losses(
        self,
        motion_loss: torch.Tensor,
        gripper_loss: torch.Tensor,
        phase_loss: torch.Tensor,
        immediate_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.log_loss_scales is None:
            total = (
                motion_loss
                + self.config.gripper_loss_weight * gripper_loss
                + self.config.phase_loss_weight * phase_loss
                + self.config.immediate_loss_weight * immediate_loss
            )
            weights = {
                "loss_weight_motion": torch.ones_like(total),
                "loss_weight_gripper": torch.full_like(total, self.config.gripper_loss_weight),
                "loss_weight_phase": torch.full_like(total, self.config.phase_loss_weight),
                "loss_weight_immediate": torch.full_like(total, self.config.immediate_loss_weight),
            }
            return total, weights

        losses = [motion_loss, gripper_loss, phase_loss, immediate_loss]
        names = ["motion", "gripper", "phase", "immediate"]
        total = torch.zeros_like(motion_loss)
        weights: Dict[str, torch.Tensor] = {}
        for idx, (name, loss_val) in enumerate(zip(names, losses)):
            log_scale = self.log_loss_scales[idx].to(loss_val.dtype)
            inv_scale = torch.exp(-log_scale)
            total = total + inv_scale * loss_val + log_scale
            weights[f"loss_weight_{name}"] = inv_scale.detach()
        return total, weights

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
        phase_targets = self._derive_phase_targets(gripper_targets)

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
        prefix_hidden, action_hidden = self._split_hidden(hidden, T)

        predicted_velocity = self._decode_motion_velocity(action_hidden)
        phase_logits = self._decode_phase_logits(action_hidden)
        gripper_logits = self._decode_gripper_logits(action_hidden, phase_logits)
        immediate_motion = self._decode_immediate_motion(prefix_hidden, action_hidden)

        motion_loss = F.mse_loss(predicted_velocity, true_velocity)
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logits, gripper_targets)
        phase_loss = F.cross_entropy(
            phase_logits.reshape(B * T, 4),
            phase_targets.reshape(B * T),
        )
        immediate_loss = F.smooth_l1_loss(immediate_motion, motion_actions[:, 0, :])

        loss, weights = self._combine_losses(
            motion_loss=motion_loss,
            gripper_loss=gripper_loss,
            phase_loss=phase_loss,
            immediate_loss=immediate_loss,
        )

        return {
            "loss": loss,
            "motion_loss": motion_loss,
            "gripper_loss": gripper_loss,
            "phase_loss": phase_loss,
            "immediate_loss": immediate_loss,
            "predicted_velocity": predicted_velocity,
            "gripper_logits": gripper_logits,
            "phase_logits": phase_logits,
            "immediate_motion": immediate_motion,
            **weights,
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
            _, action_hidden = self._split_hidden(hidden, T)
            velocity = self._decode_motion_velocity(action_hidden)
            motion = motion + velocity * dt

        final_t = torch.ones((B,), device=device, dtype=dtype)
        final_query = self._build_query(motion, final_t, state, history, text_embedding)
        final_time_emb = self.time_encoder(self.action_encoder.time_enc(final_t))
        final_hidden = self._run_blocks(final_query, projected_contexts, final_time_emb, attention_mask)
        prefix_hidden, action_hidden = self._split_hidden(final_hidden, T)

        phase_logits = self._decode_phase_logits(action_hidden)
        gripper_logits = self._decode_gripper_logits(action_hidden, phase_logits)
        immediate_motion = self._decode_immediate_motion(prefix_hidden, action_hidden)

        correction_blend = float(self.config.correction_blend)
        motion = motion.clone()
        motion[:, 0, :] = (1.0 - correction_blend) * motion[:, 0, :] + correction_blend * immediate_motion

        gripper = torch.where(gripper_logits > 0.0, 1.0, -1.0).to(dtype).unsqueeze(-1)
        return torch.cat([motion.clamp(-1.0, 1.0), gripper], dim=-1)
