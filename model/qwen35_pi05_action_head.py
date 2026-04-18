"""
pi0.5-style chunked flow matching action head conditioned on Qwen hidden states.

This preserves the key pi0.5 control protocol:
  - padded action/state dimensions
  - beta-sampled flow matching objective
  - reverse-time Euler sampler from t=1 -> 0
  - long action chunks with runtime queueing
"""

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from .cross_attention_dit import DiT
from .rtc import RTCConfig, RTCProcessor


ACTION_EXPERT_VARIANTS = {
    "gemma_300m": {"hidden_dim": 1024, "num_heads": 8, "num_layers": 18},
    "gemma_2b": {"hidden_dim": 2048, "num_heads": 8, "num_layers": 18},
}


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError(f"Expected time shape (B,), got {tuple(time.shape)}")

    device = time.device
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = (2.0 * math.pi) / period
    sin_input = scaling_factor[None, :] * time[:, None].float()
    embedding = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return embedding.to(dtype=time.dtype)


@dataclass
class QwenPI05ActionConfig:
    head_type: str = "pi05_qwen"
    action_expert_variant: str = "gemma_300m"
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 18
    dropout: float = 0.0

    action_dim: int = 7
    action_horizon: int = 50
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32

    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 0.004
    max_period: float = 4.0
    num_inference_steps: int = 10

    vlm_hidden_dim: int = 2048
    state_dim: int = 8
    state_prompt_bins: int = 256
    tokenizer_max_length: int = 200
    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0
    use_adarms: bool = True
    use_action_pos_embed: bool = False
    rtc_config: RTCConfig | None = None

    # Phase 1A: binary gripper, phase head, history conditioning
    history_len: int = 0
    history_feature_dim: int = 0
    use_state_conditioning: bool = False
    use_history_conditioning: bool = False
    use_binary_gripper: bool = False
    use_phase_head: bool = False
    phase_loss_weight: float = 0.10
    gripper_loss_weight: float = 0.20
    phase_gate_strength: float = 2.0
    grasp_phase_steps: int = 2
    # Phase 1B placeholders (disabled by default)
    use_immediate_correction: bool = False
    immediate_loss_weight: float = 0.25
    correction_blend: float = 0.50

    def __post_init__(self):
        defaults = ACTION_EXPERT_VARIANTS.get(self.action_expert_variant, None)
        if defaults is None:
            raise ValueError(
                f"Unsupported action_expert_variant={self.action_expert_variant}. "
                f"Expected one of {sorted(ACTION_EXPERT_VARIANTS)}"
            )
        if self.hidden_dim <= 0:
            self.hidden_dim = defaults["hidden_dim"]
        if self.num_heads <= 0:
            self.num_heads = defaults["num_heads"]
        if self.num_layers <= 0:
            self.num_layers = defaults["num_layers"]

        if self.chunk_size <= 0:
            self.chunk_size = int(self.action_horizon)
        if self.action_horizon <= 0:
            self.action_horizon = int(self.chunk_size)

        self.chunk_size = int(self.chunk_size)
        self.action_horizon = int(self.action_horizon)
        self.n_action_steps = int(self.n_action_steps)
        self.max_action_dim = int(self.max_action_dim)
        self.action_dim = int(self.action_dim)
        self.history_len = int(self.history_len)
        self.empty_cameras = int(self.empty_cameras)
        self.image_resolution = (
            int(self.image_resolution[0]),
            int(self.image_resolution[1]),
        )
        if self.history_feature_dim <= 0:
            self.history_feature_dim = int(self.state_dim) + int(self.action_dim) + 1

        if self.action_dim > self.max_action_dim:
            raise ValueError(
                f"action_dim ({self.action_dim}) cannot exceed max_action_dim ({self.max_action_dim})"
            )
        if self.rtc_config is not None and isinstance(self.rtc_config, dict):
            self.rtc_config = RTCConfig(**self.rtc_config)


class QwenPI05AdaRMSNorm(nn.Module):
    """Suffix-only adaptive RMSNorm with gated residuals, mirroring pi0.5 AdaRMS."""

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.cond_dim = int(cond_dim)
        self.eps = float(eps)
        self.dense = nn.Linear(self.cond_dim, self.dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    @classmethod
    def from_qwen_norm(
        cls,
        norm: nn.Module,
        cond_dim: int,
        gate_bias: float = 1.0,
    ) -> "QwenPI05AdaRMSNorm":
        eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6))
        dim = int(norm.weight.shape[0])
        module = cls(dim=dim, cond_dim=cond_dim, eps=eps)
        with torch.no_grad():
            bias = module.dense.bias.view(3, dim)
            bias.zero_()
            bias[0].copy_(norm.weight.detach().float())
            bias[2].fill_(float(gate_bias))
        return module

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = (x.float() * torch.rsqrt(var + self.eps)).to(dtype)
        cond = cond.to(device=self.dense.weight.device, dtype=self.dense.weight.dtype)
        modulation = self.dense(cond)
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        scale = scale.to(dtype)
        shift = shift.to(dtype)
        gate = gate.to(dtype)
        out = normed * (1.0 + scale) + shift
        return out, gate


def gated_residual(
    residual: Optional[torch.Tensor],
    update: Optional[torch.Tensor],
    gate: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if residual is None and update is None:
        return None
    if residual is None or update is None:
        return residual if residual is not None else update
    if gate is None:
        return residual + update
    return residual + update * gate


class QwenPI05SuffixExpertLayer(nn.Module):
    """Independent suffix stream layer, mirroring pi0.5's separate action expert."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen layer_type={self.layer_type}")
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)


class QwenPI05ActionHead(nn.Module):
    """Qwen-conditioned pi0.5-style flow matching head."""

    def __init__(self, config: QwenPI05ActionConfig):
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)

        self.context_proj = nn.Sequential(
            nn.Linear(config.vlm_hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_in_proj = nn.Linear(config.max_action_dim, hidden_dim)
        self.action_out_proj = nn.Linear(hidden_dim, config.max_action_dim)

        self.time_mlp_in = nn.Linear(hidden_dim, hidden_dim)
        self.time_mlp_out = nn.Linear(hidden_dim, hidden_dim)
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, config.chunk_size, hidden_dim) * 0.02
        )
        self.expert = DiT(
            hidden_dim=hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        self.beta_dist = torch.distributions.Beta(
            config.beta_alpha,
            config.beta_beta,
        )

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] == self.config.max_action_dim:
            return actions
        if actions.shape[-1] > self.config.max_action_dim:
            return actions[..., : self.config.max_action_dim]

        pad = actions.new_zeros(
            actions.shape[0],
            actions.shape[1],
            self.config.max_action_dim - actions.shape[-1],
        )
        return torch.cat([actions, pad], dim=-1)

    def _sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        time_beta = self.beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
        time = (
            time_beta * float(self.config.time_sampling_scale)
            + float(self.config.time_sampling_offset)
        )
        return time

    def _time_condition(self, timestep: torch.Tensor) -> torch.Tensor:
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.config.hidden_dim,
            min_period=float(self.config.min_period),
            max_period=float(self.config.max_period),
        )
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        return F.silu(time_emb)

    def _predict_velocity(
        self,
        noisy_actions: torch.Tensor,
        vlm_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_emb = self.action_in_proj(noisy_actions)
        action_emb = action_emb + self.action_pos_embed[:, : noisy_actions.shape[1], :]
        context = self.context_proj(vlm_hidden_states)
        time_cond = self._time_condition(timestep)
        hidden = self.expert(action_emb, context, time_cond, attention_mask)
        return self.action_out_proj(hidden)

    def forward(
        self,
        vlm_hidden_states: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        device = actions.device
        dtype = actions.dtype

        padded_actions = self._pad_actions(actions)
        noise = torch.randn_like(padded_actions)
        time = self._sample_time(batch_size, device=device, dtype=dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * padded_actions
        u_t = noise - padded_actions
        v_t = self._predict_velocity(
            x_t,
            vlm_hidden_states=vlm_hidden_states,
            timestep=time,
            attention_mask=attention_mask,
        )
        loss = F.mse_loss(v_t, u_t)
        return {"loss": loss, "predicted_velocity": v_t}

    @torch.no_grad()
    def predict_action(
        self,
        vlm_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        deterministic_seed: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size = vlm_hidden_states.shape[0]
        device = vlm_hidden_states.device
        dtype = vlm_hidden_states.dtype
        steps = int(num_steps if num_steps is not None else self.config.num_inference_steps)

        noise_shape = (
            batch_size,
            self.config.chunk_size,
            self.config.max_action_dim,
        )
        if deterministic_seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(deterministic_seed))
            x_t = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
        else:
            x_t = torch.randn(noise_shape, device=device, dtype=dtype)

        dt = -1.0 / float(max(steps, 1))
        for step_idx in range(steps):
            time_value = 1.0 + step_idx * dt
            timestep = torch.full((batch_size,), time_value, device=device, dtype=dtype)
            v_t = self._predict_velocity(
                x_t,
                vlm_hidden_states=vlm_hidden_states,
                timestep=timestep,
                attention_mask=attention_mask,
            )
            x_t = x_t + dt * v_t

        return x_t[..., : self.config.action_dim]


class QwenPI05ExpertHead(nn.Module):
    """pi0.5-style dual-stream Qwen action expert with separate suffix weights."""

    def __init__(self, config: QwenPI05ActionConfig):
        super().__init__()
        self.config = config
        self.model_dim = int(config.hidden_dim)
        self.motion_dim = int(config.action_dim) - 1

        self.action_in_proj = nn.Linear(config.max_action_dim, self.model_dim)
        self.action_out_proj = nn.Linear(self.model_dim, config.max_action_dim)
        self.time_mlp_in = nn.Linear(self.model_dim, self.model_dim)
        self.time_mlp_out = nn.Linear(self.model_dim, self.model_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(int(config.state_dim), self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(int(config.history_feature_dim), self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.alpha_state = nn.Parameter(torch.zeros(1))
        self.alpha_hist = nn.Parameter(torch.zeros(1))
        self.raw_gripper_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, 1),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, 4),
        )
        self.immediate_motion_head = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.motion_dim),
        )
        self.action_pos_embed = None
        if bool(config.use_action_pos_embed):
            self.action_pos_embed = nn.Parameter(
                torch.randn(1, config.chunk_size, self.model_dim) * 0.02
            )

        self.beta_dist = torch.distributions.Beta(
            config.beta_alpha,
            config.beta_beta,
        )
        self.suffix_layers = nn.ModuleList()
        self.input_adarms = nn.ModuleList()
        self.post_adarms = nn.ModuleList()
        self.final_adarms: Optional[QwenPI05AdaRMSNorm] = None
        self.selected_layer_count = 0
        self.layer_types: list[str] = []
        self.rtc_processor = RTCProcessor(config.rtc_config) if config.rtc_config is not None else None

    def _uses_state_conditioning(self) -> bool:
        return bool(getattr(self.config, "use_state_conditioning", False))

    def _uses_history_conditioning(self) -> bool:
        return bool(getattr(self.config, "use_history_conditioning", False)) and int(
            getattr(self.config, "history_len", 0)
        ) > 0

    def _uses_binary_gripper(self) -> bool:
        return bool(getattr(self.config, "use_binary_gripper", False))

    def _uses_phase_head(self) -> bool:
        return bool(getattr(self.config, "use_phase_head", False))

    def _masked_mean(self, values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=1)
        weights = mask.to(device=values.device)
        if weights.ndim == values.ndim - 1:
            weights = weights.unsqueeze(-1)
        weights = weights.to(values.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (values * weights).sum(dim=1) / denom

    def _language_model_dtype(
        self,
        language_model: nn.Module,
        fallback: torch.dtype,
    ) -> torch.dtype:
        try:
            return next(language_model.parameters()).dtype
        except StopIteration:
            return fallback

    def _module_dtype(self, module: nn.Module, fallback: torch.dtype) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return fallback

    def _expert_layer_count(self, language_model: nn.Module) -> int:
        requested = int(self.config.num_layers)
        available = len(language_model.layers)
        if requested <= 0:
            return available
        return min(requested, available)

    def initialize_time_conditioning(self, language_model: nn.Module) -> None:
        layer_count = self._expert_layer_count(language_model)
        if (
            len(self.suffix_layers) == layer_count
            and len(self.input_adarms) == layer_count
            and len(self.post_adarms) == layer_count
            and self.final_adarms is not None
        ):
            return

        expert_config = copy.deepcopy(language_model.config)
        expert_config.hidden_size = self.model_dim
        expert_config.intermediate_size = int(self.model_dim * 4)
        expert_config.num_hidden_layers = layer_count
        expert_config.layer_types = list(language_model.config.layer_types[:layer_count])
        expert_config.use_cache = False

        self.suffix_layers = nn.ModuleList(
            [QwenPI05SuffixExpertLayer(expert_config, layer_idx) for layer_idx in range(layer_count)]
        )

        eps = getattr(language_model.norm, "eps", getattr(language_model.norm, "variance_epsilon", 1e-6))
        norm_template = SimpleNamespace(weight=torch.zeros(self.model_dim), eps=eps)
        cond_dim = int(self.model_dim)
        self.input_adarms = nn.ModuleList(
            [QwenPI05AdaRMSNorm.from_qwen_norm(norm_template, cond_dim, gate_bias=1.0) for _ in range(layer_count)]
        )
        self.post_adarms = nn.ModuleList(
            [QwenPI05AdaRMSNorm.from_qwen_norm(norm_template, cond_dim, gate_bias=1.0) for _ in range(layer_count)]
        )
        self.final_adarms = QwenPI05AdaRMSNorm.from_qwen_norm(norm_template, cond_dim, gate_bias=1.0)

        module_device = next(self.action_in_proj.parameters()).device
        module_dtype = next(self.action_in_proj.parameters()).dtype
        self.suffix_layers.to(device=module_device, dtype=module_dtype)
        self.input_adarms.to(device=module_device, dtype=module_dtype)
        self.post_adarms.to(device=module_device, dtype=module_dtype)
        self.final_adarms.to(device=module_device, dtype=module_dtype)
        self.selected_layer_count = layer_count
        self.layer_types = list(expert_config.layer_types)

    def _ensure_initialized(self, language_model: nn.Module) -> None:
        self.initialize_time_conditioning(language_model)

    def _prefix_layers(self, language_model: nn.Module):
        self._ensure_initialized(language_model)
        return language_model.layers[: self.selected_layer_count]

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] == self.config.max_action_dim:
            return actions
        if actions.shape[-1] > self.config.max_action_dim:
            return actions[..., : self.config.max_action_dim]

        pad = actions.new_zeros(
            actions.shape[0],
            actions.shape[1],
            self.config.max_action_dim - actions.shape[-1],
        )
        return torch.cat([actions, pad], dim=-1)

    def _sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        time_beta = self.beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
        return (
            time_beta * float(self.config.time_sampling_scale)
            + float(self.config.time_sampling_offset)
        )

    def _time_condition(self, timestep: torch.Tensor) -> torch.Tensor:
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.model_dim,
            min_period=float(self.config.min_period),
            max_period=float(self.config.max_period),
        )
        time_emb = time_emb.to(
            device=self.time_mlp_in.weight.device,
            dtype=self.time_mlp_in.weight.dtype,
        )
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        return F.silu(time_emb)

    def _build_suffix_condition(
        self,
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._time_condition(timestep)

        if self._uses_state_conditioning() and state is not None and state.numel() > 0:
            state_hidden = self.state_encoder(
                state.to(
                    device=self.state_encoder[0].weight.device,
                    dtype=self.state_encoder[0].weight.dtype,
                )
            ).to(cond.dtype)
            cond = cond + self.alpha_state.to(cond.dtype) * state_hidden

        if self._uses_history_conditioning() and history is not None and history.numel() > 0:
            hist_hidden = self.history_encoder(
                history.to(
                    device=self.history_encoder[0].weight.device,
                    dtype=self.history_encoder[0].weight.dtype,
                )
            )
            hist_mask = history[..., -1] > 0.5
            hist_hidden = self._masked_mean(hist_hidden, hist_mask).to(cond.dtype)
            cond = cond + self.alpha_hist.to(cond.dtype) * hist_hidden

        return cond

    def build_suffix_embeddings(self, noisy_actions: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))
        if self.action_pos_embed is not None:
            action_emb = action_emb + self.action_pos_embed[:, : noisy_actions.shape[1], :]
        return action_emb

    def _normalize_prefix_position_ids(
        self,
        prefix_position_ids: Optional[torch.Tensor],
        prefix_attention_mask: Optional[torch.Tensor],
        batch_size: int,
        prefix_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if prefix_position_ids is None:
            base = torch.arange(prefix_length, device=device, dtype=torch.long)
            return base.view(1, 1, -1).expand(3, batch_size, -1)

        position_ids = prefix_position_ids.to(device=device)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        if position_ids.shape[0] != 3:
            raise ValueError(
                f"Expected prefix_position_ids with 3 or 4 leading rope dims, got shape {tuple(position_ids.shape)}"
            )
        return position_ids

    def build_suffix_position_ids(
        self,
        prefix_position_ids: Optional[torch.Tensor],
        prefix_attention_mask: Optional[torch.Tensor],
        suffix_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if prefix_attention_mask is None:
            if prefix_position_ids is None:
                raise ValueError("Either prefix_attention_mask or prefix_position_ids must be provided")
            batch_size = prefix_position_ids.shape[1] if prefix_position_ids.ndim == 3 else prefix_position_ids.shape[0]
            prefix_length = prefix_position_ids.shape[-1]
            prefix_attention_mask = torch.ones(
                batch_size,
                prefix_length,
                device=device,
                dtype=torch.long,
            )

        batch_size, prefix_length = prefix_attention_mask.shape
        prefix_pos = self._normalize_prefix_position_ids(
            prefix_position_ids=prefix_position_ids,
            prefix_attention_mask=prefix_attention_mask,
            batch_size=batch_size,
            prefix_length=prefix_length,
            device=device,
        )

        valid_mask = prefix_attention_mask.to(device=device).bool().unsqueeze(0)
        masked_prefix = prefix_pos.masked_fill(~valid_mask, -10_000)
        start = masked_prefix.max(dim=-1).values + 1
        steps = torch.arange(suffix_length, device=device, dtype=start.dtype).view(1, 1, -1)
        return start.unsqueeze(-1) + steps

    def _prefix_full_attention_mask(
        self,
        prefix_attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        prefix_mask = prefix_attention_mask.bool()
        attn = prefix_mask[:, :, None] & prefix_mask[:, None, :]
        bias = torch.zeros(attn.shape, device=prefix_attention_mask.device, dtype=dtype)
        bias.masked_fill_(~attn, torch.finfo(dtype).min)
        return bias[:, None, :, :]

    def _suffix_full_attention_mask(
        self,
        prefix_attention_mask: torch.Tensor,
        suffix_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = prefix_attention_mask.shape[0]
        device = prefix_attention_mask.device
        prefix_mask = prefix_attention_mask.bool()
        suffix_mask = torch.ones(batch_size, suffix_length, dtype=torch.bool, device=device)
        key_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        attn = suffix_mask[:, :, None] & key_mask[:, None, :]
        bias = torch.zeros(attn.shape, device=device, dtype=dtype)
        bias.masked_fill_(~attn, torch.finfo(dtype).min)
        return bias[:, None, :, :]

    def _project_full_attention(
        self,
        attn_module: Qwen3_5Attention,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(attn_module.q_proj.weight.dtype)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)

        query_states, gate = torch.chunk(
            attn_module.q_proj(hidden_states).view(*input_shape, -1, attn_module.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = attn_module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attn_module.k_norm(attn_module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states, gate

    def _apply_full_attention(
        self,
        attn_module: Qwen3_5Attention,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        gate: torch.Tensor,
        query_length: int,
    ) -> torch.Tensor:
        attn_output, _ = eager_attention_forward(
            attn_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=attn_module.scaling,
            dropout=0.0 if not self.training else attn_module.attention_dropout,
        )
        attn_output = attn_output.reshape(attn_output.shape[0], query_length, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate.to(attn_output.dtype))
        return attn_module.o_proj(attn_output.to(attn_module.o_proj.weight.dtype))

    def _run_prefix_linear_layer(
        self,
        prefix_layer: nn.Module,
        prefix_hidden: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = prefix_hidden
        prefix_norm = prefix_layer.input_layernorm(prefix_hidden)
        prefix_update = prefix_layer.linear_attn(
            hidden_states=prefix_norm,
            cache_params=None,
            attention_mask=prefix_attention_mask.bool(),
        )
        prefix_hidden = residual + prefix_update.to(residual.dtype)

        residual = prefix_hidden
        prefix_post = prefix_layer.post_attention_layernorm(prefix_hidden)
        prefix_mlp = prefix_layer.mlp(prefix_post)
        return residual + prefix_mlp.to(residual.dtype)

    def _run_prefix_full_attention_layer(
        self,
        language_model: nn.Module,
        prefix_layer: nn.Module,
        prefix_hidden: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = prefix_hidden
        prefix_norm = prefix_layer.input_layernorm(prefix_hidden)
        prefix_pos_emb = language_model.rotary_emb(prefix_norm, prefix_position_ids)
        prefix_q, prefix_k, prefix_v, prefix_gate = self._project_full_attention(
            prefix_layer.self_attn,
            prefix_norm,
            prefix_pos_emb,
        )
        prefix_mask = self._prefix_full_attention_mask(prefix_attention_mask, prefix_q.dtype)
        prefix_update = self._apply_full_attention(
            prefix_layer.self_attn,
            prefix_q,
            prefix_k,
            prefix_v,
            prefix_mask,
            prefix_gate,
            prefix_hidden.shape[1],
        )
        prefix_hidden = residual + prefix_update.to(residual.dtype)

        residual = prefix_hidden
        prefix_post = prefix_layer.post_attention_layernorm(prefix_hidden)
        prefix_mlp = prefix_layer.mlp(prefix_post)
        prefix_hidden = residual + prefix_mlp.to(residual.dtype)
        return prefix_hidden, prefix_k, prefix_v

    def _run_suffix_linear_layer(
        self,
        suffix_layer: QwenPI05SuffixExpertLayer,
        suffix_hidden: torch.Tensor,
        timestep_cond: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        residual = suffix_hidden
        suffix_norm, gate = self.input_adarms[layer_idx](suffix_hidden, timestep_cond)
        suffix_norm = suffix_norm.to(self._module_dtype(suffix_layer.linear_attn, suffix_norm.dtype))
        suffix_update = suffix_layer.linear_attn(
            hidden_states=suffix_norm,
            cache_params=None,
            attention_mask=None,
        )
        suffix_hidden = gated_residual(residual, suffix_update.to(residual.dtype), gate)

        residual = suffix_hidden
        suffix_post, gate = self.post_adarms[layer_idx](suffix_hidden, timestep_cond)
        suffix_post = suffix_post.to(self._module_dtype(suffix_layer.mlp, suffix_post.dtype))
        suffix_mlp = suffix_layer.mlp(suffix_post).to(residual.dtype)
        return gated_residual(residual, suffix_mlp, gate)

    def _run_suffix_full_attention_layer(
        self,
        language_model: nn.Module,
        suffix_layer: QwenPI05SuffixExpertLayer,
        suffix_hidden: torch.Tensor,
        timestep_cond: torch.Tensor,
        layer_idx: int,
        suffix_position_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_key_states: torch.Tensor,
        prefix_value_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = suffix_hidden
        suffix_norm, gate = self.input_adarms[layer_idx](suffix_hidden, timestep_cond)
        suffix_pos_emb = language_model.rotary_emb(suffix_norm, suffix_position_ids)
        suffix_q, suffix_k, suffix_v, suffix_attn_gate = self._project_full_attention(
            suffix_layer.self_attn,
            suffix_norm,
            suffix_pos_emb,
        )

        full_key_states = torch.cat([prefix_key_states.to(suffix_q.dtype), suffix_k], dim=2)
        full_value_states = torch.cat([prefix_value_states.to(suffix_v.dtype), suffix_v], dim=2)
        suffix_mask = self._suffix_full_attention_mask(
            prefix_attention_mask,
            suffix_hidden.shape[1],
            suffix_q.dtype,
        )
        suffix_update = self._apply_full_attention(
            suffix_layer.self_attn,
            suffix_q,
            full_key_states,
            full_value_states,
            suffix_mask,
            suffix_attn_gate,
            suffix_hidden.shape[1],
        )
        suffix_hidden = gated_residual(residual, suffix_update.to(residual.dtype), gate)

        residual = suffix_hidden
        suffix_post, gate = self.post_adarms[layer_idx](suffix_hidden, timestep_cond)
        suffix_post = suffix_post.to(self._module_dtype(suffix_layer.mlp, suffix_post.dtype))
        suffix_mlp = suffix_layer.mlp(suffix_post).to(residual.dtype)
        return gated_residual(residual, suffix_mlp, gate)

    def _run_dual_stream_layers(
        self,
        language_model: nn.Module,
        prefix_hidden: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        suffix_hidden: torch.Tensor,
        suffix_position_ids: torch.Tensor,
        timestep_cond: torch.Tensor,
    ) -> torch.Tensor:
        prefix_layers = self._prefix_layers(language_model)
        for layer_idx, prefix_layer in enumerate(prefix_layers):
            suffix_layer = self.suffix_layers[layer_idx]
            if self.layer_types[layer_idx] == "full_attention":
                prefix_hidden, prefix_k, prefix_v = self._run_prefix_full_attention_layer(
                    language_model=language_model,
                    prefix_layer=prefix_layer,
                    prefix_hidden=prefix_hidden,
                    prefix_attention_mask=prefix_attention_mask,
                    prefix_position_ids=prefix_position_ids,
                )
                suffix_hidden = self._run_suffix_full_attention_layer(
                    language_model=language_model,
                    suffix_layer=suffix_layer,
                    suffix_hidden=suffix_hidden,
                    timestep_cond=timestep_cond,
                    layer_idx=layer_idx,
                    suffix_position_ids=suffix_position_ids,
                    prefix_attention_mask=prefix_attention_mask,
                    prefix_key_states=prefix_k,
                    prefix_value_states=prefix_v,
                )
            else:
                prefix_hidden = self._run_prefix_linear_layer(
                    prefix_layer=prefix_layer,
                    prefix_hidden=prefix_hidden,
                    prefix_attention_mask=prefix_attention_mask,
                )
                suffix_hidden = self._run_suffix_linear_layer(
                    suffix_layer=suffix_layer,
                    suffix_hidden=suffix_hidden,
                    timestep_cond=timestep_cond,
                    layer_idx=layer_idx,
                )

        return self.final_adarms(suffix_hidden, timestep_cond)[0]

    @torch.no_grad()
    def build_prefix_cache(
        self,
        language_model: nn.Module,
        prefix_embeds: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        prefix_position_ids: Optional[torch.Tensor],
    ) -> dict:
        self._ensure_initialized(language_model)

        batch_size, prefix_length, _ = prefix_embeds.shape
        device = prefix_embeds.device
        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(
                batch_size,
                prefix_length,
                device=device,
                dtype=torch.long,
            )

        prefix_attention_mask = prefix_attention_mask.to(device=device)
        prefix_position_ids = self._normalize_prefix_position_ids(
            prefix_position_ids=prefix_position_ids,
            prefix_attention_mask=prefix_attention_mask,
            batch_size=batch_size,
            prefix_length=prefix_length,
            device=device,
        )

        prefix_hidden = prefix_embeds.to(self._language_model_dtype(language_model, prefix_embeds.dtype))
        cached_layers = []
        for layer_idx, prefix_layer in enumerate(self._prefix_layers(language_model)):
            if self.layer_types[layer_idx] == "full_attention":
                prefix_hidden, prefix_k, prefix_v = self._run_prefix_full_attention_layer(
                    language_model=language_model,
                    prefix_layer=prefix_layer,
                    prefix_hidden=prefix_hidden,
                    prefix_attention_mask=prefix_attention_mask,
                    prefix_position_ids=prefix_position_ids,
                )
                cached_layers.append({"key": prefix_k, "value": prefix_v})
            else:
                prefix_hidden = self._run_prefix_linear_layer(
                    prefix_layer=prefix_layer,
                    prefix_hidden=prefix_hidden,
                    prefix_attention_mask=prefix_attention_mask,
                )
                cached_layers.append(None)

        prefix_summary = self._masked_mean(prefix_hidden, prefix_attention_mask.bool())
        return {
            "layers": cached_layers,
            "prefix_attention_mask": prefix_attention_mask,
            "prefix_position_ids": prefix_position_ids,
            "prefix_summary": prefix_summary,
        }

    def _mask_velocity_dims(self, velocity: torch.Tensor) -> torch.Tensor:
        if not self._uses_binary_gripper():
            return velocity
        masked = velocity.clone()
        masked[..., self.motion_dim :] = 0
        return masked

    def _decode_phase_logits(self, suffix_hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._uses_phase_head():
            return None
        phase_input = suffix_hidden.to(self.phase_head[0].weight.dtype)
        return self.phase_head(phase_input)

    def _apply_phase_gating(
        self,
        raw_gripper_logits: torch.Tensor,
        phase_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if phase_logits is None:
            return raw_gripper_logits
        phase_probs = phase_logits.softmax(dim=-1)
        close_gate = phase_probs[..., 1] + phase_probs[..., 2]
        gate_bias = float(self.config.phase_gate_strength) * (2.0 * close_gate - 1.0)
        return raw_gripper_logits + gate_bias.to(raw_gripper_logits.dtype)

    def _decode_gripper_logits(
        self,
        suffix_hidden: torch.Tensor,
        phase_logits: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self._uses_binary_gripper():
            return None
        gripper_input = suffix_hidden.to(self.raw_gripper_head[0].weight.dtype)
        raw_gripper_logits = self.raw_gripper_head(gripper_input)[..., 0]
        return self._apply_phase_gating(raw_gripper_logits, phase_logits)

    def _derive_phase_targets(self, gripper_targets: torch.Tensor) -> torch.Tensor:
        close = gripper_targets > 0.5
        batch_size, horizon = close.shape
        targets = torch.zeros((batch_size, horizon), device=gripper_targets.device, dtype=torch.long)
        grasp_steps = max(int(self.config.grasp_phase_steps), 1)

        for batch_idx in range(batch_size):
            close_idx = torch.nonzero(close[batch_idx], as_tuple=False).flatten()
            if close_idx.numel() == 0:
                continue

            first_close = int(close_idx[0].item())
            last_close = int(close_idx[-1].item())
            grasp_end = min(first_close + grasp_steps - 1, horizon - 1)

            if first_close > 0:
                targets[batch_idx, :first_close] = 0
            targets[batch_idx, first_close : grasp_end + 1] = 1
            if grasp_end + 1 <= last_close:
                targets[batch_idx, grasp_end + 1 : last_close + 1] = 2
            if last_close + 1 < horizon:
                targets[batch_idx, last_close + 1 :] = 3

        return targets

    @torch.no_grad()
    def _run_suffix_from_cache(
        self,
        language_model: nn.Module,
        prefix_cache: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        prefix_attention_mask = prefix_cache["prefix_attention_mask"]
        prefix_position_ids = prefix_cache["prefix_position_ids"]
        suffix_position_ids = self.build_suffix_position_ids(
            prefix_position_ids=prefix_position_ids,
            prefix_attention_mask=prefix_attention_mask,
            suffix_length=x_t.shape[1],
            device=x_t.device,
        )

        suffix_hidden = self.build_suffix_embeddings(x_t)
        suffix_cond = self._build_suffix_condition(timestep, state=state, history=history)

        for layer_idx, suffix_layer in enumerate(self.suffix_layers):
            if self.layer_types[layer_idx] == "full_attention":
                cached = prefix_cache["layers"][layer_idx]
                suffix_hidden = self._run_suffix_full_attention_layer(
                    language_model=language_model,
                    suffix_layer=suffix_layer,
                    suffix_hidden=suffix_hidden,
                    timestep_cond=suffix_cond,
                    layer_idx=layer_idx,
                    suffix_position_ids=suffix_position_ids,
                    prefix_attention_mask=prefix_attention_mask,
                    prefix_key_states=cached["key"],
                    prefix_value_states=cached["value"],
                )
            else:
                suffix_hidden = self._run_suffix_linear_layer(
                    suffix_layer=suffix_layer,
                    suffix_hidden=suffix_hidden,
                    timestep_cond=suffix_cond,
                    layer_idx=layer_idx,
                )

        suffix_hidden = self.final_adarms(suffix_hidden, suffix_cond)[0]
        if return_hidden:
            return suffix_hidden

        velocity = self.action_out_proj(suffix_hidden.to(self.action_out_proj.weight.dtype))
        return self._mask_velocity_dims(velocity)

    def forward(
        self,
        language_model: nn.Module,
        prefix_embeds: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        prefix_position_ids: Optional[torch.Tensor],
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._ensure_initialized(language_model)

        batch_size = actions.shape[0]
        device = actions.device
        dtype = actions.dtype

        padded_actions = self._pad_actions(actions)
        if self._uses_binary_gripper():
            motion_targets = torch.zeros_like(padded_actions)
            motion_targets[..., : self.motion_dim] = padded_actions[..., : self.motion_dim]
            noise = torch.zeros_like(padded_actions)
            noise[..., : self.motion_dim] = torch.randn_like(motion_targets[..., : self.motion_dim])
            gripper_targets = (actions[..., self.motion_dim] > 0).to(dtype)
        else:
            motion_targets = padded_actions
            noise = torch.randn_like(padded_actions)
            gripper_targets = None
        timestep = self._sample_time(batch_size, device=device, dtype=dtype)
        timestep_expanded = timestep[:, None, None]
        x_t = timestep_expanded * noise + (1.0 - timestep_expanded) * motion_targets
        u_t = noise - motion_targets

        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(
                prefix_embeds.shape[0],
                prefix_embeds.shape[1],
                device=prefix_embeds.device,
                dtype=torch.long,
            )

        prefix_position_ids = self._normalize_prefix_position_ids(
            prefix_position_ids=prefix_position_ids,
            prefix_attention_mask=prefix_attention_mask,
            batch_size=prefix_embeds.shape[0],
            prefix_length=prefix_embeds.shape[1],
            device=prefix_embeds.device,
        )
        suffix_position_ids = self.build_suffix_position_ids(
            prefix_position_ids=prefix_position_ids,
            prefix_attention_mask=prefix_attention_mask,
            suffix_length=x_t.shape[1],
            device=prefix_embeds.device,
        )

        prefix_hidden = prefix_embeds.to(self._language_model_dtype(language_model, prefix_embeds.dtype))
        suffix_hidden = self.build_suffix_embeddings(x_t)
        suffix_cond = self._build_suffix_condition(timestep, state=state, history=history)
        suffix_hidden = self._run_dual_stream_layers(
            language_model=language_model,
            prefix_hidden=prefix_hidden,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
            suffix_hidden=suffix_hidden,
            suffix_position_ids=suffix_position_ids,
            timestep_cond=suffix_cond,
        )

        v_t = self.action_out_proj(suffix_hidden.to(self.action_out_proj.weight.dtype))
        motion_loss: torch.Tensor
        if self._uses_binary_gripper():
            v_t = self._mask_velocity_dims(v_t)
            motion_loss = F.mse_loss(v_t[..., : self.motion_dim], u_t[..., : self.motion_dim])
        else:
            motion_loss = F.mse_loss(v_t, u_t)

        zero = motion_loss.new_zeros(())
        if not self._uses_binary_gripper():
            return {
                "loss": motion_loss,
                "motion_loss": motion_loss,
                "gripper_loss": zero,
                "phase_loss": zero,
                "predicted_velocity": v_t,
            }

        phase_logits = self._decode_phase_logits(suffix_hidden)
        gripper_logits = self._decode_gripper_logits(suffix_hidden, phase_logits)
        if gripper_logits is None:
            raise RuntimeError("Binary gripper enabled but gripper logits were not produced.")
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logits, gripper_targets)
        if phase_logits is not None:
            phase_targets = self._derive_phase_targets(gripper_targets)
            phase_loss = F.cross_entropy(
                phase_logits.reshape(batch_size * phase_logits.shape[1], 4),
                phase_targets.reshape(batch_size * phase_targets.shape[1]),
            )
        else:
            phase_loss = zero

        total_loss = motion_loss + float(self.config.gripper_loss_weight) * gripper_loss
        if phase_logits is not None:
            total_loss = total_loss + float(self.config.phase_loss_weight) * phase_loss

        return {
            "loss": total_loss,
            "motion_loss": motion_loss,
            "gripper_loss": gripper_loss,
            "phase_loss": phase_loss,
            "predicted_velocity": v_t,
            "gripper_logits": gripper_logits,
            "phase_logits": phase_logits,
        }

    @torch.no_grad()
    def denoise_step(
        self,
        language_model: nn.Module,
        prefix_cache: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._ensure_initialized(language_model)
        return self._run_suffix_from_cache(
            language_model=language_model,
            prefix_cache=prefix_cache,
            x_t=x_t,
            timestep=timestep,
            state=state,
            history=history,
            return_hidden=False,
        )

    @torch.no_grad()
    def predict_action(
        self,
        language_model: nn.Module,
        prefix_embeds: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        prefix_position_ids: Optional[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        deterministic_seed: Optional[int] = None,
        inference_delay: Optional[int] = None,
        prev_chunk_left_over: Optional[torch.Tensor] = None,
        execution_horizon: Optional[int] = None,
    ) -> torch.Tensor:
        self._ensure_initialized(language_model)

        batch_size = prefix_embeds.shape[0]
        device = prefix_embeds.device
        dtype = self.action_in_proj.weight.dtype
        steps = int(num_steps if num_steps is not None else self.config.num_inference_steps)

        noise_shape = (
            batch_size,
            self.config.chunk_size,
            self.config.max_action_dim,
        )
        x_t = torch.zeros(noise_shape, device=device, dtype=dtype)
        if deterministic_seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(deterministic_seed))
            x_t[..., : self.motion_dim] = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.motion_dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            x_t[..., : self.motion_dim] = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.motion_dim,
                device=device,
                dtype=dtype,
            )
        if not self._uses_binary_gripper():
            if deterministic_seed is not None:
                generator = torch.Generator(device=device).manual_seed(int(deterministic_seed))
                x_t = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            else:
                x_t = torch.randn(noise_shape, device=device, dtype=dtype)

        prefix_cache = self.build_prefix_cache(
            language_model=language_model,
            prefix_embeds=prefix_embeds,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
        )

        dt = -1.0 / float(max(steps, 1))
        for step_idx in range(steps):
            time_value = 1.0 + step_idx * dt
            timestep = torch.full((batch_size,), time_value, device=device, dtype=dtype)
            if self.rtc_processor is not None and self.config.rtc_config is not None and self.config.rtc_config.enabled:
                prev_left_over = prev_chunk_left_over
                if prev_left_over is not None and not isinstance(prev_left_over, torch.Tensor):
                    prev_left_over = torch.as_tensor(prev_left_over, device=device, dtype=dtype)
                elif prev_left_over is not None:
                    prev_left_over = prev_left_over.to(device=device, dtype=dtype)

                def denoise_partial(input_x_t: torch.Tensor, current_timestep=timestep):
                    return self.denoise_step(
                        language_model=language_model,
                        prefix_cache=prefix_cache,
                        x_t=input_x_t,
                        timestep=current_timestep,
                        state=state,
                        history=history,
                    )

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_left_over,
                    inference_delay=inference_delay,
                    time=time_value,
                    original_denoise_step_partial=denoise_partial,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = self.denoise_step(
                    language_model=language_model,
                    prefix_cache=prefix_cache,
                    x_t=x_t,
                    timestep=timestep,
                    state=state,
                    history=history,
                )
            x_t = x_t + dt * v_t.to(dtype)

        if not self._uses_binary_gripper():
            return x_t[..., : self.config.action_dim]

        final_t = torch.full(
            (batch_size,),
            float(self.config.time_sampling_offset),
            device=device,
            dtype=dtype,
        )
        final_suffix_hidden = self._run_suffix_from_cache(
            language_model=language_model,
            prefix_cache=prefix_cache,
            x_t=x_t,
            timestep=final_t,
            state=state,
            history=history,
            return_hidden=True,
        )
        phase_logits = self._decode_phase_logits(final_suffix_hidden)
        gripper_logits = self._decode_gripper_logits(final_suffix_hidden, phase_logits)
        if gripper_logits is None:
            raise RuntimeError("Binary gripper enabled but gripper logits were not produced.")

        motion = x_t[..., : self.motion_dim].clone()
        gripper = torch.where(gripper_logits > 0.0, 1.0, -1.0).to(motion.dtype).unsqueeze(-1)
        return torch.cat([motion, gripper], dim=-1)
