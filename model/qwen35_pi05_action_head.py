"""Dual-stream Pi 0.5 action expert anchored on a Qwen3.5 backbone.

Replicates openpi Pi 0.5 (pi05=True) per-layer:
  - prefix (VLM) and suffix (action expert) tokens share attention via
    concatenated Q/K/V for `full_attention` layers, matching openpi's
    gemma.Block (single softmax, block-causal mask).
  - suffix carries its own transformer weights (gemma_300m-sized), trained
    from scratch, with adaRMSNorm + gated residual injecting the timestep.
  - Beta(1.5,1)*0.999+0.001 time sampling, x_t = t*noise + (1-t)*action,
    u_t = noise - action; reverse Euler dt = -1/N for sampling.

Qwen3.5 specifics retained:
  - Q/K RMSNorm and sigmoid attention gate inside Qwen3_5Attention.
  - Linear layers (Qwen3_5GatedDeltaNet) are run in parallel per stream (no
    suffix↔prefix coupling at that sublayer — GatedDeltaNet has no shared-KV
    semantics; suffix sees prefix only via interleaved full_attention layers).
  - 3D MRoPE position ids.
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


# gemma_300m-compatible expert: 18 layers, width 1024, 8 heads; head_dim is
# inherited from the backbone so joint attention can concat along seq.
ACTION_EXPERT_VARIANTS = {
    "gemma_300m": {"hidden_dim": 1024, "num_layers": 18},
    "gemma_2b": {"hidden_dim": 2048, "num_layers": 18},
}


@dataclass
class QwenPI05ActionConfig:
    head_type: str = "pi05_qwen"
    action_expert_variant: str = "gemma_300m"
    hidden_dim: int = 1024
    num_layers: int = 18

    action_dim: int = 7
    action_horizon: int = 10
    chunk_size: int = 10
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow-matching (matches openpi pi0.py:161,197-200,228)
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 0.004
    max_period: float = 4.0
    num_inference_steps: int = 10

    vlm_hidden_dim: int = 2048
    state_dim: int = 8
    tokenizer_max_length: int = 200
    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 1

    def __post_init__(self):
        defaults = ACTION_EXPERT_VARIANTS.get(self.action_expert_variant)
        if defaults is None:
            raise ValueError(
                f"Unsupported action_expert_variant={self.action_expert_variant}. "
                f"Expected one of {sorted(ACTION_EXPERT_VARIANTS)}"
            )
        if self.hidden_dim <= 0:
            self.hidden_dim = defaults["hidden_dim"]
        if self.num_layers <= 0:
            self.num_layers = defaults["num_layers"]

        if self.chunk_size <= 0:
            self.chunk_size = int(self.action_horizon)
        if self.action_horizon <= 0:
            self.action_horizon = int(self.chunk_size)
        if self.action_dim > self.max_action_dim:
            raise ValueError(
                f"action_dim ({self.action_dim}) cannot exceed max_action_dim ({self.max_action_dim})"
            )
        self.image_resolution = (int(self.image_resolution[0]), int(self.image_resolution[1]))


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
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    sin_input = (2.0 * math.pi / period)[None, :] * time[:, None].float()
    embedding = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return embedding.to(dtype=time.dtype)


class QwenPI05AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm injecting timestep conditioning (openpi gemma.py:112-131)."""

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6, gate_bias: float = 1.0):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.dense = nn.Linear(int(cond_dim), self.dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        with torch.no_grad():
            bias = self.dense.bias.view(3, self.dim)
            bias.zero_()
            bias[2].fill_(float(gate_bias))

    def set_scale_bias_from_rmsnorm(self, weight: torch.Tensor):
        with torch.no_grad():
            self.dense.bias.view(3, self.dim)[0].copy_(weight.detach().float())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = (x.float() * torch.rsqrt(var + self.eps)).to(dtype)
        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if x.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        return normed * (1.0 + scale.to(dtype)) + shift.to(dtype), gate.to(dtype)


def gated_residual(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return residual + update * gate


class QwenPI05SuffixExpertLayer(nn.Module):
    """A single suffix-expert layer matching the prefix layer type."""

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


class QwenPI05ExpertHead(nn.Module):
    """Pi 0.5 dual-stream expert, Qwen-backed.

    Construction is lazy: suffix weights are materialized against the backbone's
    `language_model` on first `forward` so layer-type and head_dim come from
    the live Qwen config rather than being duplicated in the dataclass.
    """

    def __init__(self, config: QwenPI05ActionConfig):
        super().__init__()
        self.config = config
        self.model_dim = int(config.hidden_dim)

        self.action_in_proj = nn.Linear(config.max_action_dim, self.model_dim)
        self.action_out_proj = nn.Linear(self.model_dim, config.max_action_dim)
        self.time_mlp_in = nn.Linear(self.model_dim, self.model_dim)
        self.time_mlp_out = nn.Linear(self.model_dim, self.model_dim)

        self.beta_dist = torch.distributions.Beta(config.beta_alpha, config.beta_beta)

        # Materialized lazily in `_ensure_initialized`.
        self.suffix_layers = nn.ModuleList()
        self.input_adarms = nn.ModuleList()
        self.post_adarms = nn.ModuleList()
        self.final_adarms: Optional[QwenPI05AdaRMSNorm] = None
        self.layer_types: list[str] = []

    # ---------- lazy suffix construction ------------------------------------

    def _ensure_initialized(self, language_model: nn.Module) -> None:
        layer_count = min(int(self.config.num_layers), len(language_model.layers))
        if self.final_adarms is not None and len(self.suffix_layers) == layer_count:
            return

        expert_config = copy.deepcopy(language_model.config)
        expert_config.hidden_size = self.model_dim
        expert_config.intermediate_size = self.model_dim * 4
        expert_config.num_hidden_layers = layer_count
        expert_config.layer_types = list(language_model.config.layer_types[:layer_count])
        expert_config.use_cache = False

        self.suffix_layers = nn.ModuleList(
            [QwenPI05SuffixExpertLayer(expert_config, i) for i in range(layer_count)]
        )

        eps = getattr(language_model.norm, "eps", getattr(language_model.norm, "variance_epsilon", 1e-6))
        cond_dim = self.model_dim
        self.input_adarms = nn.ModuleList(
            [QwenPI05AdaRMSNorm(self.model_dim, cond_dim, eps=eps) for _ in range(layer_count)]
        )
        self.post_adarms = nn.ModuleList(
            [QwenPI05AdaRMSNorm(self.model_dim, cond_dim, eps=eps) for _ in range(layer_count)]
        )
        self.final_adarms = QwenPI05AdaRMSNorm(self.model_dim, cond_dim, eps=eps)

        device = self.action_in_proj.weight.device
        dtype = self.action_in_proj.weight.dtype
        self.suffix_layers.to(device=device, dtype=dtype)
        self.input_adarms.to(device=device, dtype=dtype)
        self.post_adarms.to(device=device, dtype=dtype)
        self.final_adarms.to(device=device, dtype=dtype)
        self.layer_types = list(expert_config.layer_types)

    # ---------- flow matching primitives ------------------------------------

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        D = int(self.config.max_action_dim)
        if actions.shape[-1] == D:
            return actions
        if actions.shape[-1] > D:
            return actions[..., :D]
        pad = actions.new_zeros(actions.shape[0], actions.shape[1], D - actions.shape[-1])
        return torch.cat([actions, pad], dim=-1)

    def _sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = self.beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
        return t * float(self.config.time_sampling_scale) + float(self.config.time_sampling_offset)

    def _time_condition(self, timestep: torch.Tensor) -> torch.Tensor:
        dtype = self.time_mlp_in.weight.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep.to(dtype),
            self.model_dim,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
        )
        time_emb = F.silu(self.time_mlp_in(time_emb))
        return F.silu(self.time_mlp_out(time_emb))

    # ---------- position / mask helpers -------------------------------------

    @staticmethod
    def _normalize_prefix_position_ids(
        prefix_position_ids: Optional[torch.Tensor],
        batch_size: int,
        prefix_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if prefix_position_ids is None:
            base = torch.arange(prefix_length, device=device, dtype=torch.long)
            return base.view(1, 1, -1).expand(3, batch_size, -1)
        pos = prefix_position_ids.to(device)
        if pos.ndim == 2:
            pos = pos.unsqueeze(0).expand(3, -1, -1)
        elif pos.ndim == 3 and pos.shape[0] == 4:
            pos = pos[1:]
        if pos.shape[0] != 3:
            raise ValueError(f"prefix_position_ids must be 2D or have 3/4 rope axes, got {tuple(pos.shape)}")
        return pos

    @staticmethod
    def _build_suffix_position_ids(
        prefix_position_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        suffix_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        valid = prefix_attention_mask.to(device).bool().unsqueeze(0)  # (1, B, P)
        masked = prefix_position_ids.masked_fill(~valid, -10_000)
        start = masked.max(dim=-1).values + 1  # (3, B)
        steps = torch.arange(suffix_length, device=device, dtype=start.dtype).view(1, 1, -1)
        return start.unsqueeze(-1) + steps  # (3, B, S)

    @staticmethod
    def _joint_attention_mask(
        prefix_attention_mask: torch.Tensor,
        suffix_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Block-causal mask: prefix rows see prefix only; suffix rows see prefix+suffix."""
        B, P = prefix_attention_mask.shape
        S = suffix_length
        device = prefix_attention_mask.device
        p_mask = prefix_attention_mask.bool()
        s_mask = torch.ones(B, S, dtype=torch.bool, device=device)
        q_valid = torch.cat([p_mask, s_mask], dim=1)  # (B, P+S)

        block = torch.ones(P + S, P + S, dtype=torch.bool, device=device)
        block[:P, P:] = False  # prefix query × suffix key disallowed

        allow = q_valid[:, :, None] & q_valid[:, None, :] & block.unsqueeze(0)
        bias = torch.zeros_like(allow, dtype=dtype)
        bias.masked_fill_(~allow, torch.finfo(dtype).min)
        return bias[:, None, :, :]

    @staticmethod
    def _suffix_only_mask(
        prefix_attention_mask: torch.Tensor,
        suffix_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Mask for suffix-query-only attention over prefix+suffix keys (used at sampling)."""
        B, P = prefix_attention_mask.shape
        device = prefix_attention_mask.device
        prefix_mask = prefix_attention_mask.bool()
        suffix_mask = torch.ones(B, suffix_length, dtype=torch.bool, device=device)
        key_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        attn = suffix_mask[:, :, None] & key_mask[:, None, :]
        bias = torch.zeros(attn.shape, device=device, dtype=dtype)
        bias.masked_fill_(~attn, torch.finfo(dtype).min)
        return bias[:, None, :, :]

    # ---------- attention projection ---------------------------------------

    @staticmethod
    def _project_attn(
        attn_module: Qwen3_5Attention,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(attn_module.q_proj.weight.dtype)
        B, T = hidden_states.shape[:2]
        head_dim = attn_module.head_dim

        q_out = attn_module.q_proj(hidden_states).view(B, T, -1, head_dim * 2)
        q_raw, q_gate = torch.chunk(q_out, 2, dim=-1)
        q_gate = q_gate.reshape(B, T, -1)

        q = attn_module.q_norm(q_raw).transpose(1, 2)
        k = attn_module.k_norm(attn_module.k_proj(hidden_states).view(B, T, -1, head_dim)).transpose(1, 2)
        v = attn_module.v_proj(hidden_states).view(B, T, -1, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v, q_gate

    @staticmethod
    def _finalize_attn(
        attn_module: Qwen3_5Attention,
        attn_output: torch.Tensor,
        q_gate: torch.Tensor,
        query_length: int,
    ) -> torch.Tensor:
        # eager_attention_forward returns (B, T, H, D); reshape to (B, T, H*D).
        B = attn_output.shape[0]
        x = attn_output.reshape(B, query_length, -1).contiguous()
        x = x * torch.sigmoid(q_gate.to(x.dtype))
        return attn_module.o_proj(x.to(attn_module.o_proj.weight.dtype))

    # ---------- joint full-attention layer (training) ----------------------

    def _dual_full_attention_layer(
        self,
        language_model: nn.Module,
        prefix_layer: nn.Module,
        suffix_layer: QwenPI05SuffixExpertLayer,
        prefix_hidden: torch.Tensor,
        suffix_hidden: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        suffix_position_ids: torch.Tensor,
        timestep_cond: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        P = prefix_hidden.shape[1]
        S = suffix_hidden.shape[1]

        # Pre-attn norms.
        p_norm = prefix_layer.input_layernorm(prefix_hidden)
        s_norm, s_res_gate = self.input_adarms[layer_idx](suffix_hidden, timestep_cond)

        # Project + rope per expert (different weights, different positions).
        p_pe = language_model.rotary_emb(p_norm, prefix_position_ids)
        s_pe = language_model.rotary_emb(s_norm, suffix_position_ids)
        p_q, p_k, p_v, p_gate = self._project_attn(prefix_layer.self_attn, p_norm, p_pe)
        s_q, s_k, s_v, s_gate = self._project_attn(suffix_layer.self_attn, s_norm, s_pe)

        # Concat along seq axis (dim=2 after transpose) → single softmax.
        joint_q = torch.cat([p_q, s_q], dim=2)
        joint_k = torch.cat([p_k, s_k], dim=2)
        joint_v = torch.cat([p_v, s_v], dim=2)
        mask = self._joint_attention_mask(prefix_attention_mask, S, joint_q.dtype)

        joint_out, _ = eager_attention_forward(
            prefix_layer.self_attn,
            joint_q,
            joint_k,
            joint_v,
            mask,
            scaling=prefix_layer.self_attn.scaling,
            dropout=0.0 if not self.training else prefix_layer.self_attn.attention_dropout,
        )
        # eager_attention_forward returns (B, T, H, D) — split along T.
        p_out = joint_out[:, :P]
        s_out = joint_out[:, P:]

        p_attn = self._finalize_attn(prefix_layer.self_attn, p_out, p_gate, P)
        s_attn = self._finalize_attn(suffix_layer.self_attn, s_out, s_gate, S)

        prefix_hidden = prefix_hidden + p_attn.to(prefix_hidden.dtype)
        suffix_hidden = gated_residual(suffix_hidden, s_attn.to(suffix_hidden.dtype), s_res_gate)

        # MLP sublayer (independent per expert).
        p_mlp = prefix_layer.mlp(prefix_layer.post_attention_layernorm(prefix_hidden))
        prefix_hidden = prefix_hidden + p_mlp.to(prefix_hidden.dtype)

        s_post, s_mlp_gate = self.post_adarms[layer_idx](suffix_hidden, timestep_cond)
        s_mlp = suffix_layer.mlp(s_post)
        suffix_hidden = gated_residual(suffix_hidden, s_mlp.to(suffix_hidden.dtype), s_mlp_gate)

        return prefix_hidden, suffix_hidden

    def _dual_linear_layer(
        self,
        prefix_layer: nn.Module,
        suffix_layer: QwenPI05SuffixExpertLayer,
        prefix_hidden: torch.Tensor,
        suffix_hidden: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        timestep_cond: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefix: vanilla Qwen linear-attention block.
        p_norm = prefix_layer.input_layernorm(prefix_hidden)
        p_upd = prefix_layer.linear_attn(
            hidden_states=p_norm, cache_params=None, attention_mask=prefix_attention_mask.bool(),
        )
        prefix_hidden = prefix_hidden + p_upd.to(prefix_hidden.dtype)
        p_mlp = prefix_layer.mlp(prefix_layer.post_attention_layernorm(prefix_hidden))
        prefix_hidden = prefix_hidden + p_mlp.to(prefix_hidden.dtype)

        # Suffix: parallel linear block with adaRMSNorm.
        s_norm, s_gate_attn = self.input_adarms[layer_idx](suffix_hidden, timestep_cond)
        s_upd = suffix_layer.linear_attn(hidden_states=s_norm, cache_params=None, attention_mask=None)
        suffix_hidden = gated_residual(suffix_hidden, s_upd.to(suffix_hidden.dtype), s_gate_attn)
        s_post, s_gate_mlp = self.post_adarms[layer_idx](suffix_hidden, timestep_cond)
        s_mlp = suffix_layer.mlp(s_post)
        suffix_hidden = gated_residual(suffix_hidden, s_mlp.to(suffix_hidden.dtype), s_gate_mlp)

        return prefix_hidden, suffix_hidden

    # ---------- full dual-stream forward -----------------------------------

    def _run_dual_stream(
        self,
        language_model: nn.Module,
        prefix_hidden: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        suffix_hidden: torch.Tensor,
        suffix_position_ids: torch.Tensor,
        timestep_cond: torch.Tensor,
    ) -> torch.Tensor:
        prefix_layers = language_model.layers[: len(self.suffix_layers)]
        for layer_idx, prefix_layer in enumerate(prefix_layers):
            suffix_layer = self.suffix_layers[layer_idx]
            if self.layer_types[layer_idx] == "full_attention":
                prefix_hidden, suffix_hidden = self._dual_full_attention_layer(
                    language_model, prefix_layer, suffix_layer,
                    prefix_hidden, suffix_hidden,
                    prefix_attention_mask, prefix_position_ids, suffix_position_ids,
                    timestep_cond, layer_idx,
                )
            else:
                prefix_hidden, suffix_hidden = self._dual_linear_layer(
                    prefix_layer, suffix_layer,
                    prefix_hidden, suffix_hidden,
                    prefix_attention_mask, timestep_cond, layer_idx,
                )
        return self.final_adarms(suffix_hidden, timestep_cond)[0]

    # ---------- training forward -------------------------------------------

    def forward(
        self,
        language_model: nn.Module,
        prefix_embeds: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        prefix_position_ids: Optional[torch.Tensor],
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._ensure_initialized(language_model)

        B, _, _ = prefix_embeds.shape
        device = prefix_embeds.device
        act_dtype = self.action_in_proj.weight.dtype

        padded_actions = self._pad_actions(actions).to(act_dtype)
        noise = torch.randn_like(padded_actions)
        t = self._sample_time(B, device=device, dtype=act_dtype)
        t_exp = t[:, None, None]
        x_t = t_exp * noise + (1.0 - t_exp) * padded_actions
        u_t = noise - padded_actions

        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(B, prefix_embeds.shape[1], device=device, dtype=torch.long)
        prefix_attention_mask = prefix_attention_mask.to(device)
        prefix_position_ids = self._normalize_prefix_position_ids(
            prefix_position_ids, B, prefix_embeds.shape[1], device
        )
        suffix_position_ids = self._build_suffix_position_ids(
            prefix_position_ids, prefix_attention_mask, x_t.shape[1], device
        )

        prefix_hidden = prefix_embeds.to(next(language_model.parameters()).dtype)
        suffix_hidden = self.action_in_proj(x_t)
        timestep_cond = self._time_condition(t)

        suffix_out = self._run_dual_stream(
            language_model=language_model,
            prefix_hidden=prefix_hidden,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
            suffix_hidden=suffix_hidden,
            suffix_position_ids=suffix_position_ids,
            timestep_cond=timestep_cond,
        )
        v_t = self.action_out_proj(suffix_out.to(self.action_out_proj.weight.dtype))
        loss = F.mse_loss(v_t, u_t)
        return {"loss": loss, "predicted_velocity": v_t}

    # ---------- sampling (KV-cached prefix) --------------------------------

    @torch.no_grad()
    def _build_prefix_cache(
        self,
        language_model: nn.Module,
        prefix_embeds: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        prefix_position_ids: torch.Tensor,
    ) -> dict:
        """Run prefix through its own stream alone, caching K/V at each full-attn layer."""
        prefix_hidden = prefix_embeds.to(next(language_model.parameters()).dtype)
        cached = []
        for layer_idx, prefix_layer in enumerate(language_model.layers[: len(self.suffix_layers)]):
            if self.layer_types[layer_idx] == "full_attention":
                p_norm = prefix_layer.input_layernorm(prefix_hidden)
                p_pe = language_model.rotary_emb(p_norm, prefix_position_ids)
                p_q, p_k, p_v, p_gate = self._project_attn(prefix_layer.self_attn, p_norm, p_pe)
                # prefix-only self-attention (suffix not yet present)
                mask = self._prefix_only_mask(prefix_attention_mask, p_q.dtype)
                out, _ = eager_attention_forward(
                    prefix_layer.self_attn, p_q, p_k, p_v, mask,
                    scaling=prefix_layer.self_attn.scaling, dropout=0.0,
                )
                p_attn = self._finalize_attn(prefix_layer.self_attn, out, p_gate, prefix_hidden.shape[1])
                prefix_hidden = prefix_hidden + p_attn.to(prefix_hidden.dtype)
                p_mlp = prefix_layer.mlp(prefix_layer.post_attention_layernorm(prefix_hidden))
                prefix_hidden = prefix_hidden + p_mlp.to(prefix_hidden.dtype)
                cached.append({"k": p_k, "v": p_v})
            else:
                p_norm = prefix_layer.input_layernorm(prefix_hidden)
                p_upd = prefix_layer.linear_attn(
                    hidden_states=p_norm, cache_params=None, attention_mask=prefix_attention_mask.bool(),
                )
                prefix_hidden = prefix_hidden + p_upd.to(prefix_hidden.dtype)
                p_mlp = prefix_layer.mlp(prefix_layer.post_attention_layernorm(prefix_hidden))
                prefix_hidden = prefix_hidden + p_mlp.to(prefix_hidden.dtype)
                cached.append(None)
        return {"layers": cached, "mask": prefix_attention_mask, "positions": prefix_position_ids}

    @staticmethod
    def _prefix_only_mask(prefix_attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        m = prefix_attention_mask.bool()
        attn = m[:, :, None] & m[:, None, :]
        bias = torch.zeros(attn.shape, device=m.device, dtype=dtype)
        bias.masked_fill_(~attn, torch.finfo(dtype).min)
        return bias[:, None, :, :]

    @torch.no_grad()
    def _denoise_step(
        self,
        language_model: nn.Module,
        prefix_cache: dict,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        S = x_t.shape[1]
        device = x_t.device
        suffix_hidden = self.action_in_proj(x_t.to(self.action_in_proj.weight.dtype))
        suffix_position_ids = self._build_suffix_position_ids(
            prefix_cache["positions"], prefix_cache["mask"], S, device
        )
        timestep_cond = self._time_condition(timestep)

        for layer_idx, prefix_layer in enumerate(language_model.layers[: len(self.suffix_layers)]):
            suffix_layer = self.suffix_layers[layer_idx]
            if self.layer_types[layer_idx] == "full_attention":
                s_norm, s_gate_res = self.input_adarms[layer_idx](suffix_hidden, timestep_cond)
                s_pe = language_model.rotary_emb(s_norm, suffix_position_ids)
                s_q, s_k, s_v, s_gate = self._project_attn(suffix_layer.self_attn, s_norm, s_pe)

                p_k = prefix_cache["layers"][layer_idx]["k"].to(s_q.dtype)
                p_v = prefix_cache["layers"][layer_idx]["v"].to(s_v.dtype)
                full_k = torch.cat([p_k, s_k], dim=2)
                full_v = torch.cat([p_v, s_v], dim=2)
                mask = self._suffix_only_mask(prefix_cache["mask"], S, s_q.dtype)
                out, _ = eager_attention_forward(
                    suffix_layer.self_attn, s_q, full_k, full_v, mask,
                    scaling=suffix_layer.self_attn.scaling, dropout=0.0,
                )
                s_attn = self._finalize_attn(suffix_layer.self_attn, out, s_gate, S)
                suffix_hidden = gated_residual(suffix_hidden, s_attn.to(suffix_hidden.dtype), s_gate_res)

                s_post, s_gate_mlp = self.post_adarms[layer_idx](suffix_hidden, timestep_cond)
                s_mlp = suffix_layer.mlp(s_post)
                suffix_hidden = gated_residual(suffix_hidden, s_mlp.to(suffix_hidden.dtype), s_gate_mlp)
            else:
                s_norm, s_gate_attn = self.input_adarms[layer_idx](suffix_hidden, timestep_cond)
                s_upd = suffix_layer.linear_attn(hidden_states=s_norm, cache_params=None, attention_mask=None)
                suffix_hidden = gated_residual(suffix_hidden, s_upd.to(suffix_hidden.dtype), s_gate_attn)
                s_post, s_gate_mlp = self.post_adarms[layer_idx](suffix_hidden, timestep_cond)
                s_mlp = suffix_layer.mlp(s_post)
                suffix_hidden = gated_residual(suffix_hidden, s_mlp.to(suffix_hidden.dtype), s_gate_mlp)

        suffix_hidden = self.final_adarms(suffix_hidden, timestep_cond)[0]
        return self.action_out_proj(suffix_hidden.to(self.action_out_proj.weight.dtype))

    @torch.no_grad()
    def predict_action(
        self,
        language_model: nn.Module,
        prefix_embeds: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        prefix_position_ids: Optional[torch.Tensor],
        num_steps: Optional[int] = None,
        deterministic_seed: Optional[int] = None,
    ) -> torch.Tensor:
        self._ensure_initialized(language_model)

        B = prefix_embeds.shape[0]
        device = prefix_embeds.device
        dtype = self.action_in_proj.weight.dtype
        steps = int(num_steps if num_steps is not None else self.config.num_inference_steps)

        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(B, prefix_embeds.shape[1], device=device, dtype=torch.long)
        prefix_attention_mask = prefix_attention_mask.to(device)
        prefix_position_ids = self._normalize_prefix_position_ids(
            prefix_position_ids, B, prefix_embeds.shape[1], device
        )
        prefix_cache = self._build_prefix_cache(
            language_model, prefix_embeds, prefix_attention_mask, prefix_position_ids
        )

        shape = (B, self.config.chunk_size, self.config.max_action_dim)
        if deterministic_seed is not None:
            g = torch.Generator(device=device).manual_seed(int(deterministic_seed))
            x_t = torch.randn(shape, generator=g, device=device, dtype=dtype)
        else:
            x_t = torch.randn(shape, device=device, dtype=dtype)

        dt = -1.0 / float(max(steps, 1))
        for step_idx in range(steps):
            time_value = 1.0 + step_idx * dt
            timestep = torch.full((B,), time_value, device=device, dtype=dtype)
            v_t = self._denoise_step(language_model, prefix_cache, x_t, timestep)
            x_t = x_t + dt * v_t.to(dtype)

        return x_t[..., : self.config.action_dim]
