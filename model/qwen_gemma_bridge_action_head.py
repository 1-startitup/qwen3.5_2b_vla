import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_encoder import ActionEncoder
from .qwen35_pi05_action_head import ACTION_EXPERT_VARIANTS, create_sinusoidal_pos_embedding


@dataclass
class QwenGemmaBridgeActionConfig:
    head_type: str = "pi05_gemma_bridge"
    action_expert_variant: str = "gemma_300m"
    hidden_dim: int = 1024
    num_heads: int = 8
    num_kv_heads: int = 1
    num_layers: int = 18
    mlp_dim: int = 0
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

    tap_strategy: str = "last"
    stop_gradient_backbone: bool = True
    bridge_norm_type: str = "layernorm"
    bridge_out_dim: int = 0
    bridge_dropout: float = 0.0
    bridge_gate_bias: float = 0.0
    conditioning_mode: str = "per_layer_film"
    use_memory_summary_conditioning: bool = True
    use_text_summary_conditioning: bool = True
    use_instruction_summary_conditioning: bool = True
    use_state_conditioning: bool = True
    use_action_input_conditioning: bool = True
    memory_summary_gain_init: float = 1.0
    text_summary_gain_init: float = 2.0
    instruction_summary_gain_init: float = 3.0
    action_input_gain_init: float = 0.5
    conditioning_gate_bias: float = 1.0
    memory_norm_ratio_limit: float = 8.0

    def __post_init__(self):
        defaults = ACTION_EXPERT_VARIANTS.get(self.action_expert_variant, None)
        if defaults is None:
            raise ValueError(
                f"Unsupported action_expert_variant={self.action_expert_variant}. "
                f"Expected one of {sorted(ACTION_EXPERT_VARIANTS)}"
            )
        if self.hidden_dim <= 0:
            self.hidden_dim = int(defaults["hidden_dim"])
        if self.num_heads <= 0:
            self.num_heads = int(defaults["num_heads"])
        if self.num_layers <= 0:
            self.num_layers = int(defaults["num_layers"])
        if self.mlp_dim <= 0:
            self.mlp_dim = int(self.hidden_dim * 4)
        if self.bridge_out_dim <= 0:
            self.bridge_out_dim = int(self.hidden_dim)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        self.chunk_size = int(self.chunk_size)
        self.action_horizon = int(self.action_horizon)
        self.n_action_steps = int(self.n_action_steps)
        self.action_dim = int(self.action_dim)
        self.max_action_dim = int(self.max_action_dim)
        self.max_state_dim = int(self.max_state_dim)
        self.state_dim = int(self.state_dim)
        self.image_resolution = (
            int(self.image_resolution[0]),
            int(self.image_resolution[1]),
        )
        self.empty_cameras = int(self.empty_cameras)
        if self.action_dim > self.max_action_dim:
            raise ValueError(
                f"action_dim ({self.action_dim}) cannot exceed max_action_dim ({self.max_action_dim})"
            )
        if str(self.conditioning_mode).lower() != "per_layer_film":
            raise ValueError(
                f"Unsupported conditioning_mode={self.conditioning_mode}. Expected 'per_layer_film'"
            )


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x_float * rms
        return (out * self.weight.float()).to(dtype=x.dtype)


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


class GemmaAdaRMSNorm(nn.Module):
    """AdaRMS modulation for per-layer FiLM conditioning."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
        gate_bias: float = 1.0,
    ):
        super().__init__()
        self.norm = GemmaRMSNorm(dim=dim, eps=eps)
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0.02 / math.sqrt(max(cond_dim, 1)))
        nn.init.zeros_(self.dense.bias)
        with torch.no_grad():
            self.dense.bias.view(3, dim)[2].fill_(float(gate_bias))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm(hidden_states)
        cond = cond.to(device=self.dense.weight.device, dtype=self.dense.weight.dtype)
        modulation = self.dense(cond)
        if hidden_states.ndim == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        scale = scale.to(hidden_states.dtype)
        shift = shift.to(hidden_states.dtype)
        gate = torch.sigmoid(gate.to(hidden_states.dtype))
        return normed * (1.0 + scale) + shift, gate


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1).to(dtype=dtype)
        sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1).to(dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)


class GemmaAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)
        self.use_rope = bool(use_rope)

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim) if self.use_rope else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source = hidden_states if key_value_states is None else key_value_states
        batch_size, target_len, _ = hidden_states.shape
        source_len = source.shape[1]

        q = self.q_proj(hidden_states).view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(source).view(batch_size, source_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(source).view(batch_size, source_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None and key_value_states is None:
            cos, sin = self.rotary_emb(target_len, hidden_states.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos[:, :, :source_len, :], sin[:, :, :source_len, :])

        n_rep = self.num_heads // self.num_kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :].to(device=q.device, dtype=torch.bool)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, target_len, self.hidden_dim)
        return self.o_proj(attn_out)


class GemmaMLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(hidden_states), approximate="tanh") * self.up_proj(hidden_states))


class GemmaExpertBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        gate_bias: float = 1.0,
    ):
        super().__init__()
        self.input_adarms = GemmaAdaRMSNorm(hidden_dim, cond_dim, gate_bias=gate_bias)
        self.self_attn = GemmaAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            use_rope=True,
        )
        self.post_self_attn_adarms = GemmaAdaRMSNorm(hidden_dim, cond_dim, gate_bias=gate_bias)
        self.cross_attn = GemmaAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            use_rope=False,
        )
        self.post_cross_attn_adarms = GemmaAdaRMSNorm(hidden_dim, cond_dim, gate_bias=gate_bias)
        self.mlp = GemmaMLP(hidden_dim=hidden_dim, mlp_dim=mlp_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        expert_cond: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed, gate = self.input_adarms(hidden_states, expert_cond)
        self_update = self.self_attn(normed)
        hidden_states = gated_residual(hidden_states, self_update, gate)

        normed, gate = self.post_self_attn_adarms(hidden_states, expert_cond)
        cross_update = self.cross_attn(
            normed,
            key_value_states=memory,
            key_padding_mask=memory_mask,
        )
        hidden_states = gated_residual(hidden_states, cross_update, gate)

        normed, gate = self.post_cross_attn_adarms(hidden_states, expert_cond)
        mlp_update = self.mlp(normed)
        hidden_states = gated_residual(hidden_states, mlp_update, gate)
        return hidden_states


class GemmaActionExpert(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int,
        num_kv_heads: int,
        num_layers: int,
        mlp_dim: int,
        dropout: float = 0.0,
        gate_bias: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GemmaExpertBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    gate_bias=gate_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = GemmaRMSNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        expert_cond: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                memory=memory,
                expert_cond=expert_cond,
                memory_mask=memory_mask,
            )
        return self.final_norm(hidden_states)


class QwenMemoryBridge(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm_type: str = "layernorm",
        dropout: float = 0.0,
        gate_bias: float = 0.0,
    ):
        super().__init__()
        norm_key = str(norm_type).lower()
        if norm_key == "layernorm":
            self.norm: nn.Module = nn.LayerNorm(in_dim)
        elif norm_key == "rmsnorm":
            self.norm = GemmaRMSNorm(in_dim)
        else:
            raise ValueError(f"Unsupported bridge_norm_type={norm_type}. Expected one of ['layernorm', 'rmsnorm']")
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_bias)))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        bridged = self.proj(self.norm(memory))
        gate = torch.sigmoid(self.gate).to(dtype=bridged.dtype)
        return self.dropout(bridged) * gate * self.scale.to(dtype=bridged.dtype)


class QwenGemmaBridgeActionHead(nn.Module):
    """Flow-matching head driven by a Gemma-style expert over bridged Qwen memory."""

    def __init__(self, config: QwenGemmaBridgeActionConfig):
        super().__init__()
        self.config = config
        self.model_dim = int(config.hidden_dim)

        self.action_encoder = ActionEncoder(action_dim=config.max_action_dim, hidden_dim=self.model_dim)
        self.time_mlp_in = nn.Linear(self.model_dim, self.model_dim)
        self.time_mlp_out = nn.Linear(self.model_dim, self.model_dim)
        self.memory_bridge = QwenMemoryBridge(
            in_dim=int(config.vlm_hidden_dim),
            out_dim=int(config.bridge_out_dim),
            norm_type=str(config.bridge_norm_type),
            dropout=float(config.bridge_dropout),
            gate_bias=float(config.bridge_gate_bias),
        )
        self.bridge_to_expert = nn.Identity()
        if int(config.bridge_out_dim) != self.model_dim:
            self.bridge_to_expert = nn.Linear(int(config.bridge_out_dim), self.model_dim, bias=False)
        self.memory_summary_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.text_summary_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.instruction_summary_proj = nn.Sequential(
            nn.Linear(int(config.vlm_hidden_dim), self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(int(config.state_dim), self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.memory_summary_gain = nn.Parameter(torch.tensor(float(config.memory_summary_gain_init)))
        self.text_summary_gain = nn.Parameter(torch.tensor(float(config.text_summary_gain_init)))
        self.instruction_summary_gain = nn.Parameter(torch.tensor(float(config.instruction_summary_gain_init)))
        self.action_input_gain = nn.Parameter(torch.tensor(float(config.action_input_gain_init)))
        self.expert = GemmaActionExpert(
            hidden_dim=self.model_dim,
            cond_dim=self.model_dim,
            num_heads=int(config.num_heads),
            num_kv_heads=int(config.num_kv_heads),
            num_layers=int(config.num_layers),
            mlp_dim=int(config.mlp_dim),
            dropout=float(config.dropout),
            gate_bias=float(config.conditioning_gate_bias),
        )
        self.action_out_proj = nn.Linear(self.model_dim, int(config.max_action_dim))
        self.beta_dist = torch.distributions.Beta(float(config.beta_alpha), float(config.beta_beta))
        self.last_bridge_stats: dict[str, float] = {}

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

    def _masked_mean(
        self,
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=1)
        weights = mask.to(device=values.device)
        if weights.ndim == values.ndim - 1:
            weights = weights.unsqueeze(-1)
        weights = weights.to(values.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (values * weights).sum(dim=1) / denom

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

    def _encode_state(
        self,
        state: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not bool(self.config.use_state_conditioning) or state is None:
            return torch.zeros(batch_size, self.model_dim, device=device, dtype=dtype)
        state = state.to(
            device=self.state_encoder[0].weight.device,
            dtype=self.state_encoder[0].weight.dtype,
        )
        state = state[..., : int(self.config.state_dim)]
        return self.state_encoder(state).to(device=device, dtype=dtype)

    def _build_expert_condition(
        self,
        timestep: torch.Tensor,
        memory_summary: torch.Tensor,
        text_summary: Optional[torch.Tensor] = None,
        instruction_summary: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._time_condition(timestep)
        if bool(self.config.use_memory_summary_conditioning):
            memory_summary = memory_summary.to(
                device=self.memory_summary_proj[0].weight.device,
                dtype=self.memory_summary_proj[0].weight.dtype,
            )
            projected_memory = self.memory_summary_proj(memory_summary).to(device=cond.device, dtype=cond.dtype)
            cond = cond + self.memory_summary_gain.to(device=cond.device, dtype=cond.dtype) * projected_memory
        if bool(self.config.use_text_summary_conditioning):
            text_summary = memory_summary if text_summary is None else text_summary
            text_summary = text_summary.to(
                device=self.text_summary_proj[0].weight.device,
                dtype=self.text_summary_proj[0].weight.dtype,
            )
            projected_text = self.text_summary_proj(text_summary).to(device=cond.device, dtype=cond.dtype)
            cond = cond + self.text_summary_gain.to(device=cond.device, dtype=cond.dtype) * projected_text
        if bool(self.config.use_instruction_summary_conditioning) and instruction_summary is not None:
            instruction_summary = instruction_summary.to(
                device=self.instruction_summary_proj[0].weight.device,
                dtype=self.instruction_summary_proj[0].weight.dtype,
            )
            projected_instruction = self.instruction_summary_proj(instruction_summary).to(
                device=cond.device,
                dtype=cond.dtype,
            )
            cond = cond + self.instruction_summary_gain.to(device=cond.device, dtype=cond.dtype) * projected_instruction
        cond = cond + self._encode_state(
            state=state,
            batch_size=timestep.shape[0],
            device=cond.device,
            dtype=cond.dtype,
        )
        return cond

    def prepare_memory(
        self,
        prefix_memory: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        prefix_text_attention_mask: Optional[torch.Tensor] = None,
        use_memory_conditioning: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if bool(self.config.stop_gradient_backbone):
            prefix_memory = prefix_memory.detach()
        memory = prefix_memory.to(
            device=self.memory_bridge.proj.weight.device,
            dtype=self.memory_bridge.proj.weight.dtype,
        )
        memory = self.memory_bridge(memory)
        memory = self.bridge_to_expert(memory.to(self._module_dtype(self.bridge_to_expert, memory.dtype)))
        memory_mask = prefix_attention_mask
        if memory_mask is not None:
            memory_mask = memory_mask.to(device=memory.device, dtype=torch.bool)
        text_mask = prefix_text_attention_mask
        if text_mask is not None:
            text_mask = text_mask.to(device=memory.device, dtype=torch.bool)
            if memory_mask is not None:
                text_mask = text_mask & memory_mask
        elif memory_mask is not None:
            text_mask = memory_mask
        if not use_memory_conditioning:
            memory = torch.zeros_like(memory)
        memory_summary = self._masked_mean(memory, memory_mask)
        text_summary = self._masked_mean(memory, text_mask)
        stats = {
            "bridge_memory_norm": memory.float().norm(dim=-1).mean(),
            "text_summary_norm": text_summary.float().norm(dim=-1).mean(),
        }
        return memory, memory_mask, memory_summary, text_summary, stats

    def _module_dtype(self, module: nn.Module, fallback: torch.dtype) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return fallback

    def _predict_velocity(
        self,
        prefix_memory: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        prefix_text_attention_mask: Optional[torch.Tensor],
        instruction_summary: Optional[torch.Tensor],
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_memory_conditioning: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory, memory_mask, memory_summary, text_summary, stats = self.prepare_memory(
            prefix_memory,
            prefix_attention_mask,
            prefix_text_attention_mask=prefix_text_attention_mask,
            use_memory_conditioning=use_memory_conditioning,
        )
        noisy_actions = noisy_actions.to(
            device=self.action_out_proj.weight.device,
            dtype=self.action_out_proj.weight.dtype,
        )
        timestep = timestep.to(device=noisy_actions.device, dtype=noisy_actions.dtype)
        expert_cond = self._build_expert_condition(
            timestep,
            memory_summary,
            text_summary=text_summary,
            instruction_summary=instruction_summary,
            state=state,
        )
        action_tokens = self.action_encoder(noisy_actions, timestep)
        if bool(self.config.use_action_input_conditioning):
            action_tokens = action_tokens + (
                self.action_input_gain.to(device=action_tokens.device, dtype=action_tokens.dtype)
                * expert_cond.unsqueeze(1).to(dtype=action_tokens.dtype)
            )
        expert_hidden = self.expert(
            action_tokens,
            memory=memory,
            expert_cond=expert_cond,
            memory_mask=memory_mask,
        )
        velocity = self.action_out_proj(expert_hidden)

        action_norm = action_tokens.float().norm(dim=-1).mean()
        memory_norm = stats["bridge_memory_norm"].float()
        stats["expert_token_norm"] = action_norm
        stats["memory_norm_ratio"] = memory_norm / action_norm.clamp(min=1e-6)
        self.last_bridge_stats = {
            key: float(value.detach().cpu().item()) for key, value in stats.items()
        }
        return velocity, stats

    def forward(
        self,
        prefix_memory: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        prefix_text_attention_mask: Optional[torch.Tensor] = None,
        instruction_summary: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = actions.shape[0]
        device = actions.device
        dtype = actions.dtype

        padded_actions = self._pad_actions(actions)
        noise = torch.randn_like(padded_actions)
        timestep = self._sample_time(batch_size, device=device, dtype=dtype)
        timestep_expanded = timestep[:, None, None]
        x_t = timestep_expanded * noise + (1.0 - timestep_expanded) * padded_actions
        u_t = noise - padded_actions

        v_t, stats = self._predict_velocity(
            prefix_memory=prefix_memory,
            prefix_attention_mask=prefix_attention_mask,
            prefix_text_attention_mask=prefix_text_attention_mask,
            instruction_summary=instruction_summary,
            noisy_actions=x_t,
            timestep=timestep,
            state=state,
        )
        loss = F.mse_loss(v_t, u_t)
        return {
            "loss": loss,
            "motion_loss": loss,
            "predicted_velocity": v_t,
            "bridge_memory_norm": stats["bridge_memory_norm"],
            "expert_token_norm": stats["expert_token_norm"],
            "memory_norm_ratio": stats["memory_norm_ratio"],
        }

    @torch.no_grad()
    def predict_action(
        self,
        prefix_memory: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        prefix_text_attention_mask: Optional[torch.Tensor] = None,
        instruction_summary: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        deterministic_seed: Optional[int] = None,
        use_memory_conditioning: bool = True,
    ) -> torch.Tensor:
        batch_size = prefix_memory.shape[0]
        device = prefix_memory.device
        dtype = self.action_out_proj.weight.dtype
        steps = int(num_steps if num_steps is not None else self.config.num_inference_steps)

        noise_shape = (batch_size, self.config.chunk_size, self.config.max_action_dim)
        if deterministic_seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(deterministic_seed))
            x_t = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
        else:
            x_t = torch.randn(noise_shape, device=device, dtype=dtype)

        dt = -1.0 / float(max(steps, 1))
        for step_idx in range(steps):
            time_value = 1.0 + step_idx * dt
            timestep = torch.full((batch_size,), time_value, device=device, dtype=dtype)
            v_t, _ = self._predict_velocity(
                prefix_memory=prefix_memory,
                prefix_attention_mask=prefix_attention_mask,
                prefix_text_attention_mask=prefix_text_attention_mask,
                instruction_summary=instruction_summary,
                noisy_actions=x_t,
                timestep=timestep,
                state=state,
                use_memory_conditioning=use_memory_conditioning,
            )
            x_t = x_t + dt * v_t

        return x_t[..., : self.config.action_dim]
