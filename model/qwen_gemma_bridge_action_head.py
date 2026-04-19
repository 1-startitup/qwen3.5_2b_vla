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

    bridge_type: str = "gr00t_query_bridge"
    tap_strategy: str = "last"
    stop_gradient_backbone: bool = True
    bridge_norm_type: str = "layernorm"
    bridge_out_dim: int = 0
    bridge_dropout: float = 0.0
    bridge_gate_bias: float = 0.0
    bridge_num_queries: int = 8
    bridge_layers: int = 2
    bridge_policy_dim: int = 0
    bridge_use_state_token: bool = True
    bridge_use_tap_embeddings: bool = True
    bridge_use_token_type_embeddings: bool = True
    bridge_max_taps: int = 64
    bridge_num_token_types: int = 8
    bridge_scalar_mix: bool = False
    bridge_scalar_mix_init: str = "uniform"
    bridge_scalar_mix_use_gamma: bool = True

    use_action_input_conditioning: bool = True
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
        if self.bridge_policy_dim <= 0:
            self.bridge_policy_dim = int(self.bridge_out_dim or self.hidden_dim)
        if self.bridge_out_dim <= 0:
            self.bridge_out_dim = int(self.bridge_policy_dim)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.bridge_policy_dim % self.num_heads != 0:
            raise ValueError(
                f"bridge_policy_dim ({self.bridge_policy_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )
        if str(self.bridge_type).lower() != "gr00t_query_bridge":
            raise ValueError(
                f"Unsupported bridge_type={self.bridge_type}. Expected 'gr00t_query_bridge'"
            )
        if str(self.tap_strategy).lower() not in {"last", "all_concat", "scalar_mix"}:
            raise ValueError(
                f"Unsupported tap_strategy={self.tap_strategy}. Expected one of ['last', 'all_concat', 'scalar_mix']"
            )
        if str(self.bridge_scalar_mix_init).lower() not in {"uniform", "last_bias"}:
            raise ValueError(
                "Unsupported bridge_scalar_mix_init="
                f"{self.bridge_scalar_mix_init}. Expected one of ['uniform', 'last_bias']"
            )
        if str(self.tap_strategy).lower() == "scalar_mix":
            self.bridge_scalar_mix = True
        self.chunk_size = int(self.chunk_size)
        self.action_horizon = int(self.action_horizon)
        self.n_action_steps = int(self.n_action_steps)
        self.action_dim = int(self.action_dim)
        self.max_action_dim = int(self.max_action_dim)
        self.max_state_dim = int(self.max_state_dim)
        self.state_dim = int(self.state_dim)
        self.bridge_num_queries = int(self.bridge_num_queries)
        self.bridge_layers = int(self.bridge_layers)
        self.bridge_max_taps = int(self.bridge_max_taps)
        self.bridge_num_token_types = int(self.bridge_num_token_types)
        self.image_resolution = (
            int(self.image_resolution[0]),
            int(self.image_resolution[1]),
        )
        self.empty_cameras = int(self.empty_cameras)
        if self.action_dim > self.max_action_dim:
            raise ValueError(
                f"action_dim ({self.action_dim}) cannot exceed max_action_dim ({self.max_action_dim})"
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
        record_attention_stats: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = float(dropout)
        self.use_rope = bool(use_rope)
        self.record_attention_stats = bool(record_attention_stats)

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim) if self.use_rope else None
        self.last_attn_entropy: Optional[torch.Tensor] = None

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

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        valid_mask = None
        if key_padding_mask is not None:
            valid_mask = key_padding_mask[:, None, None, :].to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        if self.training and self.dropout > 0.0:
            probs = F.dropout(probs, p=self.dropout)
        if self.record_attention_stats:
            entropy = -(probs.clamp_min(1e-8).log() * probs).sum(dim=-1).mean()
            self.last_attn_entropy = entropy
        else:
            self.last_attn_entropy = None
        if valid_mask is not None:
            probs = probs.masked_fill(~valid_mask, 0.0)

        attn_out = torch.matmul(probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, target_len, self.hidden_dim)
        return self.o_proj(attn_out)


class GemmaMLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, hidden_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = F.gelu(self.gate_proj(hidden_states), approximate="tanh")
        return self.down_proj(gated * self.up_proj(hidden_states))


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
            record_attention_stats=True,
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
        self.last_cross_attn_entropy: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        expert_cond: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        entropies = []
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                memory=memory,
                expert_cond=expert_cond,
                memory_mask=memory_mask,
            )
            if layer.cross_attn.last_attn_entropy is not None:
                entropies.append(layer.cross_attn.last_attn_entropy)
        self.last_cross_attn_entropy = (
            torch.stack(entropies).mean() if entropies else None
        )
        return self.final_norm(hidden_states)


class QwenTokenProjector(nn.Module):
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
            raise ValueError(
                f"Unsupported bridge_norm_type={norm_type}. Expected one of ['layernorm', 'rmsnorm']"
            )
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_bias)))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        projected = self.proj(self.norm(memory))
        gate = torch.sigmoid(self.gate).to(dtype=projected.dtype)
        return self.dropout(projected) * gate * self.scale.to(dtype=projected.dtype)


class TapScalarMixer(nn.Module):
    """ELMo-style scalar mix over concatenated Qwen full-attention tap tokens."""

    def __init__(
        self,
        max_taps: int,
        init: str = "uniform",
        use_gamma: bool = True,
    ):
        super().__init__()
        self.max_taps = int(max_taps)
        self.use_gamma = bool(use_gamma)
        self.tap_logits = nn.Parameter(torch.zeros(self.max_taps))
        if str(init).lower() == "last_bias":
            with torch.no_grad():
                ramp = torch.linspace(-1.0, 1.0, steps=self.max_taps)
                self.tap_logits.copy_(ramp)
        elif str(init).lower() != "uniform":
            raise ValueError(
                f"Unsupported scalar mix init={init}. Expected one of ['uniform', 'last_bias']"
            )
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("gamma", torch.ones(1), persistent=False)

    def _unique_tap_ids(self, tap_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first_row = tap_ids[0]
        unique_ids, counts = torch.unique_consecutive(first_row, return_counts=True)
        if unique_ids.numel() == 0:
            raise ValueError("scalar_mix received empty tap_ids")
        if counts.unique().numel() != 1:
            raise ValueError(
                "scalar_mix expects concatenated taps with equal token counts per tap"
            )
        return unique_ids, counts

    def forward(
        self,
        tokens: torch.Tensor,
        memory_mask: torch.Tensor,
        tap_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if tap_ids is None:
            raise ValueError("scalar_mix requires tap_ids from the Qwen backbone adapter")
        if tokens.ndim != 3:
            raise ValueError(f"scalar_mix expects rank-3 tokens, got shape={tuple(tokens.shape)}")

        tap_ids = tap_ids.to(device=tokens.device, dtype=torch.long)
        memory_mask = memory_mask.to(device=tokens.device, dtype=torch.bool)
        unique_ids, counts = self._unique_tap_ids(tap_ids)
        num_taps = int(unique_ids.numel())
        tap_width = int(counts[0].item())
        if tokens.shape[1] != num_taps * tap_width:
            raise ValueError(
                "scalar_mix received unexpected memory layout: "
                f"seq={tokens.shape[1]} vs taps={num_taps} width={tap_width}"
            )

        tap_param_indices = unique_ids.clamp(min=0, max=self.max_taps - 1)
        weights = torch.softmax(self.tap_logits[tap_param_indices].float(), dim=0).to(dtype=tokens.dtype)
        gamma = self.gamma.to(device=tokens.device, dtype=tokens.dtype)

        reshaped_tokens = tokens.reshape(tokens.shape[0], num_taps, tap_width, tokens.shape[-1])
        reshaped_mask = memory_mask.reshape(memory_mask.shape[0], num_taps, tap_width)
        mixed_tokens = (reshaped_tokens * weights.view(1, num_taps, 1, 1)).sum(dim=1)
        mixed_tokens = mixed_tokens * gamma.view(1, 1, 1)
        mixed_mask = reshaped_mask.any(dim=1)

        stats = {
            "tap_weight_distribution": weights.float(),
            "tap_weight_entropy": (-(weights.float().clamp_min(1e-8).log() * weights.float()).sum()),
            "scalar_mix_gamma": gamma.float().view(()),
        }
        return mixed_tokens, mixed_mask, stats


class GR00TQueryBridge(nn.Module):
    def __init__(
        self,
        in_dim: int,
        policy_dim: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        state_dim: int,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        gate_bias: float = 0.0,
        use_state_token: bool = True,
        use_tap_embeddings: bool = True,
        use_token_type_embeddings: bool = True,
        max_taps: int = 64,
        num_token_types: int = 8,
        scalar_mix: bool = False,
        scalar_mix_init: str = "uniform",
        scalar_mix_use_gamma: bool = True,
    ):
        super().__init__()
        self.policy_dim = int(policy_dim)
        self.num_queries = int(num_queries)
        self.state_dim = int(state_dim)
        self.use_state_token = bool(use_state_token)
        self.use_tap_embeddings = bool(use_tap_embeddings)
        self.use_token_type_embeddings = bool(use_token_type_embeddings)
        self.use_scalar_mix = bool(scalar_mix)

        self.token_proj = QwenTokenProjector(
            in_dim=in_dim,
            out_dim=self.policy_dim,
            norm_type=norm_type,
            dropout=dropout,
            gate_bias=gate_bias,
        )
        self.query_tokens = nn.Parameter(
            torch.randn(self.num_queries, self.policy_dim) / math.sqrt(max(self.policy_dim, 1))
        )
        self.state_proj = nn.Linear(self.state_dim, self.policy_dim, bias=True)
        self.tap_embeddings = nn.Embedding(int(max_taps), self.policy_dim)
        self.token_type_embeddings = nn.Embedding(int(num_token_types), self.policy_dim)
        self.scalar_mixer = None
        if self.use_scalar_mix:
            self.scalar_mixer = TapScalarMixer(
                max_taps=int(max_taps),
                init=scalar_mix_init,
                use_gamma=scalar_mix_use_gamma,
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.policy_dim,
            nhead=int(num_heads),
            dim_feedforward=self.policy_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.final_norm = nn.LayerNorm(self.policy_dim)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        tap_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = memory.shape[0]
        if memory_mask is None:
            memory_mask = torch.ones(
                batch_size,
                memory.shape[1],
                device=memory.device,
                dtype=torch.bool,
            )
        else:
            memory_mask = memory_mask.to(device=memory.device, dtype=torch.bool)

        projected = self.token_proj(memory)
        if self.use_tap_embeddings and tap_ids is not None:
            tap_ids = tap_ids.to(device=projected.device, dtype=torch.long)
            tap_ids = tap_ids.clamp(min=0, max=self.tap_embeddings.num_embeddings - 1)
            projected = projected + self.tap_embeddings(tap_ids).to(dtype=projected.dtype)
        if self.use_token_type_embeddings and token_type_ids is not None:
            token_type_ids = token_type_ids.to(device=projected.device, dtype=torch.long)
            token_type_ids = token_type_ids.clamp(min=0, max=self.token_type_embeddings.num_embeddings - 1)
            projected = projected + self.token_type_embeddings(token_type_ids).to(dtype=projected.dtype)
        scalar_mix_stats: Dict[str, torch.Tensor] = {}
        if self.scalar_mixer is not None:
            projected, memory_mask, scalar_mix_stats = self.scalar_mixer(
                tokens=projected,
                memory_mask=memory_mask,
                tap_ids=tap_ids,
            )

        context_tokens = projected
        context_mask = memory_mask
        if self.use_state_token:
            if state is None:
                state_token = torch.zeros(
                    batch_size,
                    1,
                    self.policy_dim,
                    device=projected.device,
                    dtype=projected.dtype,
                )
            else:
                state = state.to(
                    device=self.state_proj.weight.device,
                    dtype=self.state_proj.weight.dtype,
                )
                state = state[..., : self.state_dim]
                state_token = self.state_proj(state).to(device=projected.device, dtype=projected.dtype)
                state_token = state_token.unsqueeze(1)
            context_tokens = torch.cat([state_token, context_tokens], dim=1)
            context_mask = torch.cat(
                [
                    torch.ones(batch_size, 1, device=context_mask.device, dtype=torch.bool),
                    context_mask,
                ],
                dim=1,
            )

        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=context_tokens.dtype)
        query_mask = torch.ones(batch_size, self.num_queries, device=context_mask.device, dtype=torch.bool)
        tokens = torch.cat([queries, context_tokens], dim=1)
        token_mask = torch.cat([query_mask, context_mask], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        query_tokens = self.final_norm(encoded[:, : self.num_queries])
        stats = {
            "raw_qwen_token_norm": memory.float().norm(dim=-1).mean(),
            "bridge_token_norm": context_tokens.float().norm(dim=-1).mean(),
            "query_token_norm": query_tokens.float().norm(dim=-1).mean(),
        }
        stats.update(scalar_mix_stats)
        return query_tokens, query_mask, stats


class QwenGemmaBridgeActionHead(nn.Module):
    """Flow-matching head driven by a GR00T-style token bridge plus Gemma expert."""

    def __init__(self, config: QwenGemmaBridgeActionConfig):
        super().__init__()
        self.config = config
        self.model_dim = int(config.hidden_dim)

        self.action_encoder = ActionEncoder(action_dim=config.max_action_dim, hidden_dim=self.model_dim)
        self.time_mlp_in = nn.Linear(self.model_dim, self.model_dim)
        self.time_mlp_out = nn.Linear(self.model_dim, self.model_dim)
        self.bridge = GR00TQueryBridge(
            in_dim=int(config.vlm_hidden_dim),
            policy_dim=int(config.bridge_policy_dim),
            num_queries=int(config.bridge_num_queries),
            num_layers=int(config.bridge_layers),
            num_heads=int(config.num_heads),
            state_dim=int(config.state_dim),
            dropout=float(config.bridge_dropout),
            norm_type=str(config.bridge_norm_type),
            gate_bias=float(config.bridge_gate_bias),
            use_state_token=bool(config.bridge_use_state_token),
            use_tap_embeddings=bool(config.bridge_use_tap_embeddings),
            use_token_type_embeddings=bool(config.bridge_use_token_type_embeddings),
            max_taps=int(config.bridge_max_taps),
            num_token_types=int(config.bridge_num_token_types),
            scalar_mix=bool(config.bridge_scalar_mix),
            scalar_mix_init=str(config.bridge_scalar_mix_init),
            scalar_mix_use_gamma=bool(config.bridge_scalar_mix_use_gamma),
        )
        self.bridge_to_expert = nn.Identity()
        if int(config.bridge_policy_dim) != self.model_dim:
            self.bridge_to_expert = nn.Linear(int(config.bridge_policy_dim), self.model_dim, bias=False)
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

    def _module_dtype(self, module: nn.Module, fallback: torch.dtype) -> torch.dtype:
        try:
            return next(module.parameters()).dtype
        except StopIteration:
            return fallback

    def prepare_memory(
        self,
        prefix_memory: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        tap_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        use_memory_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if bool(self.config.stop_gradient_backbone):
            prefix_memory = prefix_memory.detach()

        bridge_dtype = next(self.bridge.parameters()).dtype
        bridge_device = next(self.bridge.parameters()).device
        memory = prefix_memory.to(device=bridge_device, dtype=bridge_dtype)
        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(
                memory.shape[0],
                memory.shape[1],
                device=memory.device,
                dtype=torch.bool,
            )
        else:
            prefix_attention_mask = prefix_attention_mask.to(device=memory.device, dtype=torch.bool)
        if tap_ids is not None:
            tap_ids = tap_ids.to(device=memory.device, dtype=torch.long)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device=memory.device, dtype=torch.long)
        if not use_memory_conditioning:
            memory = torch.zeros_like(memory)

        bridge_state = None
        if bool(self.config.bridge_use_state_token) and state is not None:
            bridge_state = state.to(device=bridge_device, dtype=bridge_dtype)

        query_tokens, query_mask, stats = self.bridge(
            memory=memory,
            memory_mask=prefix_attention_mask,
            state=bridge_state,
            tap_ids=tap_ids,
            token_type_ids=token_type_ids,
        )
        query_tokens = self.bridge_to_expert(
            query_tokens.to(self._module_dtype(self.bridge_to_expert, query_tokens.dtype))
        )
        query_tokens = query_tokens.to(device=self.action_out_proj.weight.device, dtype=self.action_out_proj.weight.dtype)
        stats["bridge_memory_norm"] = query_tokens.float().norm(dim=-1).mean()
        return query_tokens, query_mask.to(device=query_tokens.device), stats

    def _predict_velocity(
        self,
        prefix_memory: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        tap_ids: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_memory_conditioning: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory, memory_mask, stats = self.prepare_memory(
            prefix_memory=prefix_memory,
            prefix_attention_mask=prefix_attention_mask,
            tap_ids=tap_ids,
            token_type_ids=token_type_ids,
            state=state,
            use_memory_conditioning=use_memory_conditioning,
        )
        noisy_actions = noisy_actions.to(
            device=self.action_out_proj.weight.device,
            dtype=self.action_out_proj.weight.dtype,
        )
        timestep = timestep.to(device=noisy_actions.device, dtype=noisy_actions.dtype)
        expert_cond = self._time_condition(timestep)
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
        per_sample_memory_norm = memory.float().norm(dim=-1).mean(dim=-1)
        per_sample_action_norm = action_tokens.float().norm(dim=-1).mean(dim=-1).clamp(min=1e-6)
        per_sample_ratio = per_sample_memory_norm / per_sample_action_norm
        stats["mean_norm_saturation_fraction"] = (
            per_sample_ratio > float(self.config.memory_norm_ratio_limit)
        ).float().mean()
        if self.expert.last_cross_attn_entropy is not None:
            stats["cross_attn_entropy"] = self.expert.last_cross_attn_entropy.float()
        serialized_stats: dict[str, float | list[float]] = {}
        for key, value in stats.items():
            if torch.is_tensor(value):
                detached = value.detach().cpu()
                if detached.numel() == 1:
                    serialized_stats[key] = float(detached.item())
                else:
                    serialized_stats[key] = detached.flatten().tolist()
            else:
                serialized_stats[key] = value
        self.last_bridge_stats = serialized_stats
        return velocity, stats

    def forward(
        self,
        prefix_memory: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        actions: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        tap_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
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
            tap_ids=tap_ids,
            token_type_ids=token_type_ids,
            noisy_actions=x_t,
            timestep=timestep,
            state=state,
        )
        loss = F.mse_loss(v_t, u_t)
        result = {
            "loss": loss,
            "motion_loss": loss,
            "predicted_velocity": v_t,
            "raw_qwen_token_norm": stats["raw_qwen_token_norm"],
            "bridge_token_norm": stats["bridge_token_norm"],
            "query_token_norm": stats["query_token_norm"],
            "bridge_memory_norm": stats["bridge_memory_norm"],
            "expert_token_norm": stats["expert_token_norm"],
            "memory_norm_ratio": stats["memory_norm_ratio"],
            "mean_norm_saturation_fraction": stats["mean_norm_saturation_fraction"],
        }
        if "cross_attn_entropy" in stats:
            result["cross_attn_entropy"] = stats["cross_attn_entropy"]
        if "tap_weight_entropy" in stats:
            result["tap_weight_entropy"] = stats["tap_weight_entropy"]
        if "scalar_mix_gamma" in stats:
            result["scalar_mix_gamma"] = stats["scalar_mix_gamma"]
        return result

    @torch.no_grad()
    def predict_action(
        self,
        prefix_memory: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        tap_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
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
                tap_ids=tap_ids,
                token_type_ids=token_type_ids,
                noisy_actions=x_t,
                timestep=timestep,
                state=state,
                use_memory_conditioning=use_memory_conditioning,
            )
            x_t = x_t + dt * v_t

        return x_t[..., : self.config.action_dim]
