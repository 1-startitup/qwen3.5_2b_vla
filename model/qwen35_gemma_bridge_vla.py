import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .qwen_backbone_adapter import QwenBackboneAdapter
from .qwen_gemma_bridge_action_head import (
    QwenGemmaBridgeActionConfig,
    QwenGemmaBridgeActionHead,
)

logger = logging.getLogger(__name__)


class Qwen35GemmaBridgeVLA(nn.Module):
    """Qwen backbone + cross-attention Gemma-style action expert."""

    architecture_name = "qwen_3.5_2b_gemma_bridge_v3_1"

    def __init__(
        self,
        vlm_model_id: str = "Qwen/Qwen3.5-2B",
        action_config: Optional[QwenGemmaBridgeActionConfig] = None,
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        vlm_loss_weight: float = 0.0,
        lora_config: Optional[Dict] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()

        if action_config is None:
            action_config = QwenGemmaBridgeActionConfig()

        self.backbone = QwenBackboneAdapter(
            model_id=vlm_model_id,
            attn_implementation=attn_implementation,
            image_resolution=getattr(action_config, "image_resolution", (224, 224)),
            empty_cameras=getattr(action_config, "empty_cameras", 0),
            tokenizer_max_length=getattr(action_config, "tokenizer_max_length", 200),
        )
        self.vlm = self.backbone.vlm
        action_config.vlm_hidden_dim = self.backbone.hidden_size

        self.action_config = action_config
        self.action_head_type = str(action_config.head_type)
        self.action_head = QwenGemmaBridgeActionHead(action_config)

        self.freeze_vision_encoder = bool(freeze_vision_encoder)
        self.freeze_vlm = bool(freeze_vlm)
        self.vlm_loss_weight = float(vlm_loss_weight)

        self.register_buffer("action_q01", torch.zeros(action_config.action_dim))
        self.register_buffer("action_q99", torch.ones(action_config.action_dim))
        self.register_buffer("local_action_q01", torch.zeros(action_config.action_dim))
        self.register_buffer("local_action_q99", torch.ones(action_config.action_dim))
        self.register_buffer("state_q01", torch.full((action_config.state_dim,), -1.0))
        self.register_buffer("state_q99", torch.full((action_config.state_dim,), 1.0))

        if lora_config is not None:
            self._apply_lora(lora_config)
        else:
            if self.freeze_vision_encoder:
                self.vlm.get_trainable_params(freeze_vision=True)
            if self.freeze_vlm:
                for param in self.vlm.parameters():
                    param.requires_grad = False
                logger.info("Frozen entire Qwen backbone for v3.1 bridge training")

        self._log_param_counts()

    def _apply_lora(self, lora_cfg: Dict):
        from peft import LoraConfig, get_peft_model

        for param in self.vlm.model.parameters():
            param.requires_grad = False

        peft_config = LoraConfig(
            r=lora_cfg.get("r", 32),
            lora_alpha=lora_cfg.get("alpha", 64),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.vlm.model = get_peft_model(self.vlm.model, peft_config)
        self.vlm.model.print_trainable_parameters()
        logger.info(
            "LoRA applied to qwen_3.5_2b_gemma_bridge_v3_1: r=%s alpha=%s",
            peft_config.r,
            peft_config.lora_alpha,
        )

    def _log_param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        action_params = sum(p.numel() for p in self.action_head.parameters())
        logger.info(
            "Qwen35GemmaBridgeVLA | Total: %.2fB | Trainable: %.1fM | Action head: %.1fM | Variant: %s | Tap strategy: %s",
            total / 1e9,
            trainable / 1e6,
            action_params / 1e6,
            self.action_config.action_expert_variant,
            self.action_config.tap_strategy,
        )

    def _action_head_dtype(self) -> torch.dtype:
        return next(self.action_head.parameters()).dtype

    def set_norm_stats(
        self,
        q01: torch.Tensor,
        q99: torch.Tensor,
        local_q01: Optional[torch.Tensor] = None,
        local_q99: Optional[torch.Tensor] = None,
        state_q01: Optional[torch.Tensor] = None,
        state_q99: Optional[torch.Tensor] = None,
    ):
        self.action_q01.copy_(q01)
        self.action_q99.copy_(q99)
        self.local_action_q01.copy_(q01 if local_q01 is None else local_q01)
        self.local_action_q99.copy_(q99 if local_q99 is None else local_q99)
        if state_q01 is not None:
            self.state_q01.copy_(state_q01)
        if state_q99 is not None:
            self.state_q99.copy_(state_q99)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        low = self.action_q01.to(device=actions.device, dtype=actions.dtype)
        high = self.action_q99.to(device=actions.device, dtype=actions.dtype)
        return 2.0 * (actions - low) / (high - low + 1e-8) - 1.0

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        low = self.action_q01.to(device=actions.device, dtype=actions.dtype)
        high = self.action_q99.to(device=actions.device, dtype=actions.dtype)
        raw = (actions + 1.0) / 2.0 * (high - low) + low
        if raw.shape[-1] >= 7:
            raw[..., 6] = torch.where(raw[..., 6] > 0.0, 1.0, -1.0)
        return raw

    def normalize_states(self, state: torch.Tensor) -> torch.Tensor:
        low = self.state_q01.to(device=state.device, dtype=state.dtype)
        high = self.state_q99.to(device=state.device, dtype=state.dtype)
        return (2.0 * (state - low) / (high - low + 1e-8) - 1.0).clamp(-1.0, 1.0)

    def _discretize_prompt_tensor(self, values: torch.Tensor, bins: int) -> torch.Tensor:
        clipped = values.float().clamp(-1.0, 1.0)
        scaled = (clipped + 1.0) * 0.5 * float(max(bins - 1, 1))
        return scaled.round().to(torch.int64)

    def _build_pi05_prompts(
        self,
        instructions: List[str],
        state: Optional[torch.Tensor],
    ) -> List[str]:
        if state is None:
            raise ValueError("v3.1 bridge requires state to build pi0.5-style prompts")

        bins = int(getattr(self.action_config, "state_prompt_bins", 256))
        norm_state = self.normalize_states(state.detach().float().cpu())
        state_bins = self._discretize_prompt_tensor(norm_state, bins)

        prompts: List[str] = []
        for idx, instruction in enumerate(instructions):
            cleaned = instruction.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, state_bins[idx].tolist()))
            prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")
        return prompts

    def forward(
        self,
        images: Optional[List[List[Any]]] = None,
        instructions: Optional[List[str]] = None,
        actions: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        vlm_inputs: Optional[Dict[str, torch.Tensor]] = None,
        vlm_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del history
        if actions is None:
            raise ValueError("Actions required for training forward pass")

        prompts = None
        if instructions is not None and state is not None:
            prompts = self._build_pi05_prompts(instructions, state)

        if vlm_inputs is None:
            if images is None or prompts is None:
                raise ValueError("images, instructions, and state are required when vlm_inputs is not provided")
            vlm_inputs = self.backbone.build_vla_inputs(images, prompts)

        if vlm_labels is None and self.vlm_loss_weight > 0 and "input_ids" in vlm_inputs:
            input_ids = vlm_inputs["input_ids"]
            vlm_labels = input_ids.clone()
            vlm_labels[vlm_labels == 0] = -100

        prefix_ctx = self.backbone.encode_prefix(
            images=images,
            prompts=prompts,
            vlm_inputs=vlm_inputs,
            vlm_labels=vlm_labels if self.vlm_loss_weight > 0 else None,
        )
        prefix_memory = self.backbone.get_prefix_memory(
            prefix_ctx,
            tap_strategy=self.action_config.tap_strategy,
        )

        action_head_dtype = self._action_head_dtype()
        norm_actions = self.normalize_actions(actions).to(action_head_dtype)
        norm_state = (
            self.normalize_states(state).to(action_head_dtype)
            if state is not None
            else None
        )
        action_out = self.action_head(
            prefix_memory=prefix_memory["memory"],
            prefix_attention_mask=prefix_memory["attention_mask"],
            prefix_text_attention_mask=prefix_memory.get("text_attention_mask"),
            instruction_summary=prefix_memory.get("instruction_summary"),
            actions=norm_actions,
            state=norm_state,
        )

        result = {"action_loss": action_out["loss"], "total_loss": action_out["loss"]}
        for key in ("motion_loss", "bridge_memory_norm", "expert_token_norm", "memory_norm_ratio"):
            if key in action_out:
                result[key] = action_out[key]

        if vlm_labels is not None and self.vlm_loss_weight > 0:
            vlm_outputs = prefix_ctx["vlm_outputs"]
            vlm_loss = (
                vlm_outputs.loss
                if getattr(vlm_outputs, "loss", None) is not None
                else torch.tensor(0.0, device=action_out["loss"].device)
            )
            result["vlm_loss"] = vlm_loss
            result["total_loss"] = action_out["loss"] + self.vlm_loss_weight * vlm_loss

        return result

    @torch.no_grad()
    def predict_action(
        self,
        images: List[List[Any]],
        instructions: List[str],
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        deterministic_seed: Optional[int] = None,
        inference_delay: Optional[int] = None,
        prev_chunk_left_over: Optional[torch.Tensor] = None,
        execution_horizon: Optional[int] = None,
    ) -> np.ndarray:
        del history, inference_delay, prev_chunk_left_over, execution_horizon
        self.eval()
        if state is None:
            raise ValueError("v3.1 bridge predict_action requires state")

        prompts = self._build_pi05_prompts(instructions, state)
        prefix_ctx = self.backbone.encode_prefix(images=images, prompts=prompts)
        prefix_memory = self.backbone.get_prefix_memory(
            prefix_ctx,
            tap_strategy=self.action_config.tap_strategy,
        )
        norm_state = self.normalize_states(state).to(self._action_head_dtype())
        norm_actions = self.action_head.predict_action(
            prefix_memory=prefix_memory["memory"],
            prefix_attention_mask=prefix_memory["attention_mask"],
            prefix_text_attention_mask=prefix_memory.get("text_attention_mask"),
            instruction_summary=prefix_memory.get("instruction_summary"),
            state=norm_state,
            num_steps=num_inference_steps,
            deterministic_seed=deterministic_seed,
        )
        raw_actions = self.unnormalize_actions(norm_actions)
        return raw_actions.float().cpu().numpy()

    def get_optimizer_groups(
        self,
        vlm_lr: float = 2.5e-5,
        action_lr: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> list:
        vlm_params = [p for p in self.vlm.parameters() if p.requires_grad]

        action_decay = []
        action_no_decay = []
        for name, param in self.action_head.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "gate" in name:
                action_no_decay.append(param)
            else:
                action_decay.append(param)

        groups = []
        if vlm_params:
            groups.append({"params": vlm_params, "lr": vlm_lr, "weight_decay": weight_decay})
        if action_decay:
            groups.append({"params": action_decay, "lr": action_lr, "weight_decay": weight_decay})
        if action_no_decay:
            groups.append({"params": action_no_decay, "lr": action_lr, "weight_decay": 0.0})
        return groups
