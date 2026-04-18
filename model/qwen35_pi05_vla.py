"""
Qwen3.5-2B + pi0.5-style VLA.

This model keeps the Qwen multimodal backbone while matching the pi0.5 control
contract:
  - chunked flow matching with reverse-time sampling
  - state-to-text prompt conditioning
  - padded internal action dimension
  - pi0.5-style runtime chunk semantics
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .qwen35_pi05_action_head import QwenPI05ActionConfig, QwenPI05ExpertHead
from .qwen35_pi05_interface import Qwen35PI05Interface

logger = logging.getLogger(__name__)


class Qwen35PI05VLA(nn.Module):
    """Qwen VLA variant aligned with pi0.5 preprocessing and flow matching."""

    architecture_name = "qwen_3.5_2b_pi0.5"

    def __init__(
        self,
        vlm_model_id: str = "Qwen/Qwen3.5-2B",
        action_config: Optional[QwenPI05ActionConfig] = None,
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        vlm_loss_weight: float = 0.0,
        lora_config: Optional[Dict] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()

        if action_config is None:
            action_config = QwenPI05ActionConfig()

        self.vlm = Qwen35PI05Interface(
            model_id=vlm_model_id,
            attn_implementation=attn_implementation,
            image_resolution=getattr(action_config, "image_resolution", (224, 224)),
            empty_cameras=getattr(action_config, "empty_cameras", 0),
            tokenizer_max_length=getattr(action_config, "tokenizer_max_length", 200),
        )
        action_config.vlm_hidden_dim = self.vlm.hidden_size

        self.action_config = action_config
        self.action_head_type = str(action_config.head_type)
        self.action_head = QwenPI05ExpertHead(action_config)

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
                logger.info("Frozen entire VLM backbone — training pi0.5-style action head only")

        self.action_head.initialize_time_conditioning(self._language_model())
        self._log_param_counts()

    def _action_head_dtype(self) -> torch.dtype:
        return next(self.action_head.parameters()).dtype

    def _language_model(self) -> nn.Module:
        model_core = self.vlm._model_core()
        if hasattr(model_core, "language_model"):
            return model_core.language_model
        raise AttributeError("Qwen35PI05Interface model core does not expose language_model")

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
            "LoRA applied to qwen_3.5_2b_pi0.5: r=%s alpha=%s",
            peft_config.r,
            peft_config.lora_alpha,
        )

    def _log_param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        action_params = sum(p.numel() for p in self.action_head.parameters())
        logger.info(
            "Qwen35PI05VLA | Total: %.2fB | Trainable: %.1fM | Action head: %.1fM | Variant: %s",
            total / 1e9,
            trainable / 1e6,
            action_params / 1e6,
            self.action_config.action_expert_variant,
        )

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

    def normalize_history(self, history: torch.Tensor) -> torch.Tensor:
        """
        Normalize history tokens while keeping the valid-flag intact.
        History layout: [state, action, valid_flag].
        """
        if history is None or history.numel() == 0:
            return history
        state_dim = int(getattr(self.action_config, "state_dim", 0))
        action_dim = int(getattr(self.action_config, "action_dim", 0))
        if history.shape[-1] < state_dim + action_dim + 1:
            return history

        norm_history = history.clone()
        norm_history[..., :state_dim] = self.normalize_states(norm_history[..., :state_dim])
        action_slice = norm_history[..., state_dim : state_dim + action_dim]
        norm_history[..., state_dim : state_dim + action_dim] = self.normalize_actions(action_slice)
        return norm_history

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
            raise ValueError("qwen_3.5_2b_pi0.5 requires state to build pi0.5-style prompts")

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
        if actions is None:
            raise ValueError("Actions required for training forward pass")

        prompts = None
        if instructions is not None and state is not None:
            prompts = self._build_pi05_prompts(instructions, state)

        if vlm_inputs is None:
            if images is None or prompts is None:
                raise ValueError("images, instructions, and state are required when vlm_inputs is not provided")
            vlm_inputs = self.vlm.build_vla_inputs(images, prompts)

        if vlm_labels is None and self.vlm_loss_weight > 0 and "input_ids" in vlm_inputs:
            input_ids = vlm_inputs["input_ids"]
            vlm_labels = input_ids.clone()
            vlm_labels[vlm_labels == 0] = -100

        prefix_inputs = self.vlm.build_prefix_from_vlm_inputs(vlm_inputs)

        fwd_kwargs = dict(output_hidden_states=True, **vlm_inputs)
        if vlm_labels is not None and self.vlm_loss_weight > 0:
            fwd_kwargs["labels"] = vlm_labels

        vlm_outputs = self.vlm(**fwd_kwargs) if self.vlm_loss_weight > 0 else None
        action_head_dtype = self._action_head_dtype()
        norm_actions = self.normalize_actions(actions).to(action_head_dtype)
        norm_state = self.normalize_states(state).to(action_head_dtype) if state is not None else None
        norm_history = self.normalize_history(history).to(action_head_dtype) if history is not None else None
        action_out = self.action_head(
            language_model=self._language_model(),
            prefix_embeds=prefix_inputs["prefix_embeds"],
            prefix_attention_mask=prefix_inputs["attention_mask"],
            prefix_position_ids=prefix_inputs["position_ids"],
            actions=norm_actions,
            state=norm_state,
            history=norm_history,
        )
        result = {"action_loss": action_out["loss"], "total_loss": action_out["loss"]}
        for key in ("motion_loss", "gripper_loss", "phase_loss"):
            if key in action_out:
                result[key] = action_out[key]

        if vlm_labels is not None and self.vlm_loss_weight > 0:
            vlm_loss = (
                vlm_outputs.loss
                if vlm_outputs.loss is not None
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
        self.eval()

        prompts = self._build_pi05_prompts(instructions, state)
        vlm_inputs = self.vlm.build_vla_inputs(images, prompts)
        prefix_inputs = self.vlm.build_prefix_from_vlm_inputs(vlm_inputs)
        action_head_dtype = self._action_head_dtype()
        norm_state = self.normalize_states(state).to(action_head_dtype) if state is not None else None
        norm_history = self.normalize_history(history).to(action_head_dtype) if history is not None else None
        norm_actions = self.action_head.predict_action(
            language_model=self._language_model(),
            prefix_embeds=prefix_inputs["prefix_embeds"],
            prefix_attention_mask=prefix_inputs["attention_mask"],
            prefix_position_ids=prefix_inputs["position_ids"],
            state=norm_state,
            history=norm_history,
            num_steps=num_inference_steps,
            deterministic_seed=deterministic_seed,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )
        if self.action_head._uses_binary_gripper():
            # action head returned cat([motion_normalized, gripper_±1])
            # Only unnormalize motion dims; preserve gripper as-is
            motion_norm = norm_actions[..., :6]
            gripper = norm_actions[..., 6:7]
            low = self.action_q01.to(device=motion_norm.device, dtype=motion_norm.dtype)
            high = self.action_q99.to(device=motion_norm.device, dtype=motion_norm.dtype)
            motion_raw = (motion_norm + 1.0) / 2.0 * (high[:6] - low[:6]) + low[:6]
            raw_actions = torch.cat([motion_raw, gripper], dim=-1)
        else:
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
            if "bias" in name or "norm" in name or "embed" in name:
                action_no_decay.append(param)
            else:
                action_decay.append(param)

        groups = []
        if vlm_params:
            groups.append(
                {"params": vlm_params, "lr": vlm_lr, "weight_decay": weight_decay}
            )
        if action_decay:
            groups.append(
                {"params": action_decay, "lr": action_lr, "weight_decay": weight_decay}
            )
        if action_no_decay:
            groups.append(
                {"params": action_no_decay, "lr": action_lr, "weight_decay": 0.0}
            )

        return groups
