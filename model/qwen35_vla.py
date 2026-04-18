"""
Qwen3.5-2B VLA Framework
========================
Full Vision-Language-Action model integrating:
  - Qwen3.5-2B as the VLM backbone
  - Flow matching or layerwise flow matching action heads
  - Optional VLM co-training (language loss)

Architecture:
  [Images + Instruction] -> Qwen3.5-2B -> hidden_states (B, S, H_vlm)
                                                |
                                          VLM projection
                                                |
                                         DiT cross-attention
                                                |
                                       Flow matching velocity
                                                |
                                         Euler integration
                                                |
                                     Action chunk (B, T, 7)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import numpy as np
import logging

from .qwen35_vl_interface import Qwen35VLInterface
from .flow_matching_head import FlowMatchingActionHead, FlowMatchingConfig
from .layerwise_flow_matching_head import (
    LayerwiseFlowMatchingActionHead,
    LayerwiseFlowMatchingConfig,
)
from .layerwise_flow_matching_head_v3 import (
    LayerwiseFlowMatchingActionHeadV3,
    LayerwiseFlowMatchingV3Config,
)
from .layerwise_flow_matching_head_v4 import (
    LayerwiseFlowMatchingActionHeadV4,
    LayerwiseFlowMatchingV4Config,
)
from .se3_utils import local_to_world_motion, world_to_local_motion

logger = logging.getLogger(__name__)


class Qwen35VLA(nn.Module):
    """
    Full VLA model: Qwen3.5-2B + Flow Matching Action Head.

    Training modes:
        1. VLA only:      action loss from flow matching
        2. Co-training:    action loss + language loss (weighted)
        3. Freeze VLM:     only train the action head

    Action normalization:
        Actions are normalized to [-1, 1] using per-dimension q01/q99 statistics.
        Gripper (index 6) is binarized to {-1, 1} after unnormalization.
    """

    def __init__(
        self,
        vlm_model_id: str = "Qwen/Qwen3.5-2B",
        action_config: Optional[FlowMatchingConfig] = None,
        action_head_type: str = "single_fm",
        freeze_vision_encoder: bool = True,
        freeze_vlm: bool = False,
        vlm_loss_weight: float = 0.1,
        cot_prompt: Optional[str] = None,
        lora_config: Optional[Dict] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()

        # 1. Load Qwen3.5 backbone
        self.vlm = Qwen35VLInterface(
            model_id=vlm_model_id,
            attn_implementation=attn_implementation,
        )

        # 2. Configure flow matching action head
        if action_config is None:
            if action_head_type == "layerwise_fm":
                action_config = LayerwiseFlowMatchingConfig(
                    vlm_hidden_dim=self.vlm.hidden_size,
                )
            elif action_head_type == "layerwise_fm_v3":
                action_config = LayerwiseFlowMatchingV3Config(
                    vlm_hidden_dim=self.vlm.hidden_size,
                )
            elif action_head_type == "layerwise_fm_v4":
                action_config = LayerwiseFlowMatchingV4Config(
                    vlm_hidden_dim=self.vlm.hidden_size,
                )
            else:
                action_config = FlowMatchingConfig.from_preset(
                    "DiT-B",
                    vlm_hidden_dim=self.vlm.hidden_size,  # 2048 for 2B
                )
        else:
            action_config.vlm_hidden_dim = self.vlm.hidden_size

        self.action_head_type = action_head_type
        if self.action_head_type == "layerwise_fm":
            if not isinstance(action_config, LayerwiseFlowMatchingConfig):
                raise TypeError(
                    f"layerwise_fm expects LayerwiseFlowMatchingConfig, got {type(action_config)}"
                )
            # Keep the action head parameters in fp32; mixed precision autocast
            # can still downcast compute when appropriate, but we avoid baking the
            # whole control head into bf16 weights.
            self.action_head = LayerwiseFlowMatchingActionHead(action_config)
        elif self.action_head_type == "layerwise_fm_v3":
            if not isinstance(action_config, LayerwiseFlowMatchingV3Config):
                raise TypeError(
                    f"layerwise_fm_v3 expects LayerwiseFlowMatchingV3Config, got {type(action_config)}"
                )
            self.action_head = LayerwiseFlowMatchingActionHeadV3(action_config)
        elif self.action_head_type == "layerwise_fm_v4":
            if not isinstance(action_config, LayerwiseFlowMatchingV4Config):
                raise TypeError(
                    f"layerwise_fm_v4 expects LayerwiseFlowMatchingV4Config, got {type(action_config)}"
                )
            self.action_head = LayerwiseFlowMatchingActionHeadV4(action_config)
        else:
            if not isinstance(action_config, FlowMatchingConfig):
                raise TypeError(
                    f"single_fm expects FlowMatchingConfig, got {type(action_config)}"
                )
            self.action_head = FlowMatchingActionHead(action_config)
        self.action_config = action_config

        # 3. Training settings
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_vlm = freeze_vlm
        self.vlm_loss_weight = vlm_loss_weight
        self.cot_prompt = cot_prompt

        # 4. Action normalization stats (loaded from dataset)
        self.register_buffer("action_q01", torch.zeros(action_config.action_dim))
        self.register_buffer("action_q99", torch.ones(action_config.action_dim))
        self.register_buffer("local_action_q01", torch.zeros(action_config.action_dim))
        self.register_buffer("local_action_q99", torch.ones(action_config.action_dim))

        # Apply LoRA if configured (before freezing, replaces full fine-tune)
        if lora_config is not None:
            self._apply_lora(lora_config)
        else:
            # Apply freezing (only when not using LoRA)
            if freeze_vision_encoder:
                self.vlm.get_trainable_params(freeze_vision=True)
            if freeze_vlm:
                for param in self.vlm.parameters():
                    param.requires_grad = False
                logger.info("Frozen entire VLM backbone — training action head only")

        self._log_param_counts()

    def _collect_vlm_hidden_states(self, vlm_outputs) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_hidden_states = list(vlm_outputs.hidden_states)
        last_hidden = all_hidden_states[-1]

        if self.action_head_type not in {"layerwise_fm", "layerwise_fm_v3", "layerwise_fm_v4"}:
            return last_hidden, [last_hidden]

        # HF hidden_states usually includes the embedding output first. For
        # explicit layer indexing we expose transformer-layer outputs only.
        transformer_hidden_states = all_hidden_states[1:] if len(all_hidden_states) > 1 else all_hidden_states
        return last_hidden, transformer_hidden_states

    def _action_head_dtype(self) -> torch.dtype:
        return next(self.action_head.parameters()).dtype

    def _apply_lora(self, lora_cfg: Dict):
        """Apply LoRA adapters to VLM backbone. Action head stays fully trainable."""
        from peft import LoraConfig, get_peft_model

        # Freeze all VLM params first
        for param in self.vlm.model.parameters():
            param.requires_grad = False

        # Apply LoRA to language model attention layers
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 32),
            lora_alpha=lora_cfg.get("alpha", 64),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.vlm.model = get_peft_model(self.vlm.model, peft_config)
        self.vlm.model.print_trainable_parameters()
        logger.info(f"LoRA applied: r={peft_config.r}, alpha={peft_config.lora_alpha}")

    def _log_param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        action_params = sum(p.numel() for p in self.action_head.parameters())
        logger.info(
            f"Qwen35VLA | Total: {total/1e9:.2f}B | "
            f"Trainable: {trainable/1e6:.1f}M | "
            f"Action head: {action_params/1e6:.1f}M | "
            f"Head type: {self.action_head_type}"
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
        """Set action normalization statistics from dataset."""
        del state_q01, state_q99
        self.action_q01.copy_(q01)
        self.action_q99.copy_(q99)
        if local_q01 is None:
            local_q01 = q01
        if local_q99 is None:
            local_q99 = q99
        self.local_action_q01.copy_(local_q01)
        self.local_action_q99.copy_(local_q99)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize raw actions to [-1, 1] using q01/q99."""
        low = self.action_q01.to(actions.dtype)
        high = self.action_q99.to(actions.dtype)
        return 2.0 * (actions - low) / (high - low + 1e-8) - 1.0

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Unnormalize actions from [-1, 1] back to original scale."""
        low = self.action_q01.to(actions.dtype)
        high = self.action_q99.to(actions.dtype)
        raw = (actions + 1.0) / 2.0 * (high - low) + low
        # Preserve the dataset's signed gripper semantics. LIBERO actions are
        # stored as {-1, +1}; collapsing them to {0, 1} breaks deployment.
        if raw.shape[-1] >= 7:
            raw[..., 6] = torch.where(raw[..., 6] > 0.0, 1.0, -1.0)
        return raw

    def normalize_local_actions(self, actions: torch.Tensor) -> torch.Tensor:
        low = self.local_action_q01.to(actions.dtype)
        high = self.local_action_q99.to(actions.dtype)
        return 2.0 * (actions - low) / (high - low + 1e-8) - 1.0

    def unnormalize_local_actions(self, actions: torch.Tensor) -> torch.Tensor:
        low = self.local_action_q01.to(actions.dtype)
        high = self.local_action_q99.to(actions.dtype)
        raw = (actions + 1.0) / 2.0 * (high - low) + low
        if raw.shape[-1] >= 7:
            raw[..., 6] = torch.where(raw[..., 6] > 0.0, 1.0, -1.0)
        return raw

    def _uses_local_action_frame(self) -> bool:
        return bool(getattr(self.action_config, "use_local_action_frame", False))

    def normalize_history(self, history: torch.Tensor) -> torch.Tensor:
        """
        Normalize the action slice inside history tokens while keeping state/flag intact.
        History tokens are laid out as [state, action, valid_flag].
        """
        if history is None or history.numel() == 0:
            return history
        if not hasattr(self.action_config, "state_dim") or history.shape[-1] < self.action_config.state_dim + self.action_config.action_dim:
            return history

        norm_history = history.clone()
        state_dim = int(self.action_config.state_dim)
        action_dim = int(self.action_config.action_dim)
        action_slice = norm_history[..., state_dim:state_dim + action_dim]
        if self._uses_local_action_frame() and history.shape[-1] >= state_dim + action_dim + 1:
            state_slice = norm_history[..., :state_dim]
            local_action_slice = action_slice.clone()
            local_action_slice[..., :6] = world_to_local_motion(action_slice[..., :6], state_slice)
            norm_history[..., state_dim:state_dim + action_dim] = self.normalize_local_actions(local_action_slice)
        else:
            norm_history[..., state_dim:state_dim + action_dim] = self.normalize_actions(action_slice)
        return norm_history

    def _use_state_prompt(self) -> bool:
        return bool(getattr(self.action_config, "use_state_prompt", False))

    def _discretize_prompt_tensor(self, values: torch.Tensor, bins: int) -> torch.Tensor:
        clipped = torch.tanh(values.float()).clamp(-1.0, 1.0)
        scaled = (clipped + 1.0) * 0.5 * float(max(bins - 1, 1))
        return scaled.round().to(torch.int64)

    def _augment_instructions_with_state(
        self,
        instructions: List[str],
        state: Optional[torch.Tensor],
        history: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """
        Build pi0.5-style prompts so the VLM sees task text and discretized state jointly.
        """
        if not self._use_state_prompt() or state is None:
            return instructions

        bins = int(getattr(self.action_config, "state_prompt_bins", 256))
        history_prompt_len = int(getattr(self.action_config, "state_prompt_history_len", 0))
        state_cpu = state.detach().float().cpu()
        history_cpu = history.detach().float().cpu() if history is not None else None

        prompts: List[str] = []
        state_dim = int(getattr(self.action_config, "state_dim", 0))
        action_dim = int(getattr(self.action_config, "action_dim", 0))

        for batch_idx, instruction in enumerate(instructions):
            state_bins = self._discretize_prompt_tensor(state_cpu[batch_idx], bins).tolist()
            prompt_lines = [
                f"Task: {instruction.strip()}",
                "State: " + " ".join(map(str, state_bins)),
            ]

            if (
                history_prompt_len > 0
                and history_cpu is not None
                and history_cpu.shape[1] > 0
                and history_cpu.shape[-1] >= state_dim + action_dim + 1
            ):
                history_item = history_cpu[batch_idx]
                valid_mask = history_item[:, -1] > 0.5
                if bool(valid_mask.any()):
                    recent = history_item[valid_mask][-history_prompt_len:]
                    recent_gripper = recent[:, state_dim + action_dim - 1]
                    recent_motion_mag = recent[:, state_dim:state_dim + action_dim - 1].norm(dim=-1)
                    summary = []
                    for motion_mag, grip in zip(recent_motion_mag.tolist(), recent_gripper.tolist()):
                        grip_label = "close" if grip > 0 else "open"
                        summary.append(f"{grip_label}:{motion_mag:.2f}")
                    prompt_lines.append("Recent: " + " | ".join(summary))

            prompt_lines.append("Action:")
            prompts.append("\n".join(prompt_lines))

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
        """
        Training forward pass.

        Args:
            images:       List of image lists per batch element
            instructions: Task instructions
            actions:      (B, T, action_dim) ground truth actions (raw, will be normalized)
            state:        (B, state_dim) proprioceptive state
            vlm_inputs:   Pre-tokenized VLM inputs (alternative to images+instructions)
            vlm_labels:   (B, L) language labels for co-training loss

        Returns:
            dict with 'action_loss', optionally 'vlm_loss' and 'total_loss'
        """
        # Build VLM inputs
        prompt_instructions = instructions
        if images is not None and instructions is not None:
            prompt_instructions = self._augment_instructions_with_state(
                instructions,
                state=state,
                history=history,
            )

        should_rebuild_inputs = (
            vlm_inputs is None
            or (
                self._use_state_prompt()
                and images is not None
                and prompt_instructions is not None
            )
        )
        if should_rebuild_inputs:
            assert images is not None and prompt_instructions is not None
            vlm_inputs = self.vlm.build_vla_inputs(
                images, prompt_instructions, cot_prompt=self.cot_prompt
            )

        # Construct causal LM labels for VLM co-training (shift input_ids by 1)
        if vlm_labels is None and self.vlm_loss_weight > 0 and "input_ids" in vlm_inputs:
            input_ids = vlm_inputs["input_ids"]
            vlm_labels = input_ids.clone()
            # Mask padding tokens (0) and image tokens with IGNORE_INDEX
            vlm_labels[vlm_labels == 0] = -100
            # Shift handled internally by HF's CausalLM loss

        # Forward through VLM backbone (pass labels for language loss)
        fwd_kwargs = dict(output_hidden_states=True, **vlm_inputs)
        if vlm_labels is not None and self.vlm_loss_weight > 0:
            fwd_kwargs["labels"] = vlm_labels

        vlm_outputs = self.vlm(**fwd_kwargs)

        # Extract hidden states for action prediction
        # Shape: (B, seq_len, H_vlm)
        hidden_states, layerwise_hidden_states = self._collect_vlm_hidden_states(vlm_outputs)
        text_embedding = self.vlm.extract_text_pooled(hidden_states, vlm_inputs)
        attention_mask = vlm_inputs.get("attention_mask", None)

        action_head_dtype = self._action_head_dtype()

        # Normalize actions in the action-head dtype. We only upcast VLM
        # hidden states when autocast is off to avoid unnecessary memory growth.
        if actions is not None:
            if self.action_head_type == "layerwise_fm_v4" and self._uses_local_action_frame() and state is not None:
                local_actions = actions.clone()
                local_actions[..., :6] = world_to_local_motion(actions[..., :6], state.to(actions.dtype).unsqueeze(1).expand(-1, actions.shape[1], -1))
                norm_actions = self.normalize_local_actions(local_actions).to(action_head_dtype)
            else:
                norm_actions = self.normalize_actions(actions).to(action_head_dtype)
        else:
            raise ValueError("Actions required for training forward pass")
        if state is not None:
            state = state.to(action_head_dtype)
        if history is not None:
            history = self.normalize_history(history).to(action_head_dtype)
        if text_embedding is not None:
            text_embedding = text_embedding.to(action_head_dtype)
        if not torch.is_autocast_enabled():
            if self.action_head_type in {"layerwise_fm", "layerwise_fm_v3", "layerwise_fm_v4"}:
                layerwise_hidden_states = [hs.to(action_head_dtype) for hs in layerwise_hidden_states]
            else:
                hidden_states = hidden_states.to(action_head_dtype)

        # Flow matching action loss
        if self.action_head_type == "layerwise_fm":
            action_out = self.action_head(
                vlm_hidden_states_list=layerwise_hidden_states,
                actions=norm_actions,
                state=state,
                attention_mask=attention_mask,
            )
        elif self.action_head_type == "layerwise_fm_v3":
            action_out = self.action_head(
                vlm_hidden_states_list=layerwise_hidden_states,
                actions=norm_actions,
                state=state,
                history=history,
                text_embedding=text_embedding,
                attention_mask=attention_mask,
            )
        elif self.action_head_type == "layerwise_fm_v4":
            action_out = self.action_head(
                vlm_hidden_states_list=layerwise_hidden_states,
                actions=norm_actions,
                state=state,
                history=history,
                text_embedding=text_embedding,
                attention_mask=attention_mask,
            )
        else:
            action_out = self.action_head(
                vlm_hidden_states=hidden_states,
                actions=norm_actions,
                state=state,
                attention_mask=attention_mask,
            )

        result = {"action_loss": action_out["loss"]}
        if "motion_loss" in action_out:
            result["motion_loss"] = action_out["motion_loss"]
        if "gripper_loss" in action_out:
            result["gripper_loss"] = action_out["gripper_loss"]
        if "consistency_loss" in action_out:
            result["consistency_loss"] = action_out["consistency_loss"]
        if "phase_loss" in action_out:
            result["phase_loss"] = action_out["phase_loss"]
        if "immediate_loss" in action_out:
            result["immediate_loss"] = action_out["immediate_loss"]
        for key in (
            "loss_weight_motion",
            "loss_weight_gripper",
            "loss_weight_phase",
            "loss_weight_immediate",
        ):
            if key in action_out:
                result[key] = action_out[key]

        # VLM co-training loss
        if vlm_labels is not None and self.vlm_loss_weight > 0:
            vlm_loss = vlm_outputs.loss if vlm_outputs.loss is not None else torch.tensor(0.0, device=action_out["loss"].device)
            result["vlm_loss"] = vlm_loss
            result["total_loss"] = action_out["loss"] + self.vlm_loss_weight * vlm_loss
        else:
            result["total_loss"] = action_out["loss"]

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
    ) -> np.ndarray:
        """
        Inference: predict action chunk from observation.

        Args:
            images:       List of image lists
            instructions: Task instructions
            state:        Optional proprioceptive state
        Returns:
            actions: (B, T, action_dim) numpy array of unnormalized actions
        """
        self.eval()

        # Build inputs and forward VLM
        prompt_instructions = self._augment_instructions_with_state(
            instructions,
            state=state,
            history=history,
        )
        vlm_inputs = self.vlm.build_vla_inputs(images, prompt_instructions)
        vlm_outputs = self.vlm(output_hidden_states=True, **vlm_inputs)
        hidden_states, layerwise_hidden_states = self._collect_vlm_hidden_states(vlm_outputs)
        text_embedding = self.vlm.extract_text_pooled(hidden_states, vlm_inputs)
        attention_mask = vlm_inputs.get("attention_mask", None)
        action_head_dtype = self._action_head_dtype()
        if state is not None:
            state = state.to(action_head_dtype)
        if history is not None:
            history = self.normalize_history(history).to(action_head_dtype)
        if text_embedding is not None:
            text_embedding = text_embedding.to(action_head_dtype)
        if not torch.is_autocast_enabled():
            if self.action_head_type in {"layerwise_fm", "layerwise_fm_v3", "layerwise_fm_v4"}:
                layerwise_hidden_states = [hs.to(action_head_dtype) for hs in layerwise_hidden_states]
            else:
                hidden_states = hidden_states.to(action_head_dtype)

        # Flow matching inference (Euler integration)
        extra_kwargs = {}
        if num_inference_steps is not None:
            extra_kwargs["num_steps"] = num_inference_steps
        if deterministic_seed is not None:
            extra_kwargs["deterministic_seed"] = deterministic_seed

        if self.action_head_type == "layerwise_fm":
            norm_actions = self.action_head.predict_action(
                vlm_hidden_states_list=layerwise_hidden_states,
                state=state,
                attention_mask=attention_mask,
                **extra_kwargs,
            )
        elif self.action_head_type == "layerwise_fm_v3":
            norm_actions = self.action_head.predict_action(
                vlm_hidden_states_list=layerwise_hidden_states,
                state=state,
                history=history,
                text_embedding=text_embedding,
                attention_mask=attention_mask,
                **extra_kwargs,
            )
        elif self.action_head_type == "layerwise_fm_v4":
            norm_actions = self.action_head.predict_action(
                vlm_hidden_states_list=layerwise_hidden_states,
                state=state,
                history=history,
                text_embedding=text_embedding,
                attention_mask=attention_mask,
                **extra_kwargs,
            )
        else:
            norm_actions = self.action_head.predict_action(
                vlm_hidden_states=hidden_states,
                state=state,
                attention_mask=attention_mask,
                **extra_kwargs,
            )

        # Unnormalize
        if self.action_head_type == "layerwise_fm_v4" and self._uses_local_action_frame() and state is not None:
            raw_actions = self.unnormalize_local_actions(norm_actions)
            expanded_state = state.to(raw_actions.dtype).unsqueeze(1).expand(-1, raw_actions.shape[1], -1)
            raw_actions = raw_actions.clone()
            raw_actions[..., :6] = local_to_world_motion(raw_actions[..., :6], expanded_state)
            raw_actions[..., 6] = torch.where(raw_actions[..., 6] > 0.0, 1.0, -1.0)
        else:
            raw_actions = self.unnormalize_actions(norm_actions)
        return raw_actions.float().cpu().numpy()

    def get_optimizer_groups(
        self, vlm_lr: float = 2.5e-5, action_lr: float = 1e-4, weight_decay: float = 0.01
    ) -> list:
        """
        Build optimizer parameter groups with separate LRs for VLM and action head.
        """
        # VLM parameters (lower learning rate)
        vlm_params = [
            p for p in self.vlm.parameters() if p.requires_grad
        ]

        # Action head parameters (higher learning rate)
        action_decay = []
        action_no_decay = []
        for name, p in self.action_head.named_parameters():
            if p.requires_grad:
                if "bias" in name or "norm" in name or "embed" in name:
                    action_no_decay.append(p)
                else:
                    action_decay.append(p)

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
