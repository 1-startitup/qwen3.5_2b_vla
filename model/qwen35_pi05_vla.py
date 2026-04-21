"""Qwen3.5-VL + Pi 0.5 dual-stream action expert.

Architecture parity with openpi pi05_libero, VLM backbone swapped to Qwen3.5-VL.
Single loss: flow-matching MSE between predicted and target velocities.
Prompt is `"Task: {instruction};\\nAction: "` — state is not discretized into
the prompt, matching Pi 0.5 libero's `discrete_state_input=False`.
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
    architecture_name = "qwen_3.5_2b_pi0.5"

    def __init__(
        self,
        vlm_model_id: str,
        action_config: Optional[QwenPI05ActionConfig] = None,
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        attn_implementation: Optional[str] = None,
        architecture_name: Optional[str] = None,
    ):
        super().__init__()
        if action_config is None:
            action_config = QwenPI05ActionConfig()
        self.architecture_name = str(architecture_name or self.architecture_name)

        self.vlm = Qwen35PI05Interface(
            model_id=vlm_model_id,
            attn_implementation=attn_implementation,
            image_resolution=action_config.image_resolution,
            empty_cameras=action_config.empty_cameras,
            tokenizer_max_length=action_config.tokenizer_max_length,
        )
        action_config.vlm_hidden_dim = self.vlm.hidden_size
        self.action_config = action_config
        self.action_head = QwenPI05ExpertHead(action_config)

        # Quantile normalization stats (filled by set_norm_stats).
        self.register_buffer("action_q01", torch.zeros(action_config.max_action_dim))
        self.register_buffer("action_q99", torch.ones(action_config.max_action_dim))
        self.register_buffer("state_q01", torch.full((action_config.state_dim,), -1.0))
        self.register_buffer("state_q99", torch.full((action_config.state_dim,), 1.0))

        if freeze_vision_encoder:
            self.vlm.freeze_vision()
        if freeze_vlm:
            for p in self.vlm.parameters():
                p.requires_grad = False
            logger.info("VLM backbone frozen — only action expert trains")

        self.action_head._ensure_initialized(self._language_model())

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        action_params = sum(p.numel() for p in self.action_head.parameters())
        logger.info(
            "%s | total=%.2fB trainable=%.1fM action_expert=%.1fM",
            self.architecture_name,
            total / 1e9,
            trainable / 1e6,
            action_params / 1e6,
        )

    # ---------- module access ----------------------------------------------

    def _language_model(self) -> nn.Module:
        core = self.vlm._model_core()
        if not hasattr(core, "language_model"):
            raise AttributeError("Qwen VL model core does not expose language_model")
        return core.language_model

    # ---------- normalization ----------------------------------------------

    def set_norm_stats(
        self,
        action_q01: torch.Tensor,
        action_q99: torch.Tensor,
        state_q01: Optional[torch.Tensor] = None,
        state_q99: Optional[torch.Tensor] = None,
    ):
        # Pad to max_action_dim so broadcasts are uniform.
        D = self.action_config.max_action_dim
        def _pad(x: torch.Tensor, fill: float) -> torch.Tensor:
            if x.shape[-1] >= D:
                return x[..., :D]
            pad = torch.full((D - x.shape[-1],), fill, dtype=x.dtype)
            return torch.cat([x, pad], dim=-1)

        self.action_q01.copy_(_pad(action_q01, -1.0))
        self.action_q99.copy_(_pad(action_q99, 1.0))
        if state_q01 is not None:
            self.state_q01.copy_(state_q01)
        if state_q99 is not None:
            self.state_q99.copy_(state_q99)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        D = actions.shape[-1]
        low = self.action_q01[:D].to(device=actions.device, dtype=actions.dtype)
        high = self.action_q99[:D].to(device=actions.device, dtype=actions.dtype)
        return 2.0 * (actions - low) / (high - low + 1e-8) - 1.0

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        D = actions.shape[-1]
        low = self.action_q01[:D].to(device=actions.device, dtype=actions.dtype)
        high = self.action_q99[:D].to(device=actions.device, dtype=actions.dtype)
        return (actions + 1.0) / 2.0 * (high - low) + low

    def normalize_states(self, state: torch.Tensor) -> torch.Tensor:
        low = self.state_q01.to(device=state.device, dtype=state.dtype)
        high = self.state_q99.to(device=state.device, dtype=state.dtype)
        return (2.0 * (state - low) / (high - low + 1e-8) - 1.0).clamp(-1.0, 1.0)

    # ---------- prompt -----------------------------------------------------

    @staticmethod
    def _build_prompts(instructions: List[str]) -> List[str]:
        # Pi 0.5 libero / prompt_from_task=True / discrete_state_input=False uses
        # the raw task instruction (cleaned of underscores/newlines) as the prompt;
        # see openpi tokenizer.py:30-33. The role separator / "\n" is absorbed by
        # Qwen-VL's chat template, so we just emit the cleaned text as user content.
        # Matches the dataset's pre-tokenized path (prompt_style="plain") so both
        # the cached and lazy tokenization codepaths produce identical tokens.
        return [instr.strip().replace("_", " ").replace("\n", " ") for instr in instructions]

    # ---------- forward ----------------------------------------------------

    def forward(
        self,
        images: Optional[List[List[Any]]] = None,
        instructions: Optional[List[str]] = None,
        actions: Optional[torch.Tensor] = None,
        vlm_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if actions is None:
            raise ValueError("actions required for training forward pass")
        if vlm_inputs is None:
            if images is None or instructions is None:
                raise ValueError("must provide (images, instructions) or vlm_inputs")
            vlm_inputs = self.vlm.build_vla_inputs(images, self._build_prompts(instructions))

        prefix = self.vlm.build_prefix_from_vlm_inputs(vlm_inputs)
        norm_actions = self.normalize_actions(actions)
        out = self.action_head(
            language_model=self._language_model(),
            prefix_embeds=prefix["prefix_embeds"],
            prefix_attention_mask=prefix["attention_mask"],
            prefix_position_ids=prefix["position_ids"],
            actions=norm_actions,
        )
        return {"loss": out["loss"], "total_loss": out["loss"]}

    @torch.no_grad()
    def predict_action(
        self,
        images: List[List[Any]],
        instructions: List[str],
        num_inference_steps: Optional[int] = None,
        deterministic_seed: Optional[int] = None,
    ) -> np.ndarray:
        self.eval()
        vlm_inputs = self.vlm.build_vla_inputs(images, self._build_prompts(instructions))
        prefix = self.vlm.build_prefix_from_vlm_inputs(vlm_inputs)
        norm_actions = self.action_head.predict_action(
            language_model=self._language_model(),
            prefix_embeds=prefix["prefix_embeds"],
            prefix_attention_mask=prefix["attention_mask"],
            prefix_position_ids=prefix["position_ids"],
            num_steps=num_inference_steps,
            deterministic_seed=deterministic_seed,
        )
        actions = self.unnormalize_actions(norm_actions)
        return actions.float().cpu().numpy()

    # ---------- optimizer --------------------------------------------------

    def get_optimizer_groups(self, lr: float, weight_decay: float) -> list:
        """Single unified learning rate for VLM + action expert, matching Pi 0.5."""
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr, "weight_decay": weight_decay}]
