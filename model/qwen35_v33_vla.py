from __future__ import annotations

from typing import Dict, Optional

from .qwen35_pi05_action_head import QwenPI05ActionConfig
from .qwen35_pi05_vla import Qwen35PI05VLA


class Qwen35V33VLA(Qwen35PI05VLA):
    architecture_name = "qwen_3.5_2b_pi05_v3_3"

    def __init__(
        self,
        vlm_model_id: str,
        action_config: Optional[QwenPI05ActionConfig] = None,
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        attn_implementation: Optional[str] = None,
        architecture_name: Optional[str] = None,
    ):
        super().__init__(
            vlm_model_id=vlm_model_id,
            action_config=action_config,
            freeze_vision_encoder=freeze_vision_encoder,
            freeze_vlm=freeze_vlm,
            attn_implementation=attn_implementation,
            architecture_name=architecture_name or self.architecture_name,
        )


__all__ = ["Qwen35V33VLA"]
