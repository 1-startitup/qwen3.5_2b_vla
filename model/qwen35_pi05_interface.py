"""Qwen3.5-VL wrapper aligned with Pi 0.5 preprocessing.

Image side: stretch-resize to a fixed square (no letterbox padding), matching
Pi 0.5 libero's ResizeImages. Optional `empty_cameras` slots appended with
all-black images to preserve the 3-camera layout used by Pi 0.5 libero.

Tokenizer: right-padded to `tokenizer_max_length`, max-length padding so all
batch entries share a consistent prefix layout.

Exposes `build_prefix_from_vlm_inputs` which materializes multimodal
embeddings (placeholder scatter for image tokens) and 3D MRoPE position ids
so the action expert can run its dual-stream forward with the prefix hidden.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class Qwen35PI05Interface(nn.Module):
    def __init__(
        self,
        model_id: str,
        attn_implementation: Optional[str] = None,
        image_resolution: Sequence[int] = (224, 224),
        empty_cameras: int = 0,
        tokenizer_max_length: int = 200,
    ):
        super().__init__()
        load_kwargs = {"torch_dtype": torch.bfloat16}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "right"

        if hasattr(self.model.config, "text_config"):
            self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self.hidden_size = self.model.config.hidden_size

        if len(image_resolution) != 2:
            raise ValueError(f"image_resolution must be length-2, got {image_resolution}")
        self.image_resolution = (int(image_resolution[0]), int(image_resolution[1]))
        self.empty_cameras = max(0, int(empty_cameras))
        self.tokenizer_max_length = int(tokenizer_max_length)

        logger.info(
            "Qwen35PI05Interface | hidden=%s dtype=%s resolution=%s empty_cams=%s",
            self.hidden_size, self.model.dtype, self.image_resolution, self.empty_cameras,
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _model_core(self) -> nn.Module:
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        return base.model if hasattr(base, "model") else base

    def freeze_vision(self) -> None:
        visual = getattr(self._model_core(), "visual", None)
        if visual is not None:
            for p in visual.parameters():
                p.requires_grad = False
            logger.info("Vision encoder frozen")

    def forward(self, output_hidden_states: bool = True, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(output_hidden_states=output_hidden_states, **kwargs)

    # ---------- image preprocessing ---------------------------------------

    def _to_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
        else:
            arr = image
        if arr is None:
            return Image.new("RGB", self.image_resolution, color="black")
        if getattr(arr, "dtype", None) is not None and str(arr.dtype).startswith("float"):
            arr = (arr.clip(0.0, 1.0) * 255.0).astype("uint8")
        return Image.fromarray(arr).convert("RGB")

    def _resize_stretch(self, image: Image.Image) -> Image.Image:
        """Pi 0.5 style: stretch-resize to the target square (no letterbox)."""
        return image.convert("RGB").resize(self.image_resolution, Image.BICUBIC)

    def preprocess_images(self, images: Sequence[Any]) -> List[Image.Image]:
        out = [self._resize_stretch(self._to_pil(img)) for img in images]
        if not out:
            out = [Image.new("RGB", self.image_resolution, color="black")]
        for _ in range(self.empty_cameras):
            out.append(Image.new("RGB", self.image_resolution, color="black"))
        return out

    # ---------- VLA input construction ------------------------------------

    def build_vla_inputs(
        self,
        images: List[List[Any]],
        instructions: List[str],
    ) -> Dict[str, torch.Tensor]:
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in self.preprocess_images(imgs)]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])

        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": "max_length",
                "truncation": True,
                "max_length": self.tokenizer_max_length,
            },
        )
        return batch.to(self.device)

    def build_prefix_from_vlm_inputs(self, vlm_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run Qwen-VL embedding + image scatter + position id computation, return prefix."""
        core = self._model_core()
        input_ids = vlm_inputs.get("input_ids")
        attention_mask = vlm_inputs.get("attention_mask")
        inputs_embeds = vlm_inputs.get("inputs_embeds")

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds must be provided")
        if inputs_embeds is None:
            inputs_embeds = core.get_input_embeddings()(input_ids)

        pixel_values = vlm_inputs.get("pixel_values")
        image_grid_thw = vlm_inputs.get("image_grid_thw")
        if pixel_values is not None:
            image_out = core.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_out.pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = core.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = vlm_inputs.get("position_ids")
        if position_ids is None:
            position_ids = core.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=vlm_inputs.get("video_grid_thw"),
                attention_mask=attention_mask,
                past_key_values=None,
                mm_token_type_ids=vlm_inputs.get("mm_token_type_ids"),
            )
        if position_ids is None:
            seq = inputs_embeds.shape[1]
            B = inputs_embeds.shape[0]
            position_ids = torch.arange(seq, device=inputs_embeds.device).view(1, 1, -1).expand(3, B, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        return {
            "prefix_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
