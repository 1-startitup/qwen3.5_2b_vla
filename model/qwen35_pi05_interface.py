"""
Qwen3.5 multimodal interface with pi0.5-style preprocessing.

This wrapper keeps the Qwen backbone but aligns the input contract with pi0.5:
  - right-padded tokenization
  - resize-with-pad image preprocessing
  - optional empty camera slots
  - hidden-state extraction for chunked flow matching
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
    """Qwen3.5 VLM wrapper with pi0.5-style preprocessing defaults."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-2B",
        attn_implementation: Optional[str] = None,
        image_resolution: Sequence[int] = (224, 224),
        empty_cameras: int = 0,
        tokenizer_max_length: int = 200,
        **kwargs,
    ):
        super().__init__()

        logger.info(
            "Loading requested backbone: %s | attn_implementation=%s",
            model_id,
            attn_implementation,
        )

        load_kwargs = {"torch_dtype": torch.bfloat16}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        try:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id,
                **load_kwargs,
            )
        except Exception as exc:
            if attn_implementation:
                logger.warning(
                    "Failed to load %s with attn_implementation=%s (%s). Falling back to default attention.",
                    model_id,
                    attn_implementation,
                    exc,
                )
                load_kwargs.pop("attn_implementation", None)
                self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_id,
                    **load_kwargs,
                )
            else:
                raise

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
            "Qwen35PI05Interface ready | hidden_size=%s | dtype=%s | image_resolution=%s | empty_cameras=%s | tokenizer_max_length=%s",
            self.hidden_size,
            self.model.dtype,
            self.image_resolution,
            self.empty_cameras,
            self.tokenizer_max_length,
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _base_model(self):
        model = self.model
        if hasattr(model, "get_base_model"):
            return model.get_base_model()
        return model

    def _model_core(self):
        base_model = self._base_model()
        if hasattr(base_model, "model"):
            return base_model.model
        return base_model

    def forward(
        self,
        output_hidden_states: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if logits_to_keep is None and kwargs.get("labels", None) is None:
            logits_to_keep = 1
        if logits_to_keep is not None:
            kwargs["logits_to_keep"] = logits_to_keep

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                output_hidden_states=output_hidden_states, **kwargs
            )
        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model.generate(**kwargs)

    def _to_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
        else:
            array = image

        if array is None:
            return Image.new("RGB", self.image_resolution, color="black")

        if getattr(array, "dtype", None) is not None and str(array.dtype).startswith("float"):
            array = (array.clip(0.0, 1.0) * 255.0).astype("uint8")

        return Image.fromarray(array).convert("RGB")

    def _resize_with_pad(self, image: Image.Image) -> Image.Image:
        target_w, target_h = self.image_resolution
        image = image.convert("RGB")
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            return Image.new("RGB", (target_w, target_h), color="black")

        scale = min(target_w / float(src_w), target_h / float(src_h))
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = image.resize((resized_w, resized_h), Image.BICUBIC)

        canvas = Image.new("RGB", (target_w, target_h), color="black")
        offset_x = (target_w - resized_w) // 2
        offset_y = (target_h - resized_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    def preprocess_images(self, images: Sequence[Any]) -> List[Image.Image]:
        processed = [self._resize_with_pad(self._to_pil(img)) for img in images]
        if not processed:
            processed = [Image.new("RGB", self.image_resolution, color="black")]

        for _ in range(self.empty_cameras):
            processed.append(Image.new("RGB", self.image_resolution, color="black"))

        return processed

    def build_vla_inputs(
        self,
        images: List[List[Any]],
        instructions: List[str],
        cot_prompt: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        messages = []
        for imgs, instruction in zip(images, instructions):
            processed_images = self.preprocess_images(imgs)
            content = [{"type": "image", "image": img} for img in processed_images]
            prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt else instruction
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        batch_inputs = self.processor.apply_chat_template(
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

        return batch_inputs.to(self.device)

    def build_prefix_from_vlm_inputs(
        self,
        vlm_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Materialize multimodal prefix embeddings plus rope position ids.

        This mirrors `Qwen3_5Model.forward` up to the point where embeddings are
        injected and handed to `language_model`, so the action suffix can be
        appended and decoded with cached prefix KV states.
        """
        model_core = self._model_core()
        input_ids = vlm_inputs.get("input_ids", None)
        attention_mask = vlm_inputs.get("attention_mask", None)
        inputs_embeds = vlm_inputs.get("inputs_embeds", None)

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids or inputs_embeds must be present in vlm_inputs")

        if inputs_embeds is None:
            inputs_embeds = model_core.get_input_embeddings()(input_ids)

        pixel_values = vlm_inputs.get("pixel_values", None)
        image_grid_thw = vlm_inputs.get("image_grid_thw", None)
        if pixel_values is not None:
            image_outputs = model_core.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True,
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            image_mask, _ = model_core.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        pixel_values_videos = vlm_inputs.get("pixel_values_videos", None)
        video_grid_thw = vlm_inputs.get("video_grid_thw", None)
        if pixel_values_videos is not None:
            video_outputs = model_core.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                return_dict=True,
            )
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )
            _, video_mask = model_core.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        position_ids = vlm_inputs.get("position_ids", None)
        if position_ids is None:
            position_ids = model_core.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=None,
                mm_token_type_ids=vlm_inputs.get("mm_token_type_ids", None),
            )

        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            batch_size = inputs_embeds.shape[0]
            position_ids = torch.arange(
                seq_len,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).view(1, 1, -1).expand(3, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        return {
            "prefix_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def get_trainable_params(self, freeze_vision: bool = True):
        model_core = self._model_core()
        visual_module = None
        if hasattr(model_core, "visual") and getattr(model_core, "visual", None) is not None:
            visual_module = model_core.visual

        if freeze_vision and visual_module is not None:
            for param in visual_module.parameters():
                param.requires_grad = False
            logger.info("Frozen vision encoder parameters")

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(
            "Trainable VLM params: %.1fM",
            sum(p.numel() for p in trainable) / 1e6,
        )
        return trainable

    def extract_text_pooled(
        self,
        hidden_states: torch.Tensor,
        vlm_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        attention_mask = vlm_inputs.get("attention_mask", None)
        mm_token_type_ids = vlm_inputs.get("mm_token_type_ids", None)

        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            attention_mask = attention_mask.to(device=hidden_states.device).bool()

        if mm_token_type_ids is None:
            text_mask = attention_mask
        else:
            text_mask = attention_mask & (mm_token_type_ids.to(hidden_states.device) == 0)

        weights = text_mask.to(hidden_states.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden_states * weights).sum(dim=1) / denom
