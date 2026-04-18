"""
Qwen3.5 multimodal interface for VLA integration.
Wraps the requested Hugging Face checkpoint and exposes a unified interface
for the VLA framework (forward, generate, build_inputs).
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class Qwen35VLInterface(nn.Module):
    """
    Lightweight wrapper around a Qwen3.5 multimodal backbone.

    The wrapper provides:
        - Unified forward / generate interface
        - Input preprocessing (images + text -> model inputs)
        - Hidden state extraction for the flow matching action head
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-2B",
        attn_implementation: Optional[str] = None,
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
        self.processor.tokenizer.padding_side = "left"

        if hasattr(self.model.config, "text_config"):
            self.model.config.hidden_size = self.model.config.text_config.hidden_size
        self.hidden_size = self.model.config.hidden_size

        model_cls = self.model.__class__.__name__
        processor_cls = self.processor.__class__.__name__
        model_type = getattr(self.model.config, "model_type", "unknown")
        visual_module = None
        if hasattr(self.model, "visual") and getattr(self.model, "visual", None) is not None:
            visual_module = self.model.visual
        elif hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            visual_module = getattr(self.model.model, "visual", None)
        has_visual = visual_module is not None
        visual_cls = visual_module.__class__.__name__ if has_visual else "none"

        logger.info(
            "Backbone loaded | requested_model_id=%s | model_class=%s | "
            "processor_class=%s | model_type=%s | has_visual=%s | visual_class=%s | "
            "attn_implementation=%s | "
            "hidden_size=%s | dtype=%s | params=%.2fB",
            model_id,
            model_cls,
            processor_cls,
            model_type,
            has_visual,
            visual_cls,
            attn_implementation,
            self.hidden_size,
            self.model.dtype,
            sum(p.numel() for p in self.model.parameters()) / 1e9,
        )

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def forward(
        self,
        output_hidden_states: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with hidden state extraction for flow matching.

        For action-only training/inference we only need hidden states, not full
        vocabulary logits over the whole sequence. Qwen supports `logits_to_keep`,
        so when language loss is disabled we keep just the last token logits to
        avoid an expensive full-vocab projection every step.
        """
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
        """Auto-regressive generation (for language output, not actions)."""
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model.generate(**kwargs)

    def build_vla_inputs(
        self,
        images: List[List[Any]],
        instructions: List[str],
        cot_prompt: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build model inputs from images + instructions for VLA training/inference.

        Args:
            images: List of image lists, one per batch element.
            instructions: Task instruction strings.
            cot_prompt: Optional chain-of-thought prompt template.
        Returns:
            BatchFeature dict with input_ids, attention_mask, pixel_values, etc.
        """
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt else instruction

            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )

        return batch_inputs.to(self.device)

    def get_trainable_params(self, freeze_vision: bool = True):
        """Return parameter groups for optimizer with optional vision encoder freezing."""
        visual_module = None
        if hasattr(self.model, "visual") and getattr(self.model, "visual", None) is not None:
            visual_module = self.model.visual
        elif hasattr(self.model, "model") and hasattr(self.model.model, "visual"):
            visual_module = getattr(self.model.model, "visual", None)

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

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize embeddings for added special tokens (e.g. action tokens)."""
        self.model.resize_token_embeddings(new_num_tokens)
        logger.info("Resized token embeddings to %s", new_num_tokens)

    def extract_text_pooled(
        self,
        hidden_states: torch.Tensor,
        vlm_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Mean-pool text tokens from the final hidden state.

        Qwen's processor emits ``mm_token_type_ids`` where text/special tokens are 0
        and image tokens are 1. We pool only the attended text tokens so the action
        head receives an explicit instruction embedding instead of relying on the
        mixed multimodal sequence alone.
        """
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

        if text_mask.ndim != 2:
            raise ValueError(
                f"Expected text mask with shape (B, S), got {tuple(text_mask.shape)}"
            )

        weights = text_mask.to(hidden_states.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (hidden_states * weights).sum(dim=1) / denom
        return pooled
