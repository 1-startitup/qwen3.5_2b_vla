import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .qwen35_pi05_interface import Qwen35PI05Interface

logger = logging.getLogger(__name__)


class QwenBackboneAdapter(nn.Module):
    """Expose Qwen prefix encoding and full-attention taps behind a stable API."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-2B",
        attn_implementation: Optional[str] = None,
        image_resolution: Sequence[int] = (224, 224),
        empty_cameras: int = 0,
        tokenizer_max_length: int = 200,
    ):
        super().__init__()
        self.vlm = Qwen35PI05Interface(
            model_id=model_id,
            attn_implementation=attn_implementation,
            image_resolution=image_resolution,
            empty_cameras=empty_cameras,
            tokenizer_max_length=tokenizer_max_length,
        )
        self.hidden_size = int(self.vlm.hidden_size)
        self._cached_full_attention_indices: list[int] | None = None

    @property
    def device(self):
        return self.vlm.device

    @property
    def dtype(self):
        return self.vlm.dtype

    def _language_model(self) -> nn.Module:
        model_core = self.vlm._model_core()
        if hasattr(model_core, "language_model"):
            return model_core.language_model
        raise AttributeError("Qwen backbone does not expose language_model")

    def build_vla_inputs(
        self,
        images: List[List[Any]],
        prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        return self.vlm.build_vla_inputs(images, prompts)

    def full_attention_layer_indices(self) -> list[int]:
        if self._cached_full_attention_indices is not None:
            return list(self._cached_full_attention_indices)

        language_model = self._language_model()
        layer_types = getattr(language_model.config, "layer_types", None)
        if not layer_types:
            self._cached_full_attention_indices = list(range(len(language_model.layers)))
        else:
            self._cached_full_attention_indices = [
                idx for idx, layer_type in enumerate(layer_types[: len(language_model.layers)])
                if str(layer_type) == "full_attention"
            ]
        return list(self._cached_full_attention_indices)

    def encode_prefix(
        self,
        images: Optional[List[List[Any]]] = None,
        prompts: Optional[List[str]] = None,
        vlm_inputs: Optional[Dict[str, torch.Tensor]] = None,
        vlm_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if vlm_inputs is None:
            if images is None or prompts is None:
                raise ValueError("images/prompts or vlm_inputs must be provided")
            vlm_inputs = self.build_vla_inputs(images, prompts)

        prefix_inputs = self.vlm.build_prefix_from_vlm_inputs(vlm_inputs)
        fwd_kwargs = dict(output_hidden_states=True, **vlm_inputs)
        if vlm_labels is not None:
            fwd_kwargs["labels"] = vlm_labels
        outputs = self.vlm(**fwd_kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None or len(hidden_states) == 0:
            raise RuntimeError("Qwen backbone did not return hidden states for tap extraction")

        return {
            "vlm_inputs": vlm_inputs,
            "vlm_outputs": outputs,
            "hidden_states": hidden_states,
            "prefix_embeds": prefix_inputs["prefix_embeds"],
            "attention_mask": prefix_inputs["attention_mask"],
            "position_ids": prefix_inputs["position_ids"],
        }

    def _text_attention_mask(
        self,
        encoded_prefix: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        attention_mask = encoded_prefix.get("attention_mask", None)
        if attention_mask is None:
            return None

        text_mask = attention_mask.to(dtype=torch.bool)
        vlm_inputs = encoded_prefix.get("vlm_inputs", {})
        mm_token_type_ids = vlm_inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            return text_mask

        mm_token_type_ids = mm_token_type_ids.to(device=text_mask.device)
        if mm_token_type_ids.shape != text_mask.shape:
            logger.warning(
                "Ignoring mm_token_type_ids with mismatched shape: %s vs %s",
                tuple(mm_token_type_ids.shape),
                tuple(text_mask.shape),
            )
            return text_mask
        return text_mask & (mm_token_type_ids == 0)

    def _token_type_ids(
        self,
        encoded_prefix: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        attention_mask = encoded_prefix.get("attention_mask", None)
        if attention_mask is None:
            return None

        vlm_inputs = encoded_prefix.get("vlm_inputs", {})
        mm_token_type_ids = vlm_inputs.get("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            return torch.zeros_like(attention_mask, dtype=torch.long)

        mm_token_type_ids = mm_token_type_ids.to(device=attention_mask.device)
        if mm_token_type_ids.shape != attention_mask.shape:
            logger.warning(
                "Ignoring mm_token_type_ids with mismatched shape: %s vs %s",
                tuple(mm_token_type_ids.shape),
                tuple(attention_mask.shape),
            )
            return torch.zeros_like(attention_mask, dtype=torch.long)
        return mm_token_type_ids.to(dtype=torch.long)

    def _instruction_summary(
        self,
        encoded_prefix: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        hidden_states = encoded_prefix.get("hidden_states", None)
        vlm_inputs = encoded_prefix.get("vlm_inputs", None)
        if hidden_states is None or vlm_inputs is None or len(hidden_states) == 0:
            return None
        return self.vlm.extract_text_pooled(hidden_states[-1], vlm_inputs)

    def get_full_attention_taps(
        self,
        encoded_prefix: Dict[str, Any],
        tap_strategy: str = "last",
    ) -> Dict[str, Any]:
        hidden_states = encoded_prefix["hidden_states"]
        tap_indices = self.full_attention_layer_indices()
        if not tap_indices:
            raise RuntimeError("No full-attention layers were discovered in the Qwen backbone")

        tap_tensors = [hidden_states[idx + 1] for idx in tap_indices]
        strategy = str(tap_strategy).lower()
        base_attention_mask = encoded_prefix["attention_mask"].to(dtype=torch.bool)
        base_token_type_ids = self._token_type_ids(encoded_prefix)
        if strategy == "last":
            selected_indices = [tap_indices[-1]]
            selected_tensors = [tap_tensors[-1]]
            memory = selected_tensors[0]
            memory_mask = base_attention_mask
            tap_ids = torch.full_like(base_attention_mask, fill_value=selected_indices[0], dtype=torch.long)
            token_type_ids = base_token_type_ids
        elif strategy in {"all_concat", "concat_all", "all", "scalar_mix"}:
            selected_indices = tap_indices
            selected_tensors = tap_tensors
            memory = torch.cat(selected_tensors, dim=1)
            memory_mask = torch.cat([base_attention_mask for _ in selected_tensors], dim=1)
            tap_ids = torch.cat(
                [
                    torch.full_like(base_attention_mask, fill_value=layer_idx, dtype=torch.long)
                    for layer_idx in selected_indices
                ],
                dim=1,
            )
            token_type_ids = None
            if base_token_type_ids is not None:
                token_type_ids = torch.cat([base_token_type_ids for _ in selected_tensors], dim=1)
        else:
            raise ValueError(
                f"Unsupported tap_strategy={tap_strategy}. Expected one of ['last', 'all_concat', 'scalar_mix']"
            )

        return {
            "tap_indices": selected_indices,
            "tap_tensors": selected_tensors,
            "memory": memory,
            "memory_mask": memory_mask,
            "attention_mask": memory_mask,
            "tap_ids": tap_ids,
            "token_type_ids": token_type_ids,
        }

    def get_prefix_memory(
        self,
        encoded_prefix: Dict[str, Any],
        tap_strategy: str = "last",
    ) -> Dict[str, Any]:
        taps = self.get_full_attention_taps(encoded_prefix, tap_strategy=tap_strategy)
        return {
            "memory": taps["memory"],
            "memory_mask": taps["memory_mask"],
            "attention_mask": taps["attention_mask"],
            "tap_ids": taps["tap_ids"],
            "token_type_ids": taps["token_type_ids"],
            "tap_indices": taps["tap_indices"],
        }
