"""
Qwen3.5-2B VLA Training Script for LIBERO
=========================================
Full training loop with:
  - Accelerate + DeepSpeed ZeRO-2
  - Flow matching action loss + optional VLM co-training loss
  - Cosine LR scheduler with warmup
  - Evaluation, checkpointing, W&B logging

Usage:
  # Single GPU
  python train.py --config config/libero_train_2b.yaml

  # Multi-GPU with DeepSpeed
  accelerate launch --config_file config/deepspeed_zero2.yaml \
    --num_processes 8 train.py --config config/libero_train_2b.yaml

  # Override config from CLI
  python train.py --config config/libero_train_2b.yaml \
    --model.vlm_model_id /path/to/local/Qwen3.5-2B \
    --dataset.name libero_all \
    --training.max_steps 50000
"""

import os
import sys
import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import set_seed

import logging
warnings.filterwarnings("ignore", message="Kwargs passed to")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
logger = logging.getLogger(__name__)


def normalize_action_head_state_dict(state_dict: dict) -> dict:
    """
    Strip wrapper prefixes that appear when the action head was saved from a compiled
    or otherwise wrapped module. This keeps checkpoints loadable in plain eval code.
    """
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        normalized[new_key] = value
    return normalized


OPTIONAL_ACTION_HEAD_KEY_PREFIXES = (
    "input_adarms.",
    "post_adarms.",
    "final_adarms.",
    "state_encoder.",
    "history_encoder.",
    "raw_gripper_head.",
    "phase_head.",
    "immediate_motion_head.",
)
OPTIONAL_ACTION_HEAD_KEYS = {
    "action_pos_embed",
    "alpha_state",
    "alpha_hist",
}


def filter_optional_action_head_load_issues(
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> tuple[list[str], list[str]]:
    def _keep(key: str) -> bool:
        if key in OPTIONAL_ACTION_HEAD_KEYS:
            return False
        return not any(key.startswith(prefix) for prefix in OPTIONAL_ACTION_HEAD_KEY_PREFIXES)

    filtered_missing = [key for key in missing_keys if _keep(key)]
    filtered_unexpected = [key for key in unexpected_keys if _keep(key)]
    return filtered_missing, filtered_unexpected


def build_policy_contract(cfg) -> tuple[dict, dict]:
    """Serialize the runtime pre/postprocess contract alongside each checkpoint."""
    architecture = str(cfg.model.get("architecture", "qwen35_vla"))
    action_cfg = cfg.action_head
    head_type = str(action_cfg.get("head_type", "single_fm"))
    use_state_prompt = bool(action_cfg.get("use_state_prompt", False))
    image_size = list(cfg.dataset.get("image_size", [224, 224]))
    if architecture in {"qwen_3.5_2b_gemma_bridge_v3_1", "qwen_3.5_2b_gemma_bridge_v3_2"}:
        pre = {
            "name": "policy_preprocessor",
            "version": "qwen_3.5_2b_gemma_bridge_v3_2",
            "images": {
                "order": ["observation.images.image", "observation.images.wrist_image"],
                "count": 2 + int(cfg.dataset.get("empty_cameras", 0)),
                "source_resolution": image_size,
                "missing_camera_policy": "append_empty_camera",
                "train_eval_contract": {
                    "input_format": "PIL_RGB",
                    "resize_mode": "resize_with_pad",
                    "target_resolution": image_size,
                    "multimodal_entrypoint": "QwenBackboneAdapter.encode_prefix",
                    "processor": {
                        "type": "Qwen3VLProcessor",
                        "model_id": str(cfg.model.vlm_model_id),
                        "padding_side": str(cfg.dataset.get("tokenizer_padding_side", "right")),
                        "max_length": int(cfg.dataset.get("tokenizer_max_length", 200)),
                    },
                },
            },
            "state": {
                "format": "eef_pos_axis_angle_gripper_qpos2",
                "dim": int(action_cfg.get("state_dim", 8)),
                "prompt_normalization": "QUANTILES_Q01_Q99",
            },
            "history": {
                "enabled": False,
                "length": 0,
                "layout": [],
            },
            "prompt": {
                "use_state_prompt": True,
                "template": "Task: {instruction}, State: {discretized_state};\\nAction: ",
                "state_prompt_bins": int(action_cfg.get("state_prompt_bins", 256)),
                "state_prompt_history_len": 0,
            },
            "normalization": {
                "action": "QUANTILES_Q01_Q99",
                "state": "QUANTILES_Q01_Q99",
                "gripper_semantics": "SIGNED_-1_1",
            },
            "motion_frame": "world",
        }
        post = {
            "name": "policy_postprocessor",
            "action_dim": int(action_cfg.get("action_dim", 7)),
            "action_horizon": int(action_cfg.get("action_horizon", action_cfg.get("chunk_size", 50))),
            "gripper_output": "SIGNED_-1_1",
            "unnormalization": "QUANTILES_Q01_Q99_FROM_ACTION_HEAD_PT",
            "runtime": {
                "chunk_size_default": int(action_cfg.get("n_action_steps", action_cfg.get("chunk_size", 50))),
                "reverse_time_sampler": True,
                "num_inference_steps": int(action_cfg.get("num_inference_steps", 10)),
                "bridge": {
                    "type": str(action_cfg.get("bridge_type", "gr00t_query_bridge")),
                    "memory_layout": "bridge_query_tokens",
                    "tap_strategy": str(action_cfg.get("tap_strategy", "all_concat")),
                    "stop_gradient_backbone": bool(action_cfg.get("stop_gradient_backbone", True)),
                    "bridge_norm_type": str(action_cfg.get("bridge_norm_type", "layernorm")),
                    "bridge_policy_dim": int(
                        action_cfg.get(
                            "bridge_policy_dim",
                            action_cfg.get("bridge_out_dim", action_cfg.get("hidden_dim", 1024)),
                        )
                    ),
                    "bridge_num_queries": int(action_cfg.get("bridge_num_queries", 8)),
                    "bridge_layers": int(action_cfg.get("bridge_layers", 2)),
                    "bridge_scalar_mix": bool(action_cfg.get("bridge_scalar_mix", False)),
                    "bridge_scalar_mix_init": str(action_cfg.get("bridge_scalar_mix_init", "uniform")),
                    "bridge_scalar_mix_use_gamma": bool(
                        action_cfg.get("bridge_scalar_mix_use_gamma", True)
                    ),
                    "bridge_use_state_token": bool(action_cfg.get("bridge_use_state_token", True)),
                    "bridge_use_tap_embeddings": bool(action_cfg.get("bridge_use_tap_embeddings", True)),
                    "bridge_use_token_type_embeddings": bool(
                        action_cfg.get("bridge_use_token_type_embeddings", True)
                    ),
                    "bridge_dropout": float(action_cfg.get("bridge_dropout", 0.0)),
                    "bridge_gate_bias": float(action_cfg.get("bridge_gate_bias", 0.0)),
                    "action_expert_variant": str(action_cfg.get("action_expert_variant", "gemma_300m")),
                    "use_action_input_conditioning": bool(
                        action_cfg.get("use_action_input_conditioning", True)
                    ),
                },
                "phase_gated_gripper": False,
                "immediate_correction": False,
                "correction_blend": 0.0,
            },
        }
        return pre, post
    if architecture == "qwen_3.5_2b_pi0.5":
        use_state_prompt = True
        pre = {
            "name": "policy_preprocessor",
            "version": "qwen_3.5_2b_pi0.5",
            "images": {
                "order": ["observation.images.image", "observation.images.wrist_image"],
                "count": 2 + int(cfg.dataset.get("empty_cameras", 0)),
                "source_resolution": image_size,
                "missing_camera_policy": "append_empty_camera",
                "train_eval_contract": {
                    "input_format": "PIL_RGB",
                    "resize_mode": "resize_with_pad",
                    "target_resolution": image_size,
                    "multimodal_entrypoint": "Qwen35PI05Interface.build_vla_inputs",
                    "processor": {
                        "type": "Qwen3VLProcessor",
                        "model_id": str(cfg.model.vlm_model_id),
                        "padding_side": str(cfg.dataset.get("tokenizer_padding_side", "right")),
                        "max_length": int(cfg.dataset.get("tokenizer_max_length", 200)),
                    },
                },
            },
            "state": {
                "format": "eef_pos_axis_angle_gripper_qpos2",
                "dim": int(action_cfg.get("state_dim", 8)),
                "prompt_normalization": "QUANTILES_Q01_Q99",
            },
            "history": {
                "enabled": int(action_cfg.get("history_len", 0)) > 0,
                "length": int(action_cfg.get("history_len", 0)),
                "layout": ["state", "action", "valid_flag"] if int(action_cfg.get("history_len", 0)) > 0 else [],
            },
            "prompt": {
                "use_state_prompt": True,
                "template": "Task: {instruction}, State: {discretized_state};\\nAction: ",
                "state_prompt_bins": int(action_cfg.get("state_prompt_bins", 256)),
                "state_prompt_history_len": 0,
            },
            "normalization": {
                "action": "QUANTILES_Q01_Q99",
                "state": "QUANTILES_Q01_Q99",
                "gripper_semantics": "SIGNED_-1_1",
            },
            "motion_frame": "world",
        }
        post = {
            "name": "policy_postprocessor",
            "action_dim": int(action_cfg.get("action_dim", 7)),
            "action_horizon": int(action_cfg.get("action_horizon", action_cfg.get("chunk_size", 50))),
            "gripper_output": "SIGNED_-1_1",
            "unnormalization": "QUANTILES_Q01_Q99_FROM_ACTION_HEAD_PT",
            "runtime": {
                "chunk_size_default": int(action_cfg.get("n_action_steps", action_cfg.get("chunk_size", 50))),
                "reverse_time_sampler": True,
                "num_inference_steps": int(action_cfg.get("num_inference_steps", 10)),
                "rtc": OmegaConf.to_container(action_cfg.get("rtc_config", None), resolve=True) if action_cfg.get("rtc_config", None) is not None else None,
                "phase_gated_gripper": bool(action_cfg.get("use_binary_gripper", False) and action_cfg.get("use_phase_head", False)),
                "immediate_correction": bool(action_cfg.get("use_immediate_correction", False)),
                "correction_blend": float(action_cfg.get("correction_blend", 0.0)),
            },
        }
        return pre, post

    pre = {
        "name": "policy_preprocessor",
        "version": "v4" if head_type == "layerwise_fm_v4" else "v3",
        "images": {
            "order": ["observation.images.image", "observation.images.wrist_image"],
            "count": 2,
            "source_resolution": image_size,
            "missing_camera_policy": "reject_sample_in_training",
            "train_eval_contract": {
                "input_format": "PIL_RGB",
                "resize_mode": "dataset_pre_resized",
                "target_resolution": image_size,
                "multimodal_entrypoint": "Qwen35VLInterface.build_vla_inputs",
                "processor": {
                    "type": "Qwen3VLProcessor",
                    "model_id": str(cfg.model.vlm_model_id),
                    "chat_template": "processor.apply_chat_template",
                },
            },
        },
        "state": {
            "format": "eef_pos_axis_angle_gripper_qpos2",
            "dim": int(action_cfg.get("state_dim", 8)),
        },
        "history": {
            "enabled": int(action_cfg.get("history_len", 0)) > 0,
            "length": int(action_cfg.get("history_len", 0)),
            "layout": ["state", "action", "valid_flag"],
        },
        "prompt": {
            "use_state_prompt": use_state_prompt,
            "template": "Task: {instruction}\\nState: {discretized_state}\\nAction:" if use_state_prompt else "{instruction}",
            "state_prompt_bins": int(action_cfg.get("state_prompt_bins", 256)),
            "state_prompt_history_len": int(action_cfg.get("state_prompt_history_len", 0)),
        },
        "normalization": {
            "action": "QUANTILES_Q01_Q99",
            "gripper_semantics": "SIGNED_-1_1",
        },
        "motion_frame": "eef_local" if head_type == "layerwise_fm_v4" and bool(action_cfg.get("use_local_action_frame", True)) else "world",
    }
    post = {
        "name": "policy_postprocessor",
        "action_dim": int(action_cfg.get("action_dim", 7)),
        "action_horizon": int(action_cfg.get("action_horizon", 8)),
        "gripper_output": "SIGNED_-1_1",
        "unnormalization": "QUANTILES_Q01_Q99_FROM_ACTION_HEAD_PT",
        "runtime": {
            "chunk_size_default": int(action_cfg.get("action_horizon", 8)),
            "phase_gated_gripper": head_type == "layerwise_fm_v4",
            "immediate_correction": head_type == "layerwise_fm_v4",
            "correction_blend": float(action_cfg.get("correction_blend", 0.0)) if head_type == "layerwise_fm_v4" else 0.0,
        },
    }
    return pre, post


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3.5-2B VLA on LIBERO")
    parser.add_argument(
        "--config",
        type=str,
        default="config/libero_train_2b.yaml",
        help="Path to YAML config file",
    )
    args, unknown = parser.parse_known_args()

    # Load YAML config
    cfg = OmegaConf.load(args.config)

    # Apply CLI overrides (e.g., --model.vlm_model_id /path/to/model)
    if unknown:
        cli_cfg = OmegaConf.from_dotlist(
            [u.lstrip("-") for u in unknown if "=" in u or not u.startswith("-")]
        )
        # Handle key=value pairs from CLI
        overrides = []
        i = 0
        while i < len(unknown):
            key = unknown[i].lstrip("-")
            if "=" in key:
                overrides.append(key)
                i += 1
            elif i + 1 < len(unknown) and not unknown[i + 1].startswith("-"):
                overrides.append(f"{key}={unknown[i+1]}")
                i += 2
            else:
                i += 1
        if overrides:
            cli_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def build_model(cfg):
    """Build the Qwen3.5 VLA model from config."""
    architecture = str(cfg.model.get("architecture", "qwen35_vla"))
    from model.qwen35_vla import Qwen35VLA
    from model.flow_matching_head import FlowMatchingConfig
    from model.layerwise_flow_matching_head import LayerwiseFlowMatchingConfig
    from model.layerwise_flow_matching_head_v3 import LayerwiseFlowMatchingV3Config
    from model.layerwise_flow_matching_head_v4 import LayerwiseFlowMatchingV4Config
    from model.qwen35_pi05_action_head import ACTION_EXPERT_VARIANTS, QwenPI05ActionConfig
    from model.qwen_gemma_bridge_action_head import QwenGemmaBridgeActionConfig
    from model.qwen35_gemma_bridge_vla import Qwen35GemmaBridgeVLA
    from model.qwen35_pi05_vla import Qwen35PI05VLA
    from model.rtc import RTCConfig

    # Build action head config
    ah_cfg = cfg.action_head
    head_type = ah_cfg.get("head_type", "single_fm")
    if architecture in {"qwen_3.5_2b_gemma_bridge_v3_1", "qwen_3.5_2b_gemma_bridge_v3_2"}:
        variant_name = str(ah_cfg.get("action_expert_variant", "gemma_300m"))
        variant_defaults = ACTION_EXPERT_VARIANTS.get(variant_name)
        if variant_defaults is None:
            raise ValueError(
                f"Unsupported action_expert_variant={variant_name}. "
                f"Expected one of {sorted(ACTION_EXPERT_VARIANTS)}"
            )
        image_resolution = ah_cfg.get("image_resolution", cfg.dataset.get("image_size", [224, 224]))
        action_config = QwenGemmaBridgeActionConfig(
            head_type=head_type,
            action_expert_variant=variant_name,
            hidden_dim=int(ah_cfg.get("hidden_dim", variant_defaults["hidden_dim"])),
            num_heads=int(ah_cfg.get("num_heads", variant_defaults["num_heads"])),
            num_kv_heads=int(ah_cfg.get("num_kv_heads", 1)),
            num_layers=int(ah_cfg.get("num_layers", variant_defaults["num_layers"])),
            mlp_dim=int(ah_cfg.get("mlp_dim", 0)),
            dropout=float(ah_cfg.get("dropout", 0.0)),
            action_dim=int(ah_cfg.get("action_dim", 7)),
            action_horizon=int(ah_cfg.get("action_horizon", ah_cfg.get("chunk_size", 50))),
            chunk_size=int(ah_cfg.get("chunk_size", ah_cfg.get("action_horizon", 50))),
            n_action_steps=int(ah_cfg.get("n_action_steps", ah_cfg.get("chunk_size", 50))),
            max_state_dim=int(ah_cfg.get("max_state_dim", 32)),
            max_action_dim=int(ah_cfg.get("max_action_dim", 32)),
            beta_alpha=float(ah_cfg.get("beta_alpha", 1.5)),
            beta_beta=float(ah_cfg.get("beta_beta", 1.0)),
            time_sampling_scale=float(ah_cfg.get("time_sampling_scale", 0.999)),
            time_sampling_offset=float(ah_cfg.get("time_sampling_offset", 0.001)),
            min_period=float(ah_cfg.get("min_period", 0.004)),
            max_period=float(ah_cfg.get("max_period", 4.0)),
            num_inference_steps=int(ah_cfg.get("num_inference_steps", 10)),
            vlm_hidden_dim=int(ah_cfg.get("vlm_hidden_dim", 2048)),
            state_dim=int(ah_cfg.get("state_dim", 8)),
            state_prompt_bins=int(ah_cfg.get("state_prompt_bins", 256)),
            tokenizer_max_length=int(
                ah_cfg.get("tokenizer_max_length", cfg.dataset.get("tokenizer_max_length", 200))
            ),
            image_resolution=(int(image_resolution[0]), int(image_resolution[1])),
            empty_cameras=int(ah_cfg.get("empty_cameras", cfg.dataset.get("empty_cameras", 0))),
            bridge_type=str(ah_cfg.get("bridge_type", "gr00t_query_bridge")),
            tap_strategy=str(ah_cfg.get("tap_strategy", "all_concat")),
            stop_gradient_backbone=bool(ah_cfg.get("stop_gradient_backbone", True)),
            bridge_norm_type=str(ah_cfg.get("bridge_norm_type", "layernorm")),
            bridge_out_dim=int(ah_cfg.get("bridge_out_dim", 0)),
            bridge_dropout=float(ah_cfg.get("bridge_dropout", 0.0)),
            bridge_gate_bias=float(ah_cfg.get("bridge_gate_bias", 0.0)),
            bridge_num_queries=int(ah_cfg.get("bridge_num_queries", 8)),
            bridge_layers=int(ah_cfg.get("bridge_layers", 2)),
            bridge_policy_dim=int(
                ah_cfg.get(
                    "bridge_policy_dim",
                    ah_cfg.get("bridge_out_dim", variant_defaults["hidden_dim"]),
                )
            ),
            bridge_use_state_token=bool(ah_cfg.get("bridge_use_state_token", True)),
            bridge_use_tap_embeddings=bool(ah_cfg.get("bridge_use_tap_embeddings", True)),
            bridge_use_token_type_embeddings=bool(
                ah_cfg.get("bridge_use_token_type_embeddings", True)
            ),
            bridge_max_taps=int(ah_cfg.get("bridge_max_taps", 64)),
            bridge_num_token_types=int(ah_cfg.get("bridge_num_token_types", 8)),
            bridge_scalar_mix=bool(ah_cfg.get("bridge_scalar_mix", False)),
            bridge_scalar_mix_init=str(ah_cfg.get("bridge_scalar_mix_init", "uniform")),
            bridge_scalar_mix_use_gamma=bool(ah_cfg.get("bridge_scalar_mix_use_gamma", True)),
            use_action_input_conditioning=bool(
                ah_cfg.get("use_action_input_conditioning", True)
            ),
            action_input_gain_init=float(ah_cfg.get("action_input_gain_init", 0.5)),
            conditioning_gate_bias=float(ah_cfg.get("conditioning_gate_bias", 1.0)),
            memory_norm_ratio_limit=float(ah_cfg.get("memory_norm_ratio_limit", 8.0)),
        )
    elif architecture == "qwen_3.5_2b_pi0.5":
        variant_name = str(ah_cfg.get("action_expert_variant", "gemma_300m"))
        variant_defaults = ACTION_EXPERT_VARIANTS.get(variant_name)
        if variant_defaults is None:
            raise ValueError(
                f"Unsupported action_expert_variant={variant_name}. "
                f"Expected one of {sorted(ACTION_EXPERT_VARIANTS)}"
            )
        image_resolution = ah_cfg.get("image_resolution", cfg.dataset.get("image_size", [224, 224]))
        rtc_cfg = None
        if ah_cfg.get("rtc_config", None) is not None:
            rtc_node = ah_cfg.get("rtc_config")
            rtc_cfg = RTCConfig(
                enabled=bool(rtc_node.get("enabled", False)),
                prefix_attention_schedule=str(rtc_node.get("prefix_attention_schedule", "linear")),
                max_guidance_weight=float(rtc_node.get("max_guidance_weight", 10.0)),
                execution_horizon=int(rtc_node.get("execution_horizon", 10)),
                debug=bool(rtc_node.get("debug", False)),
                debug_maxlen=int(rtc_node.get("debug_maxlen", 100)),
            )
        action_config = QwenPI05ActionConfig(
            head_type=head_type,
            action_expert_variant=variant_name,
            hidden_dim=int(ah_cfg.get("hidden_dim", variant_defaults["hidden_dim"])),
            num_heads=int(ah_cfg.get("num_heads", variant_defaults["num_heads"])),
            num_layers=int(ah_cfg.get("num_layers", variant_defaults["num_layers"])),
            dropout=float(ah_cfg.get("dropout", 0.0)),
            action_dim=int(ah_cfg.get("action_dim", 7)),
            action_horizon=int(ah_cfg.get("action_horizon", ah_cfg.get("chunk_size", 50))),
            chunk_size=int(ah_cfg.get("chunk_size", ah_cfg.get("action_horizon", 50))),
            n_action_steps=int(ah_cfg.get("n_action_steps", ah_cfg.get("chunk_size", 50))),
            max_state_dim=int(ah_cfg.get("max_state_dim", 32)),
            max_action_dim=int(ah_cfg.get("max_action_dim", 32)),
            beta_alpha=float(ah_cfg.get("beta_alpha", 1.5)),
            beta_beta=float(ah_cfg.get("beta_beta", 1.0)),
            time_sampling_scale=float(ah_cfg.get("time_sampling_scale", 0.999)),
            time_sampling_offset=float(ah_cfg.get("time_sampling_offset", 0.001)),
            min_period=float(ah_cfg.get("min_period", 0.004)),
            max_period=float(ah_cfg.get("max_period", 4.0)),
            num_inference_steps=int(ah_cfg.get("num_inference_steps", 10)),
            vlm_hidden_dim=int(ah_cfg.get("vlm_hidden_dim", 2048)),
            state_dim=int(ah_cfg.get("state_dim", 8)),
            state_prompt_bins=int(ah_cfg.get("state_prompt_bins", 256)),
            tokenizer_max_length=int(ah_cfg.get("tokenizer_max_length", cfg.dataset.get("tokenizer_max_length", 200))),
            image_resolution=(int(image_resolution[0]), int(image_resolution[1])),
            empty_cameras=int(ah_cfg.get("empty_cameras", cfg.dataset.get("empty_cameras", 0))),
            use_adarms=bool(ah_cfg.get("use_adarms", True)),
            use_action_pos_embed=bool(ah_cfg.get("use_action_pos_embed", False)),
            rtc_config=rtc_cfg,
            history_len=int(ah_cfg.get("history_len", 0)),
            history_feature_dim=int(
                ah_cfg.get(
                    "history_feature_dim",
                    int(ah_cfg.get("state_dim", 8)) + int(ah_cfg.get("action_dim", 7)) + 1,
                )
            ),
            use_state_conditioning=bool(ah_cfg.get("use_state_conditioning", False)),
            use_history_conditioning=bool(ah_cfg.get("use_history_conditioning", False)),
            use_binary_gripper=bool(ah_cfg.get("use_binary_gripper", False)),
            use_phase_head=bool(ah_cfg.get("use_phase_head", False)),
            phase_loss_weight=float(ah_cfg.get("phase_loss_weight", 0.10)),
            gripper_loss_weight=float(ah_cfg.get("gripper_loss_weight", 0.20)),
            phase_gate_strength=float(ah_cfg.get("phase_gate_strength", 2.0)),
            grasp_phase_steps=int(ah_cfg.get("grasp_phase_steps", 2)),
            use_immediate_correction=bool(ah_cfg.get("use_immediate_correction", False)),
            immediate_loss_weight=float(ah_cfg.get("immediate_loss_weight", 0.25)),
            correction_blend=float(ah_cfg.get("correction_blend", 0.50)),
        )
    elif head_type == "layerwise_fm":
        condition_layer_indices = ah_cfg.get("condition_layer_indices", None)
        if condition_layer_indices is not None:
            condition_layer_indices = [int(idx) for idx in condition_layer_indices]
        action_config = LayerwiseFlowMatchingConfig(
            head_type=head_type,
            hidden_dim=ah_cfg.get("hidden_dim", 1024),
            num_heads=ah_cfg.get("num_heads", 8),
            num_layers=ah_cfg.get("num_layers", 12),
            dropout=ah_cfg.dropout,
            action_dim=ah_cfg.action_dim,
            action_horizon=ah_cfg.action_horizon,
            beta_alpha=ah_cfg.beta_alpha,
            beta_beta=ah_cfg.beta_beta,
            num_inference_steps=ah_cfg.get("num_inference_steps", 10),
            timestep_buckets=ah_cfg.get("timestep_buckets", 1000),
            vlm_hidden_dim=ah_cfg.vlm_hidden_dim,
            state_dim=ah_cfg.state_dim,
            num_future_tokens=ah_cfg.get("num_future_tokens", 0),
            condition_layer_indices=condition_layer_indices,
        )
    elif head_type == "layerwise_fm_v3":
        condition_layer_indices = ah_cfg.get("condition_layer_indices", None)
        if condition_layer_indices is not None:
            condition_layer_indices = [int(idx) for idx in condition_layer_indices]
        action_config = LayerwiseFlowMatchingV3Config(
            head_type=head_type,
            hidden_dim=ah_cfg.get("hidden_dim", 1024),
            num_heads=ah_cfg.get("num_heads", 8),
            num_layers=ah_cfg.get("num_layers", 12),
            dropout=ah_cfg.dropout,
            action_dim=ah_cfg.action_dim,
            action_horizon=ah_cfg.action_horizon,
            beta_alpha=ah_cfg.beta_alpha,
            beta_beta=ah_cfg.beta_beta,
            num_inference_steps=ah_cfg.get("num_inference_steps", 10),
            timestep_buckets=ah_cfg.get("timestep_buckets", 1000),
            vlm_hidden_dim=ah_cfg.vlm_hidden_dim,
            state_dim=ah_cfg.state_dim,
            num_future_tokens=ah_cfg.get("num_future_tokens", 0),
            history_len=ah_cfg.get("history_len", 6),
            history_dim=ah_cfg.get("history_dim", 16),
            gripper_loss_weight=ah_cfg.get("gripper_loss_weight", 0.25),
            consistency_loss_weight=ah_cfg.get("consistency_loss_weight", 0.01),
            consistency_margin=ah_cfg.get("consistency_margin", 0.10),
            use_text_conditioning=ah_cfg.get("use_text_conditioning", True),
            use_state_prompt=ah_cfg.get("use_state_prompt", True),
            state_prompt_bins=ah_cfg.get("state_prompt_bins", 256),
            state_prompt_history_len=ah_cfg.get("state_prompt_history_len", 0),
            condition_layer_indices=condition_layer_indices,
        )
    elif head_type == "layerwise_fm_v4":
        condition_layer_indices = ah_cfg.get("condition_layer_indices", None)
        if condition_layer_indices is not None:
            condition_layer_indices = [int(idx) for idx in condition_layer_indices]
        action_config = LayerwiseFlowMatchingV4Config(
            head_type=head_type,
            hidden_dim=ah_cfg.get("hidden_dim", 1024),
            num_heads=ah_cfg.get("num_heads", 8),
            num_layers=ah_cfg.get("num_layers", 12),
            dropout=ah_cfg.dropout,
            action_dim=ah_cfg.action_dim,
            action_horizon=ah_cfg.action_horizon,
            beta_alpha=ah_cfg.beta_alpha,
            beta_beta=ah_cfg.beta_beta,
            num_inference_steps=ah_cfg.get("num_inference_steps", 10),
            timestep_buckets=ah_cfg.get("timestep_buckets", 1000),
            vlm_hidden_dim=ah_cfg.vlm_hidden_dim,
            state_dim=ah_cfg.state_dim,
            num_future_tokens=ah_cfg.get("num_future_tokens", 0),
            history_len=ah_cfg.get("history_len", 6),
            history_dim=ah_cfg.get("history_dim", 16),
            use_text_conditioning=ah_cfg.get("use_text_conditioning", True),
            use_state_prompt=ah_cfg.get("use_state_prompt", True),
            state_prompt_bins=ah_cfg.get("state_prompt_bins", 256),
            state_prompt_history_len=ah_cfg.get("state_prompt_history_len", 0),
            use_local_action_frame=ah_cfg.get("use_local_action_frame", True),
            phase_loss_weight=ah_cfg.get("phase_loss_weight", 0.10),
            immediate_loss_weight=ah_cfg.get("immediate_loss_weight", 0.25),
            gripper_loss_weight=ah_cfg.get("gripper_loss_weight", 0.20),
            use_dynamic_loss_balance=ah_cfg.get("use_dynamic_loss_balance", True),
            correction_blend=ah_cfg.get("correction_blend", 0.60),
            phase_gate_strength=ah_cfg.get("phase_gate_strength", 2.0),
            grasp_phase_steps=ah_cfg.get("grasp_phase_steps", 2),
            condition_layer_indices=condition_layer_indices,
        )
    else:
        action_config = FlowMatchingConfig(
            head_type=head_type,
            dit_preset=ah_cfg.dit_preset,
            hidden_dim=ah_cfg.hidden_dim,
            num_heads=ah_cfg.num_heads,
            num_layers=ah_cfg.num_layers,
            dropout=ah_cfg.dropout,
            action_dim=ah_cfg.action_dim,
            action_horizon=ah_cfg.action_horizon,
            beta_alpha=ah_cfg.beta_alpha,
            beta_beta=ah_cfg.beta_beta,
            num_inference_steps=ah_cfg.num_inference_steps,
            timestep_buckets=ah_cfg.get("timestep_buckets", 1000),
            vlm_hidden_dim=ah_cfg.vlm_hidden_dim,
            state_dim=ah_cfg.state_dim,
        )

    # LoRA config (None if not specified → full fine-tune)
    lora_cfg = None
    if cfg.get("lora") and cfg.lora.get("enabled", False):
        lora_cfg = {
            "r": cfg.lora.get("r", 32),
            "alpha": cfg.lora.get("alpha", 64),
            "dropout": cfg.lora.get("dropout", 0.05),
            "target_modules": list(cfg.lora.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ])),
        }

    if architecture in {"qwen_3.5_2b_gemma_bridge_v3_1", "qwen_3.5_2b_gemma_bridge_v3_2"}:
        model = Qwen35GemmaBridgeVLA(
            vlm_model_id=cfg.model.vlm_model_id,
            action_config=action_config,
            freeze_vision_encoder=cfg.model.freeze_vision_encoder,
            freeze_vlm=cfg.model.freeze_vlm,
            vlm_loss_weight=cfg.model.get("vlm_loss_weight", 0.0),
            lora_config=lora_cfg,
            attn_implementation=cfg.model.get("attn_implementation", None),
        )
    elif architecture == "qwen_3.5_2b_pi0.5":
        model = Qwen35PI05VLA(
            vlm_model_id=cfg.model.vlm_model_id,
            action_config=action_config,
            freeze_vision_encoder=cfg.model.freeze_vision_encoder,
            freeze_vlm=cfg.model.freeze_vlm,
            vlm_loss_weight=cfg.model.get("vlm_loss_weight", 0.0),
            lora_config=lora_cfg,
            attn_implementation=cfg.model.get("attn_implementation", None),
        )
    else:
        model = Qwen35VLA(
            vlm_model_id=cfg.model.vlm_model_id,
            action_config=action_config,
            action_head_type=head_type,
            freeze_vision_encoder=cfg.model.freeze_vision_encoder,
            freeze_vlm=cfg.model.freeze_vlm,
            vlm_loss_weight=cfg.model.vlm_loss_weight,
            cot_prompt=cfg.get("cot_prompt", None),
            lora_config=lora_cfg,
            attn_implementation=cfg.model.get("attn_implementation", None),
        )

    # Enable gradient checkpointing to reduce VRAM
    if cfg.training.get("gradient_checkpointing", False):
        # Required for LoRA + gradient checkpointing
        if lora_cfg is not None and hasattr(model.vlm.model, "enable_input_require_grads"):
            model.vlm.model.enable_input_require_grads()
        if hasattr(model.vlm.model, "gradient_checkpointing_enable"):
            model.vlm.model.gradient_checkpointing_enable()
            if hasattr(model.vlm.model, "config"):
                model.vlm.model.config.use_cache = False
            logger.info("Gradient checkpointing enabled on VLM backbone")
        else:
            logger.warning("VLM backbone does not support gradient_checkpointing_enable()")

    return model


def build_dataloader(cfg):
    """Build LIBERO dataloader from config."""
    from data.libero_dataset import build_libero_dataloader

    loader, dataset = build_libero_dataloader(
        dataset_name=cfg.dataset.name,
        data_root=cfg.dataset.get("data_root", None),
        batch_size=cfg.dataset.batch_size,
        action_horizon=cfg.action_head.action_horizon,
        num_workers=cfg.dataset.num_workers,
        image_size=tuple(cfg.dataset.image_size),
        action_type=cfg.dataset.action_type,
        vlm_model_id=cfg.model.vlm_model_id,
        prefetch_factor=cfg.dataset.get("prefetch_factor", 8),
        max_train_samples=cfg.dataset.get("max_train_samples", None),
        subset_seed=cfg.dataset.get("subset_seed", 42),
        norm_stats_sample_size=cfg.dataset.get("norm_stats_sample_size", 20000),
        history_len=cfg.action_head.get("history_len", 0),
        tokenizer_padding_side=cfg.dataset.get("tokenizer_padding_side", "left"),
        tokenizer_max_length=cfg.dataset.get("tokenizer_max_length", None),
        prompt_style=cfg.dataset.get("prompt_style", "plain"),
        state_prompt_bins=cfg.dataset.get("state_prompt_bins", cfg.action_head.get("state_prompt_bins", 256)),
        image_resize_mode=cfg.dataset.get("image_resize_mode", "stretch"),
        empty_cameras=cfg.dataset.get("empty_cameras", 0),
        shuffle=cfg.dataset.get("shuffle", True),
        drop_last=cfg.dataset.get("drop_last", True),
    )

    return loader, dataset


def build_optimizer(model, cfg):
    """Build AdamW optimizer with separate LR groups."""
    groups = model.get_optimizer_groups(
        vlm_lr=cfg.training.vlm_lr,
        action_lr=cfg.training.action_lr,
        weight_decay=cfg.training.weight_decay,
    )

    optim_name = cfg.training.get("optimizer", "adamw")
    if optim_name == "adamw_fp16":
        from torch.optim import AdamW
        # Store optimizer states in fp16 to save ~8GB VRAM
        optimizer = AdamW(
            groups,
            betas=tuple(cfg.training.betas),
            eps=cfg.training.eps,
            fused=True,
        )
        # Convert optimizer state storage to fp16 after first step via hook
        def _cast_optimizer_states_to_fp16(opt):
            for group in opt.param_groups:
                for p in group["params"]:
                    state = opt.state.get(p)
                    if state:
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor) and v.is_floating_point():
                                state[k] = v.half()
        optimizer._cast_to_fp16 = _cast_optimizer_states_to_fp16
        logger.info("Using AdamW with fp16 optimizer states (saves ~8GB VRAM)")
    else:
        optimizer = torch.optim.AdamW(
            groups,
            betas=tuple(cfg.training.betas),
            eps=cfg.training.eps,
        )

    return optimizer


def build_scheduler(optimizer, cfg):
    """Build cosine LR scheduler with linear warmup."""
    warmup_steps = cfg.training.warmup_steps
    max_steps = cfg.training.max_steps
    min_lr = cfg.training.min_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        # Scale so that LR decays to min_lr, not 0
        return max(min_lr / cfg.training.action_lr, cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def load_lora_adapters(peft_model, adapter_dir: Path):
    """Load saved LoRA adapters back into an existing PEFT-wrapped model."""
    if not adapter_dir.exists():
        return

    try:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as safe_load_file

        adapter_file = adapter_dir / "adapter_model.safetensors"
        if not adapter_file.exists():
            logger.info("No PEFT adapter weights found in %s; skipping LoRA load", adapter_dir)
            return

        state_dict = safe_load_file(str(adapter_file), device="cpu")
        set_peft_model_state_dict(peft_model, state_dict, adapter_name="default")
        logger.info(f"Loaded LoRA adapters from {adapter_dir}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load LoRA adapters from {adapter_dir}: {exc}") from exc


def load_checkpoint(model, checkpoint_dir: str, accelerator) -> tuple[int, bool]:
    """Load a checkpoint directory saved by save_checkpoint()."""
    ckpt_dir = Path(checkpoint_dir)
    action_path = ckpt_dir / "action_head.pt"
    if not action_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {action_path}")

    logger.info(f"Loading checkpoint from {ckpt_dir}")
    payload = torch.load(action_path, map_location="cpu", weights_only=False)
    unwrapped = accelerator.unwrap_model(model)
    state_dir = ckpt_dir / "accelerator_state"

    if state_dir.exists():
        logger.info(f"Loading accelerator state from {state_dir}")
        accelerator.load_state(str(state_dir))
        if "action_q01" in payload and "action_q99" in payload:
            unwrapped.set_norm_stats(
                payload["action_q01"],
                payload["action_q99"],
                payload.get("local_action_q01", payload["action_q01"]),
                payload.get("local_action_q99", payload["action_q99"]),
                payload.get("state_q01", None),
                payload.get("state_q99", None),
            )
        step = int(payload.get("step", 0))
        logger.info(f"Checkpoint loaded successfully at step {step}")
        return step, True

    normalized_action_head = normalize_action_head_state_dict(payload["action_head"])
    missing, unexpected = unwrapped.action_head.load_state_dict(
        normalized_action_head, strict=False
    )
    missing, unexpected = filter_optional_action_head_load_issues(missing, unexpected)
    if missing or unexpected:
        raise RuntimeError(
            f"Action head checkpoint mismatch while loading {ckpt_dir}: "
            f"missing={missing} unexpected={unexpected}"
        )

    if "action_q01" in payload and "action_q99" in payload:
        unwrapped.set_norm_stats(
            payload["action_q01"],
            payload["action_q99"],
            payload.get("local_action_q01", payload["action_q01"]),
            payload.get("local_action_q99", payload["action_q99"]),
            payload.get("state_q01", None),
            payload.get("state_q99", None),
        )

    lora_dir = ckpt_dir / "lora_adapters"
    if lora_dir.exists() and hasattr(unwrapped.vlm, "model"):
        load_lora_adapters(unwrapped.vlm.model, lora_dir)

    step = int(payload.get("step", 0))
    logger.info(f"Checkpoint loaded successfully at step {step}")
    return step, False


@torch.no_grad()
def evaluate(model, dataloader, accelerator, max_batches: int = 50):
    """Evaluate action prediction MSE on a subset of the data."""
    model.eval()
    total_mse = 0.0
    count = 0

    unwrapped = accelerator.unwrap_model(model)

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        actions = batch["actions"].to(accelerator.device)
        state = batch["state"].to(accelerator.device)
        history = batch.get("history", None)
        if history is not None:
            history = history.to(accelerator.device)

        if unwrapped.action_head_type in {"layerwise_fm_v3", "layerwise_fm_v4"} and "images" in batch and "instructions" in batch:
            pred_actions = unwrapped.predict_action(
                batch["images"],
                batch["instructions"],
                state=state,
                history=history,
            )
            pred_actions = torch.from_numpy(pred_actions).to(actions.device)
        elif unwrapped.action_head_type in {"pi05_qwen", "pi05_gemma_bridge"} and "images" in batch and "instructions" in batch:
            pred_actions = unwrapped.predict_action(
                batch["images"],
                batch["instructions"],
                state=state,
                history=history,
            )
            pred_actions = torch.from_numpy(pred_actions).to(actions.device)
        # Forward through model to get hidden states, then predict
        elif "vlm_inputs" in batch:
            vlm_inputs = {k: v.to(accelerator.device) for k, v in batch["vlm_inputs"].items()}
            vlm_outputs = unwrapped.vlm(output_hidden_states=True, **vlm_inputs)
            attention_mask = vlm_inputs.get("attention_mask", None)
            action_head_dtype = unwrapped._action_head_dtype()
            if unwrapped.action_head_type == "pi05_qwen":
                prefix_inputs = unwrapped.vlm.build_prefix_from_vlm_inputs(vlm_inputs)
                norm_state = unwrapped.normalize_states(state).to(action_head_dtype)
                norm_history = (
                    unwrapped.normalize_history(history).to(action_head_dtype)
                    if history is not None
                    else None
                )
                norm_actions = unwrapped.action_head.predict_action(
                    language_model=unwrapped._language_model(),
                    prefix_embeds=prefix_inputs["prefix_embeds"],
                    prefix_attention_mask=prefix_inputs["attention_mask"],
                    prefix_position_ids=prefix_inputs["position_ids"],
                    state=norm_state,
                    history=norm_history,
                )
            elif unwrapped.action_head_type == "pi05_gemma_bridge":
                prompts = unwrapped._build_pi05_prompts(batch["instructions"], state)
                prefix_ctx = unwrapped.backbone.encode_prefix(
                    images=batch.get("images"),
                    prompts=prompts,
                    vlm_inputs=vlm_inputs,
                )
                prefix_memory = unwrapped.backbone.get_prefix_memory(
                    prefix_ctx,
                    tap_strategy=unwrapped.action_config.tap_strategy,
                )
                norm_state = (
                    unwrapped.normalize_states(state).to(action_head_dtype)
                    if state is not None
                    else None
                )
                norm_actions = unwrapped.action_head.predict_action(
                    prefix_memory=prefix_memory["memory"],
                    prefix_attention_mask=prefix_memory["memory_mask"],
                    tap_ids=prefix_memory.get("tap_ids"),
                    token_type_ids=prefix_memory.get("token_type_ids"),
                    state=norm_state,
                )
            else:
                hidden_states, layerwise_hidden_states = unwrapped._collect_vlm_hidden_states(vlm_outputs)
                text_embedding = unwrapped.vlm.extract_text_pooled(hidden_states, vlm_inputs)
                state = state.to(action_head_dtype)
                if history is not None:
                    history = history.to(action_head_dtype)
                text_embedding = text_embedding.to(action_head_dtype)

                if unwrapped.action_head_type == "layerwise_fm":
                    norm_actions = unwrapped.action_head.predict_action(
                        vlm_hidden_states_list=[hs.to(action_head_dtype) for hs in layerwise_hidden_states],
                        state=state,
                        attention_mask=attention_mask,
                    )
                elif unwrapped.action_head_type == "layerwise_fm_v3":
                    norm_actions = unwrapped.action_head.predict_action(
                        vlm_hidden_states_list=[hs.to(action_head_dtype) for hs in layerwise_hidden_states],
                        state=state,
                        history=history,
                        text_embedding=text_embedding,
                        attention_mask=attention_mask,
                    )
                else:
                    norm_actions = unwrapped.action_head.predict_action(
                        vlm_hidden_states=hidden_states.to(action_head_dtype),
                        state=state,
                        attention_mask=attention_mask,
                    )
            if (
                unwrapped.action_head_type == "pi05_qwen"
                and hasattr(unwrapped.action_head, "_uses_binary_gripper")
                and unwrapped.action_head._uses_binary_gripper()
            ):
                motion_norm = norm_actions[..., :6]
                gripper = norm_actions[..., 6:7]
                low = unwrapped.action_q01.to(device=motion_norm.device, dtype=motion_norm.dtype)
                high = unwrapped.action_q99.to(device=motion_norm.device, dtype=motion_norm.dtype)
                motion_raw = (motion_norm + 1.0) / 2.0 * (high[:6] - low[:6]) + low[:6]
                pred_actions = torch.cat([motion_raw, gripper], dim=-1)
            else:
                pred_actions = unwrapped.unnormalize_actions(norm_actions)
        else:
            pred_actions = unwrapped.predict_action(
                batch["images"], batch["instructions"], state, history=history
            )
            pred_actions = torch.from_numpy(pred_actions).to(actions.device)

        gt_actions = unwrapped.unnormalize_actions(
            unwrapped.normalize_actions(actions)
        )

        mse = ((pred_actions - gt_actions) ** 2).mean().item()
        total_mse += mse
        count += 1

    model.train()
    return total_mse / max(count, 1)


def save_checkpoint(model, optimizer, scheduler, step, cfg, accelerator, eval_metrics=None):
    """Save model checkpoint."""
    output_dir = Path(cfg.training.output_dir) / f"checkpoint-{step}"
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    state_dir = output_dir / "accelerator_state"
    save_optimizer_state = bool(cfg.training.get("save_optimizer_state", True))
    if save_optimizer_state:
        accelerator.save_state(str(state_dir), safe_serialization=False)
        logger.info(
            "Saved full accelerator state to %s (includes optimizer/scheduler/random state)",
            state_dir,
        )
    else:
        logger.warning(
            "Skipping accelerator state save for %s; resume will restore weights but not optimizer state.",
            output_dir,
        )

    unwrapped = accelerator.unwrap_model(model)

    # Save action head
    torch.save(
        {
            "action_head": normalize_action_head_state_dict(unwrapped.action_head.state_dict()),
            "action_q01": unwrapped.action_q01,
            "action_q99": unwrapped.action_q99,
            "local_action_q01": unwrapped.local_action_q01,
            "local_action_q99": unwrapped.local_action_q99,
            "state_q01": getattr(unwrapped, "state_q01", None),
            "state_q99": getattr(unwrapped, "state_q99", None),
            "step": step,
        },
        output_dir / "action_head.pt",
    )

    # Save LoRA adapters if using peft
    if hasattr(unwrapped.vlm.model, "peft_config") and hasattr(unwrapped.vlm.model, "save_pretrained"):
        unwrapped.vlm.model.save_pretrained(str(output_dir / "lora_adapters"))

    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")
    policy_pre, policy_post = build_policy_contract(cfg)
    (output_dir / "policy_preprocessor.json").write_text(
        json.dumps(policy_pre, indent=2), encoding="utf-8"
    )
    (output_dir / "policy_postprocessor.json").write_text(
        json.dumps(policy_post, indent=2), encoding="utf-8"
    )
    if eval_metrics is not None:
        (output_dir / "eval_metrics.json").write_text(
            json.dumps(eval_metrics, indent=2), encoding="utf-8"
        )

    print(f"Checkpoint saved: {output_dir}", flush=True)


def main():
    cfg = parse_args()
    tcfg = cfg.training

    # Enable TF32 for massive speedup on Ampere+ GPUs (RTX 5090)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        mixed_precision=tcfg.mixed_precision,
        log_with="wandb" if tcfg.get("wandb_project") else None,
    )

    # Set seed
    set_seed(tcfg.seed)

    print("=" * 60, flush=True)
    print("Qwen VLA Training — LIBERO", flush=True)
    print("=" * 60, flush=True)
    print(f"Architecture: {cfg.model.get('architecture', 'qwen35_vla')}", flush=True)
    print(f"Backbone: {cfg.model.vlm_model_id}", flush=True)
    print(
        f"Action head: {cfg.action_head.get('head_type', 'single_fm')} | "
        f"Attention: {cfg.model.get('attn_implementation', 'flash_attention_2')}",
        flush=True,
    )
    print(
        f"Vision trainable: {not cfg.model.freeze_vision_encoder} | "
        f"Full VLM trainable: {not cfg.model.freeze_vlm}",
        flush=True,
    )
    eff_batch = cfg.dataset.batch_size * tcfg.gradient_accumulation_steps
    vlm_enabled = float(cfg.model.get("vlm_loss_weight", 0.0)) > 0
    print(f"Batch size: {cfg.dataset.batch_size} | Grad accum: {tcfg.gradient_accumulation_steps} | Effective batch: {eff_batch}", flush=True)
    print(
        f"Dataset workers: {cfg.dataset.num_workers} | Prefetch: {cfg.dataset.get('prefetch_factor', 'n/a')} | "
        f"Action horizon: {cfg.action_head.action_horizon}",
        flush=True,
    )
    print(
        f"Total steps: {tcfg.max_steps} | Save every: {tcfg.save_every} | Eval every: {tcfg.eval_every} | Log every: {tcfg.log_every}",
        flush=True,
    )
    print(
        f"LoRA: {cfg.get('lora', {}).get('enabled', False)} | "
        f"Grad ckpt: {tcfg.get('gradient_checkpointing', False)} | "
        f"VLM loss: {'enabled' if vlm_enabled else 'disabled'} | "
        f"VLM loss weight: {float(cfg.model.get('vlm_loss_weight', 0.0)):.3f}",
        flush=True,
    )
    print(
        f"Output dir: {tcfg.output_dir} | Save optimizer state: {bool(tcfg.get('save_optimizer_state', True))}",
        flush=True,
    )
    if tcfg.get("resume_from_checkpoint"):
        print(f"Resume checkpoint: {tcfg.resume_from_checkpoint}", flush=True)
    if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f} GB", flush=True)

    # Build components
    print("Building model...", flush=True)
    model = build_model(cfg)

    print("Building dataloader...", flush=True)
    dataloader, dataset = build_dataloader(cfg)
    try:
        print(f"Dataset windows: {len(dataset)}", flush=True)
    except TypeError:
        pass

    # Set normalization stats from dataset
    model.set_norm_stats(
        dataset.norm_stats["q01"].to(accelerator.device),
        dataset.norm_stats["q99"].to(accelerator.device),
        dataset.norm_stats.get("local_q01", dataset.norm_stats["q01"]).to(accelerator.device),
        dataset.norm_stats.get("local_q99", dataset.norm_stats["q99"]).to(accelerator.device),
        dataset.norm_stats.get("state_q01", None).to(accelerator.device)
        if dataset.norm_stats.get("state_q01", None) is not None
        else None,
        dataset.norm_stats.get("state_q99", None).to(accelerator.device)
        if dataset.norm_stats.get("state_q99", None) is not None
        else None,
    )

    logger.info("Building optimizer and scheduler...")
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Prepare with accelerator (handles DeepSpeed wrapping)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Optional resume: if accelerator_state exists we restore the full training
    # state (model/optimizer/scheduler/sampler/random). Otherwise we fall back
    # to weights-only loading from action_head.pt + LoRA adapters.
    resume_from = tcfg.get("resume_from_checkpoint", None)
    global_step = 0
    if resume_from:
        global_step, restored_full_state = load_checkpoint(model, resume_from, accelerator)
        if global_step > 0 and not restored_full_state:
            logger.info(f"Advancing LR scheduler to resumed step {global_step}")
            for _ in range(global_step):
                scheduler.step()
        accelerator.wait_for_everyone()
        print(f"Resumed from checkpoint: {resume_from} (step {global_step})", flush=True)

    # Init W&B
    if accelerator.is_main_process and tcfg.get("wandb_project"):
        accelerator.init_trackers(
            project_name=tcfg.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={
                "wandb": {
                    "name": tcfg.get("wandb_run_name"),
                    "dir": tcfg.output_dir,
                }
            },
        )

    # Training loop
    print(f"Starting training for {tcfg.max_steps} steps...", flush=True)
    running_loss = 0.0
    running_action_loss = 0.0
    running_vlm_loss = 0.0
    running_gripper_loss = 0.0
    running_consistency_loss = 0.0
    running_motion_loss = 0.0
    running_phase_loss = 0.0
    running_immediate_loss = 0.0
    running_weight_motion = 0.0
    running_weight_gripper = 0.0
    running_weight_phase = 0.0
    running_weight_immediate = 0.0
    running_raw_qwen_token_norm = 0.0
    running_bridge_token_norm = 0.0
    running_query_token_norm = 0.0
    running_bridge_memory_norm = 0.0
    running_expert_token_norm = 0.0
    running_cross_attn_entropy = 0.0
    running_memory_norm_ratio = 0.0
    running_tap_weight_entropy = 0.0
    running_scalar_mix_gamma = 0.0
    running_mean_norm_saturation_fraction = 0.0
    start_time = time.time()
    last_saved_step = None
    last_eval_metrics = None

    model.train()
    data_iter = iter(dataloader)
    micro_step = 0

    while global_step < tcfg.max_steps:
        # Get next batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move tensors to device
        actions = batch["actions"].to(accelerator.device)
        state = batch["state"].to(accelerator.device)
        history = batch.get("history", None)
        if history is not None:
            history = history.to(accelerator.device)

        with accelerator.accumulate(model):
            # Forward — use pre-tokenized vlm_inputs if available (from worker preprocessing)
            if "vlm_inputs" in batch:
                vlm_inputs = {k: v.to(accelerator.device) for k, v in batch["vlm_inputs"].items()}
                outputs = model(
                    images=batch.get("images"),
                    instructions=batch.get("instructions"),
                    vlm_inputs=vlm_inputs,
                    actions=actions,
                    state=state,
                    history=history,
                )
            else:
                outputs = model(
                    images=batch["images"],
                    instructions=batch["instructions"],
                    actions=actions,
                    state=state,
                    history=history,
                )

            loss = outputs["total_loss"]

            # Backward
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), tcfg.max_grad_norm
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        micro_step += 1
        # Accumulate metrics every micro-batch
        running_loss += loss.detach().item()
        running_action_loss += outputs["action_loss"].detach().item()
        if "vlm_loss" in outputs:
            running_vlm_loss += outputs["vlm_loss"].detach().item()
        if "motion_loss" in outputs:
            running_motion_loss += outputs["motion_loss"].detach().item()
        if "gripper_loss" in outputs:
            running_gripper_loss += outputs["gripper_loss"].detach().item()
        if "consistency_loss" in outputs:
            running_consistency_loss += outputs["consistency_loss"].detach().item()
        if "phase_loss" in outputs:
            running_phase_loss += outputs["phase_loss"].detach().item()
        if "immediate_loss" in outputs:
            running_immediate_loss += outputs["immediate_loss"].detach().item()
        if "loss_weight_motion" in outputs:
            running_weight_motion += outputs["loss_weight_motion"].detach().float().mean().item()
        if "loss_weight_gripper" in outputs:
            running_weight_gripper += outputs["loss_weight_gripper"].detach().float().mean().item()
        if "loss_weight_phase" in outputs:
            running_weight_phase += outputs["loss_weight_phase"].detach().float().mean().item()
        if "loss_weight_immediate" in outputs:
            running_weight_immediate += outputs["loss_weight_immediate"].detach().float().mean().item()
        if "raw_qwen_token_norm" in outputs:
            running_raw_qwen_token_norm += outputs["raw_qwen_token_norm"].detach().float().mean().item()
        if "bridge_token_norm" in outputs:
            running_bridge_token_norm += outputs["bridge_token_norm"].detach().float().mean().item()
        if "query_token_norm" in outputs:
            running_query_token_norm += outputs["query_token_norm"].detach().float().mean().item()
        if "bridge_memory_norm" in outputs:
            running_bridge_memory_norm += outputs["bridge_memory_norm"].detach().float().mean().item()
        if "expert_token_norm" in outputs:
            running_expert_token_norm += outputs["expert_token_norm"].detach().float().mean().item()
        if "cross_attn_entropy" in outputs:
            running_cross_attn_entropy += outputs["cross_attn_entropy"].detach().float().mean().item()
        if "memory_norm_ratio" in outputs:
            running_memory_norm_ratio += outputs["memory_norm_ratio"].detach().float().mean().item()
        if "tap_weight_entropy" in outputs:
            running_tap_weight_entropy += outputs["tap_weight_entropy"].detach().float().mean().item()
        if "scalar_mix_gamma" in outputs:
            running_scalar_mix_gamma += outputs["scalar_mix_gamma"].detach().float().mean().item()
        if "mean_norm_saturation_fraction" in outputs:
            running_mean_norm_saturation_fraction += (
                outputs["mean_norm_saturation_fraction"].detach().float().mean().item()
            )

        if micro_step % int(tcfg.gradient_accumulation_steps) == 0:
            global_step += 1
            if global_step == 1:
                print(f"First step completed. micro_step={micro_step}", flush=True)

            # Logging
            if global_step % int(tcfg.log_every) == 0:
                log_micros = tcfg.log_every * tcfg.gradient_accumulation_steps
                avg_loss = running_loss / log_micros
                avg_action = running_action_loss / log_micros
                avg_vlm = running_vlm_loss / log_micros
                avg_motion = running_motion_loss / log_micros
                avg_gripper = running_gripper_loss / log_micros
                avg_consistency = running_consistency_loss / log_micros
                avg_phase = running_phase_loss / log_micros
                avg_immediate = running_immediate_loss / log_micros
                avg_weight_motion = running_weight_motion / log_micros
                avg_weight_gripper = running_weight_gripper / log_micros
                avg_weight_phase = running_weight_phase / log_micros
                avg_weight_immediate = running_weight_immediate / log_micros
                avg_raw_qwen_token_norm = running_raw_qwen_token_norm / log_micros
                avg_bridge_token_norm = running_bridge_token_norm / log_micros
                avg_query_token_norm = running_query_token_norm / log_micros
                avg_bridge_memory_norm = running_bridge_memory_norm / log_micros
                avg_expert_token_norm = running_expert_token_norm / log_micros
                avg_cross_attn_entropy = running_cross_attn_entropy / log_micros
                avg_memory_norm_ratio = running_memory_norm_ratio / log_micros
                avg_tap_weight_entropy = running_tap_weight_entropy / log_micros
                avg_scalar_mix_gamma = running_scalar_mix_gamma / log_micros
                avg_mean_norm_saturation_fraction = (
                    running_mean_norm_saturation_fraction / log_micros
                )
                elapsed = time.time() - start_time
                sec_per_step = elapsed / global_step
                steps_per_sec = 1.0 / sec_per_step if sec_per_step > 0 else 0.0
                samples_per_sec = eff_batch / sec_per_step if sec_per_step > 0 else 0.0
                remaining = (tcfg.max_steps - global_step) * sec_per_step
                eta_min = remaining / 60

                lr = scheduler.get_last_lr()[0]
                timestamp = time.strftime("%H:%M:%S")
                if torch.cuda.is_available():
                    mem_alloc_gb = torch.cuda.memory_allocated() / 1e9
                    mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
                    max_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
                    gpu_stats = (
                        f" | gpu_alloc={mem_alloc_gb:.1f}GB "
                        f"gpu_reserved={mem_reserved_gb:.1f}GB "
                        f"gpu_peak_reserved={max_reserved_gb:.1f}GB"
                    )
                else:
                    gpu_stats = ""
                vlm_status = f"vlm={avg_vlm:.4f}" if vlm_enabled else "vlm=disabled"
                aux_status = ""
                if cfg.action_head.get("head_type", "single_fm") == "layerwise_fm_v3":
                    aux_status = (
                        f" motion={avg_motion:.4f} "
                        f"gripper={avg_gripper:.4f} "
                        f"cons={avg_consistency:.4f} |"
                    )
                elif cfg.action_head.get("head_type", "single_fm") == "layerwise_fm_v4":
                    aux_status = (
                        f" motion={avg_motion:.4f} "
                        f"gripper={avg_gripper:.4f} "
                        f"phase={avg_phase:.4f} "
                        f"imm={avg_immediate:.4f} "
                        f"w=({avg_weight_motion:.2f},{avg_weight_gripper:.2f},{avg_weight_phase:.2f},{avg_weight_immediate:.2f}) |"
                    )
                elif cfg.action_head.get("head_type", "single_fm") == "pi05_qwen" and bool(
                    cfg.action_head.get("use_binary_gripper", False)
                ):
                    aux_status = (
                        f" motion={avg_motion:.4f} "
                        f"gripper={avg_gripper:.4f} "
                        f"phase={avg_phase:.4f} |"
                    )
                elif cfg.action_head.get("head_type", "single_fm") == "pi05_gemma_bridge":
                    _sm_suffix = ""
                    if bool(cfg.action_head.get("bridge_scalar_mix", False)):
                        _sm_suffix = (
                            f" tapH={avg_tap_weight_entropy:.3f} "
                            f"gamma={avg_scalar_mix_gamma:.3f}"
                        )
                    aux_status = (
                        f" motion={avg_motion:.4f} "
                        f"qwen={avg_raw_qwen_token_norm:.3f} "
                        f"bridge={avg_bridge_token_norm:.3f} "
                        f"query={avg_query_token_norm:.3f} "
                        f"expert={avg_expert_token_norm:.3f} "
                        f"entropy={avg_cross_attn_entropy:.3f} "
                        f"ratio={avg_memory_norm_ratio:.3f}"
                        f"{_sm_suffix} "
                        f"sat={avg_mean_norm_saturation_fraction:.3f} |"
                    )
                print(
                    f"[{timestamp}] Step {global_step}/{tcfg.max_steps} | "
                    f"loss={avg_loss:.4f} action={avg_action:.4f} {vlm_status} |"
                    f"{aux_status} "
                    f"lr={lr:.2e} | {sec_per_step:.2f}s/step | "
                    f"{steps_per_sec:.2f} step/s | {samples_per_sec:.1f} sample/s"
                    f"{gpu_stats} | ETA {eta_min:.0f}min",
                    flush=True,
                )

                if tcfg.get("wandb_project") and accelerator.is_main_process:
                    log_payload = {
                        "train/total_loss": avg_loss,
                        "train/action_loss": avg_action,
                        "train/lr": lr,
                        "train/steps_per_sec": steps_per_sec,
                        "train/samples_per_sec": samples_per_sec,
                    }
                    if vlm_enabled:
                        log_payload["train/vlm_loss"] = avg_vlm
                    if cfg.action_head.get("head_type", "single_fm") == "layerwise_fm_v3":
                        log_payload["train/motion_loss"] = avg_motion
                        log_payload["train/gripper_loss"] = avg_gripper
                        log_payload["train/consistency_loss"] = avg_consistency
                    elif cfg.action_head.get("head_type", "single_fm") == "layerwise_fm_v4":
                        log_payload["train/motion_loss"] = avg_motion
                        log_payload["train/gripper_loss"] = avg_gripper
                        log_payload["train/phase_loss"] = avg_phase
                        log_payload["train/immediate_loss"] = avg_immediate
                        log_payload["train/loss_weight_motion"] = avg_weight_motion
                        log_payload["train/loss_weight_gripper"] = avg_weight_gripper
                        log_payload["train/loss_weight_phase"] = avg_weight_phase
                        log_payload["train/loss_weight_immediate"] = avg_weight_immediate
                    elif cfg.action_head.get("head_type", "single_fm") == "pi05_qwen" and bool(
                        cfg.action_head.get("use_binary_gripper", False)
                    ):
                        log_payload["train/motion_loss"] = avg_motion
                        log_payload["train/gripper_loss"] = avg_gripper
                        log_payload["train/phase_loss"] = avg_phase
                    elif cfg.action_head.get("head_type", "single_fm") == "pi05_gemma_bridge":
                        log_payload["train/motion_loss"] = avg_motion
                        log_payload["train/raw_qwen_token_norm"] = avg_raw_qwen_token_norm
                        log_payload["train/bridge_token_norm"] = avg_bridge_token_norm
                        log_payload["train/query_token_norm"] = avg_query_token_norm
                        log_payload["train/bridge_memory_norm"] = avg_bridge_memory_norm
                        log_payload["train/expert_token_norm"] = avg_expert_token_norm
                        log_payload["train/cross_attn_entropy"] = avg_cross_attn_entropy
                        log_payload["train/memory_norm_ratio"] = avg_memory_norm_ratio
                        log_payload["train/mean_norm_saturation_fraction"] = (
                            avg_mean_norm_saturation_fraction
                        )
                        if bool(cfg.action_head.get("bridge_scalar_mix", False)):
                            log_payload["train/tap_weight_entropy"] = avg_tap_weight_entropy
                            log_payload["train/scalar_mix_gamma"] = avg_scalar_mix_gamma
                    if torch.cuda.is_available():
                        log_payload["train/gpu_alloc_gb"] = mem_alloc_gb
                        log_payload["train/gpu_reserved_gb"] = mem_reserved_gb
                        log_payload["train/gpu_peak_reserved_gb"] = max_reserved_gb
                    accelerator.log(log_payload, step=global_step)

                running_loss = 0.0
                running_action_loss = 0.0
                running_vlm_loss = 0.0
                running_motion_loss = 0.0
                running_gripper_loss = 0.0
                running_consistency_loss = 0.0
                running_phase_loss = 0.0
                running_immediate_loss = 0.0
                running_weight_motion = 0.0
                running_weight_gripper = 0.0
                running_weight_phase = 0.0
                running_weight_immediate = 0.0
                running_raw_qwen_token_norm = 0.0
                running_bridge_token_norm = 0.0
                running_query_token_norm = 0.0
                running_bridge_memory_norm = 0.0
                running_expert_token_norm = 0.0
                running_cross_attn_entropy = 0.0
                running_memory_norm_ratio = 0.0
                running_tap_weight_entropy = 0.0
                running_scalar_mix_gamma = 0.0
                running_mean_norm_saturation_fraction = 0.0

            # Evaluation
            if global_step % tcfg.eval_every == 0:
                print(f"Evaluating at step {global_step}...", flush=True)
                eval_mse = evaluate(model, dataloader, accelerator)
                print(f"Eval MSE: {eval_mse:.6f}", flush=True)
                last_eval_metrics = {
                    "action_mse": float(eval_mse),
                    "step": int(global_step),
                }
                if tcfg.get("wandb_project"):
                    accelerator.log(
                        {"eval/action_mse": eval_mse}, step=global_step
                    )

            # Save checkpoint
            if global_step % tcfg.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, cfg, accelerator, last_eval_metrics
                )
                last_saved_step = global_step

    # Final save
    if last_saved_step != global_step:
        save_checkpoint(model, optimizer, scheduler, global_step, cfg, accelerator, last_eval_metrics)

    total_time = time.time() - start_time
    print(f"Training complete! {global_step} steps in {total_time/60:.1f} min", flush=True)
    if tcfg.get("wandb_project"):
        accelerator.end_training()


if __name__ == "__main__":
    main()
