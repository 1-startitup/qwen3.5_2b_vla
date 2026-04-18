"""
VLM Grounding Precision Diagnostic.

Runs targeted eval on multi-object tasks, extracts suffix full-attention maps
from the pi0.5 action expert attending back to the VLM prefix (image tokens),
and generates attention heatmap overlay videos + per-episode grounding reports.

Usage:
    python eval_vlm_grounding.py \
        --checkpoint_dir <CKPT> \
        --output_dir <DIR> \
        --camera_height 448 --camera_width 448
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from types import MethodType
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from eval_libero import (
    Qwen35VLALiberoPolicy,
    build_state,
    get_task_suites,
    make_libero_env,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# ---------------------------------------------------------------------------
# Attention hook: capture suffix-query -> prefix-key full attention
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Capture suffix-to-prefix full-attention maps from the pi0.5 Qwen expert."""

    def __init__(self):
        self.maps: list[torch.Tensor] = []
        self._action_head = None
        self._orig_run_suffix_full_attention_layer = None
        self.full_attention_layer_indices: list[int] = []
        self.capture_calls = 0

    @property
    def tracked_layer_count(self) -> int:
        return len(self.full_attention_layer_indices)

    def register(self, action_head, language_model):
        """Monkeypatch suffix full-attention to expose prefix attention maps."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

        from model.qwen35_pi05_action_head import gated_residual

        action_head._ensure_initialized(language_model)
        self._action_head = action_head
        self.full_attention_layer_indices = [
            idx for idx, layer_type in enumerate(action_head.layer_types) if layer_type == "full_attention"
        ]
        self._orig_run_suffix_full_attention_layer = action_head._run_suffix_full_attention_layer.__func__
        capture = self

        def patched_run_suffix_full_attention_layer(
            this,
            language_model,
            suffix_layer,
            suffix_hidden,
            timestep_cond,
            layer_idx,
            suffix_position_ids,
            prefix_attention_mask,
            prefix_key_states,
            prefix_value_states,
        ):
            residual = suffix_hidden
            suffix_norm, gate = this.input_adarms[layer_idx](suffix_hidden, timestep_cond)
            suffix_pos_emb = language_model.rotary_emb(suffix_norm, suffix_position_ids)
            suffix_q, suffix_k, suffix_v, suffix_attn_gate = this._project_full_attention(
                suffix_layer.self_attn,
                suffix_norm,
                suffix_pos_emb,
            )

            prefix_length = int(prefix_attention_mask.shape[1])
            full_key_states = torch.cat([prefix_key_states.to(suffix_q.dtype), suffix_k], dim=2)
            full_value_states = torch.cat([prefix_value_states.to(suffix_v.dtype), suffix_v], dim=2)
            suffix_mask = this._suffix_full_attention_mask(
                prefix_attention_mask,
                suffix_hidden.shape[1],
                suffix_q.dtype,
            )

            if prefix_length > 0:
                expanded_key_states = repeat_kv(
                    full_key_states,
                    suffix_layer.self_attn.num_key_value_groups,
                )
                attn_logits = (suffix_q @ expanded_key_states.transpose(-2, -1)) * suffix_layer.self_attn.scaling
                attn_logits = attn_logits + suffix_mask.to(attn_logits.dtype)
                prefix_attn = F.softmax(attn_logits.float(), dim=-1)[..., :prefix_length]
                capture.maps.append(prefix_attn.mean(dim=1).mean(dim=1).detach().cpu())
                capture.capture_calls += 1

            suffix_update = this._apply_full_attention(
                suffix_layer.self_attn,
                suffix_q,
                full_key_states,
                full_value_states,
                suffix_mask,
                suffix_attn_gate,
                suffix_hidden.shape[1],
            )
            suffix_hidden = gated_residual(residual, suffix_update.to(residual.dtype), gate)

            residual = suffix_hidden
            suffix_post, gate = this.post_adarms[layer_idx](suffix_hidden, timestep_cond)
            suffix_post = suffix_post.to(this._module_dtype(suffix_layer.mlp, suffix_post.dtype))
            suffix_mlp = suffix_layer.mlp(suffix_post).to(residual.dtype)
            return gated_residual(residual, suffix_mlp, gate)

        action_head._run_suffix_full_attention_layer = MethodType(
            patched_run_suffix_full_attention_layer,
            action_head,
        )
        logger.info(
            "Tracking %d full-attention suffix layers via QwenPI05ExpertHead runtime patch",
            self.tracked_layer_count,
        )

    def get_and_clear(self) -> Optional[torch.Tensor]:
        """Return averaged attention map across all layers/denoising steps, then clear."""
        if not self.maps:
            return None
        # Stack and average across all captured maps
        stacked = torch.stack(self.maps, dim=0)  # (num_captures, B, context_len)
        avg = stacked.mean(dim=0)  # (B, context_len)
        self.maps.clear()
        return avg

    def remove(self):
        if self._action_head is not None and self._orig_run_suffix_full_attention_layer is not None:
            self._action_head._run_suffix_full_attention_layer = MethodType(
                self._orig_run_suffix_full_attention_layer,
                self._action_head,
            )
        self._action_head = None
        self._orig_run_suffix_full_attention_layer = None


# ---------------------------------------------------------------------------
# Attention heatmap overlay
# ---------------------------------------------------------------------------

def extract_image_attention(
    attn_over_prefix: np.ndarray,
    image_mask: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Extract attention over image tokens and reshape to spatial grid."""
    image_attn = attn_over_prefix[image_mask.astype(bool)]
    # May have multiple images (agentview + wrist); take first image's tokens
    n_patches = grid_h * grid_w
    if len(image_attn) >= n_patches:
        image_attn = image_attn[:n_patches]
    else:
        # Pad if needed
        padded = np.zeros(n_patches)
        padded[: len(image_attn)] = image_attn
        image_attn = padded
    return image_attn.reshape(grid_h, grid_w)


def make_heatmap_overlay(
    frame: np.ndarray,
    attn_grid: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay attention heatmap on frame image."""
    h, w = frame.shape[:2]
    # Normalize attention to [0, 1]
    attn_min, attn_max = attn_grid.min(), attn_grid.max()
    if attn_max > attn_min:
        attn_norm = (attn_grid - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attn_grid)

    # Resize to frame size
    attn_img = Image.fromarray((attn_norm * 255).astype(np.uint8))
    attn_img = attn_img.resize((w, h), Image.BILINEAR)
    attn_arr = np.array(attn_img).astype(np.float32) / 255.0

    # Create colormap (blue -> red)
    heatmap = np.zeros((h, w, 3), dtype=np.float32)
    heatmap[..., 0] = attn_arr  # Red channel
    heatmap[..., 2] = 1.0 - attn_arr  # Blue channel

    # Blend
    blended = (1 - alpha) * frame.astype(np.float32) + alpha * heatmap * 255
    return np.clip(blended, 0, 255).astype(np.uint8)


def annotate_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Add text annotation to top of frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, 22)], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.array(img)


def encode_video(frame_dir: Path, output_path: Path, fps: int = 20):
    pattern = str(frame_dir / "frame_%04d.png")
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        return
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "fast",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("ffmpeg failed: %s", exc)


# ---------------------------------------------------------------------------
# Main grounding eval
# ---------------------------------------------------------------------------

def run_grounding_eval(
    policy: Qwen35VLALiberoPolicy,
    suite_name: str,
    suite,
    output_dir: Path,
    task_ids: list[int],
    n_episodes: int = 3,
    max_steps: int = 300,
    camera_height: int = 448,
    camera_width: int = 448,
):
    """Run grounding diagnostic on selected tasks with attention capture."""

    # Set up attention hooks on the action head
    attn_capture = AttentionCapture()
    attn_capture.register(policy.model.action_head, policy.model._language_model())

    image_token_id = None
    for cfg in (
        getattr(getattr(policy.model.vlm, "model", None), "config", None),
        getattr(policy.model.vlm, "config", None),
    ):
        if cfg is None:
            continue
        image_token_id = getattr(cfg, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(cfg, "image_token_index", None)
        if image_token_id is not None:
            break

    if image_token_id is None:
        tokenizer = getattr(getattr(policy.model.vlm, "processor", None), "tokenizer", None)
        if tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if isinstance(token_id, int) and token_id >= 0:
                image_token_id = token_id

    if image_token_id is None:
        raise ValueError("Unable to resolve Qwen image placeholder token id for grounding eval")
    image_token_id = int(image_token_id)
    logger.info("Grounding eval using image placeholder token id %d", image_token_id)

    # Capture prefix/image token metadata while the VLM builds the prefix cache.
    image_info = {}

    orig_build_prefix = policy.model.vlm.build_prefix_from_vlm_inputs

    def hooked_build_prefix(vlm_inputs):
        result = orig_build_prefix(vlm_inputs)
        input_ids = vlm_inputs.get("input_ids", None)
        image_grid_thw = vlm_inputs.get("image_grid_thw", None)
        attention_mask = result.get("attention_mask", None)
        if input_ids is not None and image_grid_thw is not None:
            # Qwen uses one <|image_pad|> token per image feature slot.
            # Match those placeholder ids directly instead of routing through
            # get_placeholder_mask(), which validates the image feature count.
            img_mask = input_ids == image_token_id
            image_info["mask"] = img_mask[0].detach().cpu().numpy()
            image_info["grid_thw"] = image_grid_thw.detach().cpu().numpy()
            image_info["image_tokens"] = int(img_mask[0].sum().item())
        if attention_mask is not None:
            image_info["prefix_tokens"] = int(attention_mask[0].sum().item())
        elif "prefix_embeds" in result:
            image_info["prefix_tokens"] = int(result["prefix_embeds"].shape[1])
        return result

    policy.model.vlm.build_prefix_from_vlm_inputs = hooked_build_prefix

    results = {
        "metadata": {
            "suite": suite_name,
            "grounding_backend": "qwen_pi05_suffix_full_attention",
            "image_token_id": image_token_id,
            "full_attention_suffix_layers": len(attn_capture.full_attention_layer_indices),
            "tracked_layer_indices": attn_capture.full_attention_layer_indices,
        },
        "tasks": {},
    }

    for task_id in task_ids:
        task = suite.get_task(task_id)
        task_name = task.name
        instruction = task.language
        init_states = suite.get_task_init_states(task_id)

        task_results = {"task_name": task_name, "instruction": instruction, "episodes": []}
        logger.info("=" * 60)
        logger.info("Grounding eval: Task %02d [%s]", task_id, task_name[:50])

        for ep in range(min(n_episodes, len(init_states))):
            policy.reset()
            env = make_libero_env(task, camera_height=camera_height, camera_width=camera_width)
            env.reset()
            env.set_init_state(init_states[ep])
            obs = env.env._get_observations()

            ep_dir = output_dir / f"task_{task_id:02d}_{task_name[:40]}" / f"episode_{ep:02d}"
            overlay_dir = ep_dir / "attn_overlay_frames"
            raw_dir = ep_dir / "agentview_frames"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)

            success = False
            eef_positions = []
            attn_stats = []
            attn_unavailable_reason = None
            logged_token_summary = False
            episode_prefix_tokens = None
            episode_image_tokens = None

            # Save initial frame
            Image.fromarray(obs["agentview_image"]).save(raw_dir / "frame_0000.png")

            for step in range(max_steps):
                obs_dict = {
                    "image": obs["agentview_image"],
                    "wrist_image": obs.get("robot0_eye_in_hand_image", obs["agentview_image"]),
                    "state": build_state(obs),
                }

                # Clear attention maps before action
                attn_capture.maps.clear()
                image_info.clear()

                action = policy.get_action(obs_dict, instruction)

                # Extract attention
                attn_map = attn_capture.get_and_clear()
                frame = obs["agentview_image"]
                if "prefix_tokens" in image_info:
                    episode_prefix_tokens = int(image_info["prefix_tokens"])
                if "image_tokens" in image_info:
                    episode_image_tokens = int(image_info["image_tokens"])
                if not logged_token_summary and episode_prefix_tokens is not None and episode_image_tokens is not None:
                    logger.info(
                        "Task %02d ep %d token summary | prefix_tokens=%d | image_tokens=%d | full_attention_suffix_layers=%d | tracked_layers=%d",
                        task_id,
                        ep,
                        episode_prefix_tokens,
                        episode_image_tokens,
                        len(attn_capture.full_attention_layer_indices),
                        attn_capture.tracked_layer_count,
                    )
                    logged_token_summary = True

                if attn_map is not None and "mask" in image_info and "grid_thw" in image_info:
                    mask = image_info["mask"]
                    grid = image_info["grid_thw"]
                    # grid_thw shape: (num_images, 3) -> (T, H, W)
                    grid_h, grid_w = int(grid[0, 1]), int(grid[0, 2])
                    attn_np = attn_map[0].numpy()

                    if len(attn_np) == len(mask):
                        attn_grid = extract_image_attention(attn_np, mask, grid_h, grid_w)
                        overlay = make_heatmap_overlay(frame, attn_grid)

                        # Get EEF position
                        eef = obs.get("robot0_eef_pos", np.zeros(3))
                        eef_positions.append(eef.tolist() if hasattr(eef, 'tolist') else list(eef))

                        # Track attention concentration
                        peak = float(attn_grid.max())
                        entropy = float(-np.sum(attn_grid / (attn_grid.sum() + 1e-8) *
                                                np.log(attn_grid / (attn_grid.sum() + 1e-8) + 1e-8)))
                        attn_stats.append({"step": step, "peak": peak, "entropy": entropy})

                        # Annotate
                        text = f"step={step} peak={peak:.3f} ent={entropy:.2f}"
                        overlay = annotate_frame(overlay, text)
                    else:
                        attn_unavailable_reason = (
                            f"prefix_length_mismatch:{len(attn_np)}_vs_mask:{len(mask)}"
                        )
                        overlay = frame
                else:
                    if attn_map is None:
                        attn_unavailable_reason = (
                            "no_attention_captured"
                            if attn_capture.tracked_layer_count > 0
                            else "no_full_attention_suffix_layers"
                        )
                    elif "mask" not in image_info or "grid_thw" not in image_info:
                        attn_unavailable_reason = "missing_image_token_metadata"
                    overlay = frame

                Image.fromarray(overlay).save(overlay_dir / f"frame_{step:04d}.png")

                obs, reward, done, info = env.step(action)
                Image.fromarray(obs["agentview_image"]).save(raw_dir / f"frame_{step + 1:04d}.png")

                if done or info.get("success", False):
                    success = bool(done or info.get("success", False))
                    break

            env.close()

            # Encode videos
            result_tag = "success" if success else "fail"
            encode_video(overlay_dir, ep_dir / f"attn_overlay_{result_tag}.mp4")
            encode_video(raw_dir, ep_dir / f"agentview_{result_tag}.mp4")

            # Grounding report
            ep_report = {
                "success": success,
                "num_steps": step + 1,
                "attn_stats": attn_stats,
                "eef_positions": eef_positions,
                "attention_available": bool(attn_stats),
                "attention_unavailable_reason": None if attn_stats else attn_unavailable_reason,
                "prefix_tokens": episode_prefix_tokens,
                "image_tokens": episode_image_tokens,
            }
            if attn_stats:
                peaks = [s["peak"] for s in attn_stats]
                entropies = [s["entropy"] for s in attn_stats]
                ep_report["avg_peak"] = float(np.mean(peaks))
                ep_report["avg_entropy"] = float(np.mean(entropies))
                ep_report["min_peak"] = float(np.min(peaks))
                ep_report["max_entropy"] = float(np.max(entropies))
            task_results["episodes"].append(ep_report)

            extra = ""
            if ep_report["attention_unavailable_reason"] is not None:
                extra = f" | attention={ep_report['attention_unavailable_reason']}"
            logger.info(
                "Task %02d ep %d: %s (%d steps) | avg_peak=%.3f avg_ent=%.2f%s",
                task_id,
                ep,
                "SUCCESS" if success else "FAIL",
                step + 1,
                ep_report.get("avg_peak", 0),
                ep_report.get("avg_entropy", 0),
                extra,
            )

        results["tasks"][f"task_{task_id:02d}"] = task_results

    # Restore original method
    policy.model.vlm.build_prefix_from_vlm_inputs = orig_build_prefix
    attn_capture.remove()

    # Save summary
    summary_path = output_dir / "grounding_report.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Grounding report saved to %s", summary_path)

    # Print summary table
    logger.info("=" * 70)
    logger.info("GROUNDING DIAGNOSTIC SUMMARY")
    logger.info("%-40s %6s %8s %8s", "Task", "SR", "AvgPeak", "AvgEnt")
    logger.info("-" * 70)
    for tid, tres in results["tasks"].items():
        eps = tres["episodes"]
        sr = sum(1 for e in eps if e["success"]) / len(eps) * 100
        avg_p = np.mean([e.get("avg_peak", 0) for e in eps])
        avg_e = np.mean([e.get("avg_entropy", 0) for e in eps])
        logger.info("%-40s %5.1f%% %8.3f %8.2f", tres["task_name"][:40], sr, avg_p, avg_e)
    logger.info("=" * 70)
    logger.info("Low peak + high entropy = diffuse attention = poor grounding")


def main():
    parser = argparse.ArgumentParser(description="VLM Grounding Precision Diagnostic")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial"])
    parser.add_argument("--task_ids", type=int, nargs="*", default=None,
                        help="Task IDs to evaluate. Default: all 10 tasks")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--camera_height", type=int, default=448)
    parser.add_argument("--camera_width", type=int, default=448)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--deterministic_seed", type=int, default=0)
    args = parser.parse_args()

    policy = Qwen35VLALiberoPolicy(
        args.checkpoint_dir,
        device=args.device,
        chunk_size=args.chunk_size,
        num_inference_steps=args.num_inference_steps,
        deterministic_seed=args.deterministic_seed,
    )

    suites = get_task_suites(args.suites)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for suite_name, suite in suites.items():
        task_ids = args.task_ids if args.task_ids is not None else list(range(suite.n_tasks))
        run_grounding_eval(
            policy=policy,
            suite_name=suite_name,
            suite=suite,
            output_dir=output_dir / suite_name,
            task_ids=task_ids,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
        )


if __name__ == "__main__":
    main()
