"""
LIBERO Dataset for VLA Training via LeRobot 0.5.x.
Wraps LeRobotDataset to provide action chunks + images for the Qwen3.5 VLA stack.
"""

import warnings
warnings.filterwarnings("ignore", message="Kwargs passed to")

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple, List, Any
import numpy as np
from PIL import Image
import logging

from model.se3_utils import world_to_local_motion

logger = logging.getLogger(__name__)

LIBERO_ACTION_DIM = 7
LIBERO_STATE_DIM = 8
LIBERO_HISTORY_FEATURE_DIM = LIBERO_STATE_DIM + LIBERO_ACTION_DIM + 1


class LiberoVLADataset(Dataset):
    """
    Wraps a LeRobot dataset for VLA training.
    Each sample: pre-tokenized VLM inputs + action_chunk + state.
    """

    def __init__(
        self,
        dataset_name: str = "lerobot/libero_spatial_image",
        data_root: Optional[str] = None,
        action_horizon: int = 8,
        image_size: Tuple[int, int] = (224, 224),
        action_type: str = "delta_qpos",
        split: str = "train",
        vlm_model_id: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        subset_seed: int = 42,
        norm_stats_sample_size: int = 20000,
        history_len: int = 0,
        tokenizer_padding_side: str = "left",
        tokenizer_max_length: Optional[int] = None,
        prompt_style: str = "plain",
        state_prompt_bins: int = 256,
        image_resize_mode: str = "stretch",
        empty_cameras: int = 0,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.vlm_model_id = vlm_model_id
        self.history_len = max(0, int(history_len))
        self.history_feature_dim = LIBERO_HISTORY_FEATURE_DIM
        self.max_train_samples = (
            int(max_train_samples) if max_train_samples is not None and int(max_train_samples) > 0 else None
        )
        self.subset_seed = int(subset_seed)
        self.norm_stats_sample_size = max(1, int(norm_stats_sample_size))
        self.tokenizer_padding_side = str(tokenizer_padding_side)
        self.tokenizer_max_length = (
            int(tokenizer_max_length) if tokenizer_max_length is not None else None
        )
        self.prompt_style = str(prompt_style)
        self.state_prompt_bins = int(state_prompt_bins)
        self.image_resize_mode = str(image_resize_mode)
        self.empty_cameras = max(0, int(empty_cameras))
        self._processor = None  # Lazy init per worker process

        # Load via LeRobot 0.5.x API
        self.dataset = None
        self.valid_indices = []
        self.valid_episode_starts = []

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            kwargs = {"repo_id": dataset_name}
            if data_root:
                kwargs["root"] = data_root
            self.dataset = LeRobotDataset(**kwargs)
            logger.info(f"Loaded {dataset_name}: {len(self.dataset)} frames")

            # Discover image keys and action key from first sample
            sample = self.dataset[0]
            self.image_keys = [k for k in sample.keys() if "image" in k.lower()]
            self.action_key = "action" if "action" in sample else None
            self.state_key = "observation.state" if "observation.state" in sample else None
            # Prefer string-valued task keys (skip task_index which is a tensor)
            self.task_key = None
            for k in sample.keys():
                if ("task" in k.lower() or "instruction" in k.lower() or "language" in k.lower()):
                    if isinstance(sample[k], str):
                        self.task_key = k
                        break

            logger.info(f"Image keys: {self.image_keys}")
            logger.info(f"Action key: {self.action_key}, State key: {self.state_key}, Task key: {self.task_key}")

            # Build valid start indices (don't cross episode boundaries)
            self._build_valid_indices()
            self._apply_subset()

        except Exception as e:
            logger.warning(f"Failed to load LeRobot dataset: {e}. Using placeholder data.")
            self.dataset = None
            self.valid_indices = list(range(1000))
            self.valid_episode_starts = [0 for _ in self.valid_indices]

        self.norm_stats = self._compute_norm_stats()
        logger.info(f"LiberoVLADataset | {dataset_name} | {len(self)} valid samples")

    @property
    def processor(self):
        """Lazy-init processor per worker process (avoids pickling issues)."""
        if self._processor is None and self.vlm_model_id:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.vlm_model_id)
            self._processor.tokenizer.padding_side = self.tokenizer_padding_side
        return self._processor

    def _build_valid_indices(self):
        """Build list of valid start indices for action chunks using episode metadata."""
        n = len(self.dataset)
        self.valid_indices = []
        self.valid_episode_starts = []

        # Use episode metadata (instant, no per-frame iteration)
        if hasattr(self.dataset, 'meta') and hasattr(self.dataset.meta, 'episodes') and self.dataset.meta.episodes is not None:
            for ep in self.dataset.meta.episodes:
                ep_start = ep["dataset_from_index"]
                ep_end = ep["dataset_to_index"]
                for j in range(ep_start, max(ep_start, ep_end - self.action_horizon + 1)):
                    self.valid_indices.append(j)
                    self.valid_episode_starts.append(ep_start)
            logger.info(f"Built {len(self.valid_indices)} valid windows from {self.dataset.meta.total_episodes} episodes ({n} frames)")
        else:
            # Fallback: assume no episode boundaries
            self.valid_indices = list(range(n - self.action_horizon + 1))
            self.valid_episode_starts = [0 for _ in self.valid_indices]
            logger.info(f"Built {len(self.valid_indices)} valid windows (no episode metadata)")

    def _apply_subset(self):
        """Optionally restrict training to a deterministic subset of valid windows."""
        if self.max_train_samples is None:
            return
        if len(self.valid_indices) <= self.max_train_samples:
            logger.info(
                "Requested max_train_samples=%d, but dataset only has %d windows; keeping full set",
                self.max_train_samples,
                len(self.valid_indices),
            )
            return

        rng = np.random.default_rng(self.subset_seed)
        chosen = np.sort(
            rng.choice(len(self.valid_indices), size=self.max_train_samples, replace=False)
        )
        self.valid_indices = [self.valid_indices[int(i)] for i in chosen]
        self.valid_episode_starts = [self.valid_episode_starts[int(i)] for i in chosen]
        logger.info(
            "Restricted train windows to %d samples (subset_seed=%d)",
            len(self.valid_indices),
            self.subset_seed,
        )

    def _compute_norm_stats(self) -> Dict[str, torch.Tensor]:
        """Compute q01/q99 action normalization statistics."""
        if self.dataset is None or self.action_key is None:
            zeros = torch.zeros(LIBERO_ACTION_DIM)
            ones = torch.ones(LIBERO_ACTION_DIM)
            state_low = torch.full((LIBERO_STATE_DIM,), -1.0)
            state_high = torch.full((LIBERO_STATE_DIM,), 1.0)
            return {
                "q01": zeros,
                "q99": ones,
                "local_q01": zeros.clone(),
                "local_q99": ones.clone(),
                "state_q01": state_low,
                "state_q99": state_high,
            }

        sample_size = min(self.norm_stats_sample_size, len(self.valid_indices))
        if sample_size <= 0:
            zeros = torch.zeros(LIBERO_ACTION_DIM)
            ones = torch.ones(LIBERO_ACTION_DIM)
            state_low = torch.full((LIBERO_STATE_DIM,), -1.0)
            state_high = torch.full((LIBERO_STATE_DIM,), 1.0)
            return {
                "q01": zeros,
                "q99": ones,
                "local_q01": zeros.clone(),
                "local_q99": ones.clone(),
                "state_q01": state_low,
                "state_q99": state_high,
            }
        indices = np.random.choice(len(self.valid_indices), sample_size, replace=False)

        all_actions = []
        all_states = []
        for i in indices:
            idx = self.valid_indices[i]
            frame = self.dataset[idx]
            action = frame[self.action_key]
            if isinstance(action, torch.Tensor):
                all_actions.append(action.float())
            elif isinstance(action, np.ndarray):
                all_actions.append(torch.from_numpy(action).float())
            if self.state_key is not None and self.state_key in frame:
                state = frame[self.state_key]
                if isinstance(state, torch.Tensor):
                    all_states.append(state.float())
                elif isinstance(state, np.ndarray):
                    all_states.append(torch.from_numpy(state).float())

        if all_actions:
            actions_tensor = torch.stack(all_actions)
            q01 = torch.quantile(actions_tensor, 0.01, dim=0)
            q99 = torch.quantile(actions_tensor, 0.99, dim=0)
            # Ensure no zero-range dimensions
            q99 = torch.where(q99 - q01 < 1e-6, q01 + 1.0, q99)

            local_q01 = q01.clone()
            local_q99 = q99.clone()
            if all_states and len(all_states) == len(all_actions):
                states_tensor = torch.stack(all_states)
                local_actions_tensor = actions_tensor.clone()
                local_actions_tensor[:, :6] = world_to_local_motion(
                    actions_tensor[:, :6], states_tensor
                )
                local_q01 = torch.quantile(local_actions_tensor, 0.01, dim=0)
                local_q99 = torch.quantile(local_actions_tensor, 0.99, dim=0)
                local_q99 = torch.where(local_q99 - local_q01 < 1e-6, local_q01 + 1.0, local_q99)
                state_q01 = torch.quantile(states_tensor, 0.01, dim=0)
                state_q99 = torch.quantile(states_tensor, 0.99, dim=0)
                state_q99 = torch.where(state_q99 - state_q01 < 1e-6, state_q01 + 1.0, state_q99)
            else:
                state_q01 = torch.full((LIBERO_STATE_DIM,), -1.0)
                state_q99 = torch.full((LIBERO_STATE_DIM,), 1.0)
        else:
            q01 = torch.zeros(LIBERO_ACTION_DIM)
            q99 = torch.ones(LIBERO_ACTION_DIM)
            local_q01 = q01.clone()
            local_q99 = q99.clone()
            state_q01 = torch.full((LIBERO_STATE_DIM,), -1.0)
            state_q99 = torch.full((LIBERO_STATE_DIM,), 1.0)

        return {
            "q01": q01,
            "q99": q99,
            "local_q01": local_q01,
            "local_q99": local_q99,
            "state_q01": state_q01,
            "state_q99": state_q99,
        }

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _get_images(self, frame) -> List[Image.Image]:
        """Extract PIL images from a frame."""
        images = []
        for key in self.image_keys:
            img = frame[key]
            if isinstance(img, torch.Tensor):
                if img.dim() == 3 and img.shape[0] in (1, 3):
                    img = img.permute(1, 2, 0)
                img = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if isinstance(img, Image.Image):
                img = self._resize_image(img)
                images.append(img)
        if not images:
            images = [Image.new("RGB", self.image_size, color="black")]
        for _ in range(self.empty_cameras):
            images.append(Image.new("RGB", self.image_size, color="black"))
        return images

    def _resize_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.image_resize_mode != "pad":
            return image.resize(self.image_size)

        target_w, target_h = self.image_size
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            return Image.new("RGB", self.image_size, color="black")

        scale = min(target_w / float(src_w), target_h / float(src_h))
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = image.resize((resized_w, resized_h), Image.BICUBIC)
        canvas = Image.new("RGB", self.image_size, color="black")
        offset_x = (target_w - resized_w) // 2
        offset_y = (target_h - resized_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    def _get_instruction(self, frame) -> str:
        """Extract instruction string from a frame."""
        instruction = "execute the task"
        if self.task_key and self.task_key in frame:
            val = frame[self.task_key]
            if isinstance(val, str):
                instruction = val
            elif isinstance(val, list) and val:
                instruction = str(val[0])
        return instruction

    def _normalize_state_for_prompt(self, state: torch.Tensor) -> torch.Tensor:
        low = self.norm_stats.get(
            "state_q01", torch.full((LIBERO_STATE_DIM,), -1.0, dtype=torch.float32)
        ).to(dtype=state.dtype)
        high = self.norm_stats.get(
            "state_q99", torch.full((LIBERO_STATE_DIM,), 1.0, dtype=torch.float32)
        ).to(dtype=state.dtype)
        return (2.0 * (state - low) / (high - low + 1e-8) - 1.0).clamp(-1.0, 1.0)

    def _build_prompt(self, instruction: str, state: torch.Tensor) -> str:
        if self.prompt_style != "pi05_state_prompt":
            return instruction

        norm_state = self._normalize_state_for_prompt(state.float())
        bins = ((norm_state + 1.0) * 0.5 * float(max(self.state_prompt_bins - 1, 1))).round().to(torch.int64)
        cleaned = instruction.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, bins.tolist()))
        return f"Task: {cleaned}, State: {state_str};\nAction: "

    def _coerce_vector(self, value: Any, expected_dim: int) -> torch.Tensor:
        """Convert a state/action vector to a fixed-width float tensor."""
        if isinstance(value, torch.Tensor):
            tensor = value.float().reshape(-1)
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).float().reshape(-1)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)

        if tensor.numel() == expected_dim:
            return tensor
        if tensor.numel() > expected_dim:
            return tensor[:expected_dim]

        padded = torch.zeros(expected_dim, dtype=torch.float32)
        padded[: tensor.numel()] = tensor
        return padded

    def _get_history(self, sample_idx: int) -> torch.Tensor:
        """Return K history tokens [state, prev_action, valid_flag] for the sample."""
        if self.history_len <= 0:
            return torch.zeros(0, self.history_feature_dim, dtype=torch.float32)

        history = torch.zeros(self.history_len, self.history_feature_dim, dtype=torch.float32)
        if self.dataset is None:
            return history

        start_idx = self.valid_indices[sample_idx]
        episode_start = self.valid_episode_starts[sample_idx] if self.valid_episode_starts else 0

        for hist_slot in range(self.history_len):
            offset = self.history_len - hist_slot
            frame_idx = start_idx - offset
            if frame_idx < episode_start:
                continue

            frame = self.dataset[frame_idx]
            state = (
                self._coerce_vector(frame[self.state_key], LIBERO_STATE_DIM)
                if self.state_key and self.state_key in frame
                else torch.zeros(LIBERO_STATE_DIM, dtype=torch.float32)
            )
            action = (
                self._coerce_vector(frame[self.action_key], LIBERO_ACTION_DIM)
                if self.action_key and self.action_key in frame
                else torch.zeros(LIBERO_ACTION_DIM, dtype=torch.float32)
            )

            history[hist_slot, :LIBERO_STATE_DIM] = state
            history[hist_slot, LIBERO_STATE_DIM:LIBERO_STATE_DIM + LIBERO_ACTION_DIM] = action
            history[hist_slot, -1] = 1.0

        return history

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.dataset is None:
            # Placeholder
            return {
                "images": [Image.new("RGB", self.image_size, color="gray")],
                "instruction": "pick up the red block and place it on the table",
                "actions": torch.randn(self.action_horizon, LIBERO_ACTION_DIM),
                "state": torch.randn(LIBERO_STATE_DIM),
                "history": torch.zeros(self.history_len, self.history_feature_dim),
            }

        start_idx = self.valid_indices[idx]
        frame = self.dataset[start_idx]

        images = self._get_images(frame)
        instruction = self._get_instruction(frame)

        # Action chunk — batch slice from underlying hf_dataset to avoid 8 individual reads
        if self.action_key:
            try:
                hf = self.dataset.hf_dataset
                end_idx = start_idx + self.action_horizon
                rows = hf.select(range(start_idx, end_idx))
                raw = rows[self.action_key]
                if isinstance(raw, list):
                    actions = torch.tensor(raw, dtype=torch.float32)
                elif isinstance(raw, torch.Tensor):
                    actions = raw.float()
                else:
                    actions = torch.as_tensor(np.array(raw), dtype=torch.float32)
            except Exception:
                act_list = []
                for t in range(self.action_horizon):
                    f = self.dataset[start_idx + t]
                    a = f[self.action_key]
                    if isinstance(a, np.ndarray):
                        a = torch.from_numpy(a).float()
                    elif isinstance(a, torch.Tensor):
                        a = a.float()
                    act_list.append(a)
                actions = torch.stack(act_list)
        else:
            actions = torch.zeros(self.action_horizon, LIBERO_ACTION_DIM)

        # State
        if self.state_key and self.state_key in frame:
            state = frame[self.state_key]
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            elif isinstance(state, torch.Tensor):
                state = state.float()
        else:
            state = torch.zeros(LIBERO_STATE_DIM)
        history = self._get_history(idx)
        prompt = self._build_prompt(instruction, state)

        # Pre-tokenize with VLM processor in worker process (offloads CPU work from GPU thread)
        if self.processor is not None:
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": img} for img in images
            ] + [{"type": "text", "text": prompt}]}]]

            processor_kwargs: Dict[str, Any] = {"padding": False}
            if self.tokenizer_max_length is not None:
                processor_kwargs = {
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": self.tokenizer_max_length,
                }

            vlm_inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=processor_kwargs,
            )
            # Squeeze batch dim (single sample)
            vlm_inputs = {k: v.squeeze(0) for k, v in vlm_inputs.items()}

            return {
                "vlm_inputs": vlm_inputs,
                "vlm_padding_side": self.tokenizer_padding_side,
                "images": images,
                "instruction": instruction,
                "actions": actions,
                "state": state,
                "history": history,
            }

        return {
            "images": images,
            "instruction": instruction,
            "actions": actions,
            "state": state,
            "history": history,
        }


def vla_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate for VLA batches with pre-tokenized inputs."""
    # Pre-tokenized path: pad VLM inputs
    if "vlm_inputs" in batch[0]:
        vlm_keys = batch[0]["vlm_inputs"].keys()
        padded_vlm = {}
        padding_side = batch[0].get("vlm_padding_side", "left")

        # Keys that are per-token sequences and need left-padding
        seq_keys = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}

        for key in vlm_keys:
            tensors = [s["vlm_inputs"][key] for s in batch]
            if tensors[0].dim() == 0:
                padded_vlm[key] = torch.stack(tensors)
            elif key in seq_keys and tensors[0].dim() >= 1:
                # Left-pad sequence tensors (pad with 0)
                max_len = max(t.shape[0] for t in tensors)
                padded = []
                for t in tensors:
                    pad_len = max_len - t.shape[0]
                    if pad_len > 0:
                        pad_tensor = torch.full((pad_len, *t.shape[1:]), 0, dtype=t.dtype)
                        if padding_side == "right":
                            padded.append(torch.cat([t, pad_tensor]))
                        else:
                            padded.append(torch.cat([pad_tensor, t]))
                    else:
                        padded.append(t)
                padded_vlm[key] = torch.stack(padded)
            else:
                # For pixel_values, image_grid_thw etc. — concatenate along batch dim
                try:
                    padded_vlm[key] = torch.cat(tensors, dim=0)
                except Exception:
                    padded_vlm[key] = torch.stack(tensors)

        return {
            "vlm_inputs": padded_vlm,
            "images": [s["images"] for s in batch],
            "instructions": [s["instruction"] for s in batch],
            "actions": torch.stack([s["actions"] for s in batch]),
            "state": torch.stack([s["state"] for s in batch]),
            "history": torch.stack([s["history"] for s in batch]),
        }

    # Fallback: raw images + instructions
    return {
        "images": [s["images"] for s in batch],
        "instructions": [s["instruction"] for s in batch],
        "actions": torch.stack([s["actions"] for s in batch]),
        "state": torch.stack([s["state"] for s in batch]),
        "history": torch.stack([s["history"] for s in batch]),
    }


def build_libero_dataloader(
    dataset_name: str = "lerobot/libero_spatial_image",
    data_root: Optional[str] = None,
    batch_size: int = 2,
    action_horizon: int = 8,
    num_workers: int = 4,
    vlm_model_id: Optional[str] = None,
    prefetch_factor: int = 8,
    max_train_samples: Optional[int] = None,
    subset_seed: int = 42,
    norm_stats_sample_size: int = 20000,
    history_len: int = 0,
    tokenizer_padding_side: str = "left",
    tokenizer_max_length: Optional[int] = None,
    prompt_style: str = "plain",
    state_prompt_bins: int = 256,
    image_resize_mode: str = "stretch",
    empty_cameras: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs,
) -> Tuple[DataLoader, LiberoVLADataset]:
    """Build LIBERO DataLoader for training."""
    dataset = LiberoVLADataset(
        dataset_name=dataset_name,
        data_root=data_root,
        action_horizon=action_horizon,
        vlm_model_id=vlm_model_id,
        max_train_samples=max_train_samples,
        subset_seed=subset_seed,
        norm_stats_sample_size=norm_stats_sample_size,
        history_len=history_len,
        tokenizer_padding_side=tokenizer_padding_side,
        tokenizer_max_length=tokenizer_max_length,
        prompt_style=prompt_style,
        state_prompt_bins=state_prompt_bins,
        image_resize_mode=image_resize_mode,
        empty_cameras=empty_cameras,
        **kwargs,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=vla_collate_fn,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return loader, dataset
