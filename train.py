"""Qwen3.5-VL + Pi 0.5 action expert — LIBERO training.

Single-path trainer: Accelerate + DeepSpeed ZeRO-2 (or single GPU), AdamW with
unified LR, cosine schedule clamped at `min_lr` (set min_lr == peak_lr for the
Pi 0.5 "constant after warmup" recipe).

Usage:
    # 8 GPUs
    accelerate launch --config_file config/deepspeed_zero2_single_fast.yaml \
        --num_processes 8 train.py --config config/libero_train_qwen_3_5_2b_pi05_v3_3.yaml

    # single GPU (raise grad-accum to compensate)
    python train.py --config config/libero_train_qwen_3_5_2b_pi05_v3_3.yaml \
        --training.gradient_accumulation_steps 8
"""

import argparse
import json
import logging
import math
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf


# ---------- EMA --------------------------------------------------------------

class EMA:
    """Exponential moving average of trainable params.

    Matches openpi Pi 0.5:
      ema_p <- decay * ema_p + (1 - decay) * current_p
    Updated once per real optimizer step (skipping microsteps during grad accum).
    `swap_in` / `swap_out` are used to eval against EMA weights without
    disturbing training state.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, offload_to_cpu: bool = False):
        self.decay = float(decay)
        self.offload = bool(offload_to_cpu)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().to("cpu" if self.offload else p.device).clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            s = self.shadow.get(name)
            if s is None:
                continue
            src = p.detach().to(s.device, non_blocking=True) if self.offload else p.detach()
            s.mul_(self.decay).add_(src, alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            s = self.shadow.get(name)
            if s is None:
                continue
            backup[name] = p.data.clone()
            p.data.copy_(s.to(p.device, non_blocking=True) if self.offload else s)
        return backup

    @torch.no_grad()
    def swap_out(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device, self.shadow[k].dtype))

warnings.filterwarnings("ignore", message="Kwargs passed to")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
logger = logging.getLogger(__name__)


# ---------- config -----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5-VL + Pi 0.5 VLA trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)

    # CLI dotlist overrides (`--training.max_steps 50000` or `key=value`).
    if unknown:
        overrides = []
        i = 0
        while i < len(unknown):
            key = unknown[i].lstrip("-")
            if "=" in key:
                overrides.append(key)
                i += 1
            elif i + 1 < len(unknown) and not unknown[i + 1].startswith("-"):
                overrides.append(f"{key}={unknown[i + 1]}")
                i += 2
            else:
                i += 1
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


# ---------- builders ---------------------------------------------------------

def build_model(cfg):
    from model import Qwen35V33VLA, QwenV33ActionConfig, ACTION_EXPERT_VARIANTS

    ah = cfg.action_head
    variant = ACTION_EXPERT_VARIANTS[str(ah.get("action_expert_variant", "gemma_300m"))]
    action_config = QwenV33ActionConfig(
        head_type=str(ah.get("head_type", "pi05_qwen")),
        action_expert_variant=str(ah.get("action_expert_variant", "gemma_300m")),
        hidden_dim=int(ah.get("hidden_dim", variant["hidden_dim"])),
        num_layers=int(ah.get("num_layers", variant["num_layers"])),
        action_dim=int(ah.get("action_dim", 7)),
        action_horizon=int(ah.get("action_horizon", 10)),
        chunk_size=int(ah.get("chunk_size", ah.get("action_horizon", 10))),
        max_state_dim=int(ah.get("max_state_dim", 32)),
        max_action_dim=int(ah.get("max_action_dim", 32)),
        beta_alpha=float(ah.get("beta_alpha", 1.5)),
        beta_beta=float(ah.get("beta_beta", 1.0)),
        time_sampling_scale=float(ah.get("time_sampling_scale", 0.999)),
        time_sampling_offset=float(ah.get("time_sampling_offset", 0.001)),
        min_period=float(ah.get("min_period", 0.004)),
        max_period=float(ah.get("max_period", 4.0)),
        num_inference_steps=int(ah.get("num_inference_steps", 10)),
        state_dim=int(ah.get("state_dim", 8)),
        tokenizer_max_length=int(ah.get("tokenizer_max_length", 200)),
        image_resolution=tuple(ah.get("image_resolution", cfg.dataset.get("image_size", [224, 224]))),
        empty_cameras=int(ah.get("empty_cameras", cfg.dataset.get("empty_cameras", 1))),
    )

    return Qwen35V33VLA(
        vlm_model_id=cfg.model.vlm_model_id,
        action_config=action_config,
        freeze_vision_encoder=bool(cfg.model.get("freeze_vision_encoder", False)),
        freeze_vlm=bool(cfg.model.get("freeze_vlm", False)),
        attn_implementation=cfg.model.get("attn_implementation", None),
        architecture_name=str(cfg.model.get("architecture", "qwen_3.5_2b_pi05_v3_3")),
    )


def build_dataloader(cfg):
    from data.libero_dataset import build_libero_dataloader

    return build_libero_dataloader(
        dataset_name=cfg.dataset.name,
        data_root=cfg.dataset.get("data_root", None),
        batch_size=int(cfg.dataset.batch_size),
        action_horizon=int(cfg.action_head.action_horizon),
        num_workers=int(cfg.dataset.get("num_workers", 4)),
        image_size=tuple(cfg.dataset.image_size),
        action_type=str(cfg.dataset.get("action_type", "delta_qpos")),
        vlm_model_id=cfg.model.vlm_model_id,
        prefetch_factor=int(cfg.dataset.get("prefetch_factor", 4)),
        max_train_samples=cfg.dataset.get("max_train_samples", None),
        subset_seed=int(cfg.dataset.get("subset_seed", 42)),
        norm_stats_sample_size=int(cfg.dataset.get("norm_stats_sample_size", 20000)),
        tokenizer_padding_side=str(cfg.dataset.get("tokenizer_padding_side", "right")),
        tokenizer_max_length=int(cfg.dataset.get("tokenizer_max_length", 200)),
        prompt_style=str(cfg.dataset.get("prompt_style", "plain")),
        image_resize_mode=str(cfg.dataset.get("image_resize_mode", "stretch")),
        empty_cameras=int(cfg.dataset.get("empty_cameras", 1)),
        shuffle=bool(cfg.dataset.get("shuffle", True)),
        drop_last=bool(cfg.dataset.get("drop_last", True)),
    )


def build_optimizer(model, cfg):
    tcfg = cfg.training
    groups = model.get_optimizer_groups(
        lr=float(tcfg.lr),
        weight_decay=float(tcfg.weight_decay),
    )
    return torch.optim.AdamW(
        groups,
        betas=tuple(tcfg.get("betas", [0.9, 0.95])),
        eps=float(tcfg.get("eps", 1e-8)),
    )


def build_scheduler(optimizer, cfg):
    tcfg = cfg.training
    warmup = int(tcfg.warmup_steps)
    max_steps = int(tcfg.max_steps)
    peak_lr = float(tcfg.lr)
    min_lr = float(tcfg.get("min_lr", peak_lr))
    floor = min_lr / peak_lr if peak_lr > 0 else 0.0

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(floor, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------- checkpoint --------------------------------------------------------

def save_checkpoint(model, accelerator, cfg, step, ema=None, eval_mse=None):
    out_dir = Path(cfg.training.output_dir) / f"checkpoint-{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()
    if bool(cfg.training.get("save_optimizer_state", True)):
        accelerator.save_state(str(out_dir / "accelerator_state"), safe_serialization=False)

    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        torch.save(
            {
                "action_head": unwrapped.action_head.state_dict(),
                "action_q01": unwrapped.action_q01,
                "action_q99": unwrapped.action_q99,
                "state_q01": unwrapped.state_q01,
                "state_q99": unwrapped.state_q99,
                "step": step,
            },
            out_dir / "action_head.pt",
        )
        if ema is not None:
            # EMA state is the canonical eval weights per Pi 0.5; see openpi
            # checkpoints.py:146-148 (ema_params replace params at eval time).
            torch.save(ema.state_dict(), out_dir / "ema.pt")
        OmegaConf.save(cfg, out_dir / "config.yaml")
        if eval_mse is not None:
            (out_dir / "eval.json").write_text(json.dumps({"mse": float(eval_mse)}))
    accelerator.wait_for_everyone()
    return out_dir


def load_checkpoint(model, ckpt_dir, accelerator, ema=None):
    ckpt_dir = Path(ckpt_dir)
    action_path = ckpt_dir / "action_head.pt"
    if not action_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {action_path}")

    state_dir = ckpt_dir / "accelerator_state"
    if state_dir.exists():
        accelerator.load_state(str(state_dir))
        payload = torch.load(action_path, map_location="cpu", weights_only=False)
    else:
        payload = torch.load(action_path, map_location="cpu", weights_only=False)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.action_head.load_state_dict(payload["action_head"], strict=False)

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.set_norm_stats(
        payload["action_q01"], payload["action_q99"],
        payload.get("state_q01"), payload.get("state_q99"),
    )

    ema_path = ckpt_dir / "ema.pt"
    if ema is not None and ema_path.exists():
        ema.load_state_dict(torch.load(ema_path, map_location="cpu", weights_only=False))
        logger.info(f"Loaded EMA state from {ema_path}")

    return int(payload.get("step", 0))


# ---------- evaluation --------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, accelerator, ema=None, max_batches: int = 20) -> float:
    """Eval with EMA weights swapped in (per Pi 0.5 protocol)."""
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    ema_backup = ema.swap_in(unwrapped) if ema is not None else None

    try:
        total_mse = 0.0
        count = 0
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            actions = batch["actions"].to(accelerator.device)
            if "vlm_inputs" in batch:
                vlm_inputs = {k: v.to(accelerator.device) for k, v in batch["vlm_inputs"].items()}
                prefix = unwrapped.vlm.build_prefix_from_vlm_inputs(vlm_inputs)
                norm_pred = unwrapped.action_head.predict_action(
                    language_model=unwrapped._language_model(),
                    prefix_embeds=prefix["prefix_embeds"],
                    prefix_attention_mask=prefix["attention_mask"],
                    prefix_position_ids=prefix["position_ids"],
                )
                pred = unwrapped.unnormalize_actions(norm_pred)
            else:
                pred_np = unwrapped.predict_action(batch["images"], batch["instructions"])
                pred = torch.from_numpy(pred_np).to(actions.device)

            gt = unwrapped.unnormalize_actions(unwrapped.normalize_actions(actions))
            total_mse += ((pred - gt) ** 2).mean().item()
            count += 1
    finally:
        if ema_backup is not None:
            ema.swap_out(unwrapped, ema_backup)
        model.train()
    return total_mse / max(count, 1)


# ---------- training loop -----------------------------------------------------

def main():
    cfg = parse_args()
    tcfg = cfg.training
    set_seed(int(tcfg.seed))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(tcfg.gradient_accumulation_steps),
        mixed_precision=str(tcfg.get("mixed_precision", "bf16")),
    )

    if accelerator.is_main_process:
        Path(tcfg.output_dir).mkdir(parents=True, exist_ok=True)
        print("=" * 60)
        print("Qwen3.5-VL + Pi 0.5 v3.3 training on LIBERO")
        print("=" * 60)
        print(f"  backbone       : {cfg.model.vlm_model_id}")
        print(f"  action expert  : {cfg.action_head.action_expert_variant}")
        print(f"  action_horizon : {cfg.action_head.action_horizon}")
        print(f"  freeze_vlm     : {cfg.model.get('freeze_vlm', False)}")
        print(f"  lr / wd / betas: {tcfg.lr} / {tcfg.weight_decay} / {list(tcfg.get('betas', [0.9, 0.95]))}")
        print(f"  warmup / steps : {tcfg.warmup_steps} / {tcfg.max_steps}")
        eff = int(cfg.dataset.batch_size) * int(tcfg.gradient_accumulation_steps) * accelerator.num_processes
        print(f"  batch (eff)    : {eff} ({cfg.dataset.batch_size} × {tcfg.gradient_accumulation_steps} × {accelerator.num_processes})")
        print(f"  output_dir     : {tcfg.output_dir}")

    logger.info("Building model ...")
    model = build_model(cfg)

    logger.info("Building dataloader ...")
    dataloader, dataset = build_dataloader(cfg)

    model.set_norm_stats(
        dataset.norm_stats["q01"],
        dataset.norm_stats["q99"],
        dataset.norm_stats.get("state_q01"),
        dataset.norm_stats.get("state_q99"),
    )

    if bool(tcfg.get("gradient_checkpointing", False)):
        if hasattr(model.vlm.model, "gradient_checkpointing_enable"):
            model.vlm.model.gradient_checkpointing_enable()
            if hasattr(model.vlm.model, "config"):
                model.vlm.model.config.use_cache = False

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # EMA — created after prepare so the shadow tracks the actual trainable params
    # (accelerator may have re-wrapped requires_grad flags).
    ema_decay = tcfg.get("ema_decay", None)
    ema_offload = bool(tcfg.get("ema_offload_to_cpu", False))
    ema = (
        EMA(accelerator.unwrap_model(model), decay=float(ema_decay), offload_to_cpu=ema_offload)
        if ema_decay else None
    )
    if ema is not None and accelerator.is_main_process:
        n_shadow = sum(v.numel() for v in ema.shadow.values())
        logger.info(f"EMA enabled: decay={ema_decay}, shadow={n_shadow/1e6:.0f}M params"
                    f"{', offload=cpu' if ema_offload else ', on-GPU'}")

    global_step = 0
    resume = tcfg.get("resume_from_checkpoint", None)
    if resume:
        global_step = load_checkpoint(model, resume, accelerator, ema=ema)
        logger.info(f"Resumed from {resume} at step {global_step}")

    if accelerator.is_main_process and tcfg.get("wandb_project"):
        accelerator.init_trackers(
            project_name=str(tcfg.wandb_project),
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": tcfg.get("wandb_run_name"), "dir": tcfg.output_dir}},
        )

    model.train()
    data_iter = iter(dataloader)
    micro_step = 0
    running_loss = 0.0
    t_start = time.time()

    while global_step < int(tcfg.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        actions = batch["actions"].to(accelerator.device)

        with accelerator.accumulate(model):
            if "vlm_inputs" in batch:
                vlm_inputs = {k: v.to(accelerator.device) for k, v in batch["vlm_inputs"].items()}
                out = model(vlm_inputs=vlm_inputs, actions=actions)
            else:
                out = model(
                    images=batch["images"],
                    instructions=batch["instructions"],
                    actions=actions,
                )

            loss = out["loss"]
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(tcfg.max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # EMA update only on real optimizer steps (skips grad-accum microsteps).
            if ema is not None and accelerator.sync_gradients:
                ema.update(accelerator.unwrap_model(model))

        running_loss += loss.detach().item()
        micro_step += 1

        if micro_step % int(tcfg.gradient_accumulation_steps) == 0:
            global_step += 1

            if global_step % int(tcfg.log_every) == 0 and accelerator.is_main_process:
                window = int(tcfg.log_every) * int(tcfg.gradient_accumulation_steps)
                avg = running_loss / window
                lr = scheduler.get_last_lr()[0]
                dt = time.time() - t_start
                sec_per_step = dt / int(tcfg.log_every)
                eta_min = (int(tcfg.max_steps) - global_step) * sec_per_step / 60.0
                print(
                    f"step {global_step:>7}/{tcfg.max_steps} | loss {avg:.4f} | lr {lr:.2e} | "
                    f"{sec_per_step:.2f}s/step | eta {eta_min:.0f}m",
                    flush=True,
                )
                if tcfg.get("wandb_project"):
                    accelerator.log({"train/loss": avg, "train/lr": lr, "train/step": global_step}, step=global_step)
                running_loss = 0.0
                t_start = time.time()

            if global_step % int(tcfg.eval_every) == 0:
                mse = evaluate(model, dataloader, accelerator, ema=ema)
                if accelerator.is_main_process:
                    print(f"eval @ step {global_step}: action_mse {mse:.5f}"
                          f"{' (EMA)' if ema is not None else ''}", flush=True)
                    if tcfg.get("wandb_project"):
                        accelerator.log({"eval/action_mse": mse}, step=global_step)

            if global_step % int(tcfg.save_every) == 0:
                save_checkpoint(model, accelerator, cfg, global_step, ema=ema)
                if accelerator.is_main_process:
                    print(f"saved checkpoint at step {global_step}", flush=True)

    save_checkpoint(model, accelerator, cfg, global_step, ema=ema)
    if accelerator.is_main_process:
        print(f"training complete at step {global_step}", flush=True)
    if tcfg.get("wandb_project"):
        accelerator.end_training()


if __name__ == "__main__":
    main()
