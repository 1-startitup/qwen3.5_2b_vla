import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from train import (
    build_dataloader,
    build_model,
    load_lora_adapters,
    normalize_action_head_state_dict,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--max_batches", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    loader, dataset = build_dataloader(cfg)
    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    ckpt_dir = Path(args.checkpoint_dir)
    payload = torch.load(ckpt_dir / "action_head.pt", map_location="cpu", weights_only=False)
    normalized_action_head = normalize_action_head_state_dict(payload["action_head"])
    missing, unexpected = model.action_head.load_state_dict(normalized_action_head, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Action head checkpoint mismatch for {ckpt_dir}: missing={missing} unexpected={unexpected}"
        )
    if "action_q01" in payload and "action_q99" in payload:
        model.set_norm_stats(
            payload["action_q01"],
            payload["action_q99"],
            payload.get("local_action_q01", payload["action_q01"]),
            payload.get("local_action_q99", payload["action_q99"]),
        )
    lora_dir = ckpt_dir / "lora_adapters"
    if lora_dir.exists() and hasattr(model.vlm, "model"):
        load_lora_adapters(model.vlm.model, lora_dir)

    tp = tn = fp = fn = 0
    tp_first = tn_first = fp_first = fn_first = 0
    total = 0
    total_first = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        pred = model.predict_action(
            images=batch["images"],
            instructions=batch["instructions"],
            state=batch["state"].to(device),
            history=batch.get("history", None).to(device) if batch.get("history", None) is not None else None,
            num_inference_steps=10,
            deterministic_seed=0,
        )
        pred = torch.from_numpy(pred).to(device)

        gt = batch["actions"].to(device)
        pred_sign = (pred[..., 6] > 0).to(torch.int64)
        gt_sign = (gt[..., 6] > 0).to(torch.int64)

        tp += int(((pred_sign == 1) & (gt_sign == 1)).sum().item())
        tn += int(((pred_sign == 0) & (gt_sign == 0)).sum().item())
        fp += int(((pred_sign == 1) & (gt_sign == 0)).sum().item())
        fn += int(((pred_sign == 0) & (gt_sign == 1)).sum().item())
        total += int(pred_sign.numel())

        pred_first = pred_sign[:, 0]
        gt_first = gt_sign[:, 0]
        tp_first += int(((pred_first == 1) & (gt_first == 1)).sum().item())
        tn_first += int(((pred_first == 0) & (gt_first == 0)).sum().item())
        fp_first += int(((pred_first == 1) & (gt_first == 0)).sum().item())
        fn_first += int(((pred_first == 0) & (gt_first == 1)).sum().item())
        total_first += int(pred_first.numel())

    def metrics(tp, tn, fp, fn, total_count):
        acc = (tp + tn) / max(total_count, 1)
        pred_pos = (tp + fp) / max(total_count, 1)
        gt_pos = (tp + fn) / max(total_count, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {
            "accuracy": acc,
            "pred_close_fraction": pred_pos,
            "gt_close_fraction": gt_pos,
            "close_precision": precision,
            "close_recall": recall,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total": total_count,
        }

    result = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "max_batches": args.max_batches,
        "overall": metrics(tp, tn, fp, fn, total),
        "first_step": metrics(tp_first, tn_first, fp_first, fn_first, total_first),
        "dataset_windows": len(dataset),
    }

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
