"""Full-pipeline smoke test for Qwen35PI05VLA with a tiny random-init Qwen3.5-VL.

Strategy:
  1. Download processor/tokenizer/chat_template from a real Qwen3.5-VL repo
     (RohitUltimate/Qwen3.5_VL_2B_full_3, public & tiny metadata).
  2. Shrink the config (hidden_size, layers, vocab, vision depth) so the
     random-init model fits on a 16GB card with room to spare.
  3. Save the shrunk random-init model + processor files to a local dir.
  4. Build Qwen35PI05VLA pointing at that dir.
  5. Run forward with the TRAINING-TIME tokenized path via build_vla_inputs:
     images → preprocessor → tokenizer → masked_scatter → prefix embeds.
  6. Backward + AdamW step.
  7. predict_action() end-to-end.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")


TINY_DIR = Path("/tmp/tiny_qwen35_vl_random")
SRC_REPO = "RohitUltimate/Qwen3.5_VL_2B_full_3"


def build_tiny_vl_checkpoint():
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    TINY_DIR.mkdir(parents=True, exist_ok=True)
    # Pull only metadata/tokenizer/processor — skip model weights (4.4GB).
    print(f"downloading metadata from {SRC_REPO} ...")
    meta_dir = Path(snapshot_download(
        repo_id=SRC_REPO,
        allow_patterns=["config.json", "tokenizer*", "processor_config.json",
                        "chat_template.jinja", "generation_config.json",
                        "preprocessor_config.json", "special_tokens_map.json",
                        "vocab.json", "merges.txt", "video_preprocessor_config.json"],
    ))
    print(f"metadata at: {meta_dir}")

    # Shrink the config.
    cfg = AutoConfig.from_pretrained(meta_dir)
    # text side
    tcfg = cfg.text_config
    orig_layers = tcfg.num_hidden_layers
    orig_layer_types = list(tcfg.layer_types) if hasattr(tcfg, "layer_types") else None
    tcfg.hidden_size = 256
    tcfg.intermediate_size = 512
    tcfg.num_hidden_layers = 4
    tcfg.num_attention_heads = 4
    tcfg.num_key_value_heads = 2
    tcfg.head_dim = 64
    if orig_layer_types is not None:
        tcfg.layer_types = orig_layer_types[:tcfg.num_hidden_layers]
    # vision side
    vcfg = cfg.vision_config
    vcfg.depth = 2
    vcfg.hidden_size = 128
    vcfg.num_heads = 4
    vcfg.intermediate_size = 256
    vcfg.out_hidden_size = tcfg.hidden_size  # must match text hidden
    # keep patch_size / spatial_merge_size / temporal_patch_size as-is so processor works

    print(f"shrunk text_config     : hidden={tcfg.hidden_size} layers={tcfg.num_hidden_layers} "
          f"heads={tcfg.num_attention_heads} head_dim={tcfg.head_dim} "
          f"layer_types={tcfg.layer_types if orig_layer_types else 'n/a'}")
    print(f"shrunk vision_config   : depth={vcfg.depth} hidden={vcfg.hidden_size} out_hidden={vcfg.out_hidden_size}")

    # Random init the shrunk model.
    print("instantiating random-init Qwen3_5ForConditionalGeneration ...")
    model = Qwen3_5ForConditionalGeneration(cfg)
    model = model.to(dtype=torch.bfloat16)

    print(f"saving to {TINY_DIR} ...")
    model.save_pretrained(TINY_DIR, safe_serialization=True)

    # Copy all tokenizer/processor files.
    import shutil
    for f in meta_dir.iterdir():
        dst = TINY_DIR / f.name
        if f.name in {"config.json", "generation_config.json"}:
            continue  # already written by save_pretrained
        if not dst.exists() and f.is_file():
            shutil.copy(f, dst)
    print(f"tiny VL checkpoint ready: {TINY_DIR}")
    print(f"   files: {sorted(p.name for p in TINY_DIR.iterdir())}")
    return str(TINY_DIR)


def main():
    torch.manual_seed(0)
    if not TINY_DIR.exists() or not (TINY_DIR / "model.safetensors").exists():
        build_tiny_vl_checkpoint()

    from model import Qwen35PI05VLA, QwenPI05ActionConfig

    action_cfg = QwenPI05ActionConfig(
        hidden_dim=128,
        num_layers=4,
        action_dim=7,
        action_horizon=10,
        chunk_size=10,
        num_inference_steps=2,
        state_dim=8,
        image_resolution=(224, 224),
        empty_cameras=1,
        tokenizer_max_length=320,
    )
    print("\n[1/4] building Qwen35PI05VLA ...")
    device = torch.device("cuda")
    model = Qwen35PI05VLA(
        vlm_model_id=str(TINY_DIR),
        action_config=action_cfg,
        freeze_vision_encoder=False,
        freeze_vlm=False,
        attn_implementation="eager",   # safer for tiny random-init model
    ).to(device)
    print(f"  model dtype: {next(model.parameters()).dtype}")

    # Norm stats: identity (low=-1, high=1).
    model.set_norm_stats(
        action_q01=-torch.ones(action_cfg.action_dim),
        action_q99=torch.ones(action_cfg.action_dim),
        state_q01=-torch.ones(action_cfg.state_dim),
        state_q99=torch.ones(action_cfg.state_dim),
    )

    # --- dummy batch --------------------------------------------------------
    print("\n[2/4] building dummy batch ...")
    B = 2
    rng = np.random.default_rng(0)
    images = []
    for _ in range(B):
        # 2 cams (third is appended as empty by empty_cameras=1 in interface).
        cams = [Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)) for _ in range(2)]
        images.append(cams)
    instructions = ["pick up the red block", "place the cup on the plate"]
    actions = torch.randn(B, action_cfg.action_horizon, action_cfg.action_dim) * 0.3

    # --- training forward --------------------------------------------------
    print("\n[3/4] training forward+backward+step ...")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    out = model(
        images=images,
        instructions=instructions,
        actions=actions.to(device),
    )
    loss = out["loss"]
    print(f"  loss: {loss.item():.4f}")
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()

    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    n_without = sum(1 for p in model.parameters() if p.grad is None)
    print(f"  params with finite grads: {n_with_grad}, without: {n_without}")
    opt.step()
    opt.zero_grad()
    print(f"  optimizer step OK")

    # --- inference ---------------------------------------------------------
    print("\n[4/4] predict_action ...")
    model.eval()
    pred = model.predict_action(
        images=[images[0]],
        instructions=[instructions[0]],
        num_inference_steps=action_cfg.num_inference_steps,
        deterministic_seed=42,
    )
    print(f"  predicted shape: {pred.shape}, dtype: {pred.dtype}")
    assert pred.shape == (1, action_cfg.chunk_size, action_cfg.action_dim), f"unexpected shape {pred.shape}"
    assert np.all(np.isfinite(pred)), "non-finite predicted actions"

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\npeak GPU memory: {mem_gb:.2f} GB")
    print(">>> full VLA smoke test PASSED <<<")


if __name__ == "__main__":
    main()
