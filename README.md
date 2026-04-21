# qwen3.5_2b_vla — Pi 0.5 replica with Qwen3.5-VL backbone

A clean rewrite of the Pi 0.5 path in this codebase, targeting **one goal**:
replicate openpi `pi05_libero` exactly, with only the VLM backbone swapped
from PaliGemma to Qwen3.5-2B-VL.

## What changed vs. the original repo

The original repo contained two architectures (`Qwen35GemmaBridgeVLA` with a
cross-attention "bridge" expert, and `Qwen35PI05VLA` — the Pi 0.5 dual-stream
expert), but **all four training yamls wired the `gemma_bridge` variant**.
`Qwen35PI05VLA` existed as code but was never actually run by `train.py`.

This branch:

1. **Deletes the non–Pi-0.5 code paths** (gemma_bridge, cross-attention DiT,
   layerwise flow-matching heads v1/v3/v4, RTC, action_encoder, old
   Qwen35VLA wrapper, legacy interface, etc.).
2. **Rewrites the Pi 0.5 dual-stream expert** against openpi for exact
   architecture parity:
   - single joint Q/K/V attention per `full_attention` layer (prefix + suffix
     concatenated along the sequence axis, one softmax, block-causal mask)
   - adaRMSNorm with zero-init scale/shift and `gate_bias=1.0`
   - gemma_300m expert sizing (18 layers × 1024 hidden, head_dim inherited
     from the Qwen backbone so joint attention shapes align)
   - Beta(1.5, 1)·0.999 + 0.001 time sampling, `x_t = t·noise + (1-t)·action`,
     `u_t = noise - action`, reverse-Euler `dt = -1/N` sampling
   - KV-cached prefix for inference
3. **Rewrites `train.py`** down to the single Pi 0.5 path, with unified LR
   across VLM + expert, cosine schedule clamped at `min_lr` (so
   `min_lr == peak_lr` gives the constant-after-warmup behaviour used by
   `pi05_libero`), and `Accelerator` + DeepSpeed ZeRO-2 wrapping.
4. **Adds EMA (`ema_decay = 0.999`)** — openpi's
   [`scripts/train.py`](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/train.py)
   uses EMA for `pi05_libero`, and
   [`checkpoints.py:146`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/checkpoints.py#L146)
   loads EMA params at eval. We now:
   - create an `EMA` shadow after `accelerator.prepare` (on-GPU by default,
     `ema_offload_to_cpu: true` in the yaml saves ~2 GB per rank for tight VRAM)
   - update once per real optimizer step (skips grad-accum microsteps)
   - `swap_in` EMA weights before `evaluate()`, `swap_out` after
   - save `ema.pt` alongside `action_head.pt` in every checkpoint
   - `eval_libero.py` prefers `ema.pt` when present
5. **Rewrites `eval_libero.py`** down to the Pi 0.5 libero exec strategy:
   predict a 10-step chunk, execute the first 5, replan. No temporal-ensemble
   smoothing, no gripper hysteresis, no instruction rephrasing, no
   adaptive-highscore replan.
6. **Fixes two bugs discovered during smoke testing on an RTX 5080:**
   - `tokenizer_max_length: 200` was too short — Qwen3.5-VL at 224×224 produces
     64 image tokens per camera, so 3 cameras alone = 192 tokens, leaving
     ~8 for text. Bumped to **320**.
   - `Qwen35PI05VLA._build_prompts` wrapped prompts with
     `"Task: {x};\nAction: "` — that is the Pi 0.5 **with-state** format
     ([openpi tokenizer.py:28](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py#L28)).
     Pi 0.5 libero uses `discrete_state_input=False` and
     `prompt_from_task=True`, which means the prompt is just the cleaned task
     instruction — line 33 of the same file. Now `_build_prompts` returns the
     cleaned instruction verbatim; this also matches the libero_dataset's
     `prompt_style: "plain"` pre-tokenization path so the two codepaths agree.

## What did NOT change (verified against openpi, already identical)

- **Last-layer prefix downstream MLP** is computed but produces zero gradient
  under action-only loss. openpi's
  [`gemma.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py)
  `Block.__call__` runs the full attn+MLP for every layer uniformly — there is
  no last-layer special case, so Pi 0.5 has the same ~5.5% "wasted" compute.
- **AdamW weight decay mask**. openpi's
  [`scripts/train.py:88`](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/train.py#L88)
  passes `weight_decay_mask=None`, applying wd=1e-10 uniformly to all params
  (including bias / norm / embed). Our single `get_optimizer_groups` group
  with wd=1e-10 matches this exactly.

## Unavoidable VLM-swap deltas (intrinsic to using Qwen3.5-VL)

- Qwen3.5's hybrid architecture: `full_attention` layers join Q/K/V with the
  suffix expert, but `linear_attention` layers (`Qwen3_5GatedDeltaNet`) run as
  two independent parallel streams — GatedDeltaNet has no softmax-attention
  semantics and thus no shared-KV concept.
- Qwen3.5's `q_norm` / `k_norm` and sigmoid attention-gate are baked into
  `Qwen3_5Attention`.
- 3D MRoPE position ids and `<image>` placeholder + `masked_scatter` for image
  token insertion, in place of PaliGemma's SigLIP-concat + 1D RoPE.
- Action expert weights cannot inherit from `pi05_base` (different backbone),
  so training runs from scratch. Expect slower convergence than
  openpi's 30k-step `pi05_libero` finetune which starts from `pi05_base`.

## Repository layout

```
config/
  libero_train_qwen_3_5_2b_pi05.yaml      # the Pi 0.5 training recipe
  smoke_train.yaml                        # tiny config used by smoke/
  accelerate_single_gpu_bf16.yaml
  deepspeed_zero2_single_fast.yaml

data/
  libero_dataset.py                       # unchanged; LeRobot LIBERO wrapper

model/
  __init__.py
  qwen35_pi05_action_head.py              # dual-stream Pi 0.5 expert
  qwen35_pi05_interface.py                # Qwen3.5-VL wrapper + prefix builder
  qwen35_pi05_vla.py                      # top-level VLA
  se3_utils.py                            # unchanged; used by libero_dataset

smoke/
  smoke_test_action_head.py               # unit: action expert forward/back/sample
  smoke_test_full.py                      # full VLA with tiny random-init Qwen3.5-VL
  smoke_test_real.py                      # full VLA with real Qwen3.5-2B weights
  smoke_test_steps.py                     # 6-step loss sanity check
  smoke_test_gradcheck.py                 # verifies which params (don't) get grad
  smoke_test_ema_semantics.py             # EMA update / swap / roundtrip unit test
  smoke_test_dataset_contract.py          # libero_dataset batch ↔ VLA compatibility

train.py                                  # single Pi 0.5 training path
eval_libero.py                            # LIBERO rollout (predict-10, exec-5)
```

## Hyperparameters (all from `libero_train_qwen_3_5_2b_pi05.yaml`)

Architecturally verified against openpi's
[`pi05_libero`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py#L744):

| | Pi 0.5 (openpi) | this repo |
|---|---|---|
| action_horizon | 10 | 10 |
| action expert | gemma_300m (18 layers × 1024) | gemma_300m |
| batch_size (effective) | 256 | 256 (8 GPUs × 32 × 1 accum) |
| num_train_steps | 30 000 | 30 000 |
| AdamW b1 / b2 / eps | 0.9 / 0.95 / 1e-8 | same |
| weight_decay | 1e-10 | same |
| grad clip | 1.0 | same |
| peak_lr / min_lr | 5e-5 / 5e-5 (constant after warmup) | same |
| warmup_steps | 10 000 | same |
| ema_decay | 0.999 | same |
| precision | bf16 | bf16 |
| image resize | stretch 224×224 | stretch 224×224 |
| cameras | 3 (base + left_wrist + empty right_wrist) | 3 (`empty_cameras: 1`) |
| state in prompt | no (`discrete_state_input=False`) | no |
| prompt | cleaned task instruction | same |

## Usage

### Training

```bash
# 8-GPU cluster with DeepSpeed ZeRO-2 (recommended — full fine-tune needs ~18 GB / rank)
accelerate launch --config_file config/deepspeed_zero2_single_fast.yaml \
  --num_processes 8 train.py --config config/libero_train_qwen_3_5_2b_pi05.yaml

# Single GPU — raise grad-accum to keep effective batch ≈ 256
python train.py --config config/libero_train_qwen_3_5_2b_pi05.yaml \
  --training.gradient_accumulation_steps 8 --dataset.batch_size 32

# CLI overrides use dotlist: `--section.key value` or `key=value`
python train.py --config config/libero_train_qwen_3_5_2b_pi05.yaml \
  --model.vlm_model_id /path/to/your/Qwen3.5-2B
```

On a small GPU (≤ 16 GB), add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
and set `training.ema_offload_to_cpu: true` in the yaml.

### Install the fast linear-attention kernels

Qwen3.5's `Qwen3_5GatedDeltaNet` falls back to a slow PyTorch implementation
without these:

```bash
pip install flash-linear-attention causal-conv1d
```

Not required for correctness; matters for throughput.

### Evaluation

```bash
python eval_libero.py \
  --checkpoint_dir checkpoints/.../checkpoint-30000 \
  --suites libero_spatial libero_object libero_goal libero_10 \
  --n_episodes 10 --max_steps 300 --replan_horizon 5
```

EMA weights (`ema.pt`) are automatically preferred over `action_head.pt`
when present in the checkpoint directory.

### Smoke tests

```bash
# Unit: action expert only, tiny random Qwen3.5 text backbone, ~1 GB, <1 min
python smoke/smoke_test_action_head.py

# Full VLA with a tiny random-init Qwen3.5-VL (downloads ~5 MB of processor config)
python smoke/smoke_test_full.py

# Full VLA with real Qwen3.5-2B weights — needs ~16 GB GPU (freezes VLM)
python smoke/smoke_test_real.py

# Verifies libero_dataset's pre-tokenized and lazy paths both work
python smoke/smoke_test_dataset_contract.py

# Verifies EMA update / swap_in / swap_out / state-dict round-trip
python smoke/smoke_test_ema_semantics.py
```

## Smoke-test results (RTX 5080, 16 GB)

- **action head isolated**: 0.27 GB peak, forward 457 ms, backward 100 ms
- **full VLA with tiny random-init Qwen3.5-VL**: 0.89 GB peak
- **full VLA with real Qwen3.5-2B (freeze_vlm, bs=1, horizon=4)**: 14.9 GB peak
- **full `train.py` loop on synthetic data (6 steps)**: loss trajectory sane,
  warmup 2 → 5 / 5 / 5 / 5 / 5 lr, EMA swap at eval works, checkpoint + ema.pt
  written, clean shutdown

## Known gaps

- **EMA is not sharded** — on each rank the shadow is a full copy of trainable
  params. For full fine-tune of Qwen3.5-2B (~2.74 B trainable) this is
  ~11 GB fp32 per rank on top of DeepSpeed ZeRO-2's sharded optimizer.
  If that's too much, set `ema_offload_to_cpu: true` in the yaml (the smoke
  tests use this).
- **No LoRA path.** openpi has `pi05_libero_lora` as a low-memory variant; we
  stripped it. Easy to add back if needed.
- **Action expert trained from scratch.** Unavoidable — can't inherit
  `pi05_base` weights after the backbone swap. Budget more training steps or
  a warm-start from a converged action-expert checkpoint if you have one.
