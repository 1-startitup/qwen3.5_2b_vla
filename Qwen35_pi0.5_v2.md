# Qwen35_pi0.5_v2

This document tracks the `Qwen35_pi0.5_v2` model line.

## Scope

- Base architecture: `qwen_3.5_2b_pi0.5`
- Phase: `Phase 1A strict`
- Goal: improve closed-loop gripper stability and phase consistency without changing the core pi0.5 dual-stream joint-attention structure

## Phase 1A changes

- Add residual state conditioning to the suffix expert
- Add residual history conditioning to the suffix expert
- Replace continuous gripper diffusion with a binary gripper head
- Add a 4-way phase head: `approach / grasp / transport / release`
- Apply phase-aware gating to gripper logits
- Keep the existing RTC-compatible `denoise_step()` interface
- Add final decode infrastructure after Euler denoising

## Explicitly out of scope for this version

- Immediate correction enabled at train or eval time
- Local action frame
- Grounding auxiliary losses
- Drawer-specific subtask prediction
- Knowledge insulation pipeline changes
- Sparse MoE

## Primary config

- [`libero_train_qwen_3_5_2b_pi0_5_v2_phase1a_bs37_30k.yaml`](/C:/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla/config/libero_train_qwen_3_5_2b_pi0_5_v2_phase1a_bs37_30k.yaml)

## Naming

- Model remark: `Qwen35_pi0.5_v2`
- Training run name: `Qwen35_pi0.5_v2_phase1a_bs37_30k`
- Output directory: `./checkpoints/Qwen35_pi0.5_v2_phase1a_bs37_30k`
