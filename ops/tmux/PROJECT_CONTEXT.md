# Qwen35 pi0.5 Experiment Context

This file is the persistent experiment memory for the `qwen35_2b_vla` project.
Use it before changing configs, re-running training, or comparing evals.

## Current canonical line

- Model line: `qwen_3.5_2b_pi0.5`
- Native repo root: `/home/frankkkz/qwen35_2b_vla`
- Canonical train config:
  `/home/frankkkz/qwen35_2b_vla/config/libero_train_qwen_3_5_2b_pi0_5_lora_r16_bs32_ga2_30k_native.yaml`
- Canonical train log:
  `/home/frankkkz/qwen35_2b_vla/train_qwen35_pi05_lora_r16_bs32_ga2_30k_native.log`
- Canonical checkpoint:
  `/home/frankkkz/qwen35_2b_vla/checkpoints/qwen_3.5_2b_pi0.5_lora_r16_bs32_ga2_30k_native/checkpoint-30000`

## Best known results for the canonical line

- Final train step: `30000`
- Final train loss at `step 30000`: `0.0017`
- Final logged eval MSE: `0.000331`
- 50-batch open-loop analysis:
  - overall MSE: `0.000412`
  - overall MAE: `0.008665`
  - gripper sign accuracy: `99.97%`
  - source:
    `/mnt/c/Users/frank/OneDrive/Desktop/qwen35_pi05_open_closed_loop_analysis_checkpoint_30000/open_loop_analysis.json`
- Native training wall time:
  - `1141.8 min`
  - about `19.0 hours`

## Closed-loop protocol results

Important: old `0%` reports were partially caused by an eval bug. The fixed
checker must respect `done == True` even when `info["success"] == False`.

- Fixed checker, `libero_spatial`, full `10 x 10`:
  - average success: `12%`
  - source:
    `/home/frankkkz/qwen35_2b_vla/tmp/eval_results_libero_spatial_checkpoint_30000_native_fixed_full.json`
- Fixed checker + RTC `execution_horizon=10`:
  - average success: `14%`
  - source:
    `/home/frankkkz/qwen35_2b_vla/tmp/eval_results_libero_spatial_checkpoint_30000_rtc_eh10.json`
- Fixed checker + adaptive high-score protocol:
  - average success: `21%`
  - source:
    `/mnt/c/Users/frank/OneDrive/Desktop/qwen35_pi05_open_closed_loop_analysis_checkpoint_30000/eval_results_libero_spatial_checkpoint_30000_adaptive_highscore.json`

Representative task strengths under the best current protocol:

- strong: task `02` = `60%`, task `01` = `50%`, task `05` = `50%`
- partial: task `03` = `20%`, task `08` = `20%`, task `00` = `10%`
- still failing: tasks `04`, `06`, `07`, `09` = `0%`

Useful sanity check:

- task `00`, episode `00`, checkpoint `30000` is a real success after the eval
  fix
- it finishes in `146` steps
- source:
  `/home/frankkkz/qwen35_2b_vla/tmp/eval_results_task0_30k_native_fixed.json`

## Biggest current issues

1. Open-loop is very strong, but closed-loop still depends heavily on runtime
   protocol.
2. The hardest failures are still drawer, stove, and cabinet variants:
   tasks `04`, `06`, `07`, `09`.
3. Adaptive replanning helps more than the base fixed protocol, which means the
   policy still needs help with online recovery and phase stability.
4. `RTC` exists in the pi0.5 path, but the canonical training config keeps it
   disabled by default, so eval protocol choice matters a lot.
5. Do not compare runs using the old success checker.

## Version timeline

| Version | Main changes | Best recorded result | Main blocker |
| --- | --- | --- | --- |
| `v1` action-only | First `layerwise_fm` action head, action-only design, no short history, no explicit state prompt, no explicit text grounding | Partial closed-loop file on tasks `00-02`: `0%` average over `3` episodes each. Source: `/home/frankkkz/qwen35_2b_vla/eval_results_closedloop_v1.json` | Action head too weak, little recovery ability, no strong task/phase grounding |
| `v2` layerwise FM | Slimmer layerwise FM head, `bs128 ga1`, DeepSpeed training, faster open-loop convergence | Final logged eval MSE reached `0.000384` by `28k`; full train completed `30000` steps in `1300.2 min`. Post-gripper-fix standard eval still `0%`. Sources: `/home/frankkkz/qwen35_2b_vla/train_layerwise_v2_bs128_ga1_deepspeed_train_30k_bs128fresh.log`, `/mnt/c/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla/POST_GRIPPERFIX_POLICY_ANALYSIS.md` | Downward bias, early closed-loop drift, gripper chatter, weak language effect |
| `v3` state-prompt/history | Added short history, text conditioning, state prompt, gripper split and consistency loss | Stored smoke evals on tasks `00` and `05` are still `0%`. Sources: `/home/frankkkz/qwen35_2b_vla/eval_results_v3_smoke1000_stateprompt_task0_task5_standard.json`, `/home/frankkkz/qwen35_2b_vla/eval_results_v3_smoke1000_stateprompt_task5_only_standard.json` | Better grounding than v2, but still not enough recovery or stable grasp execution |
| `v4` local-frame/phase | Added local action frame, phase loss, immediate loss, dynamic loss balance, correction blend | Stored smoke evals on tasks `00` and `05` remain `0%`. Sources: `/home/frankkkz/qwen35_2b_vla/eval_results_v4_smoke1000_task0_task5_standard.json`, `/home/frankkkz/qwen35_2b_vla/eval_results_v41_smoke1000_local_task0_task5_standard.json` | Architecture got richer, but closed-loop stayed brittle and task success did not move |
| `qwen_3.5_2b_pi0.5` | pi0.5-style chunked flow matching, state prompt, prefix cache, suffix expert, AdaRMS, RTC scaffolding, fixed success checker | `12%` fixed checker, `14%` fixed checker + RTC `eh10`, `21%` adaptive high-score. Open-loop MSE `4.12e-4` | Runtime protocol still matters, and hard tasks need better recovery/replanning |

## Batch size and OOM tuning rules

Training priority for this project:

1. maximize real throughput under the time budget
2. keep the GPU close to its stable memory limit
3. only after that increase effective batch with grad accumulation

Do not optimize for raw wattage alone. This model often becomes
memory-bound or kernel-selection-bound before it reaches the highest power draw.
Use `sample/s`, `s/step`, and crash boundary, not just `power.draw`.

### Known frontier on this hardware

Hardware context:

- single `RTX 5090 32GB`
- native Linux training is required for serious runs

Known probe outcomes:

- `ga1` regime:
  - `bs37` was the highest stable fast point in earlier probes
  - `bs38` failed
  - `bs40`, `bs48`, and `bs64` were slower even when they ran
- `ga2` regime:
  - `bs32` is the current canonical stable run
  - `bs33` ran but was slower
  - `bs34` and `bs35` failed with `CUDA driver error: device not ready`
  - `bs37 ga2 from0` also failed with `CUDA driver error: device not ready`
    during training

### Tuning procedure

1. Always probe in `/home/frankkkz/qwen35_2b_vla`, not under `/mnt/c`.
2. Keep model, attention implementation, precision, checkpointing, and eval/save
   cadence identical to the intended real run.
3. Increase micro-batch size first.
4. Run a short probe of about `6-10` optimizer steps.
5. Record:
   - `s/step`
   - `sample/s`
   - `gpu_reserved`
   - whether the run crashes with OOM or `device not ready`
6. Back off one notch from the first unstable point.
7. Only then consider `gradient_accumulation_steps` if you still need more
   effective batch.
8. Treat “larger batch but slower step time” as a regression.

### Canonical training recommendation

When the goal is “fastest stable run with high GPU load”:

- start from:
  `/home/frankkkz/qwen35_2b_vla/config/libero_train_qwen_3_5_2b_pi0_5_lora_r16_bs32_ga2_30k_native.yaml`
- keep:
  - `save_every: 2000`
  - `eval_every: 2000`
  - optimizer-state checkpoints enabled
- only move away from `bs32 ga2` after a fresh probe

## Eval protocol rules

1. Always use the fixed success checker.
2. When quoting a success rate, also state the protocol:
   - fixed checker full
   - fixed checker + RTC
   - adaptive high-score
3. Keep open-loop and closed-loop results side by side. High offline accuracy
   does not guarantee good rollout success.
4. Save agent-view frames for representative failures and successes.
5. For task diagnosis, focus first on tasks `04`, `06`, `07`, `09`.

## Reference files

- Historical policy analysis:
  `/mnt/c/Users/frank/OneDrive/Documents/VLA_LinearA/qwen35_2b_vla/POST_GRIPPERFIX_POLICY_ANALYSIS.md`
- Desktop analysis bundle:
  `/mnt/c/Users/frank/OneDrive/Desktop/qwen35_pi05_open_closed_loop_analysis_checkpoint_30000`
- Native tmux environment:
  `/home/frankkkz/qwen35_2b_vla/ops/tmux/qwen35_pi05_native.env`

## Update checklist

After every meaningful run, update this file and `project_context.json` with:

- config path
- train log path
- checkpoint path
- final eval MSE
- main closed-loop protocol numbers
- biggest regression or improvement
- latest stable batch frontier
