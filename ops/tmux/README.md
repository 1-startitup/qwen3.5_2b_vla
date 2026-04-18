# Qwen35 pi0.5 Native tmux Project

This folder packages the workflow we actually used in native Linux for the
`qwen_3.5_2b_pi0.5` line:

- persistent project context for Codex/tmux
- native Linux training
- open-loop evaluation
- fixed-checker closed-loop standard eval
- fixed-checker 1-episode video eval
- agentview video rendering to Desktop
- live monitoring inside tmux

## Files

- `qwen35_pi05_native.env`
  Runtime variables for the current run, checkpoint, logs, and Desktop bundles.
- `PROJECT_CONTEXT.md`
  Persistent experiment memory: version timeline, logs, eval results, blockers,
  and batch-size tuning rules.
- `project_context.json`
  Machine-readable snapshot of the same context for future automation or Codex
  sessions.
- `show_project_context.sh`
  Prints the current project context inside tmux.
- `launch_qwen35_pi05_native_tmux.sh`
  Creates a tmux session with prepared windows.
- `run_qwen35_pi05_train.sh`
  Launches the current native Linux training config.
- `run_qwen35_pi05_open_loop.sh`
  Runs the existing `open_loop_train_eval.py` entrypoint on the target checkpoint.
- `run_qwen35_pi05_fixed_standard_eval.sh`
  Runs `eval_libero.py` with the fixed success checker on `libero_spatial`.
- `run_qwen35_pi05_fixed_spatial_video_eval.sh`
  Runs a 1-episode-per-task eval with `--save_frames`.
- `render_qwen35_pi05_fixed_agentview_bundle.sh`
  Converts saved frames into mp4s and exports a Desktop bundle.

## Basic usage in native Linux

```bash
cd /home/frankkkz/qwen35_2b_vla
bash ops/tmux/launch_qwen35_pi05_native_tmux.sh
```

This creates a tmux session with these windows:

- `control`
- `context`
- `train`
- `open-loop`
- `fixed-eval`
- `video-eval`
- `render`
- `monitor`

The windows are prefilled with the correct commands, but they do not start GPU
jobs automatically by default.

The `context` window prints the current experiment memory, including:

- the canonical pi0.5 run and checkpoint
- historical version deltas from `v1` through `qwen_3.5_2b_pi0.5`
- current best closed-loop success rates
- the batch-size frontier and OOM tuning playbook

Read that window before changing batch size or eval protocol.

## Autostart examples

Start the session and immediately launch training:

```bash
bash ops/tmux/launch_qwen35_pi05_native_tmux.sh --start train
```

Create the session and immediately run the fixed standard eval:

```bash
bash ops/tmux/launch_qwen35_pi05_native_tmux.sh --start fixed-standard
```

Recreate the session from scratch:

```bash
bash ops/tmux/launch_qwen35_pi05_native_tmux.sh --kill-existing
```

## What to edit first

If you move to a new checkpoint or run name, update:

- `RUN_NAME`
- `CHECKPOINT_STEP`
- `CHECKPOINT_DIR`

in `qwen35_pi05_native.env`.

If the best-known result or stable batch frontier changes, also update:

- `PROJECT_CONTEXT.md`
- `project_context.json`

## Training tuning policy

This project optimizes for the fastest stable training run under the time
budget, not for the highest wattage number. In practice:

- probe the largest stable micro-batch first
- watch `sample/s`, `s/step`, and `gpu_reserved`
- only add `grad_accum` after finding the stable micro-batch frontier
- expect to work close to the OOM boundary and back off one step after the
  first unstable point
