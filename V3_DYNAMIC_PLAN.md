# V3 Dynamic Plan

## Goal

Push the Qwen pi0.5-style line forward with pi0.5-aligned, low-risk increments:

- one module delta per version
- official and tuned eval paths kept separate
- every next step chosen from the previous version's failure pattern

This plan treats the currently evaluated `qwen_3.5_2b_pi0.5_v2_lora_r16_bs32_ga2_30k_native`
checkpoint as `v2-old`, not as the later strict Phase 1A blueprint.

## Hard Principles

1. Single-module principle
- Each version changes exactly one mechanism relative to the previous version.
- If a version regresses, rollback means "go back one version", not "debug a pile of coupled edits".

2. Dynamic gating principle
- Versions are not unconditional.
- After each official full eval, choose the next module from the dominant remaining bucket.

3. Official vs tuned separation
- `eval_libero.py --mode official` is the benchmark-aligned path.
- `eval_libero.py --mode tuned` keeps replan / EMA / hysteresis / video diagnostics.
- Tuned results are diagnostic or model-specific only, never mixed into official benchmark numbers.

## Current Evidence Snapshot

Official reference numbers:

- `v1 ckpt-30000`: overall `12.0%`
- `v2-old ckpt-30000`: overall `11.0%`

Current bucket map:

- Bucket A: `task01`, `task03`, `task08`
  - refresh-sensitive and/or gripper-timing-sensitive
- Bucket B: `task00`, `task05`
  - long-plan continuity / phase-transition sensitive
- Bucket C: `task04`, `task06`, `task07`, `task09`
  - grounding / affordance / drawer-subtask dominated

Key diagnostics already established:

- `task01`: official `0/10`, tuned `8/10`
- `task03`: baseline `0/10`, `replan10 0/10`, tuned `2/10`
- `task08`: baseline `0/10`, `replan5 2/10`, `replan10 3/10`, `replan20 0/10`
- `task05`: strongly harmed by replanning, so it is not a simple chatter task
- `task04/06/07/09`: neither replan nor tuned runtime rescue them in smoke evals

## Version Roadmap

| Version | Single delta vs previous | Main target | Cost | Default trigger |
| --- | --- | --- | --- | --- |
| `v3.0` | No training. Official/tuned eval split. Re-run official baselines. | Reference band | 0 | Always |
| `v3.1` | BCE gripper split: `6D motion + 1D gripper logit` | Bucket A soft-gripper failures | 1 train | Always |
| `v3.2` | LoRA rank `16 -> 64`, same targets, no vision unfreeze | Remaining Bucket C plus some Bucket A | 1 train | If C is still dominant after `v3.1` |
| `v3.3` | Vision adapter plus last 2 vision blocks at small LR | Hard grounding drift in `06/07/09` | 1 train | If `v3.2` leaves C nearly flat |
| `v3.4` | Light phase head plus phase-aware gripper gating | Bucket B plus residual `03/08` stage-switch issues | 1 train + labels | If `00/05` or `03/08` still lag diagnostic references |
| `v3.5` | Local EEF-frame action representation | Elevated-support geometry in `07/09` | 1 train | If videos still show world-frame geometric drift |
| `v3.6` | Grounding auxiliary bbox head | Residual Bucket C failures | 1 train + annotation infra | Only after `v3.3/5` still leave pre-contact mislocalization |

## Exact Scope Per Version

### v3.0

- Keep `eval_libero.py --mode official` locked to:
  - `chunk_size=50`
  - `num_inference_steps=10`
  - `temporal_ensemble_decay=0`
  - `replan_horizon=None`
  - `action_ema_alpha=0`
  - `gripper_hysteresis_steps=0`
  - `gripper_min_hold_steps=0`
- Keep `--mode tuned` for diagnostic packs and model-specific evals.
- Reference band is:
  - `v1 ckpt-30000 official full`
  - `v2-old ckpt-30000 official full`

### v3.1

Only delta:

- replace 7D regression with:
  - `motion_pred[0:6]`
  - `gripper_logit`
- training loss:
  - `motion_loss = MSE(motion_pred, motion_target)`
  - `gripper_loss = BCEWithLogits(gripper_logit, binary_gripper_target)`
  - `total = motion_loss + lambda_gripper * gripper_loss`

Do not add:

- phase head
- history encoder
- state residual
- local frame
- grounding aux

Acceptance focus:

- `task01/task03` official and tuned behavior
- video check for weaker "soft grasp / regrasp loop"

### v3.2

Only delta:

- LoRA rank `r=16 -> r=64`
- keep target modules unchanged
- do not unfreeze vision blocks yet

Purpose:

- test whether current grounding weakness is mainly VLM-side capacity bottleneck

Decision:

- if Bucket C moves clearly, keep it
- if Bucket C stays mostly flat, escalate to `v3.3`

### v3.3

Only delta:

- unfreeze projector at normal VLM LR
- unfreeze last 2 vision transformer blocks at `0.1 x vlm_lr`
- keep the rest of vision frozen

Purpose:

- minimal-risk grounding improvement before any full vision unlock

Acceptance focus:

- `task06/07/09`
- video evidence that failures move from "cannot localize target" to later-stage control failures

### v3.4

Only delta:

- add a 4-way phase head:
  - `approach`
  - `grasp`
  - `transport`
  - `release`
- add phase-aware gripper gating in inference
- train with auxiliary CE phase loss

Purpose:

- explicit stage modeling for:
  - `task00/task05` continuity
  - residual `task03/task08` stage-switch errors

Trigger:

- do this only if official `00/05` or replan-sensitive `03/08` still trail their diagnostic references

### v3.5

Only delta:

- train action deltas in local EEF frame
- convert back to world frame at env step time

Purpose:

- reduce geometry brittleness on elevated targets and support surfaces

Primary tasks:

- `task07`
- `task09`

### v3.6

Only delta:

- add a lightweight grounding auxiliary head
- predict target bbox or target localization proxy from pooled vision features

Purpose:

- backstop for residual pre-contact mislocalization after `v3.3/v3.5`

This is intentionally last because it adds the most infra and the most debugging surface.

## Dynamic Decision Gates

After every version:

1. Run official full `10 tasks x 10 eps`.
2. Run targeted diagnostics:
- Bucket A pack:
  - official on `01/03/08`
  - tuned or replan references where relevant
- Bucket B pack:
  - official on `00/05`
  - replan diagnostic only as analysis
- Bucket C pack:
  - official on `04/06/07/09`
  - 2 to 3 recorded videos for the weakest tasks
3. Generate `reports/<version>_vs_<prev>.md`.
4. Decide:
- `accept` if official overall rises and the targeted bucket improves materially
- `rollback` if official falls and the targeted bucket does not improve
- `ambiguous` if overall is flat but the targeted bucket improves and another bucket regresses

Default routing:

- if Bucket A remains dominant after `v3.1`: stay on gripper/timing line only if BCE did not fire
- if Bucket C remains dominant after `v3.2`: go to `v3.3`
- if `00/05` or `03/08` still lag replan/tuned references after `v3.2/v3.3`: go to `v3.4`
- if `07/09` still show geometric drift after `v3.4`: go to `v3.5`
- if Bucket C still fails before contact after `v3.3/v3.5`: go to `v3.6`

## Dynamic Eval Pack

For every accepted version, keep these eval outputs:

- `official_full_ep10.json`
- `bucketA_diag.json`
- `bucketB_diag.json`
- `bucketC_diag.json`
- recorded video bundle for the worst Bucket C task

This gives one stable comparison slice per version and one diagnosis slice per version.

## Claude / Codex Collaboration

What is automatable now:

- `claude` CLI is available in WSL and can run non-interactive reviews with JSON Schema output.
- `tmux` is available in WSL.
- headless `codex` can be called from WSL via the absolute binary path:
  - `/mnt/c/Users/frank/.codex/.sandbox-bin/codex.exe`

Important caveat:

- `codex` is a Windows process, so any `--cd`, `--output-schema`, or `-o` path passed to it must use Windows-visible paths.
- Shared pipeline artifacts should therefore live under:
  - `/mnt/c/Users/frank/vla_pipeline`
  - equivalently `C:\Users\frank\vla_pipeline`

Recommended architecture:

1. WSL training and eval write official outputs into the shared iteration directory.
2. Claude review reads the diff report and emits schema-checked JSON.
3. Codex ping probes headless availability before the review step.
4. Codex review reads the same diff report and emits schema-checked JSON.
5. Claude merger produces a consensus JSON from both reviews.
6. Human approval is the final gate, represented by an `APPROVED` marker file.

Fallback policy:

- If codex ping fails or codex review times out, the pipeline falls back to Claude-only consensus plus a manual Codex desktop review if needed.
- If Codex headless is reachable but hits account usage limits, classify that separately from timeout or binary failures and continue with Claude-only consensus for that iteration.

## Machine-Readable Roadmap

The canonical machine-readable version map lives at:

- `ops/pipeline/v3_versions.yaml`

It encodes, for each version:

- the single allowed delta
- the target bucket
- explicit "do not add" exclusions where needed
- acceptance focus
- rollback conditions
- next-version routing rules

Both Claude and Codex pipeline reviews should treat this manifest as the source of truth for dynamic routing, with the markdown plan serving as the human-readable explanation.

## Immediate Next Steps

1. Keep `v3.0` artifacts as the reference band.
2. Prepare `v3.1` as the smallest training delta:
- BCE gripper split only
3. Keep `v3.2` ready as the next config-only delta:
- `r=64`, same target modules
4. Do not schedule `v3.3+` until `v3.1` and `v3.2` official deltas are in hand.
