# V2 Post-GripperFix Policy Analysis

## New evals

- Full standard eval after gripper semantic fix:
  - `eval_results_libero_standard_checkpoint_30000_fixedload_gripperfix.json`
  - Result: `0.0%` average, `10/10` tasks failed.
- Two-task smoke eval with temporal ensemble + gripper hysteresis:
  - `eval_results_task0_task5_standard_temporal_gripperfix.json`
  - Result: `0/1` on task 0 and `0/1` on task 5.

## What changed after the gripper fix

- The deployed gripper command now preserves the dataset's signed convention `{-1, +1}`.
- The policy is no longer silently collapsing gripper commands to `{0, 1}`.
- Train-set comparisons show the model does learn the expert's gripper phase changes in open-loop.

## What the new standard eval shows

### 1. Strong downward bias

Across the first recorded episode for all 10 standard tasks:

- End-effector `z` drops by about `0.157m` to `0.276m`.
- This happens even when the task should first require lateral alignment before grasp.
- The policy behaves like it prefers "move down toward a plausible workspace height" before it has correctly aligned to the object.

### 2. Gripper phase is unstable in closed loop

Per-task switch counts in the first recorded episode:

- task 00: `12`
- task 01: `4`
- task 02: `11`
- task 03: `53`
- task 04: `2`
- task 05: `51`
- task 06: `38`
- task 07: `38`
- task 08: `24`
- task 09: `34`

This is not "the model never tries to close". It is closer to:

- early rollout: mostly stay open
- mid/late rollout: begin toggling between open and close
- toggling often happens after the arm has already drifted away from the correct approach pose

### 3. Closed-loop drift happens early

On task 0:

- rollout is already more than `0.1` away from the nearest expert pose by step `12`
- more than `0.2` away by step `31`
- by step `40`, nearest expert pose jumps from offset `0` to around offset `54`

So the policy leaves the local expert manifold early, then tries to recover with the wrong grasp phase later.

### 4. Language conditioning is weak

From `probe_model_vulnerabilities_20samples.json`:

- wrong instruction shift L2: about `0.0198`
- blank image shift L2: about `0.6603`
- zero state shift L2: about `0.0489`

Interpretation:

- visual input dominates
- state matters somewhat
- instruction has only a small effect on the action

## What the temporal-ensemble patch changed

On the two-task smoke eval:

- task 0 gripper switches dropped from `12` to `7`
- task 5 gripper switches dropped from `51` to `7`

So the patch does reduce chatter, but it does not fix the deeper issue:

- the arm is still approaching from the wrong pose family
- grasp activation is still too late or too misaligned

## Likely root causes

1. The model learned local imitation, not robust recovery.
2. The action head is too weakly grounded in language.
3. The policy lacks memory / phase control, so once it drifts, it cannot re-enter the correct approach-grasp-place sequence.
4. The benchmark rollout needs stronger online smoothing, but policy-side smoothing alone is not enough.

## Recommended modifications

### Immediate policy-side changes

1. Keep temporal ensembling in evaluation and deployment.
2. Keep gripper hysteresis / minimum hold to suppress chatter.
3. Add mild action smoothing for translation and rotation to reduce sudden drift.

### Training / data changes

1. Train with observation perturbations or DAgger-style recovery data.
2. Add short history input:
   - previous `k` proprio states
   - previous `k` actions
   - optionally previous visual tokens
3. Add rollout-consistency supervision, not only single-window imitation.

### Architecture changes

1. Add a dedicated text-conditioning path to the action head:
   - pooled instruction token
   - or text-only cross-attention
   - do not rely only on mixed VLM hidden states
2. Add an auxiliary task-consistency loss:
   - wrong instruction should produce a different action
   - or predict task ID / instruction embedding from action-head latent
3. Add a grasp-phase head:
   - approach / grasp / transport / release
   - use it to gate or regularize gripper transitions
4. Consider a recurrent or state-space policy head for closed-loop correction.

## Recommended order

1. Keep the gripper semantic fix.
2. Keep the eval-time temporal ensemble + gripper hysteresis.
3. Add history to the policy input.
4. Add explicit text conditioning + task-consistency auxiliary loss.
5. Add phase supervision for gripper control.
6. Retrain and compare standard first, then spatial.
