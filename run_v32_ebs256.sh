#!/bin/bash
# ============================================================================
# v3.2 GR00T Bridge — Effective BS=256 Training + Probe + Eval Pipeline
#
# Tmux session: v32_ebs256
#   Window 0 (train):  Sequential training of all 3 tap strategies
#   Window 1 (probe):  Probe tests on best checkpoint (auto after training)
#   Window 2 (eval):   Closed-loop LIBERO-Spatial eval (auto after probe)
#
# Usage:
#   bash run_v32_ebs256.sh [all_concat|last|scalar_mix|all]
#
# Default: runs all_concat only (strongest smoke baseline).
# "all" runs all 3 sequentially.
# ============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

SESSION="v32_ebs256"
STRATEGY="${1:-all_concat}"

CONFIG_ALLCONCAT="config/libero_train_qwen_3_5_2b_gemma_bridge_v3_2_ebs256_30k.yaml"
CONFIG_LAST="config/libero_train_qwen_3_5_2b_gemma_bridge_v3_2_ebs256_30k_last.yaml"
CONFIG_SCALAR="config/libero_train_qwen_3_5_2b_gemma_bridge_v3_2_ebs256_30k_scalar_mix.yaml"

CKPT_ALLCONCAT="./checkpoints/v3.2_ebs256_30k_allconcat"
CKPT_LAST="./checkpoints/v3.2_ebs256_30k_last"
CKPT_SCALAR="./checkpoints/v3.2_ebs256_30k_scalarmix"

# ---------------------------------------------------------------------------
# Helper: build the training command for a given config
# ---------------------------------------------------------------------------
train_cmd() {
    local cfg="$1"
    echo "python train.py --config ${cfg}"
}

# ---------------------------------------------------------------------------
# Helper: probe command — runs tools/test_qwen35_gemma_bridge_v32.py
# ---------------------------------------------------------------------------
probe_cmd() {
    local cfg="$1"
    echo "python tools/test_qwen35_gemma_bridge_v32.py --config ${cfg} --device cuda"
}

# ---------------------------------------------------------------------------
# Helper: closed-loop eval on LIBERO-Spatial (all 10 tasks, 10 episodes each)
# ---------------------------------------------------------------------------
eval_cmd() {
    local ckpt_dir="$1"
    local tag="$2"
    # Find the last checkpoint
    local last_ckpt
    last_ckpt="\$(ls -d ${ckpt_dir}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
    echo "python eval_libero.py \\
      --checkpoint_dir \${last_ckpt:-${ckpt_dir}/checkpoint-30000} \\
      --suites libero_spatial \\
      --n_episodes 10 \\
      --max_steps 300 \\
      --deterministic_seed 0 \\
      --output ${ckpt_dir}/eval_libero_spatial_ebs256_${tag}.json \\
      --rollout_dir ${ckpt_dir}/rollouts_${tag} \\
      --record_tasks 2 --record_episodes 2 --save_frames"
}

# ---------------------------------------------------------------------------
# Build the pipeline script for a single strategy
# ---------------------------------------------------------------------------
build_single_pipeline() {
    local strategy="$1"
    local cfg ckpt_dir
    case "$strategy" in
        all_concat) cfg="$CONFIG_ALLCONCAT"; ckpt_dir="$CKPT_ALLCONCAT" ;;
        last)       cfg="$CONFIG_LAST";      ckpt_dir="$CKPT_LAST" ;;
        scalar_mix) cfg="$CONFIG_SCALAR";    ckpt_dir="$CKPT_SCALAR" ;;
        *) echo "Unknown strategy: $strategy"; exit 1 ;;
    esac

    cat <<SCRIPT
#!/bin/bash
set -euo pipefail
cd "${REPO_DIR}"

echo "========================================"
echo "  v3.2 GR00T Bridge — ${strategy}"
echo "  Effective batch size: 256 (bs=32 x ga=8)"
echo "  Config: ${cfg}"
echo "========================================"

# Phase 1: Train
echo "[Phase 1/3] Training ${strategy}..."
$(train_cmd "$cfg")
echo "[Phase 1/3] Training ${strategy} DONE"

# Phase 2: Probe
echo "[Phase 2/3] Running probe tests..."
$(probe_cmd "$cfg")
echo "[Phase 2/3] Probe DONE"

# Phase 3: Closed-loop eval
echo "[Phase 3/3] Running closed-loop LIBERO-Spatial eval..."
LAST_CKPT=\$(ls -d ${ckpt_dir}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "\$LAST_CKPT" ]; then
    echo "ERROR: No checkpoint found in ${ckpt_dir}"
    exit 1
fi
echo "  Using checkpoint: \$LAST_CKPT"
python eval_libero.py \\
    --checkpoint_dir "\$LAST_CKPT" \\
    --suites libero_spatial \\
    --n_episodes 10 \\
    --max_steps 300 \\
    --deterministic_seed 0 \\
    --output "${ckpt_dir}/eval_libero_spatial_ebs256_${strategy}.json" \\
    --rollout_dir "${ckpt_dir}/rollouts_${strategy}" \\
    --record_tasks 2 --record_episodes 2 --save_frames
echo "[Phase 3/3] Eval DONE"

echo ""
echo "========================================"
echo "  ${strategy} pipeline complete!"
echo "  Results: ${ckpt_dir}/eval_libero_spatial_ebs256_${strategy}.json"
echo "========================================"
SCRIPT
}

# ---------------------------------------------------------------------------
# Main: set up tmux session
# ---------------------------------------------------------------------------

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

if [ "$STRATEGY" = "all" ]; then
    # Sequential: all_concat -> last -> scalar_mix in one window
    COMBINED_SCRIPT=$(mktemp /tmp/v32_ebs256_all_XXXXXX.sh)
    {
        echo "#!/bin/bash"
        echo "set -euo pipefail"
        echo ""
        build_single_pipeline "all_concat"
        echo ""
        echo "echo ''"
        echo "echo '=== Moving to next strategy: last ==='"
        echo "echo ''"
        echo ""
        build_single_pipeline "last"
        echo ""
        echo "echo ''"
        echo "echo '=== Moving to next strategy: scalar_mix ==='"
        echo "echo ''"
        echo ""
        build_single_pipeline "scalar_mix"
        echo ""
        echo "echo '========================================'"
        echo "echo '  ALL 3 STRATEGIES COMPLETE'"
        echo "echo '========================================'"
    } > "$COMBINED_SCRIPT"
    chmod +x "$COMBINED_SCRIPT"

    tmux new-session -d -s "$SESSION" -n "train" "bash $COMBINED_SCRIPT; exec bash"
    echo "Tmux session '$SESSION' started: all 3 strategies sequentially"
    echo "  Attach: tmux attach -t $SESSION"
else
    # Single strategy
    SINGLE_SCRIPT=$(mktemp /tmp/v32_ebs256_${STRATEGY}_XXXXXX.sh)
    build_single_pipeline "$STRATEGY" > "$SINGLE_SCRIPT"
    chmod +x "$SINGLE_SCRIPT"

    tmux new-session -d -s "$SESSION" -n "train" "bash $SINGLE_SCRIPT; exec bash"
    echo "Tmux session '$SESSION' started: ${STRATEGY}"
    echo "  Attach: tmux attach -t $SESSION"
fi

echo ""
echo "Recipe changes vs previous 30k configs:"
echo "  - Effective batch size: 64 -> 256 (ga=2 -> ga=8)"
echo "  - Warmup steps: 1000 -> 2000 (scaled with batch)"
echo "  - Save every: 2000 -> 5000 (fewer checkpoints)"
echo ""
echo "Code fixes applied:"
echo "  - Weight decay exemption for scalar_mixer params (tap_logits, gamma)"
echo "  - Wandb/terminal: scalar_mix metrics only logged when bridge_scalar_mix=true"
