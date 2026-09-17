#!/usr/bin/env bash
# Orchestrates the RL + self-distillation loop (Fig. 3b of ADAPT-GQE,
# arXiv:2607.22468): each round runs GRPO/DAPO RL post-training, filters
# the accumulated rollout database down to the round's top-k lowest-energy
# sequences per conformer, then continues SFT on that filtered set --
# repeating with progressively stricter selection criteria.
#
# Hyperparameter schedule below matches Appendix Table 7's 12-qubit column
# (3 rounds): RL epochs [2,3,4], RL learning rate [4,3,2]e-6, distillation
# threshold [5,3.4,1.6]e-3 Ha, top-n [8,4,2]. Substitute the 14-/16-qubit
# columns' 5-round schedules for those active spaces.
#
# Usage:
#   ./run_self_distillation_rounds.sh <base_checkpoint> <tokenizer_dir> \
#       <train_jsonl> <val_jsonl> <xyz_dir> <hamiltonian_dir> \
#       <hamiltonian_scale> <circuits_jsonl> <basis> <charge> <spin> \
#       <n_active_electrons> <n_active_orbitals> <pool> <output_root>

set -euo pipefail

BASE_CKPT="${1:?}"
TOKENIZER_DIR="${2:?}"
TRAIN_JSONL="${3:?}"
VAL_JSONL="${4:?}"
XYZ_DIR="${5:?}"
HAM_DIR="${6:?}"
HAM_SCALE="${7:?}"
CIRCUITS_JSONL="${8:?}"
BASIS="${9:?}"
CHARGE="${10:?}"
SPIN="${11:?}"
N_ACTIVE_E="${12:?}"
N_ACTIVE_O="${13:?}"
POOL="${14:?}"
OUTPUT_ROOT="${15:?}"

# Per-round schedules (12-qubit column, Table 7) -- edit for your active space.
RL_EPOCHS=(2 3 4)
RL_LR=(4e-6 3e-6 2e-6)
DISTILL_THRESHOLD_MHA=(5.0 3.4 1.6)
DISTILL_TOPN=(8 4 2)
DISTILL_LR=4e-5
DISTILL_EPOCHS=5

mkdir -p "${OUTPUT_ROOT}"
CURRENT_CKPT="${BASE_CKPT}"
ROLLOUT_LOGS=()

for i in "${!RL_EPOCHS[@]}"; do
    round=$((i + 1))
    round_dir="${OUTPUT_ROOT}/round${round}"
    mkdir -p "${round_dir}"
    echo "=== Round ${round}: RL post-training ==="

    rollout_log="${round_dir}/rollouts.jsonl"
    ROLLOUT_LOGS+=("${rollout_log}")

    python rl_grpo.py \
        --checkpoint "${CURRENT_CKPT}" \
        --tokenizer-dir "${TOKENIZER_DIR}" \
        --train-jsonl "${TRAIN_JSONL}" \
        --xyz-dir "${XYZ_DIR}" \
        --basis "${BASIS}" --charge "${CHARGE}" --spin "${SPIN}" \
        --n-active-electrons "${N_ACTIVE_E}" --n-active-orbitals "${N_ACTIVE_O}" \
        --pool "${POOL}" \
        --epochs-per-batch "${RL_EPOCHS[$i]}" \
        --lr "${RL_LR[$i]}" \
        --output-dir "${round_dir}/rl_checkpoints" \
        --rollout-log "${rollout_log}"

    latest_rl_ckpt=$(ls -t "${round_dir}/rl_checkpoints"/rl_checkpoint_step*.pt | head -n1)

    echo "=== Round ${round}: building distillation dataset ==="
    rollout_log_args=()
    for log in "${ROLLOUT_LOGS[@]}"; do
        rollout_log_args+=(--rollout-log "${log}")
    done

    python build_distillation_dataset.py \
        --circuits-jsonl "${CIRCUITS_JSONL}" \
        "${rollout_log_args[@]}" \
        --hamiltonian-dir "${HAM_DIR}" \
        --hamiltonian-scale "${HAM_SCALE}" \
        --tokenizer-dir "${TOKENIZER_DIR}" \
        --top-n "${DISTILL_TOPN[$i]}" \
        --energy-threshold-mha "${DISTILL_THRESHOLD_MHA[$i]}" \
        --output-dir "${round_dir}/distill_data"

    echo "=== Round ${round}: self-distillation SFT ==="
    python distill_sft.py \
        --checkpoint "${latest_rl_ckpt}" \
        --tokenizer-dir "${TOKENIZER_DIR}" \
        --train-jsonl "${round_dir}/distill_data/train.jsonl" \
        --val-jsonl "${VAL_JSONL}" \
        --epochs "${DISTILL_EPOCHS}" \
        --lr "${DISTILL_LR}" \
        --output-dir "${round_dir}/distill_checkpoint"

    CURRENT_CKPT="${round_dir}/distill_checkpoint/best_checkpoint.pt"
    echo "Round ${round} complete. Checkpoint: ${CURRENT_CKPT}"
done

echo "All rounds complete. Final checkpoint: ${CURRENT_CKPT}"
