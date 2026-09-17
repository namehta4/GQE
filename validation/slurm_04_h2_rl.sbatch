#!/bin/bash
#SBATCH --job-name=h2-val-04-rl
#SBATCH --account=<YOUR_NERSC_ACCOUNT>
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=h2-val-04-%j.out
#SBATCH --error=h2-val-04-%j.err

# ============================================================================
# H2 validation, stage 4: GRPO/DAPO RL post-training on the pretrained tiny
# model, then re-evaluate and compare against the pretraining-only baseline
# from stage 3. This is the first real exercise of chemistry/circuit_energy.py
# being called repeatedly under load (every rollout needs an energy
# evaluation) rather than the handful of calls in test_circuit_energy.py.
#
# Given H2/STO-3G UCCSD's action space is essentially 1 operator, don't
# expect a dramatic improvement here -- the point is confirming the RL loop
# (rollout generation, reward computation, advantage normalization, DAPO
# loss, checkpointing) runs correctly end-to-end without crashing or
# producing NaNs, not achieving a large accuracy gain on a system that's
# already near-trivial to solve exactly.
#
# Run ONLY after slurm_03_h2_model.sbatch has completed.
#
# Usage: sbatch slurm_04_h2_rl.sbatch
# ============================================================================
set -euo pipefail

IMAGE="neilmehta87/gqe:26.06"
WORKDIR_HOST="${PSCRATCH}/h2-validation"

module load podman-hpc 2>/dev/null || true
echo "Job ${SLURM_JOB_ID} on $(hostname)"

srun -n 1 --gpus-per-node=1 podman-hpc run --rm --gpu \
    -v "${WORKDIR_HOST}:/workdir" \
    -w /workdir \
    "${IMAGE}" \
    bash -c '
        set -euo pipefail

        echo "--- RL post-training (short run: 50 steps) ---"
        python /opt/adapt-gqe/model/rl_grpo.py \
            --checkpoint /workdir/h2_checkpoints/best_checkpoint.pt \
            --tokenizer-dir /workdir/h2_training_examples \
            --train-jsonl /workdir/h2_training_examples/train.jsonl \
            --xyz-dir /workdir/h2_bondscan \
            --basis sto-3g --charge 0 --spin 0 \
            --n-active-electrons 2 --n-active-orbitals 2 --pool uccsd \
            --group-size 16 --conformers-per-step 4 --epochs-per-batch 2 \
            --max-new-tokens 64 --lr 1e-4 --beta 0.1 \
            --num-steps 50 --save-every-steps 10 \
            --output-dir /workdir/h2_rl_checkpoints \
            --rollout-log /workdir/h2_rl_checkpoints/rollouts.jsonl \
            --device cuda

        LATEST_CKPT=$(ls -t /workdir/h2_rl_checkpoints/rl_checkpoint_step*.pt | head -n1)
        echo "--- generating with RL checkpoint: ${LATEST_CKPT} ---"
        python /opt/adapt-gqe/model/generate.py \
            --checkpoint "${LATEST_CKPT}" \
            --tokenizer-dir /workdir/h2_training_examples \
            --test-jsonl /workdir/h2_training_examples/test.jsonl \
            --num-candidates 16 --max-new-tokens 64 --top-p 0.95 \
            --output /workdir/h2_generated_rl.jsonl --device cuda

        echo "--- evaluating RL-generated circuits ---"
        python /opt/adapt-gqe/chemistry/evaluate_generated_circuits.py \
            --generated /workdir/h2_generated_rl.jsonl \
            --xyz-dir /workdir/h2_bondscan \
            --basis sto-3g --charge 0 --spin 0 \
            --n-active-electrons 2 --n-active-orbitals 2 --pool uccsd \
            --output /workdir/h2_best_of_n_rl.jsonl

        echo "--- comparing pretrain vs. RL ---"
        python /opt/adapt-gqe/evaluation/compute_metrics.py \
            --results "pretrain:/workdir/h2_best_of_n_pretrain.jsonl" \
            --results "rl:/workdir/h2_best_of_n_rl.jsonl" \
            --epsilon-mha 0.1 \
            --output /workdir/h2_summary.csv \
            --output-json /workdir/h2_distributions.json
    '

echo "Check ${WORKDIR_HOST}/h2_summary.csv for the pretrain-vs-RL comparison."
echo "This completes the H2 Tier-1 validation ladder."
