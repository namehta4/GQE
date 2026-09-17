#!/bin/bash
#SBATCH --job-name=h2-val-03-model
#SBATCH --account=<YOUR_NERSC_ACCOUNT>
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=h2-val-03-%j.out
#SBATCH --error=h2-val-03-%j.err

# ============================================================================
# H2 validation, stage 3: assemble training examples, pretrain a
# DELIBERATELY TINY Gemma model (this is a code-path check, not an accuracy
# benchmark -- H2's Hamiltonian vector and operator sequences are far too
# small to need the paper's 1024-hidden/16-layer configuration), then
# generate + evaluate on the held-out test split.
#
# Run ONLY after slurm_02_h2_chemistry.sbatch has completed.
#
# Usage: sbatch slurm_03_h2_model.sbatch <POOL_SIZE>
#   <POOL_SIZE> = the "Operator pool 'uccsd': N candidate operators" value
#                 printed in slurm_02's job log.
# ============================================================================
set -euo pipefail

POOL_SIZE="${1:?Usage: sbatch slurm_03_h2_model.sbatch POOL_SIZE (see the slurm_02 log)}"
IMAGE="neilmehta87/gqe:26.06"
WORKDIR_HOST="${PSCRATCH}/h2-validation"

module load podman-hpc 2>/dev/null || true
echo "Job ${SLURM_JOB_ID} on $(hostname), pool_size=${POOL_SIZE}"

srun -n 1 --gpus-per-node=1 podman-hpc run --rm --gpu \
    -v "${WORKDIR_HOST}:/workdir" \
    -w /workdir \
    -e POOL_SIZE="${POOL_SIZE}" \
    "${IMAGE}" \
    bash -c '
        set -euo pipefail

        echo "--- assembling training examples ---"
        python /opt/adapt-gqe/model/build_training_examples.py \
            --circuits /workdir/h2_adaptvqe/circuits.jsonl \
            --splits /workdir/h2_adaptvqe/splits.json \
            --hamiltonian-dir /workdir/h2_hamiltonians \
            --pool-size "${POOL_SIZE}" \
            --output-dir /workdir/h2_training_examples

        echo "--- pretraining (tiny model: hidden=64, 2 layers) ---"
        python /opt/adapt-gqe/model/pretrain.py \
            --train-jsonl /workdir/h2_training_examples/train.jsonl \
            --val-jsonl /workdir/h2_training_examples/val.jsonl \
            --tokenizer-dir /workdir/h2_training_examples \
            --hidden-size 64 --num-layers 2 --num-heads 2 --num-kv-heads 2 \
            --context-length 128 --sliding-window 128 --rope-theta 10000 \
            --encoder-hidden-dim 64 --encoder-depth 2 \
            --epochs 100 --batch-size 4 --lr 1e-3 \
            --eval-every-steps 5 --patience 20 \
            --output-dir /workdir/h2_checkpoints --device cuda

        echo "--- generating on the held-out test split ---"
        python /opt/adapt-gqe/model/generate.py \
            --checkpoint /workdir/h2_checkpoints/best_checkpoint.pt \
            --tokenizer-dir /workdir/h2_training_examples \
            --test-jsonl /workdir/h2_training_examples/test.jsonl \
            --num-candidates 16 --max-new-tokens 64 --top-p 0.95 \
            --output /workdir/h2_generated_pretrain.jsonl --device cuda

        echo "--- evaluating generated circuits ---"
        python /opt/adapt-gqe/chemistry/evaluate_generated_circuits.py \
            --generated /workdir/h2_generated_pretrain.jsonl \
            --xyz-dir /workdir/h2_bondscan \
            --basis sto-3g --charge 0 --spin 0 \
            --n-active-electrons 2 --n-active-orbitals 2 --pool uccsd \
            --output /workdir/h2_best_of_n_pretrain.jsonl
    '

echo "Check ${WORKDIR_HOST}/h2_best_of_n_pretrain.jsonl -- energy_error_mha"
echo "should be small (sub-mHa to a few mHa) given how easy H2/STO-3G is."
echo "If it's large, the bug is more likely in the model/tokenizer/generation"
echo "path than in chemistry (which slurm_01/02 already validated separately)."
