#!/bin/bash
#SBATCH --job-name=h2-val-01-circuit-energy
#SBATCH --account=<YOUR_NERSC_ACCOUNT>
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=h2-val-01-%j.out
#SBATCH --error=h2-val-01-%j.err

# ============================================================================
# H2 validation, stage 1: chemistry/test_circuit_energy.py -- the mandatory
# gate before trusting anything in chemistry/circuit_energy.py (RL reward,
# best-of-N selection, everything downstream). Run this FIRST; do not
# proceed to stage 2 until it passes.
#
# This stage is CPU-bound (cudaq.set_target() for GPU statevector sim is not
# wired in yet -- see the earlier "how much can run on GPU" discussion), so
# --gpus-per-node=1 here is only to keep the environment consistent with
# later stages, not because this stage needs it. Feel free to submit to a
# CPU-only partition instead if you'd rather save GPU allocation hours.
#
# Usage: sbatch slurm_01_test_circuit_energy.sbatch
# ============================================================================
set -euo pipefail

IMAGE="neilmehta87/gqe:26.06"
WORKDIR_HOST="${PSCRATCH}/h2-validation"
mkdir -p "${WORKDIR_HOST}"

module load podman-hpc 2>/dev/null || true

echo "Job ${SLURM_JOB_ID} on $(hostname)"
srun -n 1 --gpus-per-node=1 podman-hpc run --rm --gpu \
    -v "${WORKDIR_HOST}:/workdir" \
    -w /opt/adapt-gqe/chemistry \
    "${IMAGE}" \
    python test_circuit_energy.py

echo "If the above printed PASS, proceed to slurm_02_h2_chemistry.sbatch."
echo "If it printed FAIL, stop and debug per test_circuit_energy.py's own"
echo "module docstring -- do not run the later stages until this passes."
