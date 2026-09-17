#!/bin/bash
#SBATCH --job-name=adapt-gqe
#SBATCH --account=<YOUR_NERSC_ACCOUNT>      # <-- fill in, e.g. m1234
#SBATCH --constraint=gpu
#SBATCH --qos=regular                        # use "debug" for a short smoke-test run first
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64                   # Perlmutter GPU node: 1x AMD EPYC 7763 (64 cores)
#SBATCH --time=12:00:00
#SBATCH --output=adapt-gqe-%j.out
#SBATCH --error=adapt-gqe-%j.err

# ============================================================================
# Runs orchestrator/run_pipeline.py inside the merged container image
# (docker/Dockerfile) on one Perlmutter GPU node (4x A100).
#
# BEFORE RUNNING THIS:
#   1. Build the image per docker/Dockerfile's own instructions (on a
#      Perlmutter login/build node, --platform linux/amd64).
#   2. `podman-hpc migrate <image>` if your podman-hpc build/run workflow
#      requires it to make a locally-built image usable from compute nodes
#      -- confirm against current NERSC podman-hpc documentation; this has
#      not been exercised in this environment.
#   3. Fill in --account above.
#   4. Write a real orchestrator config (see orchestrator/config.example.yaml)
#      with its `workdir:` field set to exactly `/workdir` -- that's the
#      mount point this script binds to $PSCRATCH below. Any other value
#      there will write outputs somewhere you didn't intend.
#
# VALIDATION WARNING: `podman-hpc run --gpu` is the expected NERSC flag for
# GPU passthrough as of this writing -- verify against `podman-hpc run
# --help` / current NERSC docs before relying on it; not run in this
# environment.
#
# Also note (see "how much of the pipeline can run on GPU" discussion):
# chemistry/run_adapt_vqe.py, circuit_energy.py, and
# build_adapt_vqe_dataset.py do NOT currently call cudaq.set_target(...),
# so ADAPT-VQE/energy-evaluation simulation runs on CPU even on this GPU
# node unless that's added. MD (MACE-OFF) and model training/RL already use
# the GPUs requested here.
#
# Usage:
#   sbatch run_on_perlmutter.sbatch /path/to/your_config.yaml
# ============================================================================
set -euo pipefail

CONFIG_HOST="${1:?Usage: sbatch run_on_perlmutter.sbatch <path-to-orchestrator-config.yaml>}"
IMAGE="neilmehta87/gqe:26.06"
WORKDIR_HOST="${PSCRATCH}/adapt-gqe-run-${SLURM_JOB_ID}"

mkdir -p "${WORKDIR_HOST}"
cp "${CONFIG_HOST}" "${WORKDIR_HOST}/config.yaml"

module load podman-hpc 2>/dev/null || true   # no-op if already on PATH

echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Config: ${CONFIG_HOST}"
echo "Workdir (host):      ${WORKDIR_HOST}"
echo "Workdir (in-image):  /workdir"
nvidia-smi || true

srun -n 1 --gpus-per-node=4 podman-hpc run --rm --gpu \
    -v "${WORKDIR_HOST}:/workdir" \
    -w /opt/adapt-gqe \
    "${IMAGE}" \
    python orchestrator/run_pipeline.py --config /workdir/config.yaml

echo "Done. Outputs in ${WORKDIR_HOST}"
