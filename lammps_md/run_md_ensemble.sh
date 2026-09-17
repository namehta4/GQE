#!/usr/bin/env bash
# Launch the 5 independent MD trajectories used for conformer sampling.
# Each trajectory differs only in its random seed (velocity init + Langevin noise).
#
# Updated to launch via run_md_python_driver.py rather than `lmp` directly --
# MACE-OFF is wired in through LAMMPS's ML-IAP "unified" Python interface
# (mace_mliap_unified.py), which needs a Python driver to register the model
# before the input script runs. See that file's VALIDATION WARNING before
# trusting this end-to-end; mpirun here parallelizes each trajectory's own
# LAMMPS domain decomposition exactly as before, that part is unaffected.
#
# Usage:
#   ./run_md_ensemble.sh <data_file> <element_order_space_separated> [nprocs] [mace_model]
#
# Example:
#   ./run_md_ensemble.sh imipramine.data "C H N" 8 small

set -euo pipefail

DATA_FILE="${1:?path to LAMMPS data file required}"
ELEMENT_ORDER="${2:?space-separated element order, e.g. \"C H N\" required}"
NPROCS="${3:-1}"
MACE_MODEL="${4:-small}"

SEEDS=(19823 40217 58391 67102 84459)   # 5 independent seeds; replace with your own
OUTDIR="runs"
mkdir -p "${OUTDIR}"

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    label="traj${i}"
    logfile="${OUTDIR}/${label}.log"

    echo "Launching ${label} (seed=${seed})..."
    mpirun -np "${NPROCS}" python run_md_python_driver.py \
        --seed "${seed}" \
        --run-label "${OUTDIR}/${label}" \
        --data-file "${DATA_FILE}" \
        --element-order ${ELEMENT_ORDER} \
        --mace-model "${MACE_MODEL}" \
        --device cuda \
        > "${logfile}" 2>&1 &
done

wait
echo "All 5 trajectories complete. Frame files: ${OUTDIR}/traj{0..4}.lammpstrj"
