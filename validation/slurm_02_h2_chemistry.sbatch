#!/bin/bash
#SBATCH --job-name=h2-val-02-chemistry
#SBATCH --account=<YOUR_NERSC_ACCOUNT>
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=h2-val-02-%j.out
#SBATCH --error=h2-val-02-%j.err

# ============================================================================
# H2 validation, stage 2: generate the bond-length-scan geometries, build
# canonical Hamiltonian vectors, and run ADAPT-VQE across the whole scan.
#
# Active space is FULL space for STO-3G H2 (2 electrons, 2 orbitals) -- CASCI
# equals FCI exactly here, so there's no active-space-selection ambiguity;
# at r=0.735 A this should land within numerical noise of the textbook
# E_FCI ~= -1.1373 Ha. UCCSD in a minimal basis reduces to essentially one
# non-trivial doubles amplitude for H2, so expect ADAPT-VQE to converge in
# ~1 operator -- if you see it needing many more than that, something in
# the active-space or pool setup is off, not the chemistry itself.
#
# Run ONLY after slurm_01_test_circuit_energy.sbatch has passed.
#
# Usage: sbatch slurm_02_h2_chemistry.sbatch
# ============================================================================
set -euo pipefail

IMAGE="neilmehta87/gqe:26.06"
WORKDIR_HOST="${PSCRATCH}/h2-validation"
mkdir -p "${WORKDIR_HOST}"

module load podman-hpc 2>/dev/null || true
echo "Job ${SLURM_JOB_ID} on $(hostname)"

srun -n 1 --gpus-per-node=1 podman-hpc run --rm --gpu \
    -v "${WORKDIR_HOST}:/workdir" \
    -w /workdir \
    "${IMAGE}" \
    bash -c '
        set -euo pipefail

        echo "--- generating H2 bond-length scan ---"
        python /opt/adapt-gqe/validation/generate_h2_bondscan.py \
            --output-dir /workdir/h2_bondscan --n-points 40

        echo "--- building Hamiltonian vectors (full space: 2e,2o) ---"
        python /opt/adapt-gqe/chemistry/build_hamiltonian_vectors_for_dataset.py \
            --manifest /workdir/h2_bondscan/manifest.csv \
            --basis sto-3g --charge 0 --spin 0 \
            --n-active-electrons 2 --n-active-orbitals 2 \
            --skip-ccsd \
            --term-order-file /workdir/h2_term_order.json \
            --output-dir /workdir/h2_hamiltonians

        echo "--- running ADAPT-VQE across the scan (smoke test: 3 points first) ---"
        python /opt/adapt-gqe/chemistry/build_adapt_vqe_dataset.py \
            --manifest /workdir/h2_bondscan/manifest.csv \
            --basis sto-3g --charge 0 --spin 0 \
            --n-active-electrons 2 --n-active-orbitals 2 \
            --pool uccsd --reference casci --tolerance-mha 0.1 \
            --limit 3 --output-dir /workdir/h2_adaptvqe_smoke

        echo "--- smoke test passed, running full ADAPT-VQE batch ---"
        python /opt/adapt-gqe/chemistry/build_adapt_vqe_dataset.py \
            --manifest /workdir/h2_bondscan/manifest.csv \
            --basis sto-3g --charge 0 --spin 0 \
            --n-active-electrons 2 --n-active-orbitals 2 \
            --pool uccsd --reference casci --tolerance-mha 0.1 \
            --output-dir /workdir/h2_adaptvqe --skip-existing
    '

echo "Check /workdir/h2_adaptvqe/circuits.jsonl (mounted at ${WORKDIR_HOST}/h2_adaptvqe/circuits.jsonl):"
echo "  - energy_error_mha should be ~0 for every point (tight 0.1 mHa tolerance)"
echo "  - n_operators should mostly be 1-2"
echo "  - grep the equilibrium point (h2_r0p7350) and confirm e_ref ~= -1.1373 Ha"
echo "Also note the printed 'Operator pool 'uccsd': N candidate operators' line"
echo "in the job log -- you need that N for slurm_03's --pool-size argument."
