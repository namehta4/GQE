#!/usr/bin/env python3
"""
Batch-build the canonical Hamiltonian coefficient vectors (Sec. 3.2.2 of
ADAPT-GQE, arXiv:2607.22468) for every conformer in a manifest, using the
OpenFermion-based pipeline in build_hamiltonian.py. Reuses ONE shared
--term-order-file across all conformers of a given active space so the
resulting vectors line up positionally -- required for the multimodal
Hamiltonian encoder to receive a consistent input representation.

CROSS-CHECK RECOMMENDED: this uses an INDEPENDENT Hamiltonian construction
path from cudaq_solvers.create_molecule() (the one run_adapt_vqe.py /
build_adapt_vqe_dataset.py use for the actual ADAPT-VQE circuits). Both
paths should agree numerically -- Jordan-Wigner is a canonical, parameter-
free transform, and both start from the same PySCF RHF/CASCI/CCSD setup --
but this has NOT been cross-validated end-to-end in this environment. Before
trusting the combined dataset, spot-check a handful of conformers: the
e_hf/e_casci/e_ccsd values written here should match the e_hf/e_ref values
in the corresponding rows of circuits.jsonl to numerical precision.

Usage:
  python build_hamiltonian_vectors_for_dataset.py \\
      --manifest full_dataset/manifest.csv \\
      --basis 6-31g --charge 0 --spin 0 \\
      --n-active-electrons 6 --n-active-orbitals 6 \\
      --term-order-file term_order_12q.json \\
      --output-dir hamiltonians_12q
"""
import argparse
import csv
import os

import numpy as np

from build_hamiltonian import (
    build_pyscf_mol,
    build_qubit_hamiltonian,
    load_or_init_term_order,
    run_reference_methods,
)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--n-active-electrons", type=int, required=True)
    p.add_argument("--n-active-orbitals", type=int, required=True)
    p.add_argument("--skip-ccsd", action="store_true")
    p.add_argument("--term-order-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=None)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    n_qubits = 2 * args.n_active_orbitals

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    n_failed = 0
    for i, row in enumerate(rows):
        conformer_id = row["conformer_id"]
        out_path = os.path.join(args.output_dir, f"{conformer_id}.npz")
        if os.path.exists(out_path):
            continue
        print(f"[{i + 1}/{len(rows)}] {conformer_id}")
        try:
            mol = build_pyscf_mol(row["path"], args.basis, args.charge, args.spin)
            mf, e_casci, e_ccsd = run_reference_methods(
                mol,
                args.n_active_electrons,
                args.n_active_orbitals,
                want_ccsd=not args.skip_ccsd,
            )
            qubit_hamiltonian = build_qubit_hamiltonian(
                mol, mf, args.n_active_electrons, args.n_active_orbitals
            )
            vector, order = load_or_init_term_order(
                qubit_hamiltonian, n_qubits, args.term_order_file
            )
            np.savez(
                out_path,
                coefficients=vector,
                e_hf=mf.e_tot,
                e_casci=e_casci,
                e_ccsd=e_ccsd if e_ccsd is not None else np.nan,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            n_failed += 1

    print(f"Done. {len(rows) - n_failed} succeeded, {n_failed} failed.")
    print(f"Vectors written to {args.output_dir}/")


if __name__ == "__main__":
    main()
