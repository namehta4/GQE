#!/usr/bin/env python3
"""
Generate out-of-distribution (OoD) conformers by rattling the nuclear
positions of the highest-energy reference conformer, reproducing the
OoD-perturbation construction in ADAPT-GQE (arXiv:2607.22468), Appendix A.2.1.
These structures are deliberately energetically separated from the MD- and
NEB-derived training data and serve as the most stringent generalization test.

Usage:
  python generate_ood_perturbations.py \\
      --reference-dir refs_122 --n-perturbations 200 \\
      --stdev 0.05 --output-dir ood_dataset
"""
import argparse
import csv
import os

import numpy as np


def find_highest_energy_conformer(reference_dir):
    summary_path = os.path.join(reference_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"{summary_path} not found -- run generate_reference_conformers.py "
            f"with --optimize mace first (energy ranking requires MACE energies)"
        )
    best = None
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            e = row.get("mace_energy_eV")
            if e in (None, ""):
                continue
            e = float(e)
            if best is None or e > best[1]:
                best = (row["conformer"], e)
    if best is None:
        raise RuntimeError(f"No mace_energy_eV values found in {summary_path}")
    return best  # (tag, energy)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--reference-dir",
        required=True,
        help="Output dir from generate_reference_conformers.py (the LARGER set, "
        "e.g. the 122-conformer run)",
    )
    p.add_argument(
        "--seed-xyz",
        default=None,
        help="Override: perturb this specific xyz file instead of auto-selecting "
        "the highest-energy conformer in --reference-dir",
    )
    p.add_argument("--n-perturbations", type=int, default=200)
    p.add_argument(
        "--stdev",
        type=float,
        default=0.05,
        help="Rattle standard deviation, Angstrom (paper value: 0.05)",
    )
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--output-dir", required=True)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from ase.io import read, write

    if args.seed_xyz:
        seed_path = args.seed_xyz
        seed_tag = os.path.splitext(os.path.basename(seed_path))[0]
    else:
        seed_tag, seed_energy = find_highest_energy_conformer(args.reference_dir)
        seed_path = os.path.join(args.reference_dir, f"{seed_tag}.xyz")
        print(f"Selected seed conformer: {seed_tag} ({seed_energy:.4f} eV)")

    base_atoms = read(seed_path)
    rng = np.random.default_rng(args.seed)

    for k in range(args.n_perturbations):
        atoms = base_atoms.copy()
        # ASE's rattle() perturbs each coordinate independently by
        # N(0, stdev^2) -- exactly the paper's rattle-method OoD construction.
        atoms.rattle(stdev=args.stdev, seed=int(rng.integers(0, 2**31 - 1)))

        out_path = os.path.join(args.output_dir, f"ood_{k:04d}.xyz")
        write(out_path, atoms)

    print(
        f"Wrote {args.n_perturbations} OoD perturbations of {seed_tag} "
        f"(stdev={args.stdev} A) to {args.output_dir}/"
    )


if __name__ == "__main__":
    main()
