#!/usr/bin/env python3
"""
Generate an H2 bond-length scan as a stand-in "conformer" dataset for
Tier-1 pipeline validation (see the H2/LiH/BeH2/H4 validation discussion).

H2 has no dihedral/conformational structure, so LAMMPS/RDKit conformer
sampling doesn't apply here -- instead, bond length plays the role
"conformer" plays for imipramine: each point is a distinct geometry with
its own Hamiltonian, letting build_hamiltonian_vectors_for_dataset.py,
build_adapt_vqe_dataset.py, and the model-training stages all run exactly
as they would on a real molecule, just on inputs classically trivial and
numerically exact enough to sanity-check by hand.

At r ~= 0.735 A (included by default in the scan), STO-3G H2's full-space
(2e,2o) CASCI energy is the textbook ~-1.1373 Ha value used elsewhere in
this pipeline's own validation code (chemistry/test_circuit_energy.py) --
a concrete anchor point to check chemistry/build_hamiltonian.py and
cudaq_solvers.create_molecule agree with, independent of anything scan-wide.

Writes <output-dir>/h2_r<NNNN>.xyz plus a manifest.csv with just the
columns build_hamiltonian_vectors_for_dataset.py / build_adapt_vqe_dataset.py
actually read (conformer_id, source, path) -- no dihedral columns, since
none apply to a diatomic.

Usage:
  python generate_h2_bondscan.py --output-dir h2_bondscan --n-points 40
"""
import argparse
import csv
import os


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--r-min", type=float, default=0.4, help="Angstrom")
    p.add_argument("--r-max", type=float, default=3.0, help="Angstrom")
    p.add_argument("--n-points", type=int, default=40)
    p.add_argument(
        "--include-equilibrium",
        action="store_true",
        default=True,
        help="Force-include r=0.735 A (the well-known equilibrium/FCI reference point)",
    )
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n = args.n_points - (1 if args.include_equilibrium else 0)
    bond_lengths = [args.r_min + i * (args.r_max - args.r_min) / (n - 1) for i in range(n)]
    if args.include_equilibrium:
        bond_lengths.append(0.735)
    bond_lengths.sort()

    rows = []
    for r in bond_lengths:
        tag = f"h2_r{r:.4f}".replace(".", "p")
        xyz_path = os.path.join(args.output_dir, f"{tag}.xyz")
        with open(xyz_path, "w") as f:
            f.write("2\n")
            f.write(f"H2 bond length {r:.4f} A\n")
            f.write(f"H 0.0 0.0 {-r / 2:.6f}\n")
            f.write(f"H 0.0 0.0 {r / 2:.6f}\n")
        rows.append({"conformer_id": tag, "source": "BONDSCAN", "path": xyz_path})

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["conformer_id", "source", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} H2 geometries (r in [{args.r_min}, {args.r_max}] A) to {args.output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
