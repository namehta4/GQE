#!/usr/bin/env python3
"""
Post-process completed LAMMPS NEB pair runs (from prepare_neb_pairs.py):
sanity-check each pathway's intermediate images and discard pairs that show
non-convergence artifacts -- unphysical bond stretching/compression, atomic
clashes, or (optionally) excessively high energy relative to the endpoints --
mirroring the paper's discard of 17/105 non-converged conformer pairs
(Appendix A.2.1).

Surviving images are copied into a single flat dataset directory as .xyz
files, ready for the same downstream PySCF / ADAPT-VQE stage as the
MD-derived and OoD conformers.

Usage:
  python filter_neb_pathways.py \\
      --neb-dir neb_runs --n-images 30 --element-order C H N \\
      --dataset-out neb_dataset --check-energy --mace-model small --device cuda
"""
import argparse
import csv
import glob
import os

import numpy as np
from ase.data import covalent_radii, atomic_numbers


def reference_bonds(atoms, bond_tolerance):
    """Distance-based covalent bond perception on the initial (image 0)
    structure: any pair within (r_cov_i + r_cov_j) * bond_tolerance is
    treated as bonded. Cheap and dependency-light; not a substitute for
    real bond-order perception, but sufficient as a distortion sanity check.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    radii = [covalent_radii[atomic_numbers[s]] for s in symbols]

    bonds = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            cutoff = (radii[i] + radii[j]) * bond_tolerance
            if d <= cutoff:
                bonds.append((i, j, d))
    return bonds


def check_image(atoms, bonds, max_stretch_ratio, min_nonbonded_dist):
    """Return (ok, reason, max_bond_ratio_seen) for one image given the
    reference bond list."""
    positions = atoms.get_positions()
    bonded_pairs = {(i, j) for i, j, _ in bonds}

    max_ratio = 0.0
    for i, j, d0 in bonds:
        d = np.linalg.norm(positions[i] - positions[j])
        ratio = d / d0
        max_ratio = max(max_ratio, ratio, 1.0 / ratio)
        if ratio > max_stretch_ratio or ratio < 1.0 / max_stretch_ratio:
            return False, f"bond ({i},{j}) stretched by {ratio:.2f}x", max_ratio

    n = len(atoms)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in bonded_pairs:
                continue
            d = np.linalg.norm(positions[i] - positions[j])
            min_dist = min(min_dist, d)
            if d < min_nonbonded_dist:
                return False, f"non-bonded clash ({i},{j}) at {d:.2f} A", max_ratio

    return True, None, max_ratio


def image_energy(atoms, mace_model, device):
    from mace.calculators import mace_off

    atoms = atoms.copy()
    atoms.calc = mace_off(model=mace_model, device=device)
    return float(atoms.get_potential_energy())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--neb-dir", required=True, help="Output dir from prepare_neb_pairs.py")
    p.add_argument("--n-images", type=int, required=True)
    p.add_argument("--element-order", nargs="+", required=True)
    p.add_argument("--dataset-out", required=True)

    p.add_argument("--bond-tolerance", type=float, default=1.3,
                    help="Covalent-bond distance cutoff multiplier for reference "
                    "bond perception on image 0")
    p.add_argument("--max-stretch-ratio", type=float, default=1.6,
                    help="Discard pair if any reference bond stretches/compresses "
                    "beyond this ratio in any image")
    p.add_argument("--min-nonbonded-dist", type=float, default=0.7,
                    help="Discard pair if any non-bonded atom pair comes closer "
                    "than this (Angstrom) in any image")

    p.add_argument("--check-energy", action="store_true",
                    help="Also evaluate MACE-OFF energy per image and discard "
                    "pairs with unphysically high intermediate energy")
    p.add_argument("--max-rel-energy", type=float, default=2.0,
                    help="Discard pair if any intermediate image's energy exceeds "
                    "the higher endpoint's energy by more than this, eV "
                    "(only used with --check-energy)")
    p.add_argument("--mace-model", default="small", choices=["small", "medium", "large"])
    p.add_argument("--device", default="cpu")

    args = p.parse_args()
    os.makedirs(args.dataset_out, exist_ok=True)

    pair_dirs = sorted(glob.glob(os.path.join(args.neb_dir, "pair_*")))
    print(f"Found {len(pair_dirs)} pair director(ies) to check")

    rows = []
    n_kept = 0
    for pair_dir in pair_dirs:
        pair_tag = os.path.basename(pair_dir)
        images = []
        try:
            for k in range(args.n_images):
                dump_path = os.path.join(pair_dir, f"pair_image_{k}_final.lammpstrj")
                from ase.io import read

                atoms = read(
                    dump_path, format="lammps-dump-text", specorder=args.element_order
                )
                images.append(atoms)
        except FileNotFoundError as e:
            rows.append({"pair": pair_tag, "kept": False, "reason": f"missing dump: {e}"})
            print(f"  {pair_tag}: DISCARD (missing dump file)")
            continue

        bonds = reference_bonds(images[0], args.bond_tolerance)

        discard_reason = None
        max_ratio_seen = 1.0
        for k, atoms in enumerate(images):
            ok, reason, max_ratio = check_image(
                atoms, bonds, args.max_stretch_ratio, args.min_nonbonded_dist
            )
            max_ratio_seen = max(max_ratio_seen, max_ratio)
            if not ok:
                discard_reason = f"image {k}: {reason}"
                break

        max_rel_energy = None
        if discard_reason is None and args.check_energy:
            energies = [image_energy(a, args.mace_model, args.device) for a in images]
            endpoint_ref = max(energies[0], energies[-1])
            max_rel_energy = max(e - endpoint_ref for e in energies)
            if max_rel_energy > args.max_rel_energy:
                discard_reason = (
                    f"max intermediate energy {max_rel_energy:.3f} eV above "
                    f"endpoint reference (threshold {args.max_rel_energy} eV)"
                )

        kept = discard_reason is None
        rows.append(
            {
                "pair": pair_tag,
                "kept": kept,
                "reason": discard_reason or "",
                "max_bond_stretch_ratio": round(max_ratio_seen, 3),
                "max_rel_energy_eV": max_rel_energy,
            }
        )

        if kept:
            n_kept += 1
            from ase.io import write as ase_write

            for k, atoms in enumerate(images):
                out_path = os.path.join(args.dataset_out, f"{pair_tag}_image_{k:02d}.xyz")
                ase_write(out_path, atoms)
            print(f"  {pair_tag}: KEPT ({args.n_images} images written)")
        else:
            print(f"  {pair_tag}: DISCARD ({discard_reason})")

    summary_path = os.path.join(args.dataset_out, "neb_filter_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nKept {n_kept}/{len(pair_dirs)} pairs "
          f"({len(pair_dirs) - n_kept} discarded, cf. 17/105 in the paper).")
    print(f"Images written to: {args.dataset_out}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
