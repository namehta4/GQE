#!/usr/bin/env python3
"""
Consolidate MD, NEB, and OoD conformers into a single flat dataset directory
plus a manifest CSV recording each conformer's source and its dihedral-angle
coordinates -- reproducing the Figure 1 conformational-diversity
characterization in ADAPT-GQE (arXiv:2607.22468).

This is the master index that the PySCF/ADAPT-VQE stage and the eventual
train/val/test split both read from.

Usage:
  python build_conformer_manifest.py \\
      --md-traj runs/traj0.lammpstrj:10 \\
      --md-traj runs/traj1.lammpstrj:10 \\
      --md-traj runs/traj2.lammpstrj:10 \\
      --md-traj runs/traj3.lammpstrj:10 \\
      --md-traj runs/traj4.lammpstrj:5 \\
      --neb-dir neb_dataset \\
      --ood-dir ood_dataset \\
      --element-order C H N \\
      --dihedral-defs dihedrals_imipramine_example.json \\
      --output-dir full_dataset

--dihedral-defs accepts either a plain JSON list of 4-integer atom-index
tuples, or an object with a "dihedrals" key (see
dihedrals_imipramine_example.json) -- the latter form lets you keep a
"_README" caveat alongside the definitions.
"""
import argparse
import csv
import json
import os


def iter_md_frames(traj_spec, element_order):
    from ase.io import iread

    path, stride = traj_spec.rsplit(":", 1)
    stride = int(stride)
    idx_expr = f"::{stride}"
    base = os.path.splitext(os.path.basename(path))[0]
    for i, atoms in enumerate(
        iread(path, index=idx_expr, format="lammps-dump-text", specorder=element_order)
    ):
        yield f"{base}_frame{i * stride:07d}", atoms


def iter_flat_xyz_dir(directory):
    from ase.io import read

    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".xyz"):
            tag = os.path.splitext(fname)[0]
            yield tag, read(os.path.join(directory, fname))


def compute_dihedrals(atoms, dihedral_defs):
    # ASE's get_dihedral returns degrees.
    return [atoms.get_dihedral(*tup) for tup in dihedral_defs]


def make_row(source, tag, path, atoms, dihedral_defs):
    thetas = compute_dihedrals(atoms, dihedral_defs)
    row = {"conformer_id": f"{source}_{tag}", "source": source, "path": path}
    for i, theta in enumerate(thetas):
        row[f"theta_{i}"] = theta
    return row


def load_dihedral_defs(path):
    with open(path) as f:
        raw = json.load(f)
    defs = raw["dihedrals"] if isinstance(raw, dict) else raw
    return [tuple(t) for t in defs]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--md-traj",
        action="append",
        default=[],
        help="path:stride, repeatable, one per MD trajectory "
        "(e.g. runs/traj0.lammpstrj:10)",
    )
    p.add_argument(
        "--neb-dir", default=None, help="Output dir from filter_neb_pathways.py (flat .xyz)"
    )
    p.add_argument(
        "--ood-dir", default=None, help="Output dir from generate_ood_perturbations.py (flat .xyz)"
    )
    p.add_argument(
        "--element-order",
        nargs="+",
        required=True,
        help="Required to decode LAMMPS dump atom types back to elements",
    )
    p.add_argument(
        "--dihedral-defs",
        required=True,
        help="JSON file of 4-integer atom-index tuples. MUST be expressed in the "
        "SAME atom-index scheme as everything else in the pipeline -- inspect "
        "your <output>.idx.txt, do not assume the shipped imipramine example "
        "applies to a SMILES-derived atom ordering.",
    )
    p.add_argument("--output-dir", required=True)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dihedral_defs = load_dihedral_defs(args.dihedral_defs)

    from ase.io import write as ase_write

    rows = []

    for spec in args.md_traj:
        for tag, atoms in iter_md_frames(spec, args.element_order):
            out_path = os.path.join(args.output_dir, f"MD_{tag}.xyz")
            ase_write(out_path, atoms)
            rows.append(make_row("MD", tag, out_path, atoms, dihedral_defs))

    if args.neb_dir:
        for tag, atoms in iter_flat_xyz_dir(args.neb_dir):
            out_path = os.path.join(args.output_dir, f"NEB_{tag}.xyz")
            ase_write(out_path, atoms)
            rows.append(make_row("NEB", tag, out_path, atoms, dihedral_defs))

    if args.ood_dir:
        for tag, atoms in iter_flat_xyz_dir(args.ood_dir):
            out_path = os.path.join(args.output_dir, f"OOD_{tag}.xyz")
            ase_write(out_path, atoms)
            rows.append(make_row("OOD", tag, out_path, atoms, dihedral_defs))

    if not rows:
        raise RuntimeError("No conformers found -- check --md-traj/--neb-dir/--ood-dir paths")

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    fieldnames = ["conformer_id", "source", "path"] + [
        f"theta_{i}" for i in range(len(dihedral_defs))
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(f"Consolidated {len(rows)} conformers: {counts}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
