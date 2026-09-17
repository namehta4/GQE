#!/usr/bin/env python3
"""
Convert a molecule of your choice into a LAMMPS `atom_style atomic` data
file for the MD-sampling stage, plus a canonical atom-index map.

Two input modes:
  --smiles "CN(C)CCCN1c2ccccc2CCc2ccccc21"    RDKit ETKDG-embeds 3D coords,
                                               then a cheap MMFF/UFF pre-relax
                                               (stand-in for the paper's LJ
                                               pre-optimization step).
  --input  structure.xyz | .pdb | .cif | ...  Read an existing geometry as-is
                                               via ASE (any ASE-supported format).

Both paths then get a final relaxation with the MACE-OFF foundation potential
(via its ASE calculator -- separate from the compiled LAMMPS pair_style plugin
used later in md_imipramine.in) before the LAMMPS data file is written.

IMPORTANT: the atom ordering produced here is frozen for the rest of the
pipeline. Dihedral-angle definitions (cf. Fig. 1a in the ADAPT-GQE paper) and
the canonical Hamiltonian-coefficient ordering used to condition the language
model both depend on this fixed index scheme. Inspect the companion
`<output>.idx.txt` file before defining any dihedral tuples downstream.

Examples:
  # Imipramine from SMILES, MACE-OFF relaxed, on GPU
  python mol_to_lammps_data.py \\
      --smiles "CN(C)CCCN1c2ccccc2CCc2ccccc21" \\
      --output imipramine.data --element-order C H N --device cuda

  # A molecule of your choice, starting from an existing structure file
  python mol_to_lammps_data.py \\
      --input my_molecule.xyz \\
      --output my_molecule.data --element-order C H N O --device cuda
"""
import argparse
import sys

from mol_common import (
    build_base_mol_from_smiles,
    rdkit_conf_to_ase,
    relax_with_mace,
    write_index_map,
    write_lammps_data,
)


def build_atoms_from_smiles(smiles, seed, pre_ff):
    from rdkit.Chem import AllChem

    mol = build_base_mol_from_smiles(smiles)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(
            "RDKit 3D embedding failed for this SMILES - try a different --seed"
        )

    if pre_ff == "mmff":
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    elif pre_ff == "uff":
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    # pre_ff == "none": skip pre-relaxation, rely solely on --optimize stage

    return rdkit_conf_to_ase(mol, conf_id=0)


def load_atoms_from_file(path):
    from ase.io import read

    return read(path)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="SMILES string for the molecule of interest")
    src.add_argument(
        "--input", help="Path to an existing geometry file (xyz, pdb, cif, ...)"
    )

    p.add_argument("--output", required=True, help="Output LAMMPS data file path")
    p.add_argument(
        "--index-map",
        default=None,
        help="Output path for the canonical atom-index map "
        "(default: <output>.idx.txt)",
    )
    p.add_argument(
        "--element-order",
        nargs="+",
        required=True,
        help="Element order for LAMMPS atom types, e.g. --element-order C H N. "
        "MUST match the pair_coeff element list in md_imipramine.in",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0xC0FFEE,
        help="RDKit ETKDG embedding seed (SMILES input only)",
    )
    p.add_argument(
        "--pre-relax",
        choices=["none", "mmff", "uff"],
        default="mmff",
        help="Cheap force-field pre-relaxation before MACE-OFF (SMILES input only)",
    )
    p.add_argument(
        "--optimize",
        choices=["none", "mace"],
        default="mace",
        help="Final-stage geometry optimization",
    )
    p.add_argument(
        "--mace-model",
        default="small",
        choices=["small", "medium", "large"],
        help="MACE-OFF foundation model size for the --optimize=mace stage",
    )
    p.add_argument(
        "--device", default="cpu", help="torch device for MACE-OFF relaxation (cpu/cuda)"
    )
    p.add_argument(
        "--fmax", type=float, default=0.01, help="Force convergence criterion, eV/A"
    )
    p.add_argument(
        "--box-padding",
        type=float,
        default=10.0,
        help="Vacuum padding (A) around the molecule for the LAMMPS box",
    )

    args = p.parse_args()

    if args.smiles:
        atoms = build_atoms_from_smiles(args.smiles, args.seed, args.pre_relax)
    else:
        atoms = load_atoms_from_file(args.input)

    if args.optimize == "mace":
        atoms = relax_with_mace(atoms, args.mace_model, args.device, args.fmax)

    present = sorted(set(atoms.get_chemical_symbols()))
    missing = set(present) - set(args.element_order)
    if missing:
        sys.exit(
            f"--element-order {args.element_order} is missing element(s) "
            f"{sorted(missing)} present in the molecule. Add them and rerun."
        )

    idx_map_path = args.index_map or (args.output + ".idx.txt")
    write_index_map(atoms, idx_map_path)
    write_lammps_data(atoms, args.output, args.element_order, args.box_padding)

    counts = {el: atoms.get_chemical_symbols().count(el) for el in args.element_order}
    print(f"Wrote {args.output} ({len(atoms)} atoms)")
    print(f"Atom type order (must match pair_coeff in md_imipramine.in): {args.element_order}")
    print(f"Per-element counts: {counts}")
    print(f"Canonical atom-index map written to: {idx_map_path}")
    print(
        "Inspect this file now to define dihedral-angle atom tuples for later "
        "stages -- the ordering here is frozen for the rest of the pipeline."
    )


if __name__ == "__main__":
    main()
