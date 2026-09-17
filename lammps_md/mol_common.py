"""
Shared helpers for building/relaxing molecular geometries across the
ADAPT-GQE-style data pipeline.

Every script that starts from the same SMILES string MUST call
`build_base_mol_from_smiles` (rather than re-implementing MolFromSmiles+AddHs
itself) so that RDKit's atom ordering is bit-for-bit identical across the
MD-seed geometry, the reference conformers, the NEB endpoints, and eventually
the PySCF active-space input. That ordering is what the dihedral-angle
definitions and the canonical Hamiltonian-coefficient encoding depend on.
"""
from ase import Atoms


def build_base_mol_from_smiles(smiles):
    """RDKit Mol with explicit Hs added, NOT yet embedded in 3D."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return Chem.AddHs(mol)


def rdkit_conf_to_ase(mol, conf_id=0):
    """Extract one conformer of an RDKit Mol into an ASE Atoms object,
    preserving RDKit's atom ordering exactly."""
    conf = mol.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return Atoms(symbols=symbols, positions=positions)


def relax_with_mace(atoms, model_size="small", device="cpu", fmax=0.01):
    from ase.optimize import BFGS
    from mace.calculators import mace_off

    atoms.calc = mace_off(model=model_size, device=device)
    BFGS(atoms, logfile="-").run(fmax=fmax)
    return atoms


def write_index_map(atoms, path):
    with open(path, "w") as f:
        f.write("# index0 index1_lammps element x y z\n")
        for i, atom in enumerate(atoms):
            x, y, z = atom.position
            f.write(f"{i} {i + 1} {atom.symbol} {x:.6f} {y:.6f} {z:.6f}\n")


def write_lammps_data(atoms, output, element_order, box_padding=10.0):
    from ase.io import write

    positions = atoms.get_positions()
    lo = positions.min(axis=0) - box_padding
    hi = positions.max(axis=0) + box_padding
    atoms = atoms.copy()
    atoms.set_cell(hi - lo)
    atoms.center()
    atoms.set_pbc(False)

    write(
        output,
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=element_order,
        masses=True,
    )
