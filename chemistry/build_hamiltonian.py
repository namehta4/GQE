#!/usr/bin/env python3
"""
Build the active-space qubit Hamiltonian (Jordan-Wigner mapped) for one
molecular geometry, plus CASCI/CCSD reference energies -- reproducing the
active-space construction described in Section 3.1 of ADAPT-GQE
(arXiv:2607.22468).

Active-space presets used in the paper (n_qubits = 2 * n_active_orbitals via
Jordan-Wigner spin-orbital mapping):
  12 qubits: (6e, 6o)   -- basis 6-31g, JW, UCCGSD pool downstream
  14 qubits: (6e, 7o)   -- basis 6-31g, JW, UCCGSD pool downstream
  16 qubits: (8e, 8o)   -- basis 6-31g, JW, UCCSD  pool downstream

Requires: pyscf, openfermion, numpy.

VALIDATION WARNING: this script has not been run end-to-end against a live
PySCF/OpenFermion install in this environment. The integral-convention
transpose (chemist -> OpenFermion physicist-like ordering) is a known
convention but has NOT been numerically verified here. Before trusting this
at dataset scale, run it on a small molecule with a well-known reference
energy (e.g. H2 in STO-3G, full active space n_active_electrons=2,
n_active_orbitals=2) and confirm E_HF matches ~-1.1167 Ha and E_CASCI (=FCI
here) matches ~-1.1373 Ha at the equilibrium bond length (0.735 A).

Usage:
  python build_hamiltonian.py \\
      --xyz conformer_000.xyz --basis 6-31g \\
      --n-active-electrons 6 --n-active-orbitals 6 \\
      --charge 0 --spin 0 \\
      --term-order-file term_order_12q.json \\
      --output hamiltonian_conformer_000.npz
"""
import argparse
import json
import os

import numpy as np


def build_pyscf_mol(xyz_path, basis, charge, spin):
    from ase.io import read
    from pyscf import gto

    atoms = read(xyz_path)
    atom_lines = "; ".join(
        f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}"
        for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    )
    mol = gto.M(atom=atom_lines, basis=basis, charge=charge, spin=spin, unit="Angstrom")
    return mol


def run_reference_methods(mol, n_active_electrons, n_active_orbitals, want_ccsd):
    from pyscf import cc, mcscf, scf

    mf = scf.RHF(mol).run(verbose=0)

    casci = mcscf.CASCI(mf, n_active_orbitals, n_active_electrons)
    e_casci = casci.kernel()[0]

    e_ccsd = None
    if want_ccsd:
        mycc = cc.CCSD(mf).run(verbose=0)
        e_ccsd = mycc.e_tot

    return mf, e_casci, e_ccsd


def active_space_indices(mol, n_active_electrons):
    n_electrons_total = mol.nelectron
    n_core_electrons = n_electrons_total - n_active_electrons
    if n_core_electrons % 2 != 0:
        raise ValueError(
            f"n_active_electrons={n_active_electrons} leaves an odd number of "
            f"core electrons ({n_core_electrons}) for a closed-shell RHF "
            f"reference -- check the active-space electron count."
        )
    n_core_orbitals = n_core_electrons // 2
    occupied_indices = list(range(n_core_orbitals))
    return occupied_indices


def build_qubit_hamiltonian(mol, mf, n_active_electrons, n_active_orbitals):
    from openfermion.chem.molecular_data import get_active_space_integrals, spinorb_from_spatial
    from openfermion.ops.representations import InteractionOperator
    from openfermion.transforms import get_fermion_operator, jordan_wigner
    from pyscf import ao2mo

    n_mo = mf.mo_coeff.shape[1]
    h1e_ao = mf.get_hcore()
    h1e_mo = mf.mo_coeff.T @ h1e_ao @ mf.mo_coeff

    eri_chem = ao2mo.kernel(mol, mf.mo_coeff)
    eri_chem = ao2mo.restore(1, eri_chem, n_mo)  # (pq|rs), chemist notation, full (n,n,n,n)

    # Chemist -> OpenFermion physicist-like spatial-orbital convention.
    # NOT numerically validated in this environment -- see the module
    # docstring's validation instructions before trusting this at scale.
    two_body_integrals = np.asarray(eri_chem.transpose(0, 2, 3, 1), order="C")

    occupied_indices = active_space_indices(mol, n_active_electrons)
    n_core_orbitals = len(occupied_indices)
    active_indices = list(range(n_core_orbitals, n_core_orbitals + n_active_orbitals))

    core_constant, h1e_active, h2e_active = get_active_space_integrals(
        h1e_mo, two_body_integrals, occupied_indices, active_indices
    )

    one_body_coeff, two_body_coeff = spinorb_from_spatial(h1e_active, h2e_active)

    constant = mol.energy_nuc() + core_constant
    interaction_op = InteractionOperator(constant, one_body_coeff, two_body_coeff)

    fermion_hamiltonian = get_fermion_operator(interaction_op)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress()
    return qubit_hamiltonian


def pauli_term_to_string(term, n_qubits):
    """Convert an OpenFermion QubitOperator term (tuple of (index, action))
    into a fixed-width canonical string, e.g. 'IXYZII...', usable as a stable
    sort/dictionary key across conformers of the same active space."""
    chars = ["I"] * n_qubits
    for idx, action in term:
        chars[idx] = action
    return "".join(chars)


def load_or_init_term_order(qubit_hamiltonian, n_qubits, term_order_file):
    terms = {
        pauli_term_to_string(term, n_qubits): coeff.real
        for term, coeff in qubit_hamiltonian.terms.items()
    }

    if term_order_file and os.path.exists(term_order_file):
        with open(term_order_file) as f:
            order = json.load(f)
        extra_terms = set(terms) - set(order)
        if extra_terms:
            raise ValueError(
                f"This conformer's Hamiltonian has {len(extra_terms)} Pauli "
                f"term(s) not present in the reference term order "
                f"({term_order_file}); the fixed-term-support assumption "
                f"(same active space => same term structure across "
                f"conformers) is violated. Example extra term(s): "
                f"{list(extra_terms)[:5]}"
            )
    else:
        order = sorted(terms.keys())
        if term_order_file:
            with open(term_order_file, "w") as f:
                json.dump(order, f)
            print(f"Initialized canonical term order ({len(order)} terms) -> {term_order_file}")

    vector = np.array([terms.get(p, 0.0) for p in order], dtype=np.float64)
    return vector, order


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--xyz", required=True)
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0, help="2S, PySCF convention (0 = closed shell)")
    p.add_argument("--n-active-electrons", type=int, required=True)
    p.add_argument("--n-active-orbitals", type=int, required=True)
    p.add_argument(
        "--skip-ccsd",
        action="store_true",
        help="Skip CCSD reference (only needed for the 16-qubit configuration)",
    )
    p.add_argument(
        "--term-order-file",
        default=None,
        help="Path to a canonical Pauli-term-order JSON, SHARED across all "
        "conformers of this active space. Created on first use if it does "
        "not exist yet; every subsequent conformer must be run against the "
        "SAME file so Hamiltonian coefficient vectors line up positionally "
        "(required for the Sec. 3.2.2 multimodal encoder input).",
    )
    p.add_argument("--output", required=True, help="Output .npz path")

    args = p.parse_args()
    n_qubits = 2 * args.n_active_orbitals

    mol = build_pyscf_mol(args.xyz, args.basis, args.charge, args.spin)
    mf, e_casci, e_ccsd = run_reference_methods(
        mol, args.n_active_electrons, args.n_active_orbitals, want_ccsd=not args.skip_ccsd
    )
    qubit_hamiltonian = build_qubit_hamiltonian(
        mol, mf, args.n_active_electrons, args.n_active_orbitals
    )
    vector, order = load_or_init_term_order(qubit_hamiltonian, n_qubits, args.term_order_file)

    np.savez(
        args.output,
        coefficients=vector,
        n_qubits=n_qubits,
        n_active_electrons=args.n_active_electrons,
        n_active_orbitals=args.n_active_orbitals,
        e_hf=mf.e_tot,
        e_casci=e_casci,
        e_ccsd=e_ccsd if e_ccsd is not None else np.nan,
    )
    print(
        f"Wrote {args.output}: {len(order)} Pauli terms, "
        f"E_HF={mf.e_tot:.6f}, E_CASCI={e_casci:.6f}, "
        f"E_CCSD={'skipped' if e_ccsd is None else f'{e_ccsd:.6f}'}"
    )


if __name__ == "__main__":
    main()
