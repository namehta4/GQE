#!/usr/bin/env python3
"""
Run ADAPT-VQE on one conformer's active-space Hamiltonian using CUDA-Q's own
cudaq-solvers library, reproducing the reference-circuit generation
described in Section 3.1 of ADAPT-GQE (arXiv:2607.22468).

This is a THIN WRAPPER around NVIDIA's shipped implementation, not a
from-scratch reimplementation:
  - cudaq_solvers.create_molecule()  builds the active-space Hamiltonian
    (verified directly against the pyscf/gas_phase_generator.py source
    shipped inside the cudaq-solvers wheel: RHF -> CASCI/CCSD -> spin-orbital
    integrals -> Jordan-Wigner SpinOperator).
  - cudaq_solvers.get_operator_pool() generates the UCCSD/UCCGSD/UpCCGSD
    operator pool as a flat list of cudaq.SpinOperator candidates.
  - cudaq_solvers.adapt_vqe() performs the gradient-based greedy operator
    selection plus global VQE reoptimization at each step.

STOPPING CRITERION: the paper terminates ADAPT-VQE once the circuit energy
is within a fixed tolerance epsilon of a reference energy (CASCI for
12/14-qubit systems, CCSD for 16-qubit). cudaq_solvers.adapt_vqe()'s own
`max_iter` / `grad_norm_tolerance` / `threshold_energy` options are NOT the
same criterion (they stop on internal convergence, not distance to an
external reference), and its documented return value is only the FINAL
(energy, params, operators) -- no per-iteration energy trace is exposed.
To reproduce the paper's exact stopping rule, this script performs an outer
exponential-then-binary search over `max_iter`, re-invoking adapt_vqe()
fresh at each candidate value (a deterministic, gradient-based algorithm, so
this is equivalent to -- if more wasteful than -- inspecting a single run's
internal trace) until it finds the SMALLEST operator count whose energy is
within --tolerance-mha of the reference. This trades extra compute for
staying entirely within the documented public API.

VALIDATION WARNING: neither this script nor the underlying cudaq-solvers
calls have been exercised end-to-end in this development environment (no
CUDA-Q runtime available here; the API surface above was reverse-engineered
from the actual compiled bindings' embedded docstrings and the shipped
pyscf generator source, not from a live run). Before trusting this at
dataset scale:
  1. Run it on H2 in STO-3G with n_active_electrons=2, n_active_orbitals=2,
     --reference casci, and confirm E_HF matches ~-1.1167 Ha and E_CASCI
     (=FCI here) matches ~-1.1373 Ha at the 0.735 A bond length.
  2. Confirm molecule.energies actually contains the 'R-CASCI'/'R-CCSD' keys
     assumed in get_reference_energy() below -- these were read directly out
     of gas_phase_generator.py's `results['energies'][...]` dict construction
     but could differ by cudaq-solvers version.
  3. Confirm cudaq.SpinOperator's __str__ is a reliable identity key for
     matching selected operators back to pool indices in
     operators_to_index_sequence() below -- if two distinct pool operators
     ever stringify identically this will silently misattribute an index.

Usage:
  python run_adapt_vqe.py \\
      --xyz conformer_000.xyz --basis 6-31g --charge 0 --spin 0 \\
      --n-active-electrons 6 --n-active-orbitals 6 \\
      --pool uccgsd --reference casci --tolerance-mha 5.0 \\
      --max-operators 150 --output conformer_000_circuit.json
"""
import argparse
import json


def read_geometry(xyz_path):
    from ase.io import read

    atoms = read(xyz_path)
    return [
        (s, (float(p[0]), float(p[1]), float(p[2])))
        for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    ]


def make_initial_state(n_qubits, n_electrons):
    """Hartree-Fock reference state under Jordan-Wigner mapping: occupy the
    first n_electrons (spin-)orbitals. Dimensions are baked in via closure
    since adapt_vqe's initial-state kernel is expected to take no runtime
    arguments -- verify this against a live CUDA-Q session if the call fails
    with an argument-count error."""
    import cudaq

    @cudaq.kernel
    def initial_state():
        q = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(q[i])

    return initial_state


def build_molecule(args):
    import cudaq_solvers as solvers

    geometry = read_geometry(args.xyz)
    return solvers.create_molecule(
        geometry=geometry,
        basis=args.basis,
        spin=args.spin,
        charge=args.charge,
        nele_cas=args.n_active_electrons,
        norb_cas=args.n_active_orbitals,
        casci=(args.reference == "casci"),
        ccsd=(args.reference == "ccsd"),
    )


def get_reference_energy(molecule, reference):
    key = "R-CASCI" if reference == "casci" else "R-CCSD"
    if key not in molecule.energies:
        raise KeyError(
            f"Expected '{key}' in molecule.energies, got keys "
            f"{list(molecule.energies.keys())}. These key names were read "
            f"directly out of the shipped gas_phase_generator.py source but "
            f"may differ by cudaq-solvers version -- update `key` above."
        )
    return molecule.energies[key]


def run_adapt_vqe_fixed_iter(molecule, pool, initial_state, max_iter, options):
    import cudaq_solvers as solvers

    call_options = dict(options)
    call_options["max_iter"] = max_iter
    energy, params, operators = solvers.adapt_vqe(
        initial_state, molecule.spin_op, pool, **call_options
    )
    return energy, params, operators


def find_minimal_operator_count(
    molecule, pool, initial_state, e_ref, tolerance_ha, max_operators, options
):
    """Exponential-then-binary search over max_iter for the smallest
    operator count whose ADAPT-VQE energy is within tolerance_ha of e_ref.
    See the module docstring for why this outer-loop approach is needed
    given the documented public API surface."""

    def attempt(k):
        result = run_adapt_vqe_fixed_iter(molecule, pool, initial_state, k, options)
        energy = result[0]
        ok = abs(energy - e_ref) <= tolerance_ha
        print(
            f"  max_iter={k}: energy={energy:.8f} Ha, "
            f"|E-E_ref|={abs(energy - e_ref) * 1000:.3f} mHa, converged={ok}"
        )
        return ok, result

    k = 1
    last_fail = 0
    result = None
    converged = False
    while k <= max_operators:
        converged, result = attempt(k)
        if converged:
            break
        last_fail = k
        k *= 2

    if not converged:
        print(
            f"WARNING: did not reach tolerance within --max-operators="
            f"{max_operators}; returning the best (largest) attempt."
        )
        return result, min(k, max_operators)

    hi, lo = k, last_fail
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, candidate = attempt(mid)
        if ok:
            hi, result = mid, candidate
        else:
            lo = mid
    return result, hi


def operators_to_index_sequence(pool, operators, params):
    """Map each ADAPT-VQE-selected operator back to its index in the
    ORIGINAL pool list, producing the paper's [(pool_index, coefficient)]
    sequence (Sec. 3.2.1). Matching is done by str(SpinOperator) as a
    stand-in identity key -- see the module docstring's validation item 3."""
    pool_lookup = {str(op): idx for idx, op in enumerate(pool)}
    sequence = []
    for op, coeff in zip(operators, params):
        key = str(op)
        if key not in pool_lookup:
            raise ValueError(
                f"Selected operator not found in the original pool by string "
                f"match -- cudaq.SpinOperator's __str__ may not be a reliable "
                f"identity key in your cudaq-solvers version. Selected op: {key}"
            )
        sequence.append((pool_lookup[key], coeff))
    return sequence


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--xyz", required=True)
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--n-active-electrons", type=int, required=True)
    p.add_argument("--n-active-orbitals", type=int, required=True)
    p.add_argument("--pool", choices=["uccsd", "uccgsd", "upccgsd"], required=True)
    p.add_argument(
        "--reference",
        choices=["casci", "ccsd"],
        required=True,
        help="casci for 12/14-qubit configs, ccsd for the 16-qubit config",
    )
    p.add_argument(
        "--tolerance-mha",
        type=float,
        required=True,
        help="Energy convergence tolerance epsilon, mHa (paper values: "
        "5/10 for 12q, 5/16 for 14q, 15 for 16q)",
    )
    p.add_argument("--max-operators", type=int, default=150)
    p.add_argument(
        "--grad-norm-tolerance",
        type=float,
        default=1e-6,
        help="Set well below the scale implied by --tolerance-mha so this "
        "script's own energy check -- not adapt_vqe's internal early "
        "stopping -- is what determines convergence",
    )
    p.add_argument("--threshold-energy", type=float, default=1e-8)
    p.add_argument("--dynamic-start", choices=["warm", "cold"], default="warm")
    p.add_argument("--optimizer", default="cobyla")
    p.add_argument("--output", required=True)

    args = p.parse_args()
    n_qubits = 2 * args.n_active_orbitals
    tolerance_ha = args.tolerance_mha / 1000.0

    molecule = build_molecule(args)
    e_ref = get_reference_energy(molecule, args.reference)
    print(
        f"E_HF={molecule.energies.get('hf_energy')}, "
        f"E_{args.reference}={e_ref}, n_qubits={n_qubits}"
    )

    import cudaq_solvers as solvers

    pool = solvers.get_operator_pool(
        args.pool, n_qubits=n_qubits, n_electrons=molecule.n_electrons
    )
    print(f"Operator pool '{args.pool}': {len(pool)} candidate operators")

    initial_state = make_initial_state(n_qubits, molecule.n_electrons)

    options = dict(
        optimizer=args.optimizer,
        grad_norm_tolerance=args.grad_norm_tolerance,
        threshold_energy=args.threshold_energy,
        dynamic_start=args.dynamic_start,
        verbose=False,
    )

    (energy, params, operators), n_ops = find_minimal_operator_count(
        molecule, pool, initial_state, e_ref, tolerance_ha, args.max_operators, options
    )

    index_sequence = operators_to_index_sequence(pool, operators, params)
    sequence_str = "".join(f"<op{idx}>{coeff:.6f}" for idx, coeff in index_sequence)

    result = {
        "xyz": args.xyz,
        "n_qubits": n_qubits,
        "n_active_electrons": args.n_active_electrons,
        "n_active_orbitals": args.n_active_orbitals,
        "pool": args.pool,
        "reference": args.reference,
        "e_ref": e_ref,
        "e_hf": molecule.energies.get("hf_energy"),
        "tolerance_mha": args.tolerance_mha,
        "n_operators": n_ops,
        "final_energy": energy,
        "energy_error_mha": (energy - e_ref) * 1000.0,
        "operator_sequence": index_sequence,  # [(pool_index, coefficient), ...]
        "sequence_str": sequence_str,  # Sec. 3.2.1 "Ok ck" token format
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"Converged at {n_ops} operators, E={energy:.8f} Ha "
        f"({result['energy_error_mha']:.3f} mHa from {args.reference.upper()})"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
