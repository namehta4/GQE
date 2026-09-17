#!/usr/bin/env python3
"""
RUN THIS FIRST, before trusting circuit_energy.py on real data.

Isolates the two highest-risk assumptions in circuit_energy.py -- the
cudaq.SpinOperator term-decomposition and the exp_pauli sign/phase
convention -- against an independent brute-force cross-check, on a system
small enough (H2, STO-3G, 4 qubits) that dense-matrix linear algebra is
trivial. This test needs no physical reference value and no chemistry
knowledge to interpret: it just checks that two independent ways of
computing "apply exp(theta * pool_operator) to the HF state" agree.

If this test fails, do NOT try to patch circuit_energy.py blindly --
instead, use a live Python/cudaq session to inspect:
  - `for term in pool[0]: print(term, term.get_coefficient())` to see the
    ACTUAL string format and coefficient convention your cudaq version uses,
    and fix decompose_spin_operator() to match.
  - Try both signs of the rotation angle in evaluate_energy() if the
    overlap test below comes out consistently ~0 rather than ~1 (a sign
    flip produces exp(-theta*O) instead of exp(+theta*O), which for a
    generic non-involutory operator gives a definite, reproducible mismatch
    rather than noise).

Usage:
  python test_circuit_energy.py
"""
import numpy as np


def main():
    import cudaq
    import cudaq_solvers as solvers
    from scipy.linalg import expm

    from circuit_energy import build_ansatz_kernel, decompose_spin_operator

    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    molecule = solvers.create_molecule(
        geometry=geometry, basis="sto-3g", spin=0, charge=0, casci=True
    )
    n_qubits = 2 * molecule.n_orbitals
    n_electrons = molecule.n_electrons
    print(f"H2/STO-3G: n_qubits={n_qubits}, n_electrons={n_electrons}")
    print(f"E_HF={molecule.energies.get('hf_energy')}, "
          f"E_CASCI/FCI={molecule.energies.get('R-CASCI') or molecule.energies.get('fci_energy')}")

    pool = solvers.get_operator_pool("uccsd", n_qubits=n_qubits, n_electrons=n_electrons)
    print(f"Pool size: {len(pool)}")
    if len(pool) == 0:
        raise RuntimeError("Empty pool -- cannot run this test, check get_operator_pool call")

    theta = 0.37  # arbitrary, nonzero, not a special/symmetric value
    op = pool[0]
    words, coeffs = decompose_spin_operator(op)

    # --- Path A: our kernel machinery (circuit_energy.py) ---
    ansatz = build_ansatz_kernel()
    state_a = np.array(
        cudaq.get_state(ansatz, n_qubits, n_electrons, [words], [coeffs], [theta])
    )

    # --- Path B: independent brute-force check ---
    # HF reference statevector: computational basis state with the first
    # n_electrons qubits set to |1>, matching build_ansatz_kernel()'s x(q[i])
    # loop and assuming cudaq's default qubit-index-to-bit-position mapping
    # (qubit 0 = least significant bit) -- if Path A/B disagree, this
    # ordering assumption is the first thing to double check.
    dim = 2**n_qubits
    hf_index = sum(1 << i for i in range(n_electrons))
    hf_state = np.zeros(dim, dtype=complex)
    hf_state[hf_index] = 1.0

    op_matrix = np.array(op.to_matrix())  # (dim, dim) dense matrix of the pool operator itself
    state_b = expm(theta * op_matrix) @ hf_state

    overlap = np.abs(np.vdot(state_a, state_b))
    print(f"\n|<state_from_kernel | state_from_matrix_exp>| = {overlap:.6f}")
    if overlap > 0.999:
        print("PASS: kernel-based Trotter step matches brute-force matrix exponential.")
    else:
        print(
            "FAIL: mismatch. See this file's module docstring for how to "
            "diagnose (check term string format, try flipping theta's sign, "
            "check qubit-ordering convention)."
        )


if __name__ == "__main__":
    main()
