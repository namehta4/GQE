#!/usr/bin/env python3
"""
Build a gate-level circuit from a generated operator_sequence, and optimize
it with pytket, reproducing the CANONICAL-vs-OPTIMIZED two-qubit gate count
comparison of Figure 8 in ADAPT-GQE (arXiv:2607.22468).

SCOPE HONESTLY: this targets a GENERIC universal gateset (CX-based), not
Quantinuum Helios-1's actual native gateset. Device-specific compilation
requires the `pytket-quantinuum` extension plus Quantinuum Nexus/InQuanto
account credentials to resolve the target backend's exact native gate
definitions -- neither is available in this environment, and I have not
fabricated device-specific numbers. If you have Nexus access, replace
CANONICAL_REBASE/OPTIMIZE_PASS below with
`pytket.extensions.quantinuum.QuantinuumBackend(...).get_compiled_circuit()`
against your actual device/emulator target, which will also apply
device-specific noise-aware passes this script does not attempt.

The canonical (pre-optimization) circuit uses the standard textbook
Pauli-exponential decomposition (basis-change gates + CNOT ladder + single
Rz + undo), matching the Yordanov et al. scheme the paper cites [50]. Gate
COUNTS from this part are robust regardless of sign/phase convention (a
CNOT ladder's length doesn't depend on the sign of theta). The rotation
ANGLE applied does depend on the same sign convention flagged in
circuit_energy.py -- if you need the compiled circuit's output STATE (not
just its two-qubit gate count) to match the cudaq-simulated one, validate
that angle convention the same way test_circuit_energy.py does, before
trusting anything beyond the gate-count statistics here.

Usage:
  python build_native_circuit.py \\
      --xyz conformer.xyz --basis 6-31g --charge 0 --spin 0 \\
      --n-active-electrons 6 --n-active-orbitals 6 --pool uccgsd \\
      --operator-sequence-json op_sequence.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chemistry"))
from circuit_energy import decompose_spin_operator  # noqa: E402
from evaluate_generated_circuits import build_molecule_and_pool  # noqa: E402


def pauli_word_to_qubit_ops(pauli_word_str):
    """'IXYZI...' -> list of (qubit_index, 'X'/'Y'/'Z') for non-identity positions."""
    return [(i, ch) for i, ch in enumerate(pauli_word_str) if ch != "I"]


def append_pauli_exponential(circ, qubit_ops, theta):
    """Standard exp(-i*theta/2 * P) compilation: basis-change, CNOT ladder,
    single Rz, undo -- see module docstring for the angle-convention caveat.
    """
    qubits = [q for q, _ in qubit_ops]
    paulis = [p for _, p in qubit_ops]

    for q, p in zip(qubits, paulis):
        if p == "X":
            circ.H(q)
        elif p == "Y":
            circ.V(q)  # pytket's V = sqrt(X)-like basis-change gate for Y

    for i in range(len(qubits) - 1):
        circ.CX(qubits[i], qubits[i + 1])

    if qubits:
        circ.Rz(theta, qubits[-1])

    for i in reversed(range(len(qubits) - 1)):
        circ.CX(qubits[i], qubits[i + 1])

    for q, p in zip(qubits, paulis):
        if p == "X":
            circ.H(q)
        elif p == "Y":
            circ.Vdg(q)


def build_canonical_circuit(n_qubits, n_electrons, pool, operator_sequence):
    from pytket import Circuit

    circ = Circuit(n_qubits)
    for i in range(n_electrons):
        circ.X(i)

    for idx, theta in operator_sequence:
        words, coeffs = decompose_spin_operator(pool[idx])
        for word, coeff in zip(words, coeffs):
            qubit_ops = pauli_word_to_qubit_ops(str(word))
            if not qubit_ops:
                continue  # identity term contributes only a global phase
            append_pauli_exponential(circ, qubit_ops, theta * coeff)

    return circ


def count_two_qubit_gates(circ):
    from pytket import OpType

    return circ.n_gates_of_type(OpType.CX)


def optimize_circuit(circ):
    from pytket.passes import FullPeepholeOptimise, SequencePass, DecomposeBoxes

    optimized = circ.copy()
    SequencePass([DecomposeBoxes(), FullPeepholeOptimise()]).apply(optimized)
    return optimized


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
        "--operator-sequence-json",
        required=True,
        help="JSON file: list of [pool_index, theta] pairs",
    )
    p.add_argument("--output-qasm", default=None, help="Optional: dump the optimized circuit as QASM")

    args = p.parse_args()

    with open(args.operator_sequence_json) as f:
        operator_sequence = [tuple(pair) for pair in json.load(f)]

    _molecule, pool, n_qubits, n_electrons = build_molecule_and_pool(args.xyz, args)

    canonical = build_canonical_circuit(n_qubits, n_electrons, pool, operator_sequence)
    canonical_2q = count_two_qubit_gates(canonical)

    optimized = optimize_circuit(canonical)
    optimized_2q = count_two_qubit_gates(optimized)

    print(f"Canonical two-qubit gate count: {canonical_2q}")
    print(f"Optimized two-qubit gate count: {optimized_2q}")
    if canonical_2q > 0:
        print(f"Reduction factor: {canonical_2q / max(optimized_2q, 1):.2f}x")

    if args.output_qasm:
        from pytket.qasm import circuit_to_qasm_str

        with open(args.output_qasm, "w") as f:
            f.write(circuit_to_qasm_str(optimized))
        print(f"Wrote {args.output_qasm}")


if __name__ == "__main__":
    main()
