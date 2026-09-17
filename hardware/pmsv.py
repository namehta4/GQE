"""
Simplified Partition Measurement Symmetry Verification (PMSV), a
particle-number-based error-mitigation post-selection technique, per Sec.
4.3 of ADAPT-GQE (arXiv:2607.22468), citing Yamamoto et al., Phys. Rev. Res.
4, 033110 (2022).

THIS IS NOT A FAITHFUL REPRODUCTION of the cited paper's algorithm or of
InQuanto's actual PMSV implementation (a Quantinuum commercial product I
cannot install or inspect, unlike cudaq-solvers). The real technique groups
Hamiltonian terms into simultaneously-measurable ("partition") bases and
constructs symmetry-check Pauli strings that share each partition's
measurement basis, so the check costs no extra circuit executions. What's
implemented here is the conceptually-aligned but much simpler version: given
raw measurement bitstrings (one JW-mapped qubit register per shot), discard
any shot whose Hamming weight doesn't match the expected particle number.
This captures the core error-mitigation IDEA (post-select on a conserved
symmetry) for a basic demonstration, not the paper's measurement-efficient
construction.

Also includes a NOISELESS LOCAL STATEVECTOR SHOT SIMULATOR, purely so this
module has something to demonstrate post-selection against without real
hardware access. It is not a substitute for the Helios-1 emulator's fitted
noise model used in the paper's Table 1 -- with no noise source, the
"discard rate" this simulator produces will be near zero (real hardware/
emulator noise is exactly what drives the paper's 20-50% discard rates in
Table 1), so do not compare numbers from this simulator against the paper's
reported values.
"""
import numpy as np


def particle_number_ok(bitstring, n_electrons):
    """Under Jordan-Wigner, occupation number = Hamming weight."""
    return bitstring.count("1") == n_electrons


def post_select_shots(bitstrings, n_electrons):
    """bitstrings: list of str (e.g. '01101...'). Returns (kept, discard_rate)."""
    kept = [b for b in bitstrings if particle_number_ok(b, n_electrons)]
    discard_rate = 1.0 - (len(kept) / len(bitstrings) if bitstrings else 0.0)
    return kept, discard_rate


def sample_shots_noiseless(statevector, n_shots, seed=0):
    """Sample computational-basis bitstrings from an exact statevector --
    a noiseless stand-in for real hardware shots. See module docstring:
    this will NOT reproduce the paper's noise-driven discard rates."""
    probs = np.abs(statevector) ** 2
    probs = probs / probs.sum()
    n_qubits = int(np.log2(len(statevector)))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(statevector), size=n_shots, p=probs)
    return [format(i, f"0{n_qubits}b")[::-1] for i in indices]  # qubit 0 = leftmost bit here


def estimate_expectation_with_pmsv(bitstrings, pauli_diagonal_signs, n_electrons):
    """Estimate <P> for a diagonal (Z-basis-measured) Pauli operator from raw
    shots, with and without PMSV post-selection, for comparison.

    pauli_diagonal_signs: function bitstring -> +-1, the eigenvalue of the
    (already basis-rotated) Pauli operator for that computational-basis
    outcome -- the caller is responsible for having applied whatever
    basis-rotation gates make the operator diagonal before "measurement"
    (this module only handles the post-selection statistics, not circuit
    construction -- see build_native_circuit.py for that).
    """
    raw_mean = np.mean([pauli_diagonal_signs(b) for b in bitstrings])

    kept, discard_rate = post_select_shots(bitstrings, n_electrons)
    mitigated_mean = np.mean([pauli_diagonal_signs(b) for b in kept]) if kept else float("nan")

    return {
        "raw_expectation": float(raw_mean),
        "mitigated_expectation": float(mitigated_mean),
        "discard_rate": discard_rate,
        "n_shots": len(bitstrings),
        "n_kept": len(kept),
    }
