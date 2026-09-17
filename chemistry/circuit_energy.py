"""
Apply a generated (pool_index, theta) operator sequence to the Hartree-Fock
reference state and evaluate <H>, using cudaq/cudaq_solvers primitives. This
is the missing link needed for best-of-N candidate selection (Sec. 4.1) and
the RL reward signal (Eq. 1, Sec. 3.2.3) -- a generated circuit is only as
good as its energy, and nothing upstream of this module can produce that
number.

HIGHEST-RISK MODULE IN THIS PIPELINE. Unlike run_adapt_vqe.py (which only
calls documented, black-box cudaq_solvers entry points), this module has to
reach INSIDE a cudaq.SpinOperator to Trotterize it term-by-term via
exp_pauli, which touches two things not independently confirmed against a
live cudaq install in this environment:
  1. cudaq.SpinOperator's term-iteration API (decompose_spin_operator below).
  2. The sign/phase convention exp_pauli(angle, qubits, word) uses relative
     to the anti-Hermitian UCC generator's own phase convention -- get this
     wrong and every energy this module reports is silently corrupted while
     still "running" without error.

DO NOT trust this module's output before running test_circuit_energy.py
(same directory), which isolates exactly these two risks against a
brute-force scipy matrix-exponential cross-check on a tiny system, and
passes/fails on its own without needing any physical reference value.
"""


def decompose_spin_operator(op):
    """Decompose a cudaq.SpinOperator into parallel (pauli_words,
    coefficients) lists for per-term Trotterized exponentiation.

    Assumes str(term) has the form "(<real>+<imag>j) <PAULISTRING>" and that
    UCC-style anti-Hermitian generator terms carry a purely imaginary
    coefficient (so the REAL rotation angle for exp_pauli is the imaginary
    part). See the module docstring -- validate with test_circuit_energy.py
    before trusting this on real data.
    """
    import cudaq

    words, coeffs = [], []
    for term in op:
        coeff = term.get_coefficient()
        pauli_str = str(term).strip().split()[-1]
        words.append(cudaq.pauli_word(pauli_str))
        coeffs.append(coeff.imag if abs(coeff.imag) > abs(coeff.real) else coeff.real)
    return words, coeffs


def build_ansatz_kernel():
    """Returns the parameterized ansatz kernel. Structure (loop over
    parallel words/coeffs/thetas lists calling exp_pauli) matches the
    standard CUDA-Q pattern for data-dependent, variable-length ansätze,
    consistent with how ADAPT-VQE itself must grow its own kernel
    internally at each iteration."""
    import cudaq

    @cudaq.kernel
    def ansatz(
        n_qubits: int,
        n_electrons: int,
        words: list[list[cudaq.pauli_word]],
        coeffs: list[list[float]],
        thetas: list[float],
    ):
        q = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(q[i])
        for k in range(len(thetas)):
            for j in range(len(words[k])):
                exp_pauli(thetas[k] * coeffs[k][j], q, words[k][j])

    return ansatz


def evaluate_energy(molecule, pool, operator_sequence, n_qubits, n_electrons, ansatz=None):
    """operator_sequence: list of (pool_index, theta) pairs, e.g. from
    model/generate.py's parse_generated_sequence(). Returns the expectation
    value <H> for the resulting ansatz state."""
    import cudaq

    if ansatz is None:
        ansatz = build_ansatz_kernel()

    words_per_op, coeffs_per_op, thetas = [], [], []
    for idx, theta in operator_sequence:
        words, coeffs = decompose_spin_operator(pool[idx])
        words_per_op.append(words)
        coeffs_per_op.append(coeffs)
        thetas.append(float(theta))

    result = cudaq.observe(
        ansatz, molecule.spin_op, n_qubits, n_electrons, words_per_op, coeffs_per_op, thetas
    )
    return result.expectation()


def compute_reward(e_hf, e_ref, e_circuit, r_max=30.0, eps=1e-8):
    """Eq. 1 of ADAPT-GQE: R = min((E_HF - E_ref) / (E_circuit - E_ref) - 1, R_max).
    Variationally E_circuit >= E_ref, so the denominator should be
    non-negative; `eps` is a defensive floor (not in the paper) against
    numerical noise pushing a near-converged E_circuit slightly below E_ref.
    """
    denom = e_circuit - e_ref
    denom = denom if denom > eps else eps
    return min((e_hf - e_ref) / denom - 1.0, r_max)
