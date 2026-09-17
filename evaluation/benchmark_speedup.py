#!/usr/bin/env python3
"""
Benchmark ADAPT-GQE inference time against ADAPT-VQE wall-clock time for one
conformer, reproducing the methodology of Section 4.2 in ADAPT-GQE
(arXiv:2607.22468): speedup = T_ADAPT-VQE / (T_generation + T_energy_eval).

This is a single-conformer, single-run timing (matching the paper's own
"timings reported here are measured for a single selected imipramine
conformer" methodology) -- wall-clock timings are noisy, so treat one run as
indicative, not authoritative; average over a few runs if you need a stable
number.

Usage:
  python benchmark_speedup.py \\
      --xyz conformer_000.xyz \\
      --checkpoint checkpoints_12q_eps5/best_checkpoint.pt \\
      --tokenizer-dir training_examples_12q_eps5 \\
      --term-order-file term_order_12q.json \\
      --hamiltonian-scale training_examples_12q_eps5/hamiltonian_scale.npy \\
      --basis 6-31g --charge 0 --spin 0 \\
      --n-active-electrons 6 --n-active-orbitals 6 --pool uccgsd \\
      --reference casci --tolerance-mha 5.0 \\
      --num-candidates 16
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chemistry"))

from build_hamiltonian import (  # noqa: E402
    build_pyscf_mol,
    build_qubit_hamiltonian,
    load_or_init_term_order,
    run_reference_methods,
)
from circuit_energy import build_ansatz_kernel, evaluate_energy  # noqa: E402
from evaluate_generated_circuits import build_molecule_and_pool  # noqa: E402
from generate import generate, load_model_from_checkpoint, parse_generated_sequence  # noqa: E402
from hamiltonian_encoder import HamiltonianNormalizer  # noqa: E402
from operator_tokenizer import OperatorTokenizer  # noqa: E402
from run_adapt_vqe import build_molecule, find_minimal_operator_count, get_reference_energy, make_initial_state  # noqa: E402


def time_adapt_vqe(args):
    import cudaq_solvers as solvers

    t0 = time.time()
    molecule = build_molecule(args)
    e_ref = get_reference_energy(molecule, args.reference)
    n_qubits = 2 * args.n_active_orbitals
    pool = solvers.get_operator_pool(args.pool, n_qubits=n_qubits, n_electrons=molecule.n_electrons)
    initial_state = make_initial_state(n_qubits, molecule.n_electrons)
    options = dict(
        optimizer="cobyla",
        grad_norm_tolerance=1e-6,
        threshold_energy=1e-8,
        dynamic_start="warm",
        verbose=False,
    )
    (energy, _params, _ops), n_ops = find_minimal_operator_count(
        molecule, pool, initial_state, e_ref, args.tolerance_mha / 1000.0, args.max_operators, options
    )
    elapsed = time.time() - t0
    return elapsed, n_ops, energy


def build_normalized_hamiltonian(args):
    mol = build_pyscf_mol(args.xyz, args.basis, args.charge, args.spin)
    mf, _e_casci, _e_ccsd = run_reference_methods(
        mol, args.n_active_electrons, args.n_active_orbitals, want_ccsd=False
    )
    qubit_hamiltonian = build_qubit_hamiltonian(mol, mf, args.n_active_electrons, args.n_active_orbitals)
    n_qubits = 2 * args.n_active_orbitals
    vector, _order = load_or_init_term_order(qubit_hamiltonian, n_qubits, args.term_order_file)
    normalizer = HamiltonianNormalizer.load(args.hamiltonian_scale)
    return normalizer.transform(vector)


def time_adapt_gqe(args):
    tokenizer = OperatorTokenizer.from_pretrained(args.tokenizer_dir)
    model = load_model_from_checkpoint(args.checkpoint, args.device)

    ham_vector = build_normalized_hamiltonian(args)
    ham_tensor = torch.tensor(ham_vector, dtype=torch.float32)

    t0 = time.time()
    per_conformer = generate(
        model,
        tokenizer,
        ham_tensor,
        num_candidates=args.num_candidates,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    t_gen = time.time() - t0

    molecule, pool, n_qubits, n_electrons = build_molecule_and_pool(args.xyz, args)
    ansatz = build_ansatz_kernel()

    t0 = time.time()
    energies = []
    for token_ids in per_conformer[0]:
        op_sequence, _n_invalid = parse_generated_sequence(token_ids, tokenizer)
        try:
            energies.append(
                evaluate_energy(molecule, pool, op_sequence, n_qubits, n_electrons, ansatz=ansatz)
            )
        except Exception as e:
            print(f"  candidate energy eval failed: {e}")
    t_eval = time.time() - t0

    best_energy = min(energies) if energies else float("nan")
    return t_gen, t_eval, best_energy


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--xyz", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--term-order-file", required=True)
    p.add_argument("--hamiltonian-scale", required=True)

    p.add_argument("--basis", default="6-31g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--n-active-electrons", type=int, required=True)
    p.add_argument("--n-active-orbitals", type=int, required=True)
    p.add_argument("--pool", choices=["uccsd", "uccgsd", "upccgsd"], required=True)
    p.add_argument("--reference", choices=["casci", "ccsd"], required=True)
    p.add_argument("--tolerance-mha", type=float, required=True)
    p.add_argument("--max-operators", type=int, default=150)

    p.add_argument("--num-candidates", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    print("Timing ADAPT-VQE...")
    t_adapt_vqe, n_ops_vqe, e_adapt_vqe = time_adapt_vqe(args)
    print(f"  T_ADAPT-VQE = {t_adapt_vqe:.2f} s, {n_ops_vqe} operators, E={e_adapt_vqe:.8f} Ha")

    print("Timing ADAPT-GQE (generation + energy evaluation)...")
    t_gen, t_eval, e_gqe = time_adapt_gqe(args)
    print(f"  T_generation = {t_gen:.3f} s, T_energy_eval = {t_eval:.3f} s, best E={e_gqe:.8f} Ha")

    total_gqe = t_gen + t_eval
    speedup = t_adapt_vqe / total_gqe if total_gqe > 0 else float("inf")
    print(f"\nTotal ADAPT-GQE time: {total_gqe:.3f} s")
    print(f"Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
