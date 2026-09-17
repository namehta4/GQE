#!/usr/bin/env python3
"""
Batch-run ADAPT-VQE (via run_adapt_vqe.py's functions) across every conformer
in a manifest, for ONE active-space configuration, reproducing the five
training datasets in ADAPT-GQE (arXiv:2607.22468), Table 2 / Section 3.1:

  12 qubits: (6e,6o), UCCGSD pool, epsilon in {5, 10} mHa, reference=casci
  14 qubits: (6e,7o), UCCGSD pool, epsilon in {5, 16} mHa, reference=casci
  16 qubits: (8e,8o), UCCSD  pool, epsilon = 15 mHa,        reference=ccsd

Run this script once per (qubit count, tolerance) combination -- five times
total to reproduce all of Table 2.

Each conformer requires its own exponential-then-binary search over
max_iter (see run_adapt_vqe.py's docstring for why), so this is expensive:
budget accordingly and use --limit for a smoke test before a full run, and
--skip-existing to resume an interrupted batch.

Note: the main-text results use a RANDOM 80/18/2 train/val/test split over
the pooled dataset (this script's default). The Appendix A.1.1
out-of-distribution generalization splits (by conformer class, MD->NEB
holdout, MD+NEB->OoD holdout) are a separate analysis -- the per-row
"source" field this script preserves (MD/NEB/OOD, from
build_conformer_manifest.py) is what you'd filter on to build those splits.

Usage:
  python build_adapt_vqe_dataset.py \\
      --manifest full_dataset/manifest.csv \\
      --n-active-electrons 6 --n-active-orbitals 6 \\
      --pool uccgsd --reference casci --tolerance-mha 5.0 \\
      --output-dir adaptvqe_12q_eps5
"""
import argparse
import csv
import json
import os
import random

from run_adapt_vqe import (
    build_molecule,
    find_minimal_operator_count,
    get_reference_energy,
    make_initial_state,
    operators_to_index_sequence,
)


class _Args:
    """Lightweight shim matching the attribute shape run_adapt_vqe.build_molecule
    expects, without going through argparse for every conformer."""


def process_one_conformer(row, args, options):
    import cudaq_solvers as solvers

    a = _Args()
    a.xyz = row["path"]
    a.basis = args.basis
    a.charge = args.charge
    a.spin = args.spin
    a.n_active_electrons = args.n_active_electrons
    a.n_active_orbitals = args.n_active_orbitals
    a.reference = args.reference

    molecule = build_molecule(a)
    e_ref = get_reference_energy(molecule, args.reference)

    n_qubits = 2 * args.n_active_orbitals
    pool = solvers.get_operator_pool(
        args.pool, n_qubits=n_qubits, n_electrons=molecule.n_electrons
    )
    initial_state = make_initial_state(n_qubits, molecule.n_electrons)

    (energy, params, operators), n_ops = find_minimal_operator_count(
        molecule,
        pool,
        initial_state,
        e_ref,
        args.tolerance_mha / 1000.0,
        args.max_operators,
        options,
    )
    index_sequence = operators_to_index_sequence(pool, operators, params)
    sequence_str = "".join(f"<op{idx}>{coeff:.6f}" for idx, coeff in index_sequence)

    return {
        "conformer_id": row["conformer_id"],
        "source": row["source"],
        "xyz": row["path"],
        "n_operators": n_ops,
        "final_energy": energy,
        "e_ref": e_ref,
        "e_hf": molecule.energies.get("hf_energy"),
        "energy_error_mha": (energy - e_ref) * 1000.0,
        # Structured (pool_index, coefficient) pairs -- the ML dataset builder
        # (model/build_training_examples.py) needs this directly to interleave
        # the <ham> multimodal token; sequence_str is kept only for quick
        # human inspection.
        "operator_sequence": index_sequence,
        "sequence_str": sequence_str,
    }


def write_splits(results_path, output_dir, train_frac, val_frac, seed):
    with open(results_path) as f:
        conformer_ids = [json.loads(line)["conformer_id"] for line in f]

    rng = random.Random(seed)
    rng.shuffle(conformer_ids)

    n = len(conformer_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = {
        "train": conformer_ids[:n_train],
        "val": conformer_ids[n_train : n_train + n_val],
        "test": conformer_ids[n_train + n_val :],
    }
    split_path = os.path.join(output_dir, "splits.json")
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(
        f"Splits ({n} total): train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])} -> {split_path}"
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--manifest", required=True, help="manifest.csv from build_conformer_manifest.py"
    )
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--n-active-electrons", type=int, required=True)
    p.add_argument("--n-active-orbitals", type=int, required=True)
    p.add_argument("--pool", choices=["uccsd", "uccgsd", "upccgsd"], required=True)
    p.add_argument("--reference", choices=["casci", "ccsd"], required=True)
    p.add_argument("--tolerance-mha", type=float, required=True)
    p.add_argument("--max-operators", type=int, default=150)
    p.add_argument("--grad-norm-tolerance", type=float, default=1e-6)
    p.add_argument("--threshold-energy", type=float, default=1e-8)
    p.add_argument("--dynamic-start", choices=["warm", "cold"], default="warm")
    p.add_argument("--optimizer", default="cobyla")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--limit", type=int, default=None, help="Process only the first N manifest rows"
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume: skip conformer_ids already present in circuits.jsonl",
    )
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--val-frac", type=float, default=0.18)
    p.add_argument("--split-seed", type=int, default=0xC0FFEE)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    options = dict(
        optimizer=args.optimizer,
        grad_norm_tolerance=args.grad_norm_tolerance,
        threshold_energy=args.threshold_energy,
        dynamic_start=args.dynamic_start,
        verbose=False,
    )

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    results_path = os.path.join(args.output_dir, "circuits.jsonl")
    already_done = set()
    if args.skip_existing and os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                already_done.add(json.loads(line)["conformer_id"])
        print(f"Resuming: {len(already_done)} conformer(s) already done")

    n_failed = 0
    with open(results_path, "a" if args.skip_existing else "w") as out_f:
        for i, row in enumerate(rows):
            if row["conformer_id"] in already_done:
                continue
            print(f"[{i + 1}/{len(rows)}] {row['conformer_id']}")
            try:
                result = process_one_conformer(row, args, options)
            except Exception as e:
                print(f"  FAILED: {e}")
                n_failed += 1
                continue
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    print(f"Done. {len(rows) - n_failed} succeeded, {n_failed} failed.")
    print(f"Circuits: {results_path}")

    write_splits(results_path, args.output_dir, args.train_frac, args.val_frac, args.split_seed)


if __name__ == "__main__":
    main()
