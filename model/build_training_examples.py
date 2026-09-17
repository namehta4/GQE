#!/usr/bin/env python3
"""
Assemble tokenized training examples from ADAPT-VQE circuits + Hamiltonian
coefficient vectors, for the trained-from-scratch Gemma model (Sections
3.2.1-3.2.3 of ADAPT-GQE, arXiv:2607.22468).

Reads:
  --circuits         circuits.jsonl from chemistry/build_adapt_vqe_dataset.py
  --splits           splits.json from the same
  --hamiltonian-dir  <conformer_id>.npz files from
                      chemistry/build_hamiltonian_vectors_for_dataset.py

Fits the Hamiltonian coefficient normalizer (max-abs per position) on the
TRAINING split only, then applies it to all splits -- do not skip this step
or refit per-split, or validation/test will leak scale information / be
inconsistently normalized relative to what the model was trained on.

Usage:
  python build_training_examples.py \\
      --circuits adaptvqe_12q_eps5/circuits.jsonl \\
      --splits adaptvqe_12q_eps5/splits.json \\
      --hamiltonian-dir hamiltonians_12q \\
      --pool-size 500 \\
      --output-dir training_examples_12q_eps5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hamiltonian_encoder import HamiltonianNormalizer  # noqa: E402
from operator_tokenizer import OperatorTokenizer  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--circuits", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--hamiltonian-dir", required=True)
    p.add_argument(
        "--pool-size",
        type=int,
        required=True,
        help="Operator pool size for this active space/pool combination "
        "(from len(pool) printed by run_adapt_vqe.py / "
        "build_adapt_vqe_dataset.py)",
    )
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--output-dir", required=True)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = OperatorTokenizer(pool_size=args.pool_size)
    tokenizer.save_pretrained(args.output_dir)

    with open(args.splits) as f:
        splits = json.load(f)
    split_of = {}
    for split_name, ids in splits.items():
        for cid in ids:
            split_of[cid] = split_name

    circuits = {}
    with open(args.circuits) as f:
        for line in f:
            row = json.loads(line)
            circuits[row["conformer_id"]] = row

    train_vectors = []
    for cid in splits["train"]:
        npz_path = os.path.join(args.hamiltonian_dir, f"{cid}.npz")
        if os.path.exists(npz_path):
            train_vectors.append(np.load(npz_path)["coefficients"])
    if not train_vectors:
        raise RuntimeError(
            "No Hamiltonian vectors found for any training-split conformer -- "
            "run build_hamiltonian_vectors_for_dataset.py first"
        )
    normalizer = HamiltonianNormalizer().fit(np.stack(train_vectors))
    normalizer.save(os.path.join(args.output_dir, "hamiltonian_scale.npy"))

    examples = {"train": [], "val": [], "test": []}
    n_skipped_length = 0
    n_missing_ham = 0

    for cid, row in circuits.items():
        split = split_of.get(cid)
        if split is None:
            continue

        npz_path = os.path.join(args.hamiltonian_dir, f"{cid}.npz")
        if not os.path.exists(npz_path):
            n_missing_ham += 1
            continue
        ham_vector = normalizer.transform(np.load(npz_path)["coefficients"])

        text = tokenizer.build_sequence_text(row["operator_sequence"])
        input_ids = tokenizer.encode(text)
        if len(input_ids) > args.max_length:
            n_skipped_length += 1
            continue

        examples[split].append(
            {
                "conformer_id": cid,
                "input_ids": input_ids,
                "hamiltonian": ham_vector.tolist(),
                "n_operators": row["n_operators"],
                "e_ref": row["e_ref"],
                "e_hf": row["e_hf"],
            }
        )

    for split, rows in examples.items():
        out_path = os.path.join(args.output_dir, f"{split}.jsonl")
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(rows)} example(s) -> {out_path}")

    if n_skipped_length:
        print(f"Skipped {n_skipped_length} example(s) exceeding --max-length={args.max_length}")
    if n_missing_ham:
        print(
            f"Skipped {n_missing_ham} example(s) missing a Hamiltonian vector "
            f"(run build_hamiltonian_vectors_for_dataset.py first)"
        )


if __name__ == "__main__":
    main()
