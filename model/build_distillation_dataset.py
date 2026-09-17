#!/usr/bin/env python3
"""
Build one self-distillation round's SFT dataset, reproducing the
"Self-distillation" paragraph of Section 3.2.3 in ADAPT-GQE
(arXiv:2607.22468): filter the database of all GRPO-sampled sequences
(seeded with the original ADAPT-VQE circuits) down to the top-k
lowest-energy sequences per conformer below an energy threshold, for use as
the next SFT round's training data.

Reuses the ORIGINAL Hamiltonian normalizer (--hamiltonian-scale, from the
first build_training_examples.py run) rather than refitting -- the model's
encoder was trained under that fixed normalization, and refitting here would
silently shift the input distribution it expects.

Because top-k can keep MULTIPLE sequences per conformer (unlike the
one-circuit-per-conformer original ADAPT-VQE dataset), this naturally grows
the SFT set across rounds, matching the paper's description.

Usage:
  python build_distillation_dataset.py \\
      --circuits-jsonl adaptvqe_12q_eps5/circuits.jsonl \\
      --rollout-log rl_checkpoints_12q_eps5/rollouts.jsonl \\
      --hamiltonian-dir hamiltonians_12q \\
      --hamiltonian-scale training_examples_12q_eps5/hamiltonian_scale.npy \\
      --tokenizer-dir training_examples_12q_eps5 \\
      --top-n 8 --energy-threshold-mha 5.0 \\
      --output-dir distill_round1_12q_eps5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hamiltonian_encoder import HamiltonianNormalizer  # noqa: E402
from operator_tokenizer import OperatorTokenizer  # noqa: E402


def load_circuits(path):
    """conformer_id -> dict of e_ref/e_hf, plus one seed candidate from the
    original ADAPT-VQE run."""
    conformers = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            conformers[row["conformer_id"]] = {
                "e_ref": row["e_ref"],
                "e_hf": row["e_hf"],
                "candidates": [
                    {"operator_sequence": row["operator_sequence"], "energy": row["final_energy"]}
                ],
            }
    return conformers


def add_rollouts(conformers, rollout_log_paths):
    n_skipped_unknown_conformer = 0
    for path in rollout_log_paths:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                cid = record["conformer_id"]
                if cid not in conformers:
                    n_skipped_unknown_conformer += 1
                    continue
                if record.get("energy") is None:
                    continue  # failed energy evaluation during rollout, see rl_grpo.py
                conformers[cid]["candidates"].append(
                    {"operator_sequence": record["operator_sequence"], "energy": record["energy"]}
                )
    if n_skipped_unknown_conformer:
        print(
            f"Skipped {n_skipped_unknown_conformer} rollout record(s) for conformer_ids "
            f"not present in --circuits-jsonl (e_ref/e_hf unknown for them)"
        )


def select_top_n(conformers, top_n, threshold_ha):
    selected = {}
    for cid, data in conformers.items():
        e_ref = data["e_ref"]
        within_threshold = [
            c for c in data["candidates"] if (c["energy"] - e_ref) <= threshold_ha
        ]
        within_threshold.sort(key=lambda c: c["energy"])
        kept = within_threshold[:top_n]
        if kept:
            selected[cid] = kept
    return selected


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--circuits-jsonl", required=True, help="Original ADAPT-VQE circuits.jsonl")
    p.add_argument(
        "--rollout-log",
        action="append",
        required=True,
        help="rollouts.jsonl from rl_grpo.py; repeatable to combine multiple RL runs",
    )
    p.add_argument("--hamiltonian-dir", required=True)
    p.add_argument("--hamiltonian-scale", required=True, help="From the FIRST build_training_examples.py run")
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--top-n", type=int, required=True)
    p.add_argument("--energy-threshold-mha", type=float, required=True)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--output-dir", required=True)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = OperatorTokenizer.from_pretrained(args.tokenizer_dir)
    normalizer = HamiltonianNormalizer.load(args.hamiltonian_scale)

    conformers = load_circuits(args.circuits_jsonl)
    add_rollouts(conformers, args.rollout_log)
    selected = select_top_n(conformers, args.top_n, args.energy_threshold_mha / 1000.0)

    n_skipped_length, n_missing_ham, n_examples = 0, 0, 0
    out_path = os.path.join(args.output_dir, "train.jsonl")
    with open(out_path, "w") as out_f:
        for cid, candidates in selected.items():
            npz_path = os.path.join(args.hamiltonian_dir, f"{cid}.npz")
            if not os.path.exists(npz_path):
                n_missing_ham += 1
                continue
            ham_vector = normalizer.transform(np.load(npz_path)["coefficients"])
            e_ref = conformers[cid]["e_ref"]
            e_hf = conformers[cid]["e_hf"]

            for candidate in candidates:
                text = tokenizer.build_sequence_text(candidate["operator_sequence"])
                input_ids = tokenizer.encode(text)
                if len(input_ids) > args.max_length:
                    n_skipped_length += 1
                    continue
                out_f.write(
                    json.dumps(
                        {
                            "conformer_id": cid,
                            "input_ids": input_ids,
                            "hamiltonian": ham_vector.tolist(),
                            "n_operators": len(candidate["operator_sequence"]),
                            "e_ref": e_ref,
                            "e_hf": e_hf,
                            "energy": candidate["energy"],
                        }
                    )
                    + "\n"
                )
                n_examples += 1

    print(
        f"Round dataset: {n_examples} example(s) across {len(selected)} conformer(s) "
        f"(of {len(conformers)} total) -> {out_path}"
    )
    if n_skipped_length:
        print(f"Skipped {n_skipped_length} example(s) exceeding --max-length={args.max_length}")
    if n_missing_ham:
        print(f"Skipped {n_missing_ham} conformer(s) missing a Hamiltonian vector")


if __name__ == "__main__":
    main()
