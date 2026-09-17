#!/usr/bin/env python3
"""
Compute the accuracy statistics reported in Section 4.1 of ADAPT-GQE
(arXiv:2607.22468): mean/std/median energy error vs. reference, error
relative to the ADAPT-VQE training tolerance epsilon, the fraction of
circuits within the paper's "target accuracy" criterion (within 1 mHa above
epsilon, Sec. 4.1.1), and the Pearson correlation between energy error and
generated circuit length (Sec. 4.1.3).

Accepts results files in either format already produced by this pipeline:
  - chemistry/evaluate_generated_circuits.py output (best_energy, energy_error_mha)
  - chemistry/build_adapt_vqe_dataset.py's circuits.jsonl (final_energy)

Pass one or more labeled result sets to compare them directly (e.g.
pretraining vs. post-RL vs. post-distillation), matching how Figure 5
overlays multiple methods per dataset.

Usage:
  python compute_metrics.py \\
      --results pretrain:pretrain_best_of_16.jsonl \\
      --results post_rl:post_rl_best_of_16.jsonl \\
      --epsilon-mha 5.0 --target-accuracy-mha 1.0 \\
      --output summary_12q_eps5.csv \\
      --output-json distributions_12q_eps5.json
"""
import argparse
import csv
import json

import numpy as np


def load_results(path):
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            energy = row.get("best_energy", row.get("final_energy"))
            e_ref = row["e_ref"]
            error_mha = row.get("energy_error_mha", (energy - e_ref) * 1000.0)
            rows.append(
                {
                    "conformer_id": row["conformer_id"],
                    "energy": energy,
                    "e_ref": e_ref,
                    "error_mha": error_mha,
                    "n_operators": row.get("n_operators"),
                }
            )
    return rows


def summarize(rows, epsilon_mha=None, target_accuracy_mha=1.0):
    errors = np.array([r["error_mha"] for r in rows], dtype=float)
    n_ops_pairs = [
        (r["error_mha"], r["n_operators"]) for r in rows if r["n_operators"] is not None
    ]

    summary = {
        "n": len(errors),
        "mean_error_mha": float(errors.mean()),
        "std_error_mha": float(errors.std()),
        "median_error_mha": float(np.median(errors)),
    }

    if epsilon_mha is not None:
        rel = errors - epsilon_mha
        summary["mean_error_vs_epsilon_mha"] = float(rel.mean())
        summary["std_error_vs_epsilon_mha"] = float(rel.std())
        summary["fraction_within_target_accuracy"] = float(np.mean(rel <= target_accuracy_mha))

    if len(n_ops_pairs) > 1:
        errs, ops = zip(*n_ops_pairs)
        summary["pearson_r_error_vs_n_operators"] = float(np.corrcoef(errs, ops)[0, 1])

    return summary


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results",
        action="append",
        required=True,
        help="label:path, repeatable, e.g. --results pretrain:results.jsonl",
    )
    p.add_argument(
        "--epsilon-mha",
        type=float,
        default=None,
        help="ADAPT-VQE training-tolerance for this dataset (e.g. 5.0 for the "
        "12q eps=5mHa dataset) -- enables the target-accuracy fraction metric",
    )
    p.add_argument("--target-accuracy-mha", type=float, default=1.0)
    p.add_argument("--output", required=True, help="Summary CSV path")
    p.add_argument(
        "--output-json",
        default=None,
        help="Optional: full per-conformer distributions, for plot_energy_errors.py",
    )

    args = p.parse_args()

    all_summaries = {}
    all_distributions = {}
    for spec in args.results:
        label, path = spec.split(":", 1)
        rows = load_results(path)
        all_summaries[label] = summarize(rows, args.epsilon_mha, args.target_accuracy_mha)
        all_distributions[label] = rows
        print(f"{label} (n={all_summaries[label]['n']}):")
        for k, v in all_summaries[label].items():
            if k != "n":
                print(f"  {k}: {v:.4f}")

    fieldnames = ["label"] + sorted({k for s in all_summaries.values() for k in s})
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, summary in all_summaries.items():
            writer.writerow({"label": label, **summary})
    print(f"\nSummary written to {args.output}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_distributions, f)
        print(f"Distributions written to {args.output_json}")


if __name__ == "__main__":
    main()
