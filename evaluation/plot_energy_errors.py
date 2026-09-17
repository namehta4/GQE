#!/usr/bin/env python3
"""
Reproduce the two main plot types from ADAPT-GQE (arXiv:2607.22468) Section
4.1 from compute_metrics.py's --output-json distributions file:

  (a) Figure 5 style: box-and-whisker + scatter of energy error per labeled
      method (e.g. pretraining vs. post-RL vs. post-distillation).
  (b) Figure 6 style: energy error vs. number of generated operators,
      median + interquartile range per method.

Usage:
  python plot_energy_errors.py \\
      --distributions distributions_12q_eps5.json \\
      --epsilon-mha 5.0 \\
      --output-prefix plots_12q_eps5
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_box_scatter(distributions, epsilon_mha, output_path):
    labels = list(distributions.keys())
    data = [[r["error_mha"] for r in distributions[label]] for label in labels]

    fig, ax = plt.subplots(figsize=(1.5 * len(labels) + 2, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    for i, values in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, alpha=0.4, s=10)

    if epsilon_mha is not None:
        ax.axhline(epsilon_mha, color="gray", linestyle="--", linewidth=1, label=f"epsilon={epsilon_mha} mHa")
        ax.legend()

    ax.set_ylabel("E_pred - E_ref (mHa)")
    ax.set_title("Generated circuit energy errors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_error_vs_operators(distributions, output_path, n_bins=8):
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, rows in distributions.items():
        pairs = [(r["n_operators"], r["error_mha"]) for r in rows if r["n_operators"] is not None]
        if not pairs:
            continue
        n_ops, errors = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])

        bin_edges = np.linspace(n_ops.min(), n_ops.max(), n_bins + 1)
        bin_idx = np.digitize(n_ops, bin_edges[1:-1])
        medians, q1s, q3s, centers = [], [], [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            vals = errors[mask]
            medians.append(np.median(vals))
            q1s.append(np.percentile(vals, 25))
            q3s.append(np.percentile(vals, 75))
            centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)

        centers = np.array(centers)
        medians = np.array(medians)
        yerr = np.array([medians - np.array(q1s), np.array(q3s) - medians])
        ax.errorbar(centers, medians, yerr=yerr, marker="o", capsize=3, label=label)

    ax.set_xlabel("Number of operators")
    ax.set_ylabel("E_pred - E_ref (mHa)")
    ax.set_title("Energy error vs. circuit complexity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--distributions", required=True, help="--output-json from compute_metrics.py")
    p.add_argument("--epsilon-mha", type=float, default=None)
    p.add_argument("--output-prefix", required=True)

    args = p.parse_args()
    with open(args.distributions) as f:
        distributions = json.load(f)

    plot_box_scatter(distributions, args.epsilon_mha, f"{args.output_prefix}_box.png")
    plot_error_vs_operators(distributions, f"{args.output_prefix}_vs_operators.png")


if __name__ == "__main__":
    main()
