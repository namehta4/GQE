#!/usr/bin/env python3
"""
Generate a diverse, representative set of reference conformers for a target
molecule using RDKit's ETKDG embedding with random-distance-matrix seeding,
followed by RMSD-based (Butina) clustering -- reproducing the "15 vs 122
reference conformer" construction described in ADAPT-GQE (arXiv:2607.22468,
Appendix A.2.1).

These reference conformers serve two roles later in the pipeline:
  1. Endpoints for the pairwise LAMMPS-native NEB transition-pathway runs
     (C(n,2) pairs among the smaller, e.g. 15-conformer, set).
  2. The highest-energy structure among the larger (e.g. 122-conformer) set
     seeds the out-of-distribution rattle-perturbation dataset.

IMPORTANT: --smiles here MUST be identical to the --smiles used in
mol_to_lammps_data.py, since both scripts build their base RDKit Mol via the
same mol_common.build_base_mol_from_smiles() to guarantee identical atom
ordering across the whole pipeline.

Usage:
  # "less aggressive" set (~15 representative conformers)
  python generate_reference_conformers.py \\
      --smiles "CN(C)CCCN1c2ccccc2CCc2ccccc21" \\
      --n-confs 500 --cluster-rms-thresh 1.2 --n-select 15 \\
      --output-dir refs_15 --element-order C H N --device cuda

  # "more aggressive" set (~122 representative conformers)
  python generate_reference_conformers.py \\
      --smiles "CN(C)CCCN1c2ccccc2CCc2ccccc21" \\
      --n-confs 2000 --cluster-rms-thresh 0.5 --n-select 122 \\
      --output-dir refs_122 --element-order C H N --device cuda

--cluster-rms-thresh is the knob that trades off cluster count vs. size:
smaller threshold -> more, finer clusters -> more representative conformers.
Tune it per-molecule; the values above are starting points, not universal.
"""
import argparse
import csv
import os

from ase.io import write as ase_write

from mol_common import (
    build_base_mol_from_smiles,
    rdkit_conf_to_ase,
    relax_with_mace,
    write_index_map,
    write_lammps_data,
)


def embed_candidate_pool(mol, n_confs, seed, embed_prune_rms):
    from rdkit.Chem import AllChem

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True  # "random distance matrix" seeding (Appendix A.2.1)
    params.pruneRmsThresh = embed_prune_rms  # -1 disables on-the-fly pruning
    params.numThreads = 0  # use all available cores

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(conf_ids) == 0:
        raise RuntimeError(
            "RDKit failed to embed any conformers - try a different --seed "
            "or relax --embed-prune-rms"
        )
    return list(conf_ids)


def ff_relax_and_energies(mol, conf_ids, pre_ff):
    from rdkit.Chem import AllChem

    if pre_ff == "none":
        return {cid: None for cid in conf_ids}

    if pre_ff == "mmff":
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
    elif pre_ff == "uff":
        results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
    else:
        raise ValueError(pre_ff)

    # EmbedMultipleConfs assigns conformer ids sequentially in creation order,
    # so results[i] corresponds to conf_ids[i].
    return {cid: energy for cid, (_not_converged, energy) in zip(conf_ids, results)}


def pairwise_rms_matrix(mol, conf_ids, heavy_only):
    from rdkit.Chem import AllChem

    atom_ids = (
        [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
        if heavy_only
        else None
    )

    dists = []  # condensed lower-triangle form expected by Butina
    n = len(conf_ids)
    for i in range(1, n):
        for j in range(i):
            rms = AllChem.GetConformerRMS(
                mol, conf_ids[i], conf_ids[j], atomIds=atom_ids, prealigned=False
            )
            dists.append(rms)
    return dists


def cluster_conformers(conf_ids, dists, rms_thresh):
    from rdkit.ML.Cluster import Butina

    clusters = Butina.ClusterData(
        dists, len(conf_ids), rms_thresh, isDistData=True, reordering=True
    )
    # Butina returns tuples of *positions* into conf_ids, largest cluster first
    return [tuple(conf_ids[i] for i in cluster) for cluster in clusters]


def select_representatives(clusters, energies, n_select):
    reps = []
    for cluster in clusters:
        if energies[cluster[0]] is None:
            rep_id = cluster[0]
        else:
            rep_id = min(cluster, key=lambda cid: energies[cid])
        reps.append((rep_id, len(cluster)))

    if len(reps) > n_select:
        print(
            f"Found {len(reps)} clusters; keeping the {n_select} largest "
            f"(most representative). Rerun with a smaller --cluster-rms-thresh "
            f"for finer clusters, or a larger one for fewer/broader clusters."
        )
        reps = sorted(reps, key=lambda r: r[1], reverse=True)[:n_select]
    elif len(reps) < n_select:
        print(
            f"WARNING: only {len(reps)} clusters found, fewer than the "
            f"requested --n-select {n_select}. Increase --n-confs and/or "
            f"decrease --cluster-rms-thresh to resolve more distinct clusters."
        )
    return reps


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--smiles",
        required=True,
        help="SMILES string - MUST match mol_to_lammps_data.py's --smiles",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--element-order", nargs="+", required=True)

    p.add_argument(
        "--n-confs", type=int, default=500, help="Size of the initial ETKDG candidate pool"
    )
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument(
        "--embed-prune-rms",
        type=float,
        default=-1.0,
        help="RDKit's on-the-fly embedding RMS pruning threshold "
        "(-1 disables; leaves all diversity work to the clustering step)",
    )
    p.add_argument(
        "--pre-relax",
        choices=["none", "mmff", "uff"],
        default="mmff",
        help="Cheap force-field relaxation applied to the whole candidate pool",
    )
    p.add_argument(
        "--heavy-atom-rmsd",
        action="store_true",
        default=True,
        help="Cluster on heavy-atom RMSD only (recommended; default on)",
    )
    p.add_argument(
        "--cluster-rms-thresh",
        type=float,
        required=True,
        help="Butina clustering RMSD threshold, Angstrom. Smaller -> more/finer "
        "clusters (try ~0.5 for a 122-conformer set, ~1.2 for a 15-conformer "
        "set as a starting point, then tune per molecule)",
    )
    p.add_argument(
        "--n-select",
        type=int,
        required=True,
        help="Target number of representative conformers to keep",
    )

    p.add_argument(
        "--optimize",
        choices=["none", "mace"],
        default="mace",
        help="Final-stage relaxation of the selected representatives",
    )
    p.add_argument("--mace-model", default="small", choices=["small", "medium", "large"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--box-padding", type=float, default=10.0)
    p.add_argument(
        "--write-lammps-data",
        action="store_true",
        help="Also write a LAMMPS data file per representative conformer "
        "(for direct use as NEB endpoints)",
    )

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mol = build_base_mol_from_smiles(args.smiles)

    print(f"Embedding {args.n_confs} candidate conformers via ETKDGv3 (useRandomCoords=True)...")
    conf_ids = embed_candidate_pool(mol, args.n_confs, args.seed, args.embed_prune_rms)

    print(f"Pre-relaxing candidate pool with {args.pre_relax}...")
    energies = ff_relax_and_energies(mol, conf_ids, args.pre_relax)

    print("Computing pairwise RMSD matrix for clustering...")
    dists = pairwise_rms_matrix(mol, conf_ids, args.heavy_atom_rmsd)

    print(f"Clustering at RMSD threshold {args.cluster_rms_thresh} A...")
    clusters = cluster_conformers(conf_ids, dists, args.cluster_rms_thresh)
    print(f"Found {len(clusters)} clusters from {len(conf_ids)} candidates.")

    reps = select_representatives(clusters, energies, args.n_select)

    rows = []
    for rank, (conf_id, cluster_size) in enumerate(reps):
        atoms = rdkit_conf_to_ase(mol, conf_id)

        if args.optimize == "mace":
            atoms = relax_with_mace(atoms, args.mace_model, args.device, args.fmax)

        tag = f"conformer_{rank:03d}"
        xyz_path = os.path.join(args.output_dir, f"{tag}.xyz")
        ase_write(xyz_path, atoms)

        idx_path = os.path.join(args.output_dir, f"{tag}.idx.txt")
        write_index_map(atoms, idx_path)

        final_energy = float(atoms.get_potential_energy()) if args.optimize == "mace" else None

        if args.write_lammps_data:
            data_path = os.path.join(args.output_dir, f"{tag}.data")
            write_lammps_data(atoms, data_path, args.element_order, args.box_padding)

        rows.append(
            {
                "conformer": tag,
                "rdkit_conf_id": conf_id,
                "cluster_size": cluster_size,
                "ff_energy": energies[conf_id],
                "mace_energy_eV": final_energy,
            }
        )
        print(
            f"  {tag}: cluster_size={cluster_size}, ff_energy={energies[conf_id]}, "
            f"mace_energy_eV={final_energy}"
        )

    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.optimize == "mace":
        highest = max(rows, key=lambda r: r["mace_energy_eV"])
        print(
            f"\nHighest-energy representative: {highest['conformer']} "
            f"({highest['mace_energy_eV']:.4f} eV) -- this is the seed structure "
            f"for the OoD rattle-perturbation stage."
        )

    print(f"\nWrote {len(rows)} representative conformers to {args.output_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
