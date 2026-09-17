#!/usr/bin/env python3
"""
Score every model-generated candidate (from model/generate.py's output) by
its circuit energy, and select the best-of-N per conformer -- reproducing
the evaluation methodology of Sec. 4.1 in ADAPT-GQE (arXiv:2607.22468):
"each model generates 16 candidate circuits ... the circuit with the lowest
energy ... is used in the reported metrics."

Do NOT run this at scale before test_circuit_energy.py passes -- see that
file and circuit_energy.py's module docstrings for why.

Usage:
  python evaluate_generated_circuits.py \\
      --generated generated_12q_eps5.jsonl \\
      --basis 6-31g --charge 0 --spin 0 \\
      --n-active-electrons 6 --n-active-orbitals 6 --pool uccgsd \\
      --xyz-dir full_dataset \\
      --output best_of_n_12q_eps5.jsonl
"""
import argparse
import json
import os

from circuit_energy import build_ansatz_kernel, compute_reward, evaluate_energy


def build_molecule_and_pool(xyz_path, args):
    import cudaq_solvers as solvers

    from ase.io import read

    atoms = read(xyz_path)
    geometry = [
        (s, (float(p[0]), float(p[1]), float(p[2])))
        for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    ]
    molecule = solvers.create_molecule(
        geometry=geometry,
        basis=args.basis,
        spin=args.spin,
        charge=args.charge,
        nele_cas=args.n_active_electrons,
        norb_cas=args.n_active_orbitals,
        casci=True,
    )
    n_qubits = 2 * args.n_active_orbitals
    pool = solvers.get_operator_pool(args.pool, n_qubits=n_qubits, n_electrons=molecule.n_electrons)
    return molecule, pool, n_qubits, molecule.n_electrons


def resolve_xyz_path(conformer_id, xyz_dir):
    """conformer_id looks like 'MD_traj0_frame0000010' / 'NEB_pair_000_001_image_05'
    / 'OOD_ood_0042' (see build_conformer_manifest.py); the corresponding file
    is '<conformer_id>.xyz' in xyz_dir."""
    path = os.path.join(xyz_dir, f"{conformer_id}.xyz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No geometry file found for {conformer_id} at {path}")
    return path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--generated", required=True, help="Output of model/generate.py")
    p.add_argument("--xyz-dir", required=True, help="Directory of <conformer_id>.xyz files")
    p.add_argument("--basis", default="6-31g")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--n-active-electrons", type=int, required=True)
    p.add_argument("--n-active-orbitals", type=int, required=True)
    p.add_argument("--pool", choices=["uccsd", "uccgsd", "upccgsd"], required=True)
    p.add_argument("--reward-r-max", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", required=True)

    args = p.parse_args()

    ansatz = build_ansatz_kernel()

    rows = []
    with open(args.generated) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    n_failed = 0
    with open(args.output, "w") as out_f:
        for i, row in enumerate(rows):
            cid = row["conformer_id"]
            print(f"[{i + 1}/{len(rows)}] {cid}")
            try:
                xyz_path = resolve_xyz_path(cid, args.xyz_dir)
                molecule, pool, n_qubits, n_electrons = build_molecule_and_pool(xyz_path, args)
            except Exception as e:
                print(f"  FAILED (molecule/pool build): {e}")
                n_failed += 1
                continue

            best = None
            for c, candidate in enumerate(row["candidates"]):
                op_sequence = candidate["operator_sequence"]
                try:
                    e_circuit = evaluate_energy(
                        molecule, pool, op_sequence, n_qubits, n_electrons, ansatz=ansatz
                    )
                except Exception as e:
                    print(f"  candidate {c} FAILED energy evaluation: {e}")
                    continue

                if best is None or e_circuit < best["energy"]:
                    best = {
                        "candidate_index": c,
                        "energy": e_circuit,
                        "n_operators": len(op_sequence),
                        "n_invalid_pairs": candidate.get("n_invalid_pairs", 0),
                    }

            if best is None:
                print("  All candidates failed energy evaluation, skipping conformer.")
                n_failed += 1
                continue

            e_ref, e_hf = row["e_ref"], row["e_hf"]
            result = {
                "conformer_id": cid,
                "e_ref": e_ref,
                "e_hf": e_hf,
                "best_candidate_index": best["candidate_index"],
                "best_energy": best["energy"],
                "energy_error_mha": (best["energy"] - e_ref) * 1000.0,
                "n_operators": best["n_operators"],
                "reward": compute_reward(e_hf, e_ref, best["energy"], r_max=args.reward_r_max),
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            print(
                f"  best: candidate {best['candidate_index']}, "
                f"error={result['energy_error_mha']:.3f} mHa, "
                f"n_operators={best['n_operators']}"
            )

    print(f"Done. {len(rows) - n_failed}/{len(rows)} conformers scored.")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
