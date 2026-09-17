"""
Declarative stage graph for the ADAPT-GQE pipeline, mirroring PIPELINE.md's
manual run order. Each Stage is either a subprocess command or (for the
self-distillation loop, which is a dynamic round count rather than a fixed
DAG node) a Python callable.

This module only BUILDS the graph -- run_pipeline.py executes it (with
skip/resume, retries, and logging).

Scripts are invoked by absolute path so the orchestrator's own working
directory doesn't matter; every script that needs sibling-module imports
either relies on Python's automatic same-directory sys.path insertion or
already does its own os.path.dirname(__file__)-relative sys.path.insert
(chemistry/, model/, evaluation/ scripts were written this way already).

KNOWN GAP, deliberately not silently worked around: NEB requires an
MPI-partition-aware Python driver to register the ML-IAP MACE model (see
lammps_md/neb_template.in's "NOT YET RESOLVED" note) that does not exist
yet. The NEB stage is included but disabled by default
(dataset.include_neb: false in the config) rather than pretending it works.
"""
import itertools
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Stage:
    name: str
    depends_on: List[str] = field(default_factory=list)
    command: Optional[List[str]] = None
    run_fn: Optional[Callable] = None  # (ctx) -> None, for non-subprocess stages
    outputs: List[str] = field(default_factory=list)  # all must exist to count as "done"
    retryable: bool = True


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script(*parts):
    return os.path.join(_repo_root(), *parts)


def build_shared_stages(cfg, workdir):
    """Stages that run once per molecule, independent of any downstream
    (qubit count, tolerance) dataset configuration."""
    mol = cfg["molecule"]
    lammps_cfg = cfg.get("lammps", {})
    element_order = mol["element_order"]
    smiles = mol["smiles"]
    device = lammps_cfg.get("device", "cuda")
    mace_model = lammps_cfg.get("mace_model", "small")

    data_file = os.path.join(workdir, "imipramine.data")
    refs15_dir = os.path.join(workdir, "refs_15")
    refs122_dir = os.path.join(workdir, "refs_122")
    md_dir = os.path.join(workdir, "runs")
    neb_dir = os.path.join(workdir, "neb_runs")
    neb_dataset_dir = os.path.join(workdir, "neb_dataset")
    ood_dir = os.path.join(workdir, "ood_dataset")
    manifest_dir = os.path.join(workdir, "full_dataset")

    stages = [
        Stage(
            name="mol_to_lammps_data",
            command=[
                "python", _script("lammps_md", "mol_to_lammps_data.py"),
                "--smiles", smiles, "--output", data_file,
                "--element-order", *element_order, "--device", device,
            ],
            outputs=[data_file],
        ),
        Stage(
            name="refs_15",
            command=[
                "python", _script("lammps_md", "generate_reference_conformers.py"),
                "--smiles", smiles,
                "--n-confs", str(lammps_cfg.get("refs15_n_confs", 500)),
                "--cluster-rms-thresh", str(lammps_cfg.get("refs15_cluster_rms_thresh", 1.2)),
                "--n-select", "15", "--output-dir", refs15_dir,
                "--element-order", *element_order, "--device", device,
                "--write-lammps-data",
            ],
            outputs=[os.path.join(refs15_dir, "summary.csv")],
        ),
        Stage(
            name="refs_122",
            command=[
                "python", _script("lammps_md", "generate_reference_conformers.py"),
                "--smiles", smiles,
                "--n-confs", str(lammps_cfg.get("refs122_n_confs", 2000)),
                "--cluster-rms-thresh", str(lammps_cfg.get("refs122_cluster_rms_thresh", 0.5)),
                "--n-select", "122", "--output-dir", refs122_dir,
                "--element-order", *element_order, "--device", device,
            ],
            outputs=[os.path.join(refs122_dir, "summary.csv")],
        ),
        Stage(
            name="md_ensemble",
            depends_on=["mol_to_lammps_data"],
            command=[
                "bash", _script("lammps_md", "run_md_ensemble.sh"),
                data_file, " ".join(element_order),
                str(lammps_cfg.get("md_nprocs", 1)), mace_model,
            ],
            outputs=[os.path.join(md_dir, "traj4.lammpstrj")],
        ),
        Stage(
            name="ood_perturb",
            depends_on=["refs_122"],
            command=[
                "python", _script("lammps_md", "generate_ood_perturbations.py"),
                "--reference-dir", refs122_dir,
                "--n-perturbations", str(lammps_cfg.get("n_ood_perturbations", 200)),
                "--stdev", "0.05", "--output-dir", ood_dir,
            ],
            outputs=[ood_dir],
        ),
    ]

    include_neb = lammps_cfg.get("include_neb", False)
    if include_neb:
        stages += [
            Stage(
                name="neb_prepare",
                depends_on=["refs_15"],
                command=[
                    "python", _script("lammps_md", "prepare_neb_pairs.py"),
                    "--conformer-dir", refs15_dir, "--output-dir", neb_dir,
                    "--lammps-bin", lammps_cfg.get("lammps_bin", "lmp"),
                    "--n-images", str(lammps_cfg.get("neb_n_images", 30)),
                ],
                outputs=[os.path.join(neb_dir, "run_neb_pairs.sh")],
            ),
            Stage(
                name="neb_run",
                depends_on=["neb_prepare"],
                command=["bash", os.path.join(neb_dir, "run_neb_pairs.sh")],
                outputs=[],  # no single reliable marker; relies on neb_filter's own checks
                retryable=False,  # known-broken pending the MPI-partition ML-IAP driver
            ),
            Stage(
                name="neb_filter",
                depends_on=["neb_run"],
                command=[
                    "python", _script("lammps_md", "filter_neb_pathways.py"),
                    "--neb-dir", neb_dir, "--n-images", str(lammps_cfg.get("neb_n_images", 30)),
                    "--element-order", *element_order, "--dataset-out", neb_dataset_dir,
                ],
                outputs=[os.path.join(neb_dataset_dir, "neb_filter_summary.csv")],
            ),
        ]

    manifest_cmd = [
        "python", _script("lammps_md", "build_conformer_manifest.py"),
    ]
    for i, stride in enumerate(lammps_cfg.get("md_strides", [10, 10, 10, 10, 5])):
        manifest_cmd += ["--md-traj", f"{os.path.join(md_dir, f'traj{i}.lammpstrj')}:{stride}"]
    if include_neb:
        manifest_cmd += ["--neb-dir", neb_dataset_dir]
    manifest_cmd += [
        "--ood-dir", ood_dir,
        "--element-order", *element_order,
        "--dihedral-defs", lammps_cfg.get(
            "dihedral_defs", _script("lammps_md", "dihedrals_imipramine_example.json")
        ),
        "--output-dir", manifest_dir,
    ]
    manifest_depends = ["md_ensemble", "ood_perturb"] + (["neb_filter"] if include_neb else [])
    stages.append(
        Stage(
            name="manifest",
            depends_on=manifest_depends,
            command=manifest_cmd,
            outputs=[os.path.join(manifest_dir, "manifest.csv")],
        )
    )
    return stages, manifest_dir


def build_dataset_config_stages(cfg, workdir, manifest_dir, dcfg):
    """Stages for one (qubit count, tolerance, pool) dataset configuration,
    per the table in PIPELINE.md Section 3."""
    name = dcfg["name"]
    n_e, n_o = dcfg["n_active_electrons"], dcfg["n_active_orbitals"]
    qubit_key = f"{n_e}e{n_o}o"
    basis = dcfg.get("basis", "6-31g")
    charge, spin = dcfg.get("charge", 0), dcfg.get("spin", 0)

    ham_dir = os.path.join(workdir, f"hamiltonians_{qubit_key}")
    term_order_file = os.path.join(workdir, f"term_order_{qubit_key}.json")
    adaptvqe_dir = os.path.join(workdir, f"adaptvqe_{name}")
    smoke_dir = os.path.join(workdir, f"adaptvqe_{name}_smoke")
    training_dir = os.path.join(workdir, f"training_examples_{name}")
    ckpt_dir = os.path.join(workdir, f"checkpoints_{name}")

    stages = []

    # Hamiltonian vectors are shared across every tolerance variant of the
    # same (n_active_electrons, n_active_orbitals) -- keyed by qubit_key, not
    # `name`, so this only actually runs once even if multiple dataset_configs
    # share an active space (e.g. 12q eps=5 and eps=10).
    ham_stage_name = f"hamiltonians_{qubit_key}"
    if ham_stage_name not in _SEEN_STAGE_NAMES:
        _SEEN_STAGE_NAMES.add(ham_stage_name)
        stages.append(
            Stage(
                name=ham_stage_name,
                depends_on=["manifest"],
                command=[
                    "python", _script("chemistry", "build_hamiltonian_vectors_for_dataset.py"),
                    "--manifest", os.path.join(manifest_dir, "manifest.csv"),
                    "--basis", basis, "--charge", str(charge), "--spin", str(spin),
                    "--n-active-electrons", str(n_e), "--n-active-orbitals", str(n_o),
                    "--term-order-file", term_order_file, "--output-dir", ham_dir,
                ],
                outputs=[ham_dir],
            )
        )

    # Smoke test (--limit) before committing to the full, expensive
    # ADAPT-VQE batch -- this is the orchestrator behavior requested
    # explicitly: verify the stage works on a handful of conformers first.
    adaptvqe_common = [
        "--manifest", os.path.join(manifest_dir, "manifest.csv"),
        "--basis", basis, "--charge", str(charge), "--spin", str(spin),
        "--n-active-electrons", str(n_e), "--n-active-orbitals", str(n_o),
        "--pool", dcfg["pool"], "--reference", dcfg["reference"],
        "--tolerance-mha", str(dcfg["tolerance_mha"]),
    ]
    stages.append(
        Stage(
            name=f"adaptvqe_{name}_smoke",
            depends_on=["manifest"],
            command=(
                ["python", _script("chemistry", "build_adapt_vqe_dataset.py")]
                + adaptvqe_common
                + ["--limit", "3", "--output-dir", smoke_dir]
            ),
            outputs=[os.path.join(smoke_dir, "circuits.jsonl")],
        )
    )
    stages.append(
        Stage(
            name=f"adaptvqe_{name}",
            depends_on=[f"adaptvqe_{name}_smoke"],
            command=(
                ["python", _script("chemistry", "build_adapt_vqe_dataset.py")]
                + adaptvqe_common
                + ["--output-dir", adaptvqe_dir, "--skip-existing"]
            ),
            outputs=[os.path.join(adaptvqe_dir, "splits.json")],
        )
    )

    stages.append(
        Stage(
            name=f"training_examples_{name}",
            depends_on=[f"adaptvqe_{name}", ham_stage_name],
            command=[
                "python", _script("model", "build_training_examples.py"),
                "--circuits", os.path.join(adaptvqe_dir, "circuits.jsonl"),
                "--splits", os.path.join(adaptvqe_dir, "splits.json"),
                "--hamiltonian-dir", ham_dir,
                "--pool-size", str(dcfg["pool_size"]),
                "--output-dir", training_dir,
            ],
            outputs=[os.path.join(training_dir, "train.jsonl")],
        )
    )

    model_cfg = dcfg.get("model", {})
    stages.append(
        Stage(
            name=f"pretrain_{name}",
            depends_on=[f"training_examples_{name}"],
            command=[
                "python", _script("model", "pretrain.py"),
                "--train-jsonl", os.path.join(training_dir, "train.jsonl"),
                "--val-jsonl", os.path.join(training_dir, "val.jsonl"),
                "--tokenizer-dir", training_dir,
                "--hidden-size", str(model_cfg.get("hidden_size", 1024)),
                "--num-layers", str(model_cfg.get("num_layers", 16)),
                "--num-heads", str(model_cfg.get("num_heads", 8)),
                "--num-kv-heads", str(model_cfg.get("num_kv_heads", 8)),
                "--context-length", str(model_cfg.get("context_length", 1024)),
                "--sliding-window", str(model_cfg.get("sliding_window", 1024)),
                "--rope-theta", str(model_cfg.get("rope_theta", 10000)),
                "--encoder-hidden-dim", str(model_cfg.get("encoder_hidden_dim", 2048)),
                "--epochs", str(model_cfg.get("epochs", 20)),
                "--batch-size", str(model_cfg.get("batch_size", 8)),
                "--lr", str(model_cfg.get("lr", 4e-5)),
                "--output-dir", ckpt_dir,
            ],
            outputs=[os.path.join(ckpt_dir, "best_checkpoint.pt")],
        )
    )

    return stages, {
        "name": name,
        "n_active_electrons": n_e,
        "n_active_orbitals": n_o,
        "basis": basis,
        "charge": charge,
        "spin": spin,
        "pool": dcfg["pool"],
        "training_dir": training_dir,
        "pretrain_ckpt_dir": ckpt_dir,
        "manifest_dir": manifest_dir,
    }


_SEEN_STAGE_NAMES = set()


def build_all_stages(cfg, workdir):
    """Returns (stages, dataset_ctxs) -- dataset_ctxs is consumed by
    run_pipeline.py to drive the post-pretraining evaluation and
    self-distillation steps for each dataset config."""
    _SEEN_STAGE_NAMES.clear()
    shared, manifest_dir = build_shared_stages(cfg, workdir)
    all_stages = list(shared)
    dataset_ctxs = []
    for dcfg in cfg["dataset_configs"]:
        stages, dctx = build_dataset_config_stages(cfg, workdir, manifest_dir, dcfg)
        all_stages += stages
        dataset_ctxs.append(dctx)
    return all_stages, dataset_ctxs
