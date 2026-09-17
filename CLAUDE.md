# CLAUDE.md — Project Briefing for Claude Code

This file is for a fresh Claude Code instance picking up this project with
no memory of how it was built. Read this in full before doing anything —
it tells you what's real, what's assumed, and exactly where to start.

## What this project is

An implementation of the pipeline in *Learning to Prepare Molecular Ground
States with Transformer Models* (ADAPT-GQE), arXiv:2607.22468: train a
small transformer to generate molecular ground-state quantum circuits
directly (conditioned on the Hamiltonian), using ADAPT-VQE-generated data
as the initial training signal, then improve the model beyond that data via
GRPO/DAPO reinforcement learning and self-distillation.

Start with `README.md` (plain-language pipeline overview + repo layout)
and `PIPELINE.md` (exact end-to-end run order, every command, and a
consolidated Appendix of every validation risk flagged during
construction). Read both before writing new code.

## Current status — READ THIS FIRST

**Nothing in this pipeline has been functionally validated.** Every script
was written from the paper, from documentation, and — for `cudaq-solvers`
specifically — from its actual downloaded/inspected shipped source, but
with no live CUDA-Q/LAMMPS/GPU environment to test against at the time of
writing. What HAS been confirmed, as of this handoff:

- The merged container image (`docker/Dockerfile`) **builds successfully**
  for `linux/amd64` and is pushed to Docker Hub as
  **`docker.io/neilmehta87/gqe:26.06`** — pull it directly, do not rebuild
  unless you have a specific reason to (it takes hours: LLVM from source,
  CUDA-Q wheel, cudaqx, LAMMPS+Kokkos+CUDA).
- Every key Python package imports correctly inside that image with sane
  versions: `cudaq`, `cudaq_solvers` (built from `github.com/namehta4/cudaqx`
  with `qec;solvers` libs enabled — this is what `chemistry/run_adapt_vqe.py`
  depends on), `rdkit`, `ase`, `pytket`, `qiskit`, `qiskit_ibm_runtime`,
  `torch` (cu129), `transformers`, `pyscf`, `openfermion`, `mace`.
- **Unresolved**: after all imports succeeded, the verification process
  crashed on exit with `munmap_chunk(): invalid pointer` (a glibc heap
  corruption message during Python interpreter shutdown, after all real
  work had completed). This happened under Rosetta-emulated amd64-on-arm64
  on a Mac with no NVIDIA GPU/driver — it may or may not reproduce natively
  on Perlmutter. Worth checking early, since `model/rl_grpo.py` needs both
  `torch` and `cudaq` loaded in the same process, which is exactly the kind
  of combination that triggers library-conflict crashes like this.
- **Nothing else has been run.** No LAMMPS MD, no ADAPT-VQE, no model
  training, no RL, no hardware compilation.

## Your immediate next steps, in order

1. **`chemistry/test_circuit_energy.py`** — run this first, on a real GPU
   node. It validates the two highest-risk assumptions in
   `chemistry/circuit_energy.py` (how `cudaq.SpinOperator` exposes term
   coefficients, and whether `exp_pauli`'s sign convention matches the UCC
   generator's phase convention) against a brute-force cross-check on
   H2/STO-3G. **Do not trust ADAPT-VQE, RL, or evaluation results until
   this passes.** SLURM script: `validation/slurm_01_test_circuit_energy.sbatch`.
2. **The H2 validation ladder** (`validation/slurm_02` through `slurm_04`) —
   runs the full chemistry → model → RL pipeline on a trivial H2
   bond-length scan (used in place of real conformer sampling, since H2 has
   no dihedral structure). Each script explains what a healthy result looks
   like. Fill in `--account=<YOUR_NERSC_ACCOUNT>` in every `.sbatch` file
   before submitting anything.
3. **`lammps_md/mace_mliap_unified.py`'s tiny-system test** (described in
   its own module docstring) — validates the ML-IAP MACE integration before
   trusting any real MD/NEB run. This is a bigger unknown than #1: it was
   written from general knowledge of LAMMPS's `MLIAPUnified` ABC, not from
   a live LAMMPS session, and multiple specific method/attribute names are
   marked "VERIFY:" in that file. Diff it against
   `<lammps_source>/examples/mliap/mliap_unified_lj.py` and
   `<lammps_source>/python/lammps/mliap/mliap_unified_abc.py` in the actual
   built LAMMPS tree (inside the container) if it doesn't work as written.
4. Only after 1-3 pass: move to the real molecule (imipramine, or another
   of your choosing) via `PIPELINE.md`'s full run order, ideally through
   `orchestrator/run_pipeline.py` rather than by hand.

## Repository layout

| Directory | Contents |
|---|---|
| `lammps_md/` | Molecule input, LAMMPS MD (MACE-OFF via ML-IAP), NEB, OoD perturbations, dataset consolidation |
| `chemistry/` | Hamiltonian construction, ADAPT-VQE via `cudaq-solvers`, circuit energy evaluation |
| `model/` | Tokenizer, Hamiltonian encoder, Gemma 3 model, pretraining, generation, GRPO/DAPO RL, self-distillation |
| `evaluation/` | Accuracy metrics, plots, compute-speedup benchmarking |
| `hardware/` | pytket circuit compilation; Quantinuum PMSV is a simplified reimplementation, not InQuanto's actual algorithm; IBM Quantum submission via `qiskit-ibm-runtime` not yet built |
| `docker/` | The merged container image definition (already built + pushed, see above) |
| `orchestrator/` | Automated stage-graph runner (resume/retry/smoke-test/adaptive self-distillation) + Perlmutter SLURM scripts |
| `validation/` | The H2 Tier-1 validation ladder — start here |

## Known unresolved risks (see `PIPELINE.md`'s Appendix for the full table)

Ranked by how badly a mistake here would hurt, silently:

1. **`chemistry/circuit_energy.py`** — highest risk. A wrong sign convention
   produces a plausible-looking but wrong energy for every circuit, which
   then poisons the RL reward and every evaluation metric. Gated by
   `test_circuit_energy.py`.
2. **`lammps_md/mace_mliap_unified.py`** — the ML-IAP/MACE integration is an
   untested draft (see its own docstring's "VERIFY:" comments).
3. **`chemistry/build_hamiltonian.py`**'s OpenFermion chemist→physicist
   integral transpose — cross-check against `cudaq_solvers.create_molecule`'s
   independently-computed energies for the same conformer.
4. **`model/gemma_hamiltonian_model.py`**'s Gemma 3 config — class names
   were current as of `transformers` at write time; verify against your
   installed version if `build_gemma3_backbone` fails to import.
5. **`model/generate.py`**'s custom KV-cache loop — HF's `past_key_values`
   behavior when driving a model purely via `inputs_embeds` is
   version-dependent; watch for repeated/garbage tokens after the first few
   generated tokens.
6. NEB (`lammps_md/neb_template.in`) is **disabled by default** in the
   orchestrator (`include_neb: false`) — it needs an MPI-partition-aware
   Python driver for the ML-IAP model that does not exist yet. Don't enable
   it until that's written.

## Conventions used throughout this codebase

- **Every genuinely uncertain assumption is documented inline**, usually as
  a "VALIDATION WARNING" or "VERIFY:" comment in the relevant file's module
  docstring, explaining exactly what's uncertain and how to check it. Keep
  doing this for new uncertain code — don't present a guess as settled fact.
- **Write a validation test alongside risky code where one is feasible**
  (see `chemistry/test_circuit_energy.py` for the pattern: isolate the risk
  into the smallest possible checkable claim, independent of any specific
  literature reference value).
- Atom ordering is established once (via `mol_to_lammps_data.py` /
  `mol_common.py`) and must survive unchanged through every later stage —
  LAMMPS, RDKit, PySCF, and the Hamiltonian encoder all depend on this.
- No code comments explaining *what* code does (names should do that);
  comments are reserved for non-obvious *why* — a hidden constraint, a
  workaround, a version-specific gotcha.
- Nemotron (the paper's second, pretrained-LLM model) and the InQuanto/Nexus
  hardware submission path are explicitly out of scope — don't build these
  unless the user asks.

## Git / deployment

- Repo: `https://github.com/namehta4/GQE.git`, `main` branch. Commit
  messages end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
  Only commit/push when the user explicitly asks.
- Container image: `docker.io/neilmehta87/gqe:26.06` (already built and
  pushed — pull with `podman-hpc pull docker.io/neilmehta87/gqe:26.06` on
  Perlmutter rather than rebuilding).
- `orchestrator/run_on_perlmutter.sbatch` and every script in `validation/`
  need `--account=<YOUR_NERSC_ACCOUNT>` filled in before submission.
