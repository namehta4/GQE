# ADAPT-GQE Pipeline — End-to-End Run Order

Reproduces the pipeline in *Learning to Prepare Molecular Ground States with
Transformer Models* (arXiv:2607.22468), using LAMMPS+MACE-OFF for MD,
`cudaq-solvers` for ADAPT-VQE, and a from-scratch Gemma 3 model. Nemotron
(the paper's second, pretrained-LLM model) remains out of scope. A merged
container image for NERSC Perlmutter is available at `docker/Dockerfile`.

**Automated runner available:** `orchestrator/run_pipeline.py` drives every
stage below from a single YAML config (`orchestrator/config.example.yaml`),
with resume/skip, retries, a smoke-test-before-full-run pattern for the two
most expensive stages, and adaptive self-distillation stopping based on
validation-loss improvement rather than a fixed round count. The manual
commands below are still the ground truth for what each stage actually does
and are useful for running a single stage by hand or debugging a failure the
orchestrator hit.

**Nothing in this pipeline has been executed end-to-end.** Every stage was
built from documentation, the actual shipped source of `cudaq-solvers`, and
careful reasoning, but this environment has no LAMMPS+MACE build, no CUDA-Q
runtime, and no live PySCF/OpenFermion/transformers+GPU stack to test
against. See **Appendix: Validation Checkpoints** before trusting any stage
at real scale — it consolidates every flagged risk into one place.

---

## 0. One-time setup (per environment, not per molecule)

**If you're using `docker/Dockerfile`** (built on Perlmutter per that file's
own instructions), steps 1-3 below are already done for you — it builds
LAMMPS with ML-IAP, `cudaq`/`cudaq-solvers`, `pytket`, `rdkit`, `ase`, and
`mace-torch` from source, and copies this repo's code into
`/opt/adapt-gqe`. Skip straight to step 4 — the image gives you an
environment *capable* of running that test, it does not run it for you.

1. Build LAMMPS with `PKG_ML-IAP` + `MLIAP_ENABLE_PYTHON` (see
   `docker/Dockerfile`'s LAMMPS build stage). MACE-OFF is wired in via
   `lammps_md/mace_mliap_unified.py`, LAMMPS's ML-IAP "unified" Python
   interface — NOT a compiled `pair_style mace` plugin (an earlier, now
   superseded plan). See that file's VALIDATION WARNING: this integration
   is an untested draft, not a confirmed-working one.
2. Install: `ase`, `rdkit`, `mace-torch`, `pyscf`, `openfermion`,
   `cudaq`, `cudaq-solvers`, `pytket`, `torch`, `transformers`,
   `numpy`, `matplotlib`, `pyyaml` (for the orchestrator). See
   `docker/Dockerfile` for the full, version-pinned list and the
   CUDA-version compatibility notes across libtorch/PyTorch/CUDA-Q.
3. `python lammps_md/run_md_python_driver.py` (not `lmp -in` directly) is
   how MD runs now, since ML-IAP "unified" potentials are Python objects
   that must be registered with a running `lammps` instance before the
   input script executes.
4. **Run `chemistry/test_circuit_energy.py` and confirm it passes.**
   Everything from Section 4 onward (RL, evaluation, benchmarking, hardware)
   depends on `chemistry/circuit_energy.py` being correct. Do not proceed
   past Section 3 until this passes. Separately, before trusting any real
   MD/NEB run: validate `mace_mliap_unified.py` on a tiny system per its own
   docstring — a container that builds successfully says nothing about
   whether that integration actually produces correct forces.

---

## 1. Choose your molecule (once)

All later steps reuse the SAME atom ordering established here.

```bash
cd lammps_md
python mol_to_lammps_data.py \
    --smiles "CN(C)CCCN1c2ccccc2CCc2ccccc21" \
    --output imipramine.data --element-order C H N --device cuda
```
→ `imipramine.data`, `imipramine.data.idx.txt` (inspect the latter before
defining dihedral tuples in Section 2.4).

---

## 2. Data generation (once per molecule)

### 2.1 Molecular dynamics
```bash
./run_md_ensemble.sh <lammps_binary> <mace_off_model.pt> imipramine.data 8
```
→ `runs/traj{0..4}.lammpstrj`

### 2.2 Reference conformers (two runs: a coarse 15-set and a fine 122-set)
```bash
python generate_reference_conformers.py \
    --smiles "CN(C)CCCN1c2ccccc2CCc2ccccc21" \
    --n-confs 500 --cluster-rms-thresh 1.2 --n-select 15 \
    --output-dir refs_15 --element-order C H N --device cuda --write-lammps-data

python generate_reference_conformers.py \
    --smiles "CN(C)CCCN1c2ccccc2CCc2ccccc21" \
    --n-confs 2000 --cluster-rms-thresh 0.5 --n-select 122 \
    --output-dir refs_122 --element-order C H N --device cuda
```
→ `refs_15/conformer_*.{xyz,data,idx.txt}`, `refs_122/...`, both with `summary.csv`

### 2.3 NEB transition pathways (uses the 15-set)
```bash
python prepare_neb_pairs.py \
    --conformer-dir refs_15 --output-dir neb_runs \
    --lammps-bin lmp_mace --model-file mace_off_model.pt --n-images 30

bash neb_runs/run_neb_pairs.sh   # or submit per-pair as a cluster array job

python filter_neb_pathways.py \
    --neb-dir neb_runs --n-images 30 --element-order C H N \
    --dataset-out neb_dataset --check-energy --mace-model small --device cuda
```
→ `neb_dataset/*.xyz`, `neb_filter_summary.csv`

### 2.4 Out-of-distribution perturbations (uses the 122-set's highest-energy conformer)
```bash
python generate_ood_perturbations.py \
    --reference-dir refs_122 --n-perturbations 200 --stdev 0.05 \
    --output-dir ood_dataset
```

### 2.5 Consolidate + characterize
```bash
python build_conformer_manifest.py \
    --md-traj runs/traj0.lammpstrj:10 --md-traj runs/traj1.lammpstrj:10 \
    --md-traj runs/traj2.lammpstrj:10 --md-traj runs/traj3.lammpstrj:10 \
    --md-traj runs/traj4.lammpstrj:5 \
    --neb-dir neb_dataset --ood-dir ood_dataset \
    --element-order C H N \
    --dihedral-defs dihedrals_imipramine_example.json \
    --output-dir full_dataset
```
→ `full_dataset/*.xyz`, `full_dataset/manifest.csv` — **the master index
every later stage reads from.**

⚠️ `dihedrals_imipramine_example.json` uses the paper's own Fig. 15 atom
ordering, which will NOT match RDKit's SMILES-derived ordering from Step 1.
Rebuild it from your own `.idx.txt` before trusting the dihedral columns
(this does not block anything downstream — dihedral angles are diagnostic
only, not used by ADAPT-VQE or the model).

---

## 3. Chemistry: Hamiltonians + ADAPT-VQE reference circuits

Run once **per active-space qubit count** (3x: 12/14/16 qubits) for the
Hamiltonian vectors, and once **per (qubit count, tolerance) dataset** (5x,
per the paper's Table 2) for the ADAPT-VQE circuits:

| Config | n_active_electrons | n_active_orbitals | pool | reference | tolerance (mHa) |
|---|---|---|---|---|---|
| 12q, ε=5  | 6 | 6 | uccgsd | casci | 5 |
| 12q, ε=10 | 6 | 6 | uccgsd | casci | 10 |
| 14q, ε=5  | 6 | 7 | uccgsd | casci | 5 |
| 14q, ε=16 | 6 | 7 | uccgsd | casci | 16 |
| 16q, ε=15 | 8 | 8 | uccsd  | ccsd  | 15 |

```bash
cd ../chemistry

# Hamiltonian vectors -- once per qubit count (shared across its tolerance variants)
python build_hamiltonian_vectors_for_dataset.py \
    --manifest ../lammps_md/full_dataset/manifest.csv \
    --n-active-electrons 6 --n-active-orbitals 6 \
    --term-order-file term_order_12q.json \
    --output-dir hamiltonians_12q

# ADAPT-VQE circuits -- once per (qubit count, tolerance) row above
python build_adapt_vqe_dataset.py \
    --manifest ../lammps_md/full_dataset/manifest.csv \
    --n-active-electrons 6 --n-active-orbitals 6 \
    --pool uccgsd --reference casci --tolerance-mha 5.0 \
    --output-dir adaptvqe_12q_eps5 \
    --skip-existing   # add on resume after an interrupted run
```
→ `hamiltonians_12q/<conformer_id>.npz` (reused by every tolerance variant
of the 12q config); `adaptvqe_12q_eps5/circuits.jsonl` + `splits.json`.

Repeat the `build_adapt_vqe_dataset.py` line for each of the 5 rows in the
table (swap `--n-active-electrons/orbitals`, `--pool`, `--reference`,
`--tolerance-mha`, `--output-dir`), and the `build_hamiltonian_vectors_for_dataset.py`
line once per distinct qubit count (12/14/16).

This is the most expensive stage in the whole pipeline (every conformer
needs its own exponential-then-binary search over ADAPT-VQE's `max_iter`) —
use `--limit N` for a smoke test before committing to a full manifest run.

---

## 4. Model: pretraining (repeat per dataset config)

```bash
cd ../model

python build_training_examples.py \
    --circuits ../chemistry/adaptvqe_12q_eps5/circuits.jsonl \
    --splits ../chemistry/adaptvqe_12q_eps5/splits.json \
    --hamiltonian-dir ../chemistry/hamiltonians_12q \
    --pool-size <N>   `# printed by build_adapt_vqe_dataset.py's run log` \
    --output-dir training_examples_12q_eps5

python pretrain.py \
    --train-jsonl training_examples_12q_eps5/train.jsonl \
    --val-jsonl training_examples_12q_eps5/val.jsonl \
    --tokenizer-dir training_examples_12q_eps5 \
    --hidden-size 1024 --num-layers 16 --num-heads 8 --num-kv-heads 8 \
    --context-length 1024 --sliding-window 1024 --rope-theta 10000 \
    --encoder-hidden-dim 2048 --encoder-depth 4 --encoder-ffn-mult 2.0 \
    --encoder-dropout 0.2 \
    --epochs 20 --batch-size 8 --lr 4e-5 --weight-decay 0.1 \
    --output-dir checkpoints_12q_eps5
```
Architecture hyperparameters above match Appendix Table 6's 12q/ε=5mHa
column — see that table (or `hamiltonian_encoder.py`'s docstring, Table 3,
for the encoder side) for the other four configs' values.

→ `training_examples_12q_eps5/{train,val,test}.jsonl` + tokenizer files +
`hamiltonian_scale.npy`; `checkpoints_12q_eps5/best_checkpoint.pt`.

---

## 5. Baseline evaluation (pretraining-only accuracy)

```bash
python generate.py \
    --checkpoint checkpoints_12q_eps5/best_checkpoint.pt \
    --tokenizer-dir training_examples_12q_eps5 \
    --test-jsonl training_examples_12q_eps5/test.jsonl \
    --num-candidates 16 --max-new-tokens 600 --top-p 0.95 \
    --output generated_12q_eps5_pretrain.jsonl

cd ../chemistry
python evaluate_generated_circuits.py \
    --generated ../model/generated_12q_eps5_pretrain.jsonl \
    --xyz-dir ../lammps_md/full_dataset \
    --n-active-electrons 6 --n-active-orbitals 6 --pool uccgsd \
    --output best_of_n_12q_eps5_pretrain.jsonl
```

---

## 6. RL post-training (GRPO/DAPO)

```bash
cd ../model
python rl_grpo.py \
    --checkpoint checkpoints_12q_eps5/best_checkpoint.pt \
    --tokenizer-dir training_examples_12q_eps5 \
    --train-jsonl training_examples_12q_eps5/train.jsonl \
    --xyz-dir ../lammps_md/full_dataset \
    --n-active-electrons 6 --n-active-orbitals 6 --pool uccgsd \
    --group-size 16 --conformers-per-step 4 --epochs-per-batch 2 \
    --lr 4e-6 --beta 0.28 \
    --output-dir rl_checkpoints_12q_eps5 \
    --rollout-log rl_checkpoints_12q_eps5/rollouts.jsonl
```

## 7. Self-distillation rounds (RL → filter → SFT, repeated)

Either drive it manually per round with `build_distillation_dataset.py` +
`distill_sft.py`, or use the orchestrator (12-qubit, 3-round schedule from
Appendix Table 7 — edit the array variables at the top of the script for
the 14q/16q 5-round schedules):

```bash
./run_self_distillation_rounds.sh \
    checkpoints_12q_eps5/best_checkpoint.pt \
    training_examples_12q_eps5 \
    training_examples_12q_eps5/train.jsonl \
    training_examples_12q_eps5/val.jsonl \
    ../lammps_md/full_dataset \
    ../chemistry/hamiltonians_12q \
    training_examples_12q_eps5/hamiltonian_scale.npy \
    ../chemistry/adaptvqe_12q_eps5/circuits.jsonl \
    6-31g 0 0 6 6 uccgsd \
    distill_12q_eps5
```
→ final checkpoint at `distill_12q_eps5/round3/distill_checkpoint/best_checkpoint.pt`

---

## 8. Post-RL/distillation evaluation

Repeat Section 5's two commands with the final checkpoint from Step 7 (or
any intermediate RL checkpoint from Step 6) in place of the pretrained one,
producing e.g. `best_of_n_12q_eps5_post_distill.jsonl`.

## 9. Compare methods + reproduce paper-style plots

```bash
cd ../evaluation
python compute_metrics.py \
    --results pretrain:../chemistry/best_of_n_12q_eps5_pretrain.jsonl \
    --results post_distill:../chemistry/best_of_n_12q_eps5_post_distill.jsonl \
    --epsilon-mha 5.0 --target-accuracy-mha 1.0 \
    --output summary_12q_eps5.csv --output-json distributions_12q_eps5.json

python plot_energy_errors.py \
    --distributions distributions_12q_eps5.json --epsilon-mha 5.0 \
    --output-prefix plots_12q_eps5
```

## 10. Compute-speedup benchmark

```bash
python benchmark_speedup.py \
    --xyz ../lammps_md/full_dataset/<one_conformer>.xyz \
    --checkpoint ../model/checkpoints_12q_eps5/best_checkpoint.pt \
    --tokenizer-dir ../model/training_examples_12q_eps5 \
    --term-order-file ../chemistry/term_order_12q.json \
    --hamiltonian-scale ../model/training_examples_12q_eps5/hamiltonian_scale.npy \
    --n-active-electrons 6 --n-active-orbitals 6 --pool uccgsd \
    --reference casci --tolerance-mha 5.0
```

## 11. Hardware-track circuit optimization (optional)

Requires one generated `operator_sequence` saved as JSON (`[[idx, theta], ...]`
— pull one out of a `generate.py`/`evaluate_generated_circuits.py` output row):

```bash
cd ../hardware
python build_native_circuit.py \
    --xyz ../lammps_md/full_dataset/<conformer>.xyz \
    --n-active-electrons 6 --n-active-orbitals 6 --pool uccgsd \
    --operator-sequence-json op_sequence.json \
    --output-qasm circuit.qasm
```
Real device/emulator submission needs InQuanto + Quantinuum Nexus
credentials this environment doesn't have — see
`quantinuum_submission_stub.py` for exactly what would plug in there.

---

## Repeat for every dataset config

Steps 3 (ADAPT-VQE circuits only, not the shared Hamiltonian build) through
10 run once per row of the Section 3 table — 5 times total to reproduce the
paper's full result set. Step 2 (data generation) and the per-qubit-count
half of Step 3 run once each, reused across each qubit count's tolerance
variants.

---

## Appendix: Validation Checkpoints (consolidated)

In the order you'll hit them:

| Stage | File | Risk | How to check |
|---|---|---|---|
| MD | `lammps_md/md_imipramine.in` | LAMMPS-MACE `pair_style`/`pair_coeff` syntax varies by plugin build | Compare against your build's own example inputs |
| NEB | `lammps_md/neb_template.in` | `neb` command's final-coordinates file format is version-sensitive | Check against `examples/neb` in your LAMMPS install |
| Hamiltonian | `chemistry/build_hamiltonian.py` | OpenFermion chemist→physicist integral transpose unverified | Cross-check `e_hf`/`e_casci` against `cudaq_solvers.create_molecule`'s independently-computed energies for the same conformer |
| ADAPT-VQE | `chemistry/run_adapt_vqe.py` | `molecule.energies` dict key names (`R-CASCI`/`R-CCSD`) reverse-engineered from shipped source, not confirmed live | First run will raise a clear `KeyError` listing actual keys if wrong |
| ADAPT-VQE | `chemistry/run_adapt_vqe.py` | `SpinOperator.__str__` used as an identity key to recover pool indices | If operator-index mapping looks wrong, inspect `str(pool[i])` directly in a live session |
| Model | `model/gemma_hamiltonian_model.py` | Gemma 3 config/model class names shifted across `transformers` versions | Import error will point at the exact two lines to fix |
| Generation | `model/generate.py` | HF `past_key_values`/`use_cache` behavior under pure `inputs_embeds` driving is implementation-dependent | Watch for repeated/garbage tokens after the first few generation steps |
| **Energy eval** | `chemistry/circuit_energy.py` | **Highest risk in the pipeline**: `SpinOperator` term decomposition + `exp_pauli` sign convention, silently wrong if mismatched | **Run `chemistry/test_circuit_energy.py` — do not proceed past Section 3 without this passing** |
| Hardware | `hardware/pmsv.py` | Simplified reimplementation of the *concept*, not the cited paper's or InQuanto's actual algorithm | Not independently verifiable here; treat as illustrative only |
| Hardware | `hardware/quantinuum_submission_stub.py` | Real device submission not implemented (proprietary, no access) | N/A — intentionally a stub |

The energy-evaluation checkpoint is the load-bearing one: RL reward,
best-of-N selection, the evaluation harness, and the speedup benchmark all
depend on it being correct.
