# GQE

## What this project does (plain-language overview)

Every molecule has a natural "lowest-energy" arrangement of its electrons,
called its **ground state**. Knowing this precisely is the whole point of
computational chemistry — it's how you predict a drug's stability, a
material's properties, or a reaction's outcome. Quantum computers can, in
principle, find this state directly, but you first need to build a
**quantum circuit** (a recipe of quantum operations) that prepares it — and
building that recipe well is itself hard and slow.

This project trains an AI to build those recipes automatically, instead of
solving the slow method from scratch every time. Concretely, the pipeline
does this in order:

1. **Generate many shapes of the molecule.** Real molecules aren't rigid —
   they wiggle, rotate around bonds, and settle into different 3D shapes
   ("conformers"). We simulate this jiggling (molecular dynamics) to collect
   a large, varied set of realistic shapes for one target molecule.
2. **Solve each shape the slow, trustworthy way.** For every shape, we run
   an established quantum-chemistry algorithm called **ADAPT-VQE**, which
   iteratively builds a circuit, step by step, that prepares that shape's
   ground state accurately. This works well but is computationally
   expensive and has to be redone completely for every new shape.
3. **Teach an AI model the pattern.** We then treat each (molecule shape →
   circuit recipe) pair as a training example and train a small language
   model — conceptually similar to how ChatGPT learns to predict the next
   word, except here it learns to predict the next step of a quantum
   circuit, given the molecule's shape as input.
4. **Let the trained model generate new recipes instantly.** Once trained,
   the model can produce a full circuit for a *new, unseen* shape in a
   single fast pass — no slow step-by-step search required.
5. **Make the model better than its teacher.** Using reinforcement learning
   (trial, feedback, repeat — here the "feedback" is the energy the
   generated circuit actually achieves), we keep improving the model until
   it outperforms the ADAPT-VQE examples it was originally trained to copy.
6. **Check the work.** We compare the AI-generated circuits against the
   original slow method on accuracy and on speed, to confirm the shortcut
   is actually trustworthy and worthwhile.
7. **(Optional) Try it on real quantum hardware.** Finally, a generated
   circuit can be compiled down and run on an actual quantum computer, to
   see how it holds up outside of an idealized simulation.

Each numbered step above corresponds to a directory in this repository
(see the layout table below), and `PIPELINE.md` walks through the exact
commands for all of it. If you just want to try the smallest possible
version of the whole thing end to end, see `validation/` — it runs this
same seven-step process on hydrogen (H2), the simplest molecule there is,
specifically so you can sanity-check that everything works before pointing
it at something as complex as a real drug molecule.

---

An implementation of the pipeline described in *Learning to Prepare
Molecular Ground States with Transformer Models* (ADAPT-GQE),
[arXiv:2607.22468](https://arxiv.org/abs/2607.22468): a generative AI
framework that learns to synthesize molecular ground-state preparation
circuits, trained on ADAPT-VQE reference data and improved beyond it via
reinforcement learning and self-distillation.

This repository builds the full pipeline end-to-end using LAMMPS for
conformer sampling, `cudaq-solvers` for ADAPT-VQE, and a from-scratch
Gemma 3 model for circuit generation. NVIDIA's Nemotron path (the paper's
second, pretrained-LLM model) is out of scope here; IBM Quantum hardware
support is planned but not yet built (Quantinuum hardware compilation is
partially built — see `hardware/`).

**See [`PIPELINE.md`](PIPELINE.md) for the full end-to-end run order**,
including exact commands, which stages depend on which, and a consolidated
list of every validation checkpoint that should be cleared before trusting
a stage at real scale.

## Status

Nothing in this pipeline has been executed end-to-end. It was built from
the paper, from documentation, and — for `cudaq-solvers` specifically —
from its actual shipped source (downloaded and inspected directly rather
than guessed at). Every file that carries real correctness risk says so
in its own docstring, with a validation test provided where one could be
written (`chemistry/test_circuit_energy.py` in particular gates everything
downstream of it and should be the first thing run against a live
CUDA-Q environment).

## Repository layout

| Directory | Contents |
|---|---|
| `lammps_md/` | Molecule input, LAMMPS MD sampling (MACE-OFF via ML-IAP), NEB transition pathways, out-of-distribution perturbations, dataset consolidation |
| `chemistry/` | Active-space Hamiltonian construction, ADAPT-VQE via `cudaq-solvers`, generated-circuit energy evaluation |
| `model/` | Tokenizer, Hamiltonian encoder, Gemma 3 model, pretraining, custom generation loop, GRPO/DAPO RL post-training, self-distillation |
| `evaluation/` | Accuracy metrics, plots, compute-speedup benchmarking |
| `hardware/` | Circuit compilation/optimization (pytket), simplified symmetry-verification error mitigation |
| `docker/` | Merged container image (CUDA-Q + cudaqx solvers, LAMMPS, IBM Qiskit stack, pipeline dependencies) for NERSC Perlmutter |
| `orchestrator/` | Automated stage-graph runner (resume, retries, smoke tests, adaptive self-distillation stopping) plus Perlmutter SLURM submission scripts |
| `validation/` | Small, fast end-to-end validation run on H2 (bond-length scan in place of full conformer sampling) with its own SLURM scripts — run this before pointing the pipeline at a real molecule |

## License

BSD 3-Clause — see [`LICENSE`](LICENSE).
