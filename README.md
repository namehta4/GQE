# GQE

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

## License

BSD 3-Clause — see [`LICENSE`](LICENSE).
