#!/usr/bin/env python3
"""
Python-launched driver for md_imipramine.in, needed because ML-IAP
"unified" potentials (mace_mliap_unified.py) are Python OBJECTS that must
be registered with a running `lammps` instance before the rest of the input
script executes -- they cannot be referenced by name from a standalone
`lmp -in script.in` CLI invocation the way a compiled pair_style can.

VALIDATION WARNING: the exact registration call (activate_mliappy / however
your LAMMPS version exposes it) is unverified here -- see
mace_mliap_unified.py's module docstring for the full caveat. This driver
is the other half of that same unverified integration; get
mace_mliap_unified.py working first (per its own recommended tiny-system
test) before trusting this script end-to-end.

Replaces the direct `lmp -in md_imipramine.in -var ...` invocation in
run_md_ensemble.sh with `python run_md_python_driver.py ...` (same -var
values, passed as CLI args below instead).

Usage:
  python run_md_python_driver.py \\
      --seed 12345 --run-label traj0 \\
      --data-file imipramine.data --element-order C H N \\
      --mace-model small --device cuda
"""
import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-label", required=True)
    p.add_argument("--data-file", required=True)
    p.add_argument("--element-order", nargs="+", required=True)
    p.add_argument("--mace-model", default="small", choices=["small", "medium", "large"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--input-script", default="md_imipramine.in")
    args = p.parse_args()

    import lammps
    from mace_mliap_unified import MACEUnified

    lmp = lammps.lammps()

    # VERIFY: this is the step with the least certainty in the whole
    # integration -- the exact function LAMMPS exposes to hand a Python
    # MLIAPUnified instance to a running lammps() session (name/module path
    # varies by version; `lammps.mliap.mliap_unified_couple` is a plausible
    # location based on general ML-IAP documentation, not confirmed here).
    from lammps.mliap.mliap_unified_couple import activate_mliappy

    activate_mliappy(lmp)
    unified = MACEUnified(
        element_types=args.element_order, model_size=args.mace_model, device=args.device
    )
    lmp.mliap.load_model(unified)  # VERIFY: exact registration call/name

    # Hand the rest of the run to the ordinary input script, with the same
    # -var substitutions md_imipramine.in already expects. md_imipramine.in
    # itself needs its `pair_style mace ...` / `pair_coeff ...` lines
    # replaced with `pair_style mliap unified` (no model path argument --
    # the Python object above IS the model) before this will work.
    lmp.command(f"variable seed index {args.seed}")
    lmp.command(f"variable run_label index {args.run_label}")
    lmp.command(f"variable data_file index {args.data_file}")
    lmp.file(args.input_script)


if __name__ == "__main__":
    main()
