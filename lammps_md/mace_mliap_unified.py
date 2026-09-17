"""
LAMMPS ML-IAP "unified" wrapper exposing a MACE-OFF model as a LAMMPS
interatomic potential, used instead of a dedicated pair_style mace plugin
(per your instruction to use ML-IAP, since neither of your container
recipes builds a custom MACE pair_style).

======================== VALIDATION WARNING ========================
THIS IS A BEST-EFFORT DRAFT, NOT A TESTED INTEGRATION -- treat it as a
starting point for hands-on debugging, not working code. LAMMPS's
MLIAPUnified ABC (lammps.mliap.mliap_unified_abc.MLIAPUnified) and the
MLIAPData object passed into its methods have attribute/method names that
have changed across LAMMPS versions, and I have no live LAMMPS build in
this environment to check against. Every "VERIFY:" comment below marks a
specific point of real uncertainty. Before trusting this file:

  1. Diff it against the reference LAMMPS ships in its own source tree at
     <lammps_source>/examples/mliap/mliap_unified_lj.py and
     <lammps_source>/python/lammps/mliap/mliap_unified_abc.py -- THOSE are
     ground truth for your built LAMMPS version, this file is not.
  2. The force write-back convention (data.update_pair_forces vs. a
     direct per-atom force array vs. something else entirely) is the single
     highest-risk guess in this file -- LJ's example will show you the
     PAIRWISE convention; a many-body potential like MACE may need a
     different entry point that ML-IAP's design is supposed to support but
     whose exact name I could not verify here.
  3. Test on a tiny system (a handful of atoms, one timestep) and compare
     the reported energy/forces against calling ASE's mace_off calculator
     directly on the same geometry, before running any real MD or NEB.

DESIGN CHOICE: rather than hand-reconstructing MACE's internal graph
tensors (edge_index, shifts, node_attrs, z_table one-hot encoding, etc. --
an internal format I cannot verify from memory), this wrapper reconstructs
an ASE Atoms object from the MLIAPData pairwise neighbor information on
every call and reuses MACE's own already-validated ASE calculator
(mace.calculators.mace_off). This trades per-step overhead for correctness
confidence.

EXECUTION MODEL CAVEAT: "unified" ML-IAP potentials are Python OBJECTS, not
plain strings you can reference from a standalone `lmp -in script.in`
invocation. In practice this likely needs to be driven from a small Python
script that constructs LAMMPS via the `lammps` Python module, registers
this class instance with it, and THEN runs the rest of the input script --
see run_md_python_driver.py (companion file) for the intended shape. This
changes md_imipramine.in's/neb_template.in's execution model from a direct
CLI call to a Python-launched one; run_md_ensemble.sh's invocation will
need to change accordingly once this is validated.
======================================================================
"""
import numpy as np
from ase import Atoms
from lammps.mliap.mliap_unified_abc import MLIAPUnified


class MACEUnified(MLIAPUnified):
    def __init__(self, element_types, model_size="small", device="cuda", rcutfac=6.0):
        super().__init__()
        from mace.calculators import mace_off

        self.element_types = element_types  # e.g. ["C", "H", "N"], index-matched to
                                             # LAMMPS atom types AND the data file's
                                             # specorder from mol_to_lammps_data.py
        self.ndescriptors = 1  # unused -- MACE has no separate descriptor stage here
        self.nparams = 1  # unused -- MACE's own parameters aren't exposed through this ABC
        self.rcutfac = rcutfac  # VERIFY: must be >= MACE-OFF's actual cutoff radius
        self.calc = mace_off(model=model_size, device=device)

    def compute_descriptors(self, data):
        pass  # not used -- see DESIGN CHOICE above

    def compute_gradients(self, data):
        pass  # not used -- see DESIGN CHOICE above

    def compute_forces(self, data):
        # VERIFY: attribute names below (nlocal/ntotal/elems/pair_i/pair_j/rij)
        # against your LAMMPS version's actual MLIAPData object.
        n_local = data.nlocal
        n_total = data.ntotal
        elems = np.asarray(data.elems)  # 0-based indices into self.element_types
        pair_i = np.asarray(data.pair_i)
        pair_j = np.asarray(data.pair_j)
        rij = np.asarray(data.rij)

        # Reconstruct absolute positions for all local+ghost atoms from
        # pairwise displacement vectors, anchored at atom 0 -- avoids
        # assuming `data` exposes absolute coordinates directly (uncertain
        # across versions); only relies on the pairwise contract, which
        # the unified interface is documented to always provide.
        positions = np.zeros((n_total, 3))
        placed = np.zeros(n_total, dtype=bool)
        placed[0] = True
        for _ in range(4):  # VERIFY: enough passes to place a fully connected graph
            for i, j, r in zip(pair_i, pair_j, rij):
                if placed[i] and not placed[j]:
                    positions[j] = positions[i] + r
                    placed[j] = True
                elif placed[j] and not placed[i]:
                    positions[i] = positions[j] - r
                    placed[i] = True

        symbols = [self.element_types[e] for e in elems]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = self.calc

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # VERIFY (highest-risk line in this file): the actual write-back API.
        # `data.energy` and a per-local-atom `data.update_pair_forces(...)`
        # call are a best guess modeled on the pairwise LJ example; MACE's
        # forces are already per-atom totals (not pairwise contributions),
        # so this may need a different ML-IAP entry point for many-body
        # potentials if pairwise write-back is enforced by your version.
        data.energy = float(energy)
        for i in range(n_local):
            data.update_pair_forces(i, forces[i])
