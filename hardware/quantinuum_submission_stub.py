"""
NOT IMPLEMENTED: real submission to Quantinuum Helios-1 (or its emulator).

Section 4.3 of ADAPT-GQE (arXiv:2607.22468) submits circuits through
Quantinuum's InQuanto computational-chemistry platform and the Nexus cloud
access platform. Both are commercial Quantinuum products requiring an
account and credentials -- unlike cudaq-solvers (which is open, pip-
installable, and which I downloaded and inspected directly to ground
run_adapt_vqe.py in this pipeline), I have no way to install, inspect, or
test against InQuanto/Nexus in this environment. Writing a submission
function against a guessed API would be indistinguishable from working code
until the first time someone with real credentials tries to run it, so
rather than fabricate one, this file documents the shape of the missing
piece and where it plugs into the rest of this pipeline.

What this pipeline DOES give you, ready to plug in once you have access:
  - hardware/build_native_circuit.py produces an optimized pytket Circuit
    from a generated operator_sequence.
  - hardware/pmsv.py's post_select_shots() implements the post-processing
    half of symmetry-verification error mitigation, independent of where
    the shots came from.

What you would need to add, with InQuanto/Nexus access:
  1. Compile `circuit` (from build_native_circuit.py) against the ACTUAL
     Helios-1 native gateset via InQuanto/pytket-quantinuum's
     device-specific backend, rather than this pipeline's generic
     CX-based optimization target.
  2. Submit the compiled circuit + Hamiltonian (operator averaging, per
     Pauli-term measurement bases) to the Nexus-hosted Helios-1 device or
     its fitted-noise-model emulator, at your chosen shot count.
  3. Feed the returned per-partition measurement bitstrings into
     pmsv.post_select_shots() (or a measurement-efficient PMSV
     implementation, if InQuanto exposes one) before computing the final
     energy estimate.
"""


def submit_and_evaluate(circuit, hamiltonian, shots, device="Helios-1"):
    raise NotImplementedError(
        "Requires InQuanto + Quantinuum Nexus account access. See this "
        "module's docstring for the three steps this function would need "
        "to perform once you have that access."
    )
