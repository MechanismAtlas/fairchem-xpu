# UMA Nitro on Intel XPU

This package ports UMA Nitro inference to Intel XPU using PyTorch and Intel Triton.
It retains the FairChem XPU fork history and upstream license notices. The tested
configuration is UMA-S-1.2.1 with the `omol` task on Aurora GPU Max 1550 tiles.
Model checkpoints are obtained separately under their own terms.

## Entry point

In an environment with the compatible XPU PyTorch/Triton stack and this fork's
`fairchem-core` package installed:

```python
from fairchem.core import FAIRChemCalculator
from fairchem.core.models.uma.uma_nitro_xpu import load_predict_unit

predictor = load_predict_unit("/path/to/uma-s-1p2p1.pt")
calculator = FAIRChemCalculator(predictor, task_name="omol")
```

The validated host stack used Python 3.12, Aurora frameworks 2026.1.0,
PyTorch `2.13.0a0+gitcf30153`, and the associated Intel Triton runtime. This is
not a claim of compatibility with arbitrary PyPI Torch/Triton installations.
For OpenMM OpenCL interoperability, initialize the Intel Triton backend before
creating an OpenCL Context. Reactra's UMA adapter performs this initialization.
Lazy first initialization inside an existing Context corrupted position readback
in the retained native reproducer; per-step synchronization is not required.

## Qualified default

The default combines FP32 forward activations, two-half forward GEMMs,
deterministic Wigner gauge, ordered exact neighbor buckets and a fixed-tree
charge reduction. Selected adjoints retain measured narrower arithmetic.
It is not bitwise-equivalent to general float64 inference. Capacity overflow
continues to refuse rather than silently truncate neighbors.

The qualified v44 experiment measured approximately 32.03 ms per PMD energy/force
call (416 atoms) and 23.10 ms for CM (299 atoms). Three fresh-process PMD
24-axis derivative checks passed the unchanged 0.0001 nm displacement and
5 kJ/mol/nm plus 1% policy. Reference-force, three rotation and graph/overflow
checks passed on both fixtures. See `validation-summary.json` for source hashes
and numerical summaries. These checks do not establish general chemical accuracy,
long trajectory stability, reaction crossings or free-energy barriers.

Reactra owns the enzyme fixtures, complete workflow benchmarks, EMLE acceleration
and scheduling. This fork owns provider inference; it does not duplicate those
engine components. Raw native experiment captures remain in the Reactra research
archive. Benchmark timings exclude setup unless explicitly stated.

## Provenance

The base XPU revision is `5ea25ebb68f043395b299344760852726d2a2bce` from
`abagusetty/fairchem`; the Nitro source reference is
`645af7d03e91138ef07e51de387316c9e2e152dc` from `facebookresearch/fairchem`.
Preserve `UMA_NITRO_LICENSE_PROVENANCE.md` and embedded source notices.
