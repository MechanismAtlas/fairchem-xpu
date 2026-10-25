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

The initial qualified v44 experiment measured approximately 32.03 ms per PMD energy/force
call (416 atoms) and 23.10 ms for CM (299 atoms). Three fresh-process PMD
24-axis derivative checks passed the unchanged 0.0001 nm displacement and
5 kJ/mol/nm plus 1% policy. Reference-force, three rotation and graph/overflow
checks passed on both fixtures. See `validation-summary.json` for source hashes
and numerical summaries. These checks do not establish general chemical accuracy,
long trajectory stability, reaction crossings or free-energy barriers.

The current default also uses contiguous operands for the existing frozen FP32
linear layers. Native job 8960614 compared four fresh processes in an A–B–B–A
sequence, with 512 synchronized callbacks per system and case. Candidate medians
were 31.21 ms PMD and 22.71 ms CM, approximately 1.2% and 0.9% faster than the
paired controls. Both candidate runs beat both controls; the conservative gain is
about 1.1% PMD and 0.5% CM. Reference and three-rotation checks pass, as does the
original PMD 24-axis derivative policy. Startup is excluded; this incremental
callback gain is not a measured full-dynamics speedup. The four-product GEMM and
shape-specific tiling alternatives were rejected. See `current_profile` in the
validation summary for exact source identity and numerical evidence.

The current default also factors the first merged radial affine head exactly.
Its 32 Gaussian inputs depend on distance; the frozen source/target element
embeddings are projected once into a model-owned ordered-type cache. Runtime
evaluates the smaller Gaussian projection and gathers the constant contribution.
The original backward calculation is retained. Native job 8960750 measured
30.67–30.72 ms PMD and 22.32–22.36 ms CM in an A–B–B–A comparison, a further
1.3%/1.6% callback gain. References, rotations and the original 24-axis gate pass;
worst smooth derivative error is 2.0764 kJ/mol/nm under the unchanged policy.
Observed allocator peak increases by about 24 MiB. This adds no user settings
or service; it does not establish an additional full-dynamics speedup. Exact
source identities and controls are recorded in `current_profile`.

The current default additionally clones input data before transfer. This retains
all existing validation while avoiding GPU cloning and scalar synchronization for
CPU inputs, and leaves caller-owned data on its original device. Native job
8961256, with four fresh A–B–B–A processes and 512 changing-coordinate ordinary
Reactra callbacks per fixture/case, measured 30.53–30.56 ms PMD and 22.16–22.24 ms
CM: a further 1.4%/1.7% gain. Both candidates beat both controls. All eight
reference/rotation rows and the original 24-axis derivative gate pass unchanged
(worst smooth error 2.11066 kJ/mol/nm). Separate native probe 8961245 passes
18 field/dtype/aliasing/striding/batching/refusal/collation checks, including inputs
already on XPU. No new setting, cache or validation path is introduced. This is
not an additional full-dynamics speedup or current production chemistry claim.

Inference additionally avoids importing training telemetry: the WandB logger
import now lives at its two existing training-use sites. Logger conditions and
all model calculations remain unchanged. Native job 8961320 passes all eight
reference/rotation rows and the unchanged original 24-axis derivative gate
(worst smooth error 2.19494 kJ/mol/nm), with WandB absent throughout candidate
inference. Thirty-two lightweight logger-block cases preserve training results
and errors. This is a dependency cleanup: paired first-load timings show strong
cache effects and do not establish a repeatable whole-startup speedup. No
per-step or full-dynamics gain is claimed. Existing runtime prefixes remain frozen.

Nonperiodic inference now uses zero periodic images, including capacity retries.
Periodicity is bound into the existing capacity keys; periodic behavior and all
overflow guards remain. Native job 8961481 passes 21 exact graph/cache/overflow checks,
including same-density nonperiodic→periodic→nonperiodic transitions. Native job 8961502
measured 29.99–30.08 ms PMD / 21.86–21.90 ms CM across four fresh A–B–B–A processes,
512 ordinary changing-coordinate callbacks per fixture/case: a further 1.960%/1.802%
gain (conservative 1.794%/1.615%). All eight reference/rotation rows and the unchanged
original 24-axis derivative gate pass, worst smooth error 2.15355 kJ/mol/nm. All five
maintained runtime files exactly match the native candidate. These results do not
establish an additional full EMLE/MM speedup. No new user setting or cache is added.

The fused conv1 backward now stores its completed Wigner-weight gradient before
forming the node gradient, shortening register lifetimes. It only reorders
existing statements; no operation, setting or precision is added. Native job
8961648 completed successfully in 9m16s. Four fresh A–B–B–A processes, with 512
changing-coordinate callbacks per system/case, measured PMD 30.138/30.016 ms
controls versus 29.603/29.660 ms candidates; CM 21.925/21.939 versus
21.586/21.499 ms. Median gains are 1.502%/1.808%, conservative comparisons
1.200%/1.570%. Both candidates beat both controls. All eight reference/rotation
rows and the unchanged original 24-axis gate pass (worst smooth error
2.06140 kJ/mol/nm); peak GPU allocation is unchanged. This is a callback gain,
not an additional complete EMLE/MM measurement. The formatted source removes
one blank line from retained native source, with identical ASTs; both hashes are
recorded explicitly. Earlier profiles remain in the validation summary, and
existing runtime prefixes remain frozen.

Reactra owns the enzyme fixtures, complete workflow benchmarks, EMLE acceleration
and scheduling. This fork owns provider inference; it does not duplicate those
engine components. Raw native experiment captures remain in the Reactra research
archive. Benchmark timings exclude setup unless explicitly stated.

## Provenance

The base XPU revision is `5ea25ebb68f043395b299344760852726d2a2bce` from
`abagusetty/fairchem`; the Nitro source reference is
`645af7d03e91138ef07e51de387316c9e2e152dc` from `facebookresearch/fairchem`.
Preserve `UMA_NITRO_LICENSE_PROVENANCE.md` and embedded source notices.
