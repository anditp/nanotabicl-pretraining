# nanoTabICL regression pretraining

Native PyTorch pretraining with **uv**, for a CUDA GPU on Linux or Apple's MPS
GPU on macOS. This repository trains the baseline for the separate WSP regression
experiment. It does not run the WSP construction agent.

## Start on a GPU machine

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then clone this private repository using your authenticated GitHub account:

```sh
git clone https://github.com/anditp/nanotabicl-pretraining.git
cd nanotabicl-pretraining
uv sync --locked
uv run --frozen python preflight.py --device cuda
uv run --frozen python train.py --device cuda --run-dir runs/cuda-seed0
```

On an Apple Silicon Mac, replace `cuda` with `mps`:

```sh
uv run --frozen python preflight.py --device mps
caffeinate -i uv run --frozen python train.py --device mps --run-dir runs/mps-seed0
```

`uv` manages Python 3.12, dependencies and the local `.venv`. PyTorch performs
GPU computation. The lock uses PyTorch 2.15.0 from PyPI, which supplies the
native Mac build and a CUDA-enabled Linux build. A Linux machine needs an NVIDIA
driver compatible with that build. Explicit `--device cuda`/`mps` fails if the
accelerator is unavailable; it never silently falls back to CPU. `--device auto`
selects CUDA, then MPS, then CPU. The trainer uses one GPU per process.

Run long jobs in tmux or your cluster's normal job scheduler. `caffeinate`
prevents Mac idle sleep while its child runs; closing the lid can still stop it.

## Same model, prior and training recipe

The original nanoTabICLv2 backbone is unchanged: width 128, 3 column blocks,
3 row blocks, 12 ICL blocks, eight heads, four row CLS tokens, 128 inducing
vectors and scalar regression output. There are 27,537,265 trainable parameters.
Model and prior are pinned to upstream commit
`4a7f9c7648f05c3efb34555e088105405a707469`.

The fixed recipe is 20,000 optimizer steps, eight independently sampled tables
per step using gradient accumulation, numeric targets from upstream
`rand_dataset_filtered`, seed zero, AdamW with learning rate 0.001, weight decay
0.01, 1,000-step warmup, cosine decay and clipping at one. Context sizes range
from 32 to 1,024, feature counts from 1 to 100, and each table has 128 query rows.
Feature and target preprocessing matches the WSP evaluator exactly.

Synthetic generation, ExtraTrees-based prior filtering and preprocessing still
run on CPU. The network's forward/backward pass and optimizer use the selected
GPU. `training.jsonl` records `prior_and_transfer_seconds` and
`optimization_seconds` so a CPU sampling bottleneck is visible. This first
version does not use distributed training, mixed precision or a different model.

No CTR23 dataset values, benchmark split files, caches or WSP weights are needed
or included. Benchmark outcomes never select checkpoints. The final fixed-budget
checkpoint is the primary baseline; this is local pretraining, not an official
pretrained checkpoint or an upstream training-recipe reproduction claim.

## Preserve the existing experiment identity

`experiment/` holds exact copies of all source files included in the original
experiment identity. A few unused benchmark metadata/evaluation files are kept
there solely so that those original file checksums remain verifiable; the
trainer does not read them as training data.

`reference_contract.json` records the original identity. `train.py` verifies
every file, then invokes the unchanged original trainer. It replaces only the
old benchmark-preparation gate with source/configuration verification, adds
timing, and records native-entrypoint provenance in each checkpoint. It does
not alter the original training recipe or read OpenML data. OpenML is not a
dependency of this native training environment.

The existing WSP project, completed weights, frozen benchmark manifest, running
Docker training and WSP sandbox remain separate from this repository.

## Checkpoints and resume

Outputs under the selected run directory:

- `latest.pt`: optimizer/model/completed-step and Python/NumPy/PyTorch RNG state,
  saved atomically every 100 steps.
- `final.pt`: final fixed-budget model, safe tensor/simple-metadata format.
- `state.json`: completion status, reference identity and final checkpoint hash.
- `native_provenance.json`: native entrypoint/source hashes, Git revision,
  Python/platform and accelerator.
- `training.jsonl`: losses, learning rates, gradients and timings.

Resume a native run on the same device/environment:

```sh
uv run --frozen python train.py --device cuda --run-dir runs/cuda-seed0 --resume
```

The CPU Docker job can continue independently. A GPU job starts a new seed-zero
run by default. This adapter rejects CPU-Docker-to-GPU migration as an exact
resume; PyTorch does not promise bitwise reproducibility between CPU and GPU.
It resumes from the last completed native checkpoint and replays later work,
without saving a partially applied update. Git ignores all generated checkpoints.

## Bring the completed baseline back for comparison

Copy the completed run directory, including `state.json`, `final.pt` and
`native_provenance.json`, into the original project's `baselines/` directory:

```sh
# In the original nanoTabICL-regression project:
python3 manage.py evaluate --wsp-run runs/20261122T151512Z-1d2ebe --baseline-run baselines/cuda-seed0
```

The unchanged evaluator verifies the preserved reference identity, model
configuration, full 20,000-step budget and checkpoint checksum. The extra native
provenance records the environment that produced the weights. Do not copy an
unfinished checkpoint into place as the final baseline.

## Checks

```sh
uv run --frozen python -B -m unittest -v test_native test_runtime
uv run --frozen python preflight.py --device cuda
```

Tests cover frozen-source validation, accelerator selection, native adapter and
checkpoint provenance, and exact CPU optimizer/RNG resume. The preflight uses
the full original backbone on a small disposable synthetic table and performs
forward/backward/AdamW. It neither loads nor changes actual WSP weights.

Verified here with uv 0.13.9 and Python 3.12.16: all six tests pass, and the
full-backbone CPU preflight completes successfully. CUDA and MPS are unavailable
to the editing session, so run the accelerator preflight on your training host.
Reference files retain their original bytes, including line endings, through
Git attributes.

## References

- [nanoTabICL source](https://github.com/soda-inria/nanotabicl/tree/4a7f9c7648f05c3efb34555e088105405a707469)
- [uv and PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

The upstream BSD license is retained in `LICENSE` and `experiment/source/LICENSE`.
