# nanoTabICL regression baseline: pretraining and evaluation

Native PyTorch pretraining with **uv**, for a CUDA 12.1 GPU on Linux or Apple's MPS
GPU on macOS. This repository trains the baseline for the separate WSP regression
experiment. It does not run the WSP construction agent.

## Start on a Linux CUDA 12.1 cluster

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then clone this public repository:

```sh
git clone https://github.com/anditp/nanotabicl-pretraining.git
cd nanotabicl-pretraining
uv sync --locked
uv run --frozen python preflight.py --device cuda
uv run --frozen python train.py --device cuda --run-dir runs/cuda-seed0
```

Run the preflight and training commands inside your scheduler's GPU allocation.
To update an existing clone, run `git pull --ff-only` followed by
`uv sync --locked`; uv replaces the previous PyTorch build in the environment.

### Submit with Slurm

After `uv sync --locked` finishes on the login node, submit from the repository
root:

```sh
sbatch train.slurm
```

`train.slurm` requests one GPU, four CPU cores (matching the trainer's thread
count), 64 GB of host memory and 24 hours. It runs the CUDA preflight, then
starts the fixed-budget baseline in `runs/cuda-seed0`. A failed preflight stops
the job. Output goes to `slurm-nanotabicl-JOB_ID.out` in the repository root.
The job uses the installed environment without synchronizing dependencies;
the checkout, `.venv` and uv-managed Python must be accessible on compute nodes,
and `uv` must be on the job's PATH.

Pass your site's partition/account and any resource overrides to `sbatch`,
before the script name, for example:

```sh
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT train.slurm
```

Trainer arguments go after the script name. To resume the last completed
checkpoint under the same environment:

```sh
sbatch train.slurm --run-dir runs/cuda-seed0 --resume
```

Add the same site options to the resume submission when required. The script
does not submit further jobs automatically.

On an Apple Silicon Mac, replace `cuda` with `mps`:

```sh
uv run --frozen python preflight.py --device mps
caffeinate -i uv run --frozen python train.py --device mps --run-dir runs/mps-seed0
```

`uv` manages Python 3.12, dependencies and the local `.venv`. PyTorch performs
GPU computation. Both training and evaluation lock PyTorch **2.5.1+cu121** on
Linux, from the explicit official `https://download.pytorch.org/whl/cu121`
index. The CUDA 12.1 wheel targets Linux x86_64 and includes its CUDA runtime
dependencies; the node still needs a compatible NVIDIA driver. PyTorch 2.15.0
does not provide this CUDA 12.1 build, so the PyTorch version is intentionally
changed. macOS uses PyTorch 2.5.1 from PyPI. Explicit `--device cuda`/`mps` fails if the
accelerator is unavailable; it never silently falls back to CPU. `--device auto`
selects CUDA, then MPS, then CPU. The trainer uses one GPU per process.

The scientific dependencies are pinned to published releases with Python 3.12
Linux wheels: NumPy 2.1.3, SciPy 1.14.1, scikit-learn 1.5.2 and pandas 2.2.3.
Both projects limit dependency resolution to releases published before
2025-06-01, so regenerating the locks retains an established package snapshot.
This replaces the original scientific pins copied from the Docker environment;
package versions are recorded with each training/evaluation run.

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

No CTR23 dataset values, caches or WSP weights are included. Public task
metadata and official split indices are frozen for evaluation only; pretraining
does not use them. Benchmark outcomes never select checkpoints. The final fixed-budget
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

The WSP construction project and its completed weights remain separate. This
repository adds evaluation of the pretrained baseline only.

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

The CUDA 12.1 update changes the package versions and training lock. A run
created with the old PyTorch 2.15 environment cannot be resumed exactly under
this environment; retain its original checkout/environment for exact resume,
or start a new run directory with the CUDA 12.1 environment.

## Evaluate the pretrained baseline

Evaluation is native and **baseline-only**. It compares the completed pretrained
model with five published CTR23 learners: ranger, rpart, xgboost, cv_glmnet and
gam. It does not load WSP weights or expose a WSP evaluation command.

A separate locked uv project adds OpenML for evaluation without changing the
training environment, training lock or checkpoint-resume fingerprints:

```sh
uv sync --project evaluation --locked
uv run --project evaluation --frozen python evaluate.py --baseline-run runs/cuda-seed0
```

The checkpoint must have completed the full 20,000-step training budget.
Evaluation freezes a checksum-verified copy before loading it. By default it uses
CPU float32, matching the reference protocol. To evaluate on your GPU instead,
append `--device cuda` or `--device mps`; the selected device is recorded and
must remain the same when resuming.

The benchmark is the frozen **19-dataset CTR23 subset, 640 official splits**,
selected from the 35-task suite using at most 10,000 total rows and 100 raw
predictors. It is not a claim of evaluating the entire 35-dataset suite. Task
IDs, dataset versions, target names, checksums and all repeats/folds are included
under `experiment/benchmarks/frozen/`. These files contain metadata and row
indices, not dataset feature or target values. OpenML downloads the actual data
into an ignored local cache when evaluation starts.

Every training row is retained as context. The fixed preprocessing is fitted
on context rows only; target normalization is inverted before computing RMSE,
MAE and R-squared in original units. Query batching uses the unchanged wrapper.
Each dataset's RMSE is the arithmetic mean of official fold RMSEs, matching the
published baseline aggregation. Cross-dataset comparisons use geometric means
of paired RMSE ratios and normalized RMSE, rather than a mean of raw errors
with incompatible units. Undefined constant-target metrics are counted.

The output directory contains `scores.csv`, `summary.json`, per-fold metrics,
predictions, train/test indices, checkpoint/source hashes, timing and sparsity.
Published scores are the authors' results, not local reruns. Failed tasks/folds
prevent a final aggregate; completed prediction files are checksummed for reuse.

```sh
uv run --project evaluation --frozen python evaluate.py --resume-evaluation evaluations/EVALUATION_ID
```

The final checkpoint retains compatibility with the original experiment through
its preserved source identity and configuration. WSP construction/evaluation
continues to live separately.

## Checks

```sh
uv run --frozen python -B -m unittest -v test_native test_runtime
uv run --frozen python preflight.py --device cuda
uv run --project evaluation --frozen python -B -m unittest -v test_evaluation
```

Tests cover frozen-source validation, accelerator selection, native adapter and
checkpoint provenance, and exact CPU optimizer/RNG resume. The preflight uses
the full original backbone on a small disposable synthetic table and performs
forward/backward/AdamW. It neither loads nor changes actual WSP weights.

Verified here with uv 0.13.9, Python 3.12.16 and PyTorch 2.5.1: all six tests pass, and the
full-backbone CPU preflight completes successfully. CUDA and MPS are unavailable
to the editing session, so run the accelerator preflight on your training host.
Evaluation tests use offline synthetic fixtures and cover incomplete-checkpoint
rejection, failed-fold recovery, prediction reuse and published-score aggregation.
All 12 training/evaluation tests pass; the evaluation integration tests use
offline synthetic fixtures and do not claim a completed CTR23 benchmark run.
Reference files retain their original bytes, including line endings, through
Git attributes.
Linux x86_64 dry-run synchronization of both locked environments selects
`torch==2.5.1+cu121` and `nvidia-cuda-runtime-cu12==12.1.105`.

## References

- [nanoTabICL source](https://github.com/soda-inria/nanotabicl/tree/4a7f9c7648f05c3efb34555e088105405a707469)
- [uv and PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [Official PyTorch 2.5.1 CUDA 12.1 installation](https://pytorch.org/get-started/previous-versions/#v251)
- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

The upstream BSD license is retained in `LICENSE` and `experiment/source/LICENSE`.
