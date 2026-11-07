# Arithmetic Addition: RoPE baseline vs. Abacus

This example contains a controlled comparison between two decoder-only
Transformers on synthetic multi-digit addition:

| Experiment | Positional signal | Optimizer | Config |
| --- | --- | --- | --- |
| Baseline | RoPE (rotary relative positions) | AdamW | `config.yaml` |
| Abacus | Learned Abacus digit-column embeddings | AdamW | `config_abacus.yaml` |

There are three matched size/budget variants:

| Variant | Configs | Train | OOD test | Epochs | Use |
| --- | --- | --- | --- | --- | --- |
| Small | `config.yaml`, `config_abacus.yaml` | 2-6 / 100k | 7-10 / 2k | 300 | quick local runs |
| Large | `config_large.yaml`, `config_abacus_large.yaml` | 2-6 / 200k | 7-12 / 5k | 150 | Modal / fast GPU |
| Local | `config_large_local.yaml`, `config_abacus_local.yaml` | 2-6 / 150k | 7-12 / 5k | 150 | RTX 2050, both models in ~4.5h |

The small and large variants use the same architecture, data format, loss, and
optimizer; they differ only in dataset size, OOD range, and model width (4x128
for small, 4x256 for large). The local variant keeps the large model and OOD
test but uses a 150k training pool and a 150-epoch budget so both models finish
in roughly 4.5 hours on the RTX 2050.

The experiment isolates the positional-encoding change. Both models use the
same generated data, the same disjoint train/validation split, the same OOD
test split, the same tokenizer, the same preprocessing, the same training and
evaluation sequence lengths, the same depth/width/heads/FFN/dropout, and the
same optimizer and training budget.

## Data and fairness

The local dataset (`dataset_impl/arithmetic_addition.py`) writes operands and
answers **least-significant digit first** (LSB-first / reversed). Reversal is
required for Abacus: it makes the column id of an already-generated token
independent of how many future digits will be generated. The same reversed
format is used by the RoPE baseline, so the comparison is not confounded by
data order.

Splits:

- **Small configs**: 2-6 digit train/ID validation, generated from one pool with
  `seed=42`, sliced into 100,000 train and 1,000 validation examples (disjoint);
  OOD test is 7-10 digits (`seed=200`, 2,000 examples).
- **Large configs**: the same 2-6 digit pool scaled to 200,000 train and 5,000
  validation examples; OOD test is 7-12 digits (`seed=200`, 5,000 examples).
- **Local configs**: the same setup with a 150,000-example training pool and
  5,000 validation examples.

The original (MSB-first) strings are kept as `original_source` /
`original_target` metadata and only used for readable logs; all model inputs and
metrics use the LSB-first representation.

## Abacus implementation

`models/abacus_transformer.py` follows the official implementation for
“Transformers Can Do Arithmetic with the Right Embeddings” (McLeish et al.,
2024, https://arxiv.org/abs/2405.17399):

- one learned embedding table, indexed by digit column;
- column ids are consecutive and one-based from the start of each number
  (the units digit, because data is LSB-first);
- all non-digit tokens share slot `0`;
- a single offset `k ~ U[0, max_offset]` is sampled per training batch and added
  to digit positions only; inference uses offset `0`;
- the Abacus embedding is **added** to the token embedding, not concatenated.

Paper add-ons outside the positional-encoding change — input injection,
progressive loss, and the depth-recurrent/looped transformer — are intentionally
excluded, so the shared trunk and training compute match the baseline. The only
added parameters are the Abacus column table itself.

The critical consistency property is: appending generated digits never changes
an earlier token's column id, so teacher-forced training and autoregressive
generation assign identical ids.

The shared Transformer trunk (hidden size, depth, heads, FFN, dropout) is
identical to the baseline. Abacus additionally owns its learned column table
(1024 x hidden size), which is the method's positional-parameter overhead; the
comparison reports this transparently rather than hiding it.

## Setup

The example is self-contained: it does **not** import `trainite.*` at runtime.
PyTorch, PyTorch-Ignite, Accelerate, TensorBoard, and W&B are used.

Using the existing environment:

```bash
cd examples/arithmetic_addition
source .venv/bin/activate
```

Using `uv` from scratch:

```bash
cd examples/arithmetic_addition
uv sync
```

## Training

Run from `examples/arithmetic_addition`:

```bash
./.venv/bin/python main.py config.yaml          # baseline-adamw
./.venv/bin/python main.py config_abacus.yaml   # abacus-adamw
```

Large run on Modal (L4 by default, ~4-5h for both; L40S ~2.5-3h):

```bash
modal run modal_train.py                    # config_large.yaml then config_abacus_large.yaml
modal run modal_train.py --gpu L40S
```

Modal requires a payment method on file even though the $30/month credit covers
this run. Outputs are written to the `arithmetic-addition-outputs` Volume:

```bash
modal volume get arithmetic-addition-outputs / ./modal_outputs
```

Local run (RTX 2050, both models sequentially, 150k samples, 150 epochs each):

```bash
cd examples/arithmetic_addition
./.venv/bin/python main.py config_large_local.yaml
./.venv/bin/python main.py config_abacus_local.yaml
```

Or both in one shell:

```bash
./.venv/bin/python main.py config_large_local.yaml && \
  ./.venv/bin/python main.py config_abacus_local.yaml
```

Each run creates a timestamped directory:

```text
outputs/<run_name>/<timestamp>/
├── best.pt                 # lowest validation loss
├── last.pt                 # most recent epoch
├── config.yaml             # resolved config for this run
├── output.log              # console output, appended by evaluate.py too
├── eval_context.json       # best epoch/global step and final eval metrics
├── wandb_run_id.txt        # W&B run id used by evaluate.py
├── tensorboard/            # TensorBoard event files
└── wandb/                  # W&B run files (offline runs live here)
```

Checkpoint selection is identical for both experiments: the checkpoint with the
lowest ID validation loss. Final ID and OOD metrics are logged at that best
checkpoint's epoch/global step, and `evaluate.py` reuses the same association
from `eval_context.json`.

### W&B behavior

`wandb.mode: auto` means:

- `online` when `WANDB_API_KEY` is set;
- otherwise `offline` (no network required).

Run names are `baseline-adamw` / `abacus-adamw` for the small configs,
`baseline-adamw-large` / `abacus-adamw-large` for the large configs, and
`baseline-adamw-large-local` / `abacus-adamw-large-local` for the local
configs, all in project `arithmetic-addition`. Logging failures are caught and
downgraded to warnings so training never depends on W&B. To force offline mode:

```bash
WANDB_MODE=offline ./.venv/bin/python main.py config.yaml
```

To upload a completed offline run:

```bash
./.venv/bin/wandb sync outputs/baseline-adamw/<timestamp>/wandb/offline-run-*
```

## Evaluation

`evaluate.py` reloads a run's best checkpoint, evaluates ID (validation) and OOD
(test) splits with the same `evaluation.evaluate_split` helper used by training,
appends to that run's `output.log`, and writes to the **same** TensorBoard
directory and W&B run (via `wandb_run_id.txt`). It never creates a new run
directory or W&B run.

```bash
./.venv/bin/python evaluate.py config.yaml
./.venv/bin/python evaluate.py config_abacus.yaml
```

Useful options:

```bash
--checkpoint PATH        # explicit best.pt (default: newest run for the config)
--run-dir DIRECTORY      # run directory containing best.pt
--max-examples N         # limit samples per split (smoke testing)
--skip-wandb
--skip-tensorboard
```

Exact match is the primary metric; teacher-forced token accuracy is secondary.

## Dashboards

TensorBoard:

```bash
./.venv/bin/tensorboard --logdir outputs
```

W&B (when online): project `arithmetic-addition`, with run names matching the
config variant (`*-large`, `*-large-local`). Offline runs can be viewed after
`wandb sync`.

Logged metrics:

| Tag | Meaning |
| --- | --- |
| `training/batch_loss` | smoothed per-iteration training loss |
| `training/loss`, `validation/loss` | epoch-level train/validation loss |
| `training/token_accuracy`, `validation/token_accuracy` | epoch-level token accuracy |
| `training/learning_rate` | learning rate at epoch end |
| `progress/epoch`, `progress/global_step` | epoch and optimizer step |
| `eval/id/exact_match`, `eval/ood/exact_match` | primary final metrics |
| `eval/id/token_accuracy`, `eval/ood/token_accuracy` | secondary final metrics |
| `eval/{id,ood}/exact_match/len_AxB` | exact match per operand-length bucket |
| `eval/best_val_loss` | validation loss of the selected checkpoint |
| `inference/training`, `inference/validation` | qualitative sample text summaries |

## Verification status

Verified locally:

- both experiments complete short smoke runs with TensorBoard and offline W&B
  logging;
- two identical seeded short runs produced identical losses for both models;
- `evaluate.py` loads the best checkpoint, appends to `output.log`, and writes
  to the same TensorBoard directory/W&B run id;
- the small baseline completed a full 300-epoch run on the RTX 2050 (ID exact
  match 1.0000, 7-10 digit OOD token accuracy 0.3750);
- measured RTX 2050 epoch times: 4x128 / 100k is ~19-21 s; 4x256 / 200k is
  ~75 s (baseline) and ~66 s (Abacus). At 150k that is ~57 s and ~50 s
  respectively, putting the local variant (150 epochs each) at ~4.5 h for
  both models.

**Not verified here:** convergence and final OOD quality for the large and
local variants. Run both configs in a variant on the same hardware and
compare their best-validation-loss checkpoints before drawing conclusions.

The Muon and zero-padding variants, and any separate reversed-operand
experiment, are left untouched. This comparison deliberately uses LSB-first
data for both models, as described above.
