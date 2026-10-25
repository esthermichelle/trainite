# Trainite Project: arithmetic-addition-starter

> Generated with Trainite v0.1.0 using:
> ```bash
> trainite init arithmetic-addition-starter --model rope-transformer --dataset arithmetic-addition --trainer decoder-trainer
> ```

Welcome to your new Trainite-generated training project!

Trainite is a toolbox for training language models with PyTorch-Ignite. This project contains real, readable Python files that you own and can modify as needed.

## Project Structure

- `config.yaml`: The central configuration for your experiment. Edit this to change hyperparameters, dataset paths, or output settings.
- `main.py`: The entrypoint for training. Run it with `python main.py config.yaml`.
- `models/`: Contains the model architecture definition.
- `dataset_impl/`: Contains the selected dataset and transform implementations.
- `trainer.py`: Defines the training and evaluation logic. You can override `_train_step` or `_eval_step` here.
- `config.py`: Contains base Pydantic models for configuration structure validation.
- `utils.py`: Shared utilities for configuration and logging.

## Getting Started

You can set up and run this project using either **uv** (recommended for speed) or standard **pip**.

### Option A: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver.

1. **Install uv** (if you don't have it):
   * **macOS/Linux**:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   * **Windows**:
     ```powershell
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   * Or via pip: `pip install uv`

2. **Install Dependencies & Set Up Virtual Environment**:
   ```bash
   uv sync
   ```

3. **Run Training**:
   ```bash
   uv run python main.py config.yaml
   ```

4. **Monitor Progress**:
   ```bash
   uv run tensorboard --logdir outputs
   ```

### Option B: Using standard pip & venv

If you prefer standard Python tools, you can create a virtual environment and use `pip`:

1. **Create and Activate a Virtual Environment**:
   * **macOS/Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   * **Windows**:
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate
     ```

2. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

   > [!NOTE]
   > `pip install -e .` only installs this project's dependencies — it does **not**
   > build the project into an installable Python package. This project is meant
   > to be run from source via `python main.py config.yaml`.
   >
   > To make it installable as a real package, restructure your code into a single
   > package — e.g., move `models/`, `dataset_impl/`, `preprocessors/` and the
   > top-level modules (`main.py`, `config.py`, `trainer.py`, `utils.py`) into a
   > `src/<your_package>/` directory with `__init__.py` files — then edit
   > `pyproject.toml` to replace the `[tool.setuptools]` block with:
   >
   > ```toml
   > [tool.setuptools.packages.find]
   > where = ["src"]
   > include = ["<your_package>*"]
   > ```
   >
   > and run `pip install -e .` again. See the
   > [setuptools package discovery docs](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)
   > for other layouts.

3. **Run Training**:
   ```bash
   python main.py config.yaml
   ```

4. **Monitor Progress**:
   ```bash
   tensorboard --logdir outputs
   ```

### Option C: Cloud Training (SkyPilot)

To launch this experiment on cloud GPUs (AWS, GCP, Azure, Lambda, etc.) using [SkyPilot](https://docs.skypilot.ai/):

1. **Initialize SkyPilot configuration** (if not already generated):
   ```bash
   trainite add:sky
   ```
   This generates `sky.yaml` and adds `skypilot` to `pyproject.toml`.

2. **Update dependencies & verify cloud**:
   ```bash
   pip install -e .    # or: uv sync
   sky check
   ```

3. **Launch on Cloud**:
   ```bash
   sky launch sky.yaml
   ```

> [!TIP]
> Trainite installs the base `skypilot` package. Running `sky check` will detect your active cloud credentials and provide the exact one-line command to install your specific cloud provider's driver if needed (e.g. `pip install "skypilot[aws]"` or `pip install "skypilot[kubernetes]"`).
>
> You can also run jobs on your own on-premises servers or custom clusters via SSH by following SkyPilot's [Existing Machines guide](https://docs.skypilot.co/en/latest/reservations/existing-machines.html).

## Experiment Logging and Checkpoints

The default `logger: tensorboard` stores metrics and checkpoints locally. To use ClearML instead:

1. Run `clearml-init` to configure your ClearML server and credentials.
2. Set `logger: clearml` in `config.yaml`.

ClearML runs still keep checkpoints in the local output directory. By default, `ClearMLSaver` also uploads them to
ClearML's file server with `output_uri=True`.

To change checkpoint storage, edit the `ClearMLSaver` call in `trainer.py`:

- Use a storage URI such as `s3://bucket/path` to upload somewhere else.
- Use `None` to leave the destination to `CLEARML_DEFAULT_OUTPUT_URI` or
  `sdk.development.default_output_uri` in `clearml.conf`.

To disable checkpoint uploads while retaining ClearML metric logging, pass `output_uri=False` to `ClearMLLogger` in
`utils.py` and use `output_uri=None` for `ClearMLSaver` in `trainer.py`.

## Components

### Models: rope_transformer
#### rope_transformer
# RoPE Transformer model

It is a decoder-only Transformer with Rotary Position Embeddings (RoPE): given a sequence of token IDs, it predicts the next token at every position.

## What goes in / out

- **Input**:
  - `input_ids` with shape `(batch, seq_len)`
  - `attention_mask` (optional) with shape `(batch, seq_len)`
- **Output**: logits with shape `(batch, seq_len, vocab_size)`

The trainer uses these logits with cross-entropy loss.

## What the model is made of

The model is small and standard:

1. **Embedding**
   Converts token IDs into vectors.
2. **Rotary Positional Encoding (RoPE)**
   Applies relative position rotation to query and key vectors.
3. **Transformer blocks**
   Repeated attention + feedforward layers.
4. **Final projection**
   Maps hidden states back to vocabulary logits.

## Practical notes

### Padding
Padding ID is `0`. The model ignores padded positions in attention.

### Sequence length
`max_seq_len` represents the precomputed cache size for RoPE. Inputs longer than this will fall back to slower on-the-fly calculations.

### Hidden size and heads
`hidden_size` must be divisible by `num_heads`.

### Rotary positional encoding
`head_dim` (which is `hidden_size // num_heads`) must be even because RoPE rotates pairs of dimensions.

## Config knobs

### `hidden_size`
Size of the token embeddings and hidden states.

Bigger values usually give more capacity, but cost more memory and time.

### `num_layers`
How many Transformer blocks to stack.

More layers = more capacity, slower training.

### `num_heads`
How many attention heads to use.

Choose a value that divides `hidden_size` cleanly.

### `feedforward_dim`
Size of the feedforward layer inside each block.

A common choice is `2x` to `4x` of `hidden_size`.

### `dropout`
Dropout rate used in attention and feedforward layers.

### `max_seq_len`
Precomputed cache size for the rotary position embeddings.

## Minimal config example

```yaml
model:
  _target_: models.rope_transformer.RoPETransformerModel
  hidden_size: 128
  num_layers: 4
  num_heads: 4
  feedforward_dim: 256
  dropout: 0.1
  max_seq_len: 64
```

## When to change this file

Edit `models/rope_transformer.py` if you want to:

- make the model wider or deeper
- swap attention or feedforward behavior
- change how padding is handled
- add caching or other sequence features

## Good starting rule

If the dataset is tiny, start with a small model:

- `hidden_size=64`
- `num_layers=2`
- `num_heads=2`
- `feedforward_dim=128`

If training is stable and the model underfits, scale up from there.


### Dataset: arithmetic_addition


### Trainer: decoder_trainer
# Trainer

`Trainer` (registered as `decoder-trainer`) is the default training loop for this project.

It is built on PyTorch-Ignite and already handles the things you usually want on day one:

- training and evaluation loops
- optimizer setup
- mixed precision (AMP: float32, fp16, bf16) via Hugging Face Accelerate
- multi-GPU distributed data parallel (DDP) support via Hugging Face Accelerate
- learning-rate warmup + decay
- checkpoints
- early stopping
- TensorBoard logging

## What it expects

`Trainer` works with a model that returns logits shaped like:

```python
(batch, seq_len, vocab_size)
```

and a batch that looks like:

```python
{
    "input_ids": Tensor,
    "labels": Tensor,
    "attention_mask": Tensor,  # Optional
}
```

It uses cross-entropy loss and ignores label value `-100`.

## What happens during training

For each batch:

1. move batch to device
2. run forward pass
3. compute cross-entropy loss
4. backpropagate
5. clip gradients if `grad_clip_norm` is set
6. step optimizer

At the end of each epoch it also runs evaluation on:

- training data
- validation data, if present

At the end of the training run, it runs a final evaluation on the test data, if present.

## Logging and checkpoints

`Trainer` saves a run directory like this:

```text
outputs/<run_name>/<timestamp>/
```

Inside it you get:

- `config.yaml`
- `output.log`
- `last_*` checkpoint
- `best_*` checkpoint if validation exists
- TensorBoard logs

## Config knobs

### `epochs`
How many full passes over the training data to run.

### `precision`
Mixed precision training mode: `"float32"` (default), `"fp16"`, or `"bf16"`.
Managed automatically via Hugging Face Accelerate.

- `"float32"`: Standard full precision.
- `"fp16"`: Half precision with dynamic loss scaling (recommended for NVIDIA GPUs with Tensor Cores).
- `"bf16"`: Bfloat16 mixed precision without loss scaling (recommended for Ampere+ GPUs or modern CPUs/TPUs).

### `log_every_steps`
How often to print training updates.

### `grad_clip_norm`
If set, gradients are clipped before the optimizer step.

This can help when training becomes unstable.

### `early_stopping_patience`
Number of epochs to wait for validation loss improvement before stopping training early. Set to `null` to disable early stopping.

### `inference_every_epochs`
Run qualitative generation/inference tests on the model every N epochs. Set to `null` to disable.

### `inference_num_samples`
Number of random prompt samples to generate and log during the evaluation inference phase.

### `max_inference_new_tokens`
Maximum number of new tokens to generate per sample.

## Distributed Training (Multi-GPU)

`Trainer` uses [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/index) to scale seamlessly across multiple GPUs without code changes.

Launch distributed training using `accelerate launch`:

```bash
# Launch directly specifying the number of processes:
accelerate launch --num_processes=4 main.py config.yaml

# Or configure your environment interactively once:
accelerate config
# And launch with saved settings:
accelerate launch main.py config.yaml
```

Accelerate automatically handles DDP wrapping, data partitioning (distributed samplers), and metric synchronization across all processes. Checkpoints and evaluation logs are automatically isolated to the main process (rank 0).

For further reading, see:
- [Launching Distributed Code](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
- [Accelerate Command Line Reference](https://huggingface.co/docs/accelerate/package_reference/cli)

## What to tweak first

If you are just getting started, the usual first changes are:

- `epochs`
- `precision`
- `log_every_steps`
- `grad_clip_norm`
- `early_stopping_patience`
- optimizer settings in `config.yaml`

## If you want to customize behavior

Edit `trainer.py` if you want to:

- change how loss is computed
- add gradient accumulation
- add new metrics
- change checkpoint rules
- change validation frequency
- change logging behavior

The main places to look are:

- `_train_step`
- `_eval_step`
- `_attach_metrics`

## Minimal config example

```yaml
trainer:
  epochs: 10
  precision: bf16  # float32, fp16, or bf16
  log_every_steps: 20
  grad_clip_norm: 1.0
```

## Good starting rule

If the run fails or looks unstable, check these first:

- input and label shapes match
- `vocab_size` is large enough for the dataset
- `max_seq_len` is long enough
- learning rate is not too high


### Preprocessor: char_tokenizer
# Char Tokenizer

`CharTokenizer` is a simple character-level tokenizer with a configurable vocabulary.

## Features

- Character-to-ID mapping from a charset preset or literal custom charset.
- Special tokens:
  - `<PAD>` (ID 0)
  - `<BOS>` (ID 1)
  - `<SEP>` (ID 2)
  - `<EOS>` (ID 3)
  - `<UNK>` (ID 4)
- Supports padding, truncation, and PyTorch tensor generation.

## Config knobs

### `_target_`

Must be set to `preprocessors.char_tokenizer.CharTokenizer` in the template config.

### `charset`

Accepts the same presets as the string-reverse dataset, including `@universal`, `@alpha`, `@digits`, and `@alphanumeric`. A literal string defines a custom vocabulary.

## Minimal config example

```yaml
preprocessor:
  _target_: preprocessors.char_tokenizer.CharTokenizer
  charset: "@alphanumeric"
```


## Understanding `utils.py`

`utils.py` contains the bootstrapping and setup logic that links your configurations to executable code. It primarily acts as the file where all the helper functions are located that are being used in `trainer.py`. It handles parsing of dynamic `_target_` paths from `config.yaml`, builds your models and datasets, manages dataset splitting, and attaches standard PyTorch-Ignite event handlers (such as checkpointing, early stopping, learning rate scheduling, and logging).

Since the generated code in `utils.py` belongs to you, you can modify it directly to customize checkpoint rules, adjust early stopping metrics, or add new loggers.

## Customization

### Dataset Configuration

You can configure your data in two ways in `config.yaml`:

#### Option 1: Explicit Splits
Define exactly what goes into each split. Use this for established datasets with fixed splits.
```yaml
data:
  train:
    dataset:
      _target_: ...
    dataloader:
      batch_size: 32
      shuffle: true
  val:
    dataset:
      _target_: ...
    dataloader:
      batch_size: 32
```

#### Option 2: Automatic Splitting
Define a single dataset and split ratios. Trainite will split it randomly (with a fixed seed).
```yaml
data:
  dataset:
    _target_: ...
  test_ratio: 0.1
  val_ratio: 0.1
  dataloader:
    batch_size: 32
```
*Note: In Option 2, the training split defaults to `shuffle: true`, while others default to `shuffle: false`.*

### Adding Configuration Parameters
1. Add the parameter to `config.yaml`.
2. (Optional) Update the corresponding Pydantic model in `config.py` if you want strict validation and IDE autocomplete for custom fields.

## Design Philosophy
The code generated here is yours. There are no hidden abstractions or magic imports. Feel free to refactor or rewrite any part of it to suit your research needs.
