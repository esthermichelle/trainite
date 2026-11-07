"""Run the arithmetic-addition experiments on Modal.

Local usage::

    pip install modal
    modal setup                 # once per machine
    modal run modal_train.py

The entrypoint runs, sequentially, on one L4:

    config_large.yaml          # RoPE baseline
    config_abacus_large.yaml   # Abacus

Outputs are written to the Modal Volume ``arithmetic-addition-outputs``
mounted at ``/root/arithmetic_addition/outputs``. Download them with::

    modal volume get arithmetic-addition-outputs / ./modal_outputs

W&B is offline by default. To enable online logging, create a secret::

    modal secret create wandb WANDB_API_KEY=...

then add ``secrets=[modal.Secret.from_name("wandb")]`` to the function decorator
and remove ``WANDB_MODE`` from ``env``.

Override the GPU or config list from the command line::

    modal run modal_train.py --gpu A100
    modal run modal_train.py --configs config_large.yaml
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import modal

APP_NAME = "arithmetic-addition"
VOLUME_NAME = "arithmetic-addition-outputs"
LOCAL_DIR = Path(__file__).resolve().parent
REMOTE_DIR = "/root/arithmetic_addition"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "pytorch-ignite",
        "accelerate>=1.0.0",
        "omegaconf>=2.3.0",
        "pydantic>=2",
        "PyYAML",
        "tensorboard",
        "wandb>=0.17",
        "clearml",
    )
    .add_local_file(str(LOCAL_DIR / "main.py"), f"{REMOTE_DIR}/main.py")
    .add_local_file(str(LOCAL_DIR / "trainer.py"), f"{REMOTE_DIR}/trainer.py")
    .add_local_file(str(LOCAL_DIR / "utils.py"), f"{REMOTE_DIR}/utils.py")
    .add_local_file(str(LOCAL_DIR / "config.py"), f"{REMOTE_DIR}/config.py")
    .add_local_file(str(LOCAL_DIR / "evaluation.py"), f"{REMOTE_DIR}/evaluation.py")
    .add_local_file(str(LOCAL_DIR / "evaluate.py"), f"{REMOTE_DIR}/evaluate.py")
    .add_local_file(str(LOCAL_DIR / "config.yaml"), f"{REMOTE_DIR}/config.yaml")
    .add_local_file(str(LOCAL_DIR / "config_abacus.yaml"), f"{REMOTE_DIR}/config_abacus.yaml")
    .add_local_file(str(LOCAL_DIR / "config_large.yaml"), f"{REMOTE_DIR}/config_large.yaml")
    .add_local_file(
        str(LOCAL_DIR / "config_abacus_large.yaml"),
        f"{REMOTE_DIR}/config_abacus_large.yaml",
    )
    .add_local_dir(str(LOCAL_DIR / "dataset_impl"), f"{REMOTE_DIR}/dataset_impl")
    .add_local_dir(str(LOCAL_DIR / "models"), f"{REMOTE_DIR}/models")
    .add_local_dir(str(LOCAL_DIR / "preprocessors"), f"{REMOTE_DIR}/preprocessors")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _commit_periodically(stop: threading.Event, interval: float = 600.0) -> None:
    """Commit the volume every ``interval`` seconds until ``stop`` is set."""
    while not stop.wait(interval):
        try:
            volume.commit()
            print("Committed outputs to the Modal Volume.", flush=True)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Volume commit failed: {exc}", flush=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=16384,
    timeout=12 * 60 * 60,
    volumes={f"{REMOTE_DIR}/outputs": volume},
    env={"PYTHONUNBUFFERED": "1", "WANDB_MODE": "offline"},
)
def train_all(configs: list[str]) -> None:
    os.chdir(REMOTE_DIR)
    os.makedirs("outputs", exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("WANDB_MODE", "offline")

    stop = threading.Event()
    committer = threading.Thread(
        target=_commit_periodically, args=(stop,), daemon=True
    )
    committer.start()

    try:
        for config in configs:
            print(f"\n=== {config} ===", flush=True)
            subprocess.run(
                [sys.executable, "main.py", config],
                check=True,
                env=env,
            )
            # Commit after each config so the first run survives if the second fails.
            volume.commit()
    finally:
        stop.set()
        committer.join(timeout=5)
        volume.commit()

    print("\nAll runs complete; outputs committed to the Modal Volume.", flush=True)


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    configs: str = "config_large.yaml,config_abacus_large.yaml",
) -> None:
    run_list = [name.strip() for name in configs.split(",") if name.strip()]
    if not run_list:
        raise ValueError("No configs provided.")

    train_all.with_options(gpu=gpu).remote(run_list)
