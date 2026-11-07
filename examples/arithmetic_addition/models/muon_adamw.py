"""Hybrid Muon + AdamW optimizer.

Muon (arXiv 2502.16971 / torch.optim.Muon) orthogonalizes gradients via
Newton–Schulz and only accepts strictly 2D "hidden layer" weight matrices —
it rejects 1D biases/norm weights and does not cover embeddings in practice.
The standard recipe therefore pairs Muon (2D matrices) with AdamW for
everything else (embeddings, biases, layer-norm weights).

This is a drop-in replacement for a plain optimizer in ``config.yaml``::

    optimizer:
      _target_: models.muon_adamw.MuonAdamW
      lr: 0.001            # AdamW lr, used for embeddings/biases/norms
      muon_lr: 0.02        # Muon lr, used for 2D matrices
      muon_weight_decay: 0.01
      weight_decay: 0.01
      adjust_lr_fn: original   # or "match_rms_adamw" (Muon lr = lr*0.2*sqrt(max(A,B)))

The two sub-optimizers share the same parameter-group dicts with the parent
(``self.param_groups``), so Ignite's LR scheduler mutates each group's own
learning rate and the correct per-group scale is preserved during decay.
Momentum buffers etc. live inside the sub-optimizers only.
"""

import torch


class MuonAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        muon_lr: float = 0.02,
        muon_weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str = "original",
        adambetas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        params = list(params)
        matrices = [p for p in params if p.ndim == 2]
        other = [p for p in params if p.ndim != 2]

        super().__init__(
            [
                {
                    "params": matrices,
                    "lr": muon_lr,
                    "weight_decay": muon_weight_decay,
                    "momentum": momentum,
                    "nesterov": nesterov,
                    "ns_steps": ns_steps,
                    "adjust_lr_fn": adjust_lr_fn,
                },
                {
                    "params": other,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "betas": adambetas,
                },
            ],
            {},
        )

        self._sub_optimizers = []
        if matrices:
            self._sub_optimizers.append(
                torch.optim.Muon(
                    matrices,
                    lr=muon_lr,
                    weight_decay=muon_weight_decay,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_steps=ns_steps,
                    adjust_lr_fn=adjust_lr_fn,
                )
            )
        if other:
            self._sub_optimizers.append(
                torch.optim.AdamW(other, lr=lr, weight_decay=weight_decay, betas=adambetas)
            )

    @property
    def muon(self):
        return self._sub_optimizers[0]

    @property
    def adamw(self):
        return self._sub_optimizers[-1]

    def step(self, closure=None):
        loss = None
        for opt in self._sub_optimizers:
            out = opt.step(closure)
            if out is not None:
                loss = out
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._sub_optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"sub_optimizers": [opt.state_dict() for opt in self._sub_optimizers]}

    def load_state_dict(self, state_dict):
        for opt, opt_state in zip(self._sub_optimizers, state_dict["sub_optimizers"]):
            opt.load_state_dict(opt_state)
