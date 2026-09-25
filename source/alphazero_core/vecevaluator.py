"""Batched evaluation for the vector-shaped games.

The counterpart of :mod:`alphazero_core.evaluator`, and the same idea: the only
game-specific thing an evaluator needs is how to turn positions into network
input, so that is a callable passed in rather than an import.  Everything else
-- the JIT trace, the legality mask, the softmax -- is the same for any game.

Priors come back masked to the legal actions and renormalised, so a search never
has to think about the illegal ones.  Values come back as the network produced
them: **seat-relative**, output ``k`` for the seat ``k`` turns after the mover.
Turning that into a seat-indexed vector is
:func:`alphazero_core.vecstate.rotate_to_absolute`, which the search does once
per backup.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from .vecnet import VecNet
from .vecstate import VecState

# positions -> (N, inputs) float32
Encoder = Callable[[Sequence[VecState]], np.ndarray]


class BatchVecEvaluator:
    """Runs the net on a list of positions and returns ``(priors, values)``."""

    def __init__(self, net: VecNet, encode: Encoder, device: str = "cpu",
                 jit: bool = True):
        dev_lower = device.lower()
        torch_dev = "cpu" if dev_lower in ("gpu", "ov-gpu", "auto") else device
        self.net = net.to(torch_dev).eval()
        self.encode = encode
        self.device = torch_dev
        self.cfg = net.cfg
        self.module = self.net
        self._ov = None

        if dev_lower in ("gpu", "ov-gpu", "auto"):
            try:
                from .ov_evaluator import VecOVEvaluator
                self._ov = VecOVEvaluator(net, encode, device="GPU")
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("OpenVINO GPU initialization failed (%s); falling back to CPU.", exc)

        if self._ov is None and jit and torch_dev == "cpu":
            try:
                example = torch.zeros(8, net.cfg.inputs)
                traced = torch.jit.trace(self.net, example)
                self.module = torch.jit.optimize_for_inference(traced)
            except Exception:
                self.module = self.net  # fall back to eager rather than fail

    @torch.inference_mode()
    def evaluate(self, states: Sequence[VecState]) -> tuple[np.ndarray, np.ndarray]:
        if self._ov is not None:
            try:
                return self._ov.evaluate(states)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("OpenVINO GPU evaluation failed (%s); this batch ran on CPU.", exc)
        x = torch.from_numpy(self.encode(states)).to(self.device)
        logits, values = self.module(x)
        mask = np.stack([s.legal_mask() for s in states])
        logits = logits.masked_fill(torch.from_numpy(~mask).to(self.device), -1e9)
        priors = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        # A position with no legal action at all would softmax to uniform over
        # -1e9, which is uniform over nonsense.  The search never asks about
        # one -- terminal positions are scored, not evaluated -- but a NaN here
        # would be silent, so it is spelled out.
        dead = ~mask.any(axis=1)
        if dead.any():
            priors[dead] = 0.0
        return priors, values.cpu().numpy().astype(np.float32)
