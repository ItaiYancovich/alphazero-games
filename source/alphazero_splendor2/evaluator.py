"""Batched network evaluation for the v2 game."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .features import encode
from .game import Game
from .net import SplendorNet


class BatchEvaluator:
    """Runs the net over a list of positions and returns ``(priors, values)``.

    Priors come back masked to the legal actions and renormalised, so the search
    never has to think about the illegal ones.  Values come back as the network
    produced them: seat-relative, output ``k`` for the seat ``k`` turns after
    the mover.
    """

    def __init__(self, net: SplendorNet, device: str = "cpu", jit: bool = True):
        self.net = net.to(device).eval()
        self.device = device
        self.cfg = net.cfg
        self.module = self.net
        if jit and device == "cpu":
            try:
                traced = torch.jit.trace(self.net, torch.zeros(8, net.cfg.inputs))
                self.module = torch.jit.optimize_for_inference(traced)
            except Exception:
                self.module = self.net  # fall back to eager rather than fail

    @torch.inference_mode()
    def evaluate(self, states: Sequence[Game]) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(encode(states))
        if self.device != "cpu":
            x = x.to(self.device)
        logits, values = self.module(x)
        mask = np.stack([s.legal_mask() for s in states])
        logits = logits.masked_fill(torch.from_numpy(~mask).to(self.device), -1e9)
        priors = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        dead = ~mask.any(axis=1)
        if dead.any():
            priors[dead] = 0.0
        return priors, values.cpu().numpy().astype(np.float32)
