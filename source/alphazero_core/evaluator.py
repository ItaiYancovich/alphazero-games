"""Batched network evaluation, for any game.

The only game-specific thing an evaluator needs is how to turn a batch of
canonical boards into input planes, so that is a callable passed in rather than
an import.  Everything else -- the JIT trace, the legality mask, the softmax --
is identical for Hex and Connect Four.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from .net import LEGACY_PLANES, PVNet
from .state import GameState

Encoder = Callable[[np.ndarray, int], np.ndarray]


class BatchEvaluator:
    """Runs the net on a list of positions and returns (priors, values).

    Priors come back in the *canonical* frame (the frame the network sees), and
    are already masked to legal cells and renormalised.

    ``encode`` maps ``(N, rows, cols) uint8`` canonical boards and a plane count
    to an ``(N, planes, rows, cols) float32`` tensor; ``example_shape`` is the
    board shape to trace with.
    """

    def __init__(
        self,
        net: PVNet,
        encode: Encoder,
        device: str = "cpu",
        jit: bool = True,
        example_shape: tuple[int, int] = (11, 11),
    ):
        dev_lower = device.lower()
        torch_dev = "cpu" if dev_lower in ("gpu", "ov-gpu", "auto") else device
        self.net = net.to(torch_dev).eval()
        self.encode = encode
        self.device = torch_dev
        self.in_planes = getattr(net.cfg, "in_planes", LEGACY_PLANES)
        self.module = self.net
        self._ov = None

        if dev_lower in ("gpu", "ov-gpu", "auto"):
            try:
                from .ov_evaluator import OVEvaluator
                self._ov = OVEvaluator(net, encode, device="GPU", board_shape=example_shape)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("OpenVINO GPU initialization failed (%s); falling back to CPU.", exc)

        if self._ov is None and jit and torch_dev == "cpu":
            # Folding BatchNorm into the convolutions roughly halves inference
            # cost, which is the whole self-play budget on a CPU.  Traced
            # convnets stay shape-polymorphic, so one trace serves every board
            # size and batch size in the curriculum.
            try:
                example = torch.zeros(8, self.in_planes, *example_shape)
                traced = torch.jit.trace(self.net, example)
                self.module = torch.jit.optimize_for_inference(traced)
            except Exception:
                self.module = self.net  # fall back to eager rather than fail

    @torch.inference_mode()
    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        if self._ov is not None:
            try:
                return self._ov.evaluate(states)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("OpenVINO GPU evaluation failed (%s); this batch ran on CPU.", exc)
        planes, context = self._inputs(states)
        logits, values = self.module(torch.from_numpy(planes).to(self.device))
        return self._outputs(context, logits, values)

    @property
    def background(self) -> bool:
        """Can the network answer while the search goes on?

        Only the web build's can (its network runs in other workers); a search
        with such an engine keeps a batch in flight -- see ``run_search``.
        """
        return self._ov is None and bool(getattr(self.module, "background", False))

    def evaluate_start(self, states: list[GameState]):
        """Hand ``states`` to the network; returns a function that waits for
        the answer, as :meth:`evaluate` gives it.  Needs :attr:`background`."""
        planes, context = self._inputs(states)
        wait = self.module.start(torch.from_numpy(planes))
        return lambda: self._outputs(context, *wait())

    def _inputs(self, states: list[GameState]):
        """The network's input planes for ``states``, and what :meth:`_outputs`
        needs to turn its answer into priors."""
        # One batched encode rather than one per position: the feature planes
        # are cheap in bulk and dominated by numpy call overhead one at a time.
        boards = np.stack([s.canonical_board() for s in states])
        return self.encode(boards, self.in_planes), (boards, states)

    def _outputs(self, context, logits, values) -> tuple[np.ndarray, np.ndarray]:
        boards, states = context
        # Legality mask, in the same canonical frame the boards are already in.
        # An empty cell is playable in Hex; Connect Four masks further (only the
        # lowest empty cell of a column can be played), which is why the mask is
        # intersected with what the states themselves report.
        mask = (boards == 0).reshape(len(states), -1)
        for i, state in enumerate(states):
            playable = getattr(state, "canonical_policy_mask", None)
            if playable is not None:
                mask[i] &= playable()
        logits = logits.masked_fill(torch.from_numpy(~mask).to(self.device), -1e9)
        priors = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        return priors, values.cpu().numpy().astype(np.float32)
