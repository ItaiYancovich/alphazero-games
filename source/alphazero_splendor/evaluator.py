"""Batched network evaluation for Splendor: the shared one, wired to the encoder."""

from __future__ import annotations

from alphazero_core.vecevaluator import BatchVecEvaluator as _BatchVecEvaluator

from .features import encode


class BatchEvaluator(_BatchVecEvaluator):
    """:class:`~alphazero_core.vecevaluator.BatchVecEvaluator` with this game's encoder."""

    def __init__(self, net, device: str = "cpu", jit: bool = True):
        super().__init__(net, encode, device=device, jit=jit)
