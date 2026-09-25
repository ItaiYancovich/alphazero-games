"""Splendor's network: the shared vector net, sized for this game.

Nothing here is Splendor-specific except the two numbers -- how long the feature
vector is and how many actions there are.  The architecture, the checkpoint
format and the loaders are :mod:`alphazero_core.vecnet`, exactly as
``alphazero_hex.net`` and ``alphazero_c4.net`` are both thin views of the shared
convolutional tower.
"""

from __future__ import annotations

from alphazero_core.vecnet import (VecNet, VecNetConfig,  # noqa: F401
                                   count_parameters, load_checkpoint,
                                   save_checkpoint)

from .features import INPUTS, NAME as FEATURES
from .splendor_game import MAX_SEATS, NACTIONS


def default_config(width: int = 256, blocks: int = 4) -> VecNetConfig:
    """A fresh network's shape.  The defaults are ~500k parameters on a CPU."""
    return VecNetConfig(inputs=INPUTS, actions=NACTIONS, seats=MAX_SEATS,
                        width=width, blocks=blocks, features=FEATURES)


def new_net(width: int = 256, blocks: int = 4) -> VecNet:
    return VecNet(default_config(width, blocks))
