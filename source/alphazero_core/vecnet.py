"""A game-agnostic policy/value network over a feature *vector*.

:mod:`alphazero_core.net` is a convolutional tower, and a convolution is a claim
about the game: that neighbouring cells relate to each other the same way
everywhere on the board.  Hex and Connect Four are like that.  A game of cards,
tokens and tableaux is not -- there is no geometry for a kernel to slide over --
so this is a residual **MLP**, which assumes nothing about the input beyond its
being a fixed-length vector of numbers.

Two heads, as usual, but both generalised past two players:

* **Policy**: ``actions`` logits over a fixed action list.  The game supplies
  the legality mask; the network never needs to know what an action *is*.
* **Value**: ``seats`` outputs rather than one, read *seat-relative* -- output
  ``k`` is the expected result for the player ``k`` turns after whoever is to
  move.  A two-player game uses the first two and its targets are ``+v, -v``;
  a four-player game uses all four.  One set of weights therefore plays every
  table size, which is the only way a network trained on a mixture of two-,
  three- and four-player games can exist at all.

``LayerNorm`` rather than ``BatchNorm``: a search evaluates batches of wildly
varying size, down to one position, and batch statistics over a batch of one are
not statistics.

Nothing in this file mentions Splendor.  A second vector-shaped game needs an
encoder and an action list, not a second network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class VecNetConfig:
    inputs: int = 1          # length of the feature vector
    actions: int = 1         # size of the fixed action space
    seats: int = 4           # value outputs; the largest table the net serves
    width: int = 256
    blocks: int = 4          # residual blocks, two linear layers each
    value_hidden: int = 128
    # Which encoding the weights were trained against.  Stored in the
    # checkpoint, because a network fed the wrong feature set does not fail --
    # it plays badly, which is far harder to notice.  The same reasoning as
    # ``BGNetConfig.features``.
    features: str = "v1"

    def to_dict(self) -> dict:
        return asdict(self)


class VecResBlock(nn.Module):
    """``x -> x + MLP(x)``, normalised, starting life as the identity."""

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.act = nn.ReLU(inplace=True)
        # As in the convolutional tower: a deep residual stack that starts as a
        # random perturbation of its input trains noticeably less stably.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = self.act(self.norm1(self.fc1(x)))
        y = self.norm2(self.fc2(y))
        return self.act(x + y)


class VecNet(nn.Module):
    """Feature vector -> (policy logits, one per action; value, one per seat)."""

    def __init__(self, cfg: VecNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or VecNetConfig()
        w = self.cfg.width
        self.stem = nn.Sequential(
            nn.Linear(self.cfg.inputs, w), nn.LayerNorm(w), nn.ReLU(inplace=True)
        )
        self.tower = nn.Sequential(*[VecResBlock(w) for _ in range(self.cfg.blocks)])
        self.policy_head = nn.Linear(w, self.cfg.actions)
        self.value_head = nn.Sequential(
            nn.Linear(w, self.cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.value_hidden, self.cfg.seats),
        )
        # Start every position at dead level rather than at a random opinion:
        # self-play bootstrapping off a confidently wrong initialisation takes a
        # surprisingly long time to wash out.
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))
        return self.policy_head(h), torch.tanh(self.value_head(h))


def save_checkpoint(path, net: VecNet, extra: dict | None = None) -> None:
    torch.save(
        {"cfg": net.cfg.to_dict(), "state_dict": net.state_dict(), "extra": extra or {}},
        path,
    )


def load_checkpoint(path, map_location="cpu") -> tuple[VecNet, dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    net = VecNet(VecNetConfig(**blob["cfg"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob.get("extra", {})


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
