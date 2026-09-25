"""The backgammon value network.

A plain multi-layer perceptron, and deliberately not the residual convolutional
tower the other two games share.  A convolution assumes that neighbouring cells
are related in the same way everywhere on the board; backgammon has no such
geometry -- what matters is how many checkers sit on a point, how far it is from
home, and whether the opponent can reach it, which the encoding in
:mod:`alphazero_bg.features` already spells out unit by unit.  A small dense net
over those features is what TD-Gammon used, and it is still the right shape.

Only a value head.  There is no policy head because there is nothing for it to
index: a backgammon move is a sequence of hops chosen from a list that changes
with every roll, so there is no fixed set of actions to put probabilities on.
The agent instead evaluates the position each candidate move leads to and takes
the best -- afterstate evaluation, which is what makes a value-only network
enough here.

The output is *equity* for the side to move, in -1..+1, where a plain win is
1/3, a gammon 2/3 and a backgammon 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from .features import INPUTS, size_of
from .outcomes import OUTPUTS, equity


@dataclass
class BGNetConfig:
    inputs: int = INPUTS
    hidden: tuple[int, ...] = (256, 128, 64)
    # Which encoding the weights were trained against.  Stored in the
    # checkpoint, because a network fed the wrong feature set does not fail --
    # it just plays badly, which is far harder to notice.
    features: str = "basic"
    # 1 = a single equity scalar; 5 = the outcome probabilities in
    # :mod:`alphazero_bg.outcomes`, which is what a backgammon result actually
    # is.  Stored in the checkpoint, because a five-output network read as a
    # one-output one would silently play on its P(win) alone.
    outputs: int = 1
    # TD-Gammon got a long way on a single hidden layer of 40-80 units; three
    # narrowing layers cost almost nothing on a CPU (a forward pass here is
    # ~70k multiply-adds) and give the net somewhere to put the interactions
    # between race, blot exposure and board strength.

    def __post_init__(self):
        expected = size_of(self.features)
        if self.inputs != expected:
            # The pair has to agree; the feature set is the thing that decides.
            object.__setattr__(self, "inputs", expected)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["hidden"] = list(self.hidden)
        return out


class BGNet(nn.Module):
    """Position -> equity for the side to move, in [-1, 1]."""

    def __init__(self, cfg: BGNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or BGNetConfig()
        layers: list[nn.Module] = []
        width = self.cfg.inputs
        for size in self.cfg.hidden:
            layers += [nn.Linear(width, size), nn.ReLU(inplace=True)]
            width = size
        layers.append(nn.Linear(width, self.cfg.outputs))
        self.stack = nn.Sequential(*layers)
        # Start every position at dead level rather than at a random opinion:
        # with self-play bootstrapping off its own values, a confident random
        # initialisation takes a surprisingly long time to wash out.
        nn.init.zeros_(self.stack[-1].weight)
        nn.init.zeros_(self.stack[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Equity in [-1, 1] for one output; five probabilities for five.

        The heads differ in more than width.  A single output is squashed with
        tanh because equity is signed; the five are sigmoids because each is a
        probability, and they are deliberately *not* a softmax -- they are five
        yes/no questions about the same game, and winning a backgammon is also
        winning a gammon.
        """
        raw = self.stack(x)
        if self.cfg.outputs == 1:
            return torch.tanh(raw).squeeze(-1)
        return torch.sigmoid(raw)

    def equities(self, x: torch.Tensor) -> torch.Tensor:
        """Equity in [-1, 1], whichever head this network has."""
        out = self.forward(x)
        if self.cfg.outputs == 1:
            return out
        points = (2.0 * out[..., 0] - 1.0 + out[..., 1] + out[..., 2]
                  - out[..., 3] - out[..., 4])
        return points / 3.0


def save_checkpoint(path, net: BGNet, extra: dict | None = None) -> None:
    torch.save(
        {"cfg": net.cfg.to_dict(), "state_dict": net.state_dict(), "extra": extra or {}},
        path,
    )


def load_checkpoint(path, map_location="cpu") -> tuple[BGNet, dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(blob["cfg"])
    cfg["hidden"] = tuple(cfg["hidden"])
    cfg.setdefault("features", "basic")   # checkpoints from before there was a choice
    cfg.setdefault("outputs", 1)          # ...and from before there were five
    net = BGNet(BGNetConfig(**cfg))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob.get("extra", {})


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
