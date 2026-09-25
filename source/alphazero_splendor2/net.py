"""The v2 network: the shared residual MLP, with one Splendor-shaped extra.

The tower, the policy head and the seat-relative value head are
:class:`alphazero_core.vecnet.VecNet` unchanged -- there is nothing about a
Splendor position that wants a different architecture from any other game
described by a feature vector.

What is added is an **auxiliary head**, and the reason is the shape of the
game's learning problem rather than a general improvement.  A Splendor game runs
sixty to a hundred plies and ends in a single bit: somebody got to fifteen
first.  That is a very thin signal to hang a value function on, and the first
thirty plies of a game have almost no gradient in it -- every reasonable opening
leads to roughly a coin flip, so the value head learns "0.0" and stops.
Meanwhile the position is *full* of things that are predictable and that a
player has to be able to estimate: how many points each seat will finish with,
and how long the game has left to run.  Fitting those as well costs one small
linear layer and gives every position a dense target, which is the standard
answer (KataGo's ownership and score heads, and every auxiliary-task result
before it) to a reward that only arrives at the end.

The auxiliary outputs are used **only in training**.  ``forward`` returns the
same ``(policy, value)`` pair the shared evaluator and the JIT trace expect, and
``forward_all`` is what the trainer calls.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from alphazero_core.vecnet import (VecNet, VecNetConfig,  # noqa: F401
                                   count_parameters)

from .features import INPUTS, NAME as FEATURES
from .game import MAX_SEATS, NACTIONS

# Final prestige is capped for scaling; a runaway seat rarely passes this.
POINT_SCALE = 22.0
PLIES_SCALE = 120.0


class SplendorNet(VecNet):
    """:class:`VecNet` plus a head that predicts how the game will finish.

    ``aux`` is ``MAX_SEATS`` final prestige totals, seat-relative like the value
    head, followed by one number for how many plies the game has left.
    """

    AUX_OUTPUTS = MAX_SEATS + 1

    def __init__(self, cfg: VecNetConfig | None = None):
        super().__init__(cfg)
        self.aux_head = nn.Sequential(
            nn.Linear(self.cfg.width, self.cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.value_hidden, self.AUX_OUTPUTS),
        )
        nn.init.zeros_(self.aux_head[-1].weight)
        nn.init.zeros_(self.aux_head[-1].bias)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.tower(self.stem(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), torch.tanh(self.value_head(h))

    def forward_all(self, x: torch.Tensor):
        """``(policy, value, aux)`` -- the trainer's view of the same pass."""
        h = self.trunk(x)
        return (self.policy_head(h), torch.tanh(self.value_head(h)),
                self.aux_head(h))


def default_config(width: int = 320, blocks: int = 6) -> VecNetConfig:
    """A fresh network's shape.

    Wider and deeper than v1's 256x4, because the profile said the network was
    two per cent of self-play: the tower was never what the run could not
    afford, and a game with this much arithmetic in its positions has plenty for
    the capacity to do.
    """
    return VecNetConfig(inputs=INPUTS, actions=NACTIONS, seats=MAX_SEATS,
                        width=width, blocks=blocks, features=FEATURES)


def new_net(width: int = 320, blocks: int = 6) -> SplendorNet:
    return SplendorNet(default_config(width, blocks))


def save_checkpoint(path, net: SplendorNet, extra: dict | None = None) -> None:
    torch.save({"cfg": net.cfg.to_dict(), "state_dict": net.state_dict(),
                "extra": extra or {}}, path)


def load_checkpoint(path, map_location="cpu") -> tuple[SplendorNet, dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = VecNetConfig(**blob["cfg"])
    if cfg.features != FEATURES:
        raise ValueError(
            f"checkpoint was trained on feature set {cfg.features!r}, not "
            f"{FEATURES!r}: it would play badly rather than fail, so it is "
            f"refused here instead")
    net = SplendorNet(cfg)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob.get("extra", {})
