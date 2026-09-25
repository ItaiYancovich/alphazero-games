"""The v3 Hex network, built out of v2 without changing a single output.

v2 is the strongest thing this project has, and every hour of its training is
in its weights.  Distilling it into a new architecture would spend hours of
compute re-learning what it already knows, and would still start a little
weaker.  Instead every addition here is **function-preserving**: v3 is
constructed from v2 so that, on any position, its move probabilities and its
value are v2's to within floating-point noise.  Training then starts from
exactly v2's strength, and each new part is free to become useful from there.

What is added, and why each starts as a no-op:

* **Extra input planes.**  The stem convolution gains input channels whose
  weights are zero, so whatever the new planes contain, the trunk sees what v2
  saw.  Gradient reaches those weights immediately, so the network starts using
  the planes as soon as they help.
* **Extra residual blocks** at the end of the tower.  Their last batch-norm
  scale is zero, so each adds exactly nothing to a non-negative input.
* **A swap head** for the pie rule: a linear layer on the pooled value
  features giving ``s = P(swap)``.  Its weights start at zero and its bias at
  ``logit(0.1)``, which reproduces
  :class:`alphazero_hex.evaluator.SwapAwareEvaluator` exactly -- the fixed
  prior every v2 agent plays swap games with.
* **An opponent-reply head** (KataGo's auxiliary policy): predicts the move the
  opponent answers with.  Training-only; ``forward`` never computes it.

``forward`` returns ``n*n + 1`` log-probabilities -- the cells, then the swap --
rather than raw logits, so masking and renormalising over whatever is legal
gives the right distribution whether or not the swap is available.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero_core.net import NetConfig, PVNet, count_parameters  # noqa: F401

SWAP_PRIOR = 0.1


@dataclass
class V3Config(NetConfig):
    swap_head: bool = True
    opp_head: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class HexNetV3(PVNet):
    """PVNet plus a swap head and an opponent-reply head."""

    def __init__(self, cfg: V3Config):
        if cfg.wdl or cfg.ply_head or cfg.policy_planes != 1:
            raise ValueError("HexNetV3 is a scalar-value, one-logit-per-cell network")
        super().__init__(cfg)
        h = cfg.head_channels
        self.swap_fc = nn.Linear(2 * h, 1) if cfg.swap_head else None
        if self.swap_fc is not None:
            nn.init.zeros_(self.swap_fc.weight)
            nn.init.constant_(self.swap_fc.bias, math.log(SWAP_PRIOR / (1.0 - SWAP_PRIOR)))
        self.opp_head = nn.Sequential(
            nn.Conv2d(cfg.channels, h, 1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, 1, 1),
        ) if cfg.opp_head else None

    def _heads(self, x: torch.Tensor):
        b = x.shape[0]
        trunk = self.tower(self.stem(x))
        cells = self.policy_head(trunk).reshape(b, -1)
        v = self.value_conv(trunk)
        pooled = torch.cat([v.mean(dim=(2, 3)), v.amax(dim=(2, 3))], dim=1)
        value = torch.tanh(self.value_fc(pooled)).squeeze(1)
        return trunk, cells, pooled, value

    def _policy(self, cells: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        if self.swap_fc is None:
            return cells
        s = self.swap_fc(pooled)
        # log((1 - s) * softmax(cells)) and log(s): one distribution over
        # cells-and-swap, from which masking recovers either sub-case exactly.
        return torch.cat([F.log_softmax(cells, dim=1) + F.logsigmoid(-s),
                          F.logsigmoid(s)], dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, cells, pooled, value = self._heads(x)
        return self._policy(cells, pooled), value

    def forward_train(self, x: torch.Tensor):
        """``(policy, value, occupancy logits, opponent-reply logits)``."""
        trunk, cells, pooled, value = self._heads(x)
        b = x.shape[0]
        aux = self.aux(trunk) if self.aux is not None else None
        opp = self.opp_head(trunk).reshape(b, -1) if self.opp_head is not None else None
        return self._policy(cells, pooled), value, aux, opp


def grow_from(v2: PVNet, extra_planes: int = 0, extra_blocks: int = 0,
              swap_head: bool = True, opp_head: bool = True) -> HexNetV3:
    """A :class:`HexNetV3` that computes exactly what ``v2`` computes."""
    old = v2.cfg
    base = {f.name: getattr(old, f.name) for f in fields(NetConfig)}
    base.update(in_planes=old.in_planes + extra_planes, blocks=old.blocks + extra_blocks)
    cfg = V3Config(**base, swap_head=swap_head, opp_head=opp_head)
    net = HexNetV3(cfg)

    state = {k: v.clone() for k, v in v2.state_dict().items()}
    stem_key = "stem.0.weight"
    w_old = state[stem_key]
    w_new = torch.zeros(w_old.shape[0], cfg.in_planes, *w_old.shape[2:], dtype=w_old.dtype)
    w_new[:, :old.in_planes] = w_old
    state[stem_key] = w_new
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"v2 weights with no place in v3: {unexpected}")
    allowed = ("tower.", "swap_fc.", "opp_head.")
    bad = [k for k in missing if not k.startswith(allowed)]
    if bad:
        raise ValueError(f"v3 weights v2 should have provided: {bad}")
    for i in range(old.blocks, cfg.blocks):
        # New blocks must add nothing: zero the last scale *and* shift.
        nn.init.zeros_(net.tower[i].bn2.weight)
        nn.init.zeros_(net.tower[i].bn2.bias)
    net.eval()
    return net


def save_v3(path, net: HexNetV3, extra: dict | None = None) -> None:
    torch.save({"arch": "hex_v3", "cfg": net.cfg.to_dict(), "state_dict": net.state_dict(),
                "extra": extra or {}}, path)


def load_any(path, map_location="cpu") -> tuple[PVNet, dict]:
    """Load a v3 checkpoint, or any earlier Hex checkpoint as the PVNet it is."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if blob.get("arch") == "hex_v3":
        net: PVNet = HexNetV3(V3Config(**blob["cfg"]))
    else:
        net = PVNet(NetConfig(**blob["cfg"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob.get("extra", {})
