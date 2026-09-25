"""Size-agnostic policy/value residual network.

Shared by both games: Hex reads it as ``alphazero_hex.net`` and Connect Four as
``alphazero_c4.net``, and a checkpoint stores only its ``NetConfig`` and a state
dict, so which game a file belongs to is a matter of where it was written, not
of what class saved it.

The network is fully convolutional and the value head pools globally, so the
*same weights* work on any board size -- and on non-square boards, which is what
lets the identical tower serve an 11x11 rhombus and a 6x7 grid.  That is what makes the size curriculum
(train on 5x5, transfer to 7x7, 9x9, then 11x11) possible: convolutional Hex
patterns -- bridges, ladders, edge templates -- are local and transfer almost
verbatim to bigger boards.

Two additions on top of the plain AlphaZero tower, both aimed at the same
weakness: a 3x3 convolution moves information one cell per layer, but Hex is
decided by whole-board connectivity.

* **Global-pooling blocks** (``gp_every``).  Every k-th residual block pools its
  first convolution over the whole board and broadcasts the result back as a
  per-channel bias, so any cell can see any other cell immediately.  The pool
  costs O(C^2) per block instead of the O(C^2 * H * W) of a convolution --
  about 2% of trunk cost at 11x11.
* **Auxiliary occupancy head** (``aux_head``).  Predicts, per cell, who owns it
  when the game ends.  A position then supplies n*n labels of supervision
  instead of the single win/loss scalar, which is how a bigger trunk can be
  fed without generating proportionally more games.

``NetConfig`` defaults reproduce the original 3-plane, no-pooling, no-auxiliary
network, so checkpoints saved before any of this still load unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# The original own/opponent/ones encoding, and the default every checkpoint
# saved before the Hex feature planes existed was trained with.  Defined here
# rather than imported so the core does not depend on either game's features.
LEGACY_PLANES = 3

IN_PLANES = LEGACY_PLANES  # backwards-compatible alias


@dataclass
class NetConfig:
    channels: int = 64
    blocks: int = 6
    head_channels: int = 32
    value_hidden: int = 64
    in_planes: int = LEGACY_PLANES  # 3 = board only, 10 = features.PLANES
    gp_every: int = 0  # 0 = none, k = every k-th block is a global-pooling block
    gp_channels: int = 32
    aux_head: bool = False  # per-cell final-occupancy auxiliary target
    # Three-way value head -- P(win), P(draw), P(loss) for the side to move --
    # instead of a single tanh scalar.  A scalar cannot tell "dead drawn" from
    # "sharp and even": both are 0, and regressing towards 0 is the same
    # gradient for each.  In a game where half the results are draws that is
    # most of the value signal thrown away.  ``forward`` still reports the
    # scalar P(win) - P(loss), so search, the GUI and the JIT trace are
    # unchanged; only the loss sees the three logits.  False reproduces the
    # scalar head exactly, which is what every checkpoint before this was
    # trained with.
    wdl: bool = False
    # Predict how many plies remain until the game ends, as a fraction of the
    # board.  In a game that draws six times in ten, ``z = 0`` is most of the
    # value signal and it carries almost no gradient: a dead-drawn position and
    # a sharp one forty plies from resolution are the same label.  Distance to
    # termination separates them, costs one scalar, and is known exactly for
    # every stored position without any extra search.
    ply_head: bool = False
    # Logits the policy head emits per cell.  Every game here but one answers
    # "where does the next mark go", so one logit per cell *is* the action
    # space and this stays 1 -- which is what every checkpoint before this was
    # trained with.  Intransitive's move is a piece plus one of eight
    # directions, so it asks for 8 and reads the flattened head as
    # ``direction * ncells + cell``: the head stays convolutional, the trunk is
    # untouched, and a move is still a local question about a square and its
    # neighbours.
    policy_planes: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        # Start as the identity: with ten-plus blocks a residual tower that
        # begins as a random perturbation of its input trains noticeably less
        # stably from scratch.
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class GlobalPoolBlock(nn.Module):
    """Residual block whose second convolution carries a whole-board bias."""

    def __init__(self, c: int, gp: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.gp_conv = nn.Conv2d(c, gp, 1, bias=False)
        self.gp_bn = nn.BatchNorm2d(gp)
        self.gp_fc = nn.Linear(2 * gp, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        g = F.relu(self.gp_bn(self.gp_conv(y)))
        # mean and max over the board: "how much of this is about anywhere"
        g = torch.cat([g.mean(dim=(2, 3)), g.amax(dim=(2, 3))], dim=1)
        y = y + self.gp_fc(g)[:, :, None, None]
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PVNet(nn.Module):
    """Outputs (policy logits, one per cell, and a value in [-1, 1])."""

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        c = self.cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(self.cfg.in_planes, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        gp_every = self.cfg.gp_every
        blocks = [
            GlobalPoolBlock(c, self.cfg.gp_channels)
            if gp_every and (i + 1) % gp_every == 0
            else ResBlock(c)
            for i in range(self.cfg.blocks)
        ]
        self.tower = nn.Sequential(*blocks)

        h = self.cfg.head_channels
        self.policy_head = nn.Sequential(
            nn.Conv2d(c, h, 1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            # ``policy_planes`` logits per cell -> board-size agnostic.  One is
            # the games whose move is a cell; eight is Intransitive, where a
            # move is a piece and a direction and channel ``d`` over square
            # ``s`` means "the piece on s goes direction d".
            nn.Conv2d(h, self.cfg.policy_planes, 1),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, h, 1, bias=False),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(2 * h, self.cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.value_hidden, 3 if self.cfg.wdl else 1),
        )
        # 3 classes per cell: empty / side-to-move / opponent at game end.
        self.aux = nn.Conv2d(c, 3, 1) if self.cfg.aux_head else None
        # Remaining plies, as a fraction of the board, from the pooled trunk.
        self.ply = nn.Sequential(
            nn.Linear(2 * h, self.cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.value_hidden, 1),
        ) if self.cfg.ply_head else None

    def _trunk(self, x: torch.Tensor):
        """``(policy, scalar value, wdl logits, plies-left, trunk features)``.

        The last two are ``None`` unless their head is configured.
        """
        b = x.shape[0]
        x = self.tower(self.stem(x))
        policy = self.policy_head(x).reshape(b, -1)
        v = self.value_conv(x)
        # global average + global max pooling keeps the head size-independent
        pooled = torch.cat([v.mean(dim=(2, 3)), v.amax(dim=(2, 3))], dim=1)
        # Shares the pooled features with the value head deliberately: the point
        # is to make those features carry how far the game still has to run, not
        # to learn it off to one side.
        plies = self.ply(pooled).squeeze(1) if self.ply is not None else None
        v = self.value_fc(pooled)
        if self.cfg.wdl:
            # Logits are ordered (win, draw, loss) from the side to move's
            # point of view.  The scalar every caller already expects is the
            # expected result under that distribution, P(win) - P(loss), which
            # lands in [-1, 1] like the tanh it replaces -- so a WDL net drops
            # into the search and the arena with nothing else changed.
            wdl = v
            p = torch.softmax(v, dim=1)
            value = p[:, 0] - p[:, 2]
        else:
            wdl = None
            value = torch.tanh(v).squeeze(1)
        return policy, value, wdl, plies, x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Deliberately a 2-tuple: search, the GUI and the JIT trace all depend
        # on this signature, and none of them wants the auxiliary heads.
        policy, value, _, _, _ = self._trunk(x)
        return policy, value

    def forward_with_aux(self, x: torch.Tensor):
        """Training-only forward: ``(policy, value, occupancy logits or None)``."""
        policy, value, _, _, trunk = self._trunk(x)
        return policy, value, (self.aux(trunk) if self.aux is not None else None)

    def forward_full(self, x: torch.Tensor):
        """``(policy, value, occupancy logits, wdl logits, plies-left)``.

        Everything the loss can use; the last three are ``None`` when their head
        is not configured.  Separate from :meth:`forward_with_aux` rather than
        replacing it, because four other games call that method and none of them
        has these heads to ask about.
        """
        policy, value, wdl, plies, trunk = self._trunk(x)
        return (policy, value,
                (self.aux(trunk) if self.aux is not None else None), wdl, plies)


def save_checkpoint(path, net: PVNet, extra: dict | None = None) -> None:
    torch.save(
        {"cfg": net.cfg.to_dict(), "state_dict": net.state_dict(), "extra": extra or {}},
        path,
    )


def load_checkpoint(path, map_location="cpu") -> tuple[PVNet, dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    net = PVNet(NetConfig(**blob["cfg"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob.get("extra", {})


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


# The name every Hex checkpoint, script and test was written against.  Nothing
# in the class was ever Hex-specific, so this is an alias rather than a subclass.
HexNet = PVNet
