"""The v3 network: tokens where they buy structure, a dense trunk where they don't.

v2 is a flat vector into a residual MLP with seventy-two fixed output logits.
That cannot express either thing Splendor says about its own state -- that the
four face-up cards of a tier are a *set*, and that the five colours are
interchangeable -- so the twenty-four cards, six colours and five nobles are
**tokens** here, all cards through one shared encoder, and the logit for "buy
this card" is a **pointer**: a dot product between that card's token and a query
the trunk produces.  A two-point card costing three blue is learned once instead
of twelve times, and "take two blue" and "take two red" come out of the same
weights applied to different tokens.

The shape of the thing is set by a measurement rather than by taste.  A pure
token transformer -- three attention blocks over thirty-six tokens at width 64 --
costs 200 microseconds a position on one CPU thread against v2's 28, because
thirty-six-row matmuls are almost all dispatch overhead and get about a quarter
of the arithmetic throughput that a 320-wide dense layer gets.  This run is CPU
only and evaluates a couple of hundred positions a move, so a 7x forward pass is
not a trade worth making for structure that data augmentation can also supply.

So: **one** attention block over the tokens, which is where card and colour
identity gets shared; then pool to a flat vector and spend the rest of the
budget in wide dense layers, which is where a CPU is fast; then read the policy
back off the tokens with pointer queries built from the trunk.  Hard equivariance
in the policy head, learned equivariance in the trunk, and the exact colour and
slot symmetries applied as augmentation on top (:mod:`~.features`).

**Three heads, because one bit is a thin training signal.**  A game runs seventy
plies and ends in one bit, and v2's value loss sat between 0.296 and 0.301 for
fifty straight iterations.  That number wants reading carefully: it is a mean
*per active seat* against targets of plus or minus one, so a head predicting
nothing would score 1.0 and v2's was explaining about seventy per cent of the
variance.  The complaint is not that it was bad, it is that it had **stopped
moving**, and so had the run's strength.  Beside the value head, an auxiliary
head predicts
each seat's final prestige, the plies remaining, and each seat's **turns to
fifteen** as :mod:`alphazero_splendor3.oracle` computes it.  That last is close
to a sufficient statistic for who wins, and fitting it makes the trunk do the
planning arithmetic instead of being handed it.

Auxiliary outputs are training-only: ``forward`` returns the ``(policy, value)``
the search wants, ``forward_all`` is the trainer's view of the same pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .features import (CARD_TOKENS, CARD_W, COLOUR_TOKENS, COLOUR_W, GLOBAL_W,
                       INPUTS, NAME as FEATURES, NOBLE_TOKENS, NOBLE_W,
                       O_CARDS, O_COLOUR, O_NOBLES)
from .game import (BOARD_SLOTS, GEMS, MAX_RESERVED, MAX_SEATS, NACTIONS, TIERS,
                   TAKE2D, TAKE3)

POINT_SCALE = 22.0
PLIES_SCALE = 120.0
TURNS_SCALE = 30.0

N_TOKENS = 1 + COLOUR_TOKENS + CARD_TOKENS + NOBLE_TOKENS       # 36
# The nine pointer queries the trunk produces, in this order.
Q_TRIO, Q_PAIR, Q_TAKE1, Q_TAKE2S, Q_DISCARD = 0, 1, 2, 3, 4
Q_BUY, Q_RESERVE, Q_BUYRES, Q_NOBLE = 5, 6, 7, 8
N_QUERIES = 9


@dataclass
class NetConfig:
    inputs: int = INPUTS
    actions: int = NACTIONS
    seats: int = MAX_SEATS
    token_width: int = 32
    token_blocks: int = 1
    heads: int = 4
    width: int = 256
    blocks: int = 3
    value_hidden: int = 128
    features: str = FEATURES

    def to_dict(self) -> dict:
        return asdict(self)


class TokenBlock(nn.Module):
    """Pre-norm attention plus a feed-forward, over the token axis."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                 nn.Linear(width * 2, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.n2(x))


class DenseBlock(nn.Module):
    """A residual pair of wide linears: what a CPU is actually fast at."""

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(self.norm(x)))
        return x + self.fc2(h)


class SplendorNet(nn.Module):
    """Token encoder, dense trunk, pointer policy, value and auxiliary heads."""

    AUX_OUTPUTS = 2 * MAX_SEATS + 1        # final points, turns to win, plies

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        d = self.cfg.token_width
        w = self.cfg.width

        self.proj_global = nn.Linear(GLOBAL_W, d)
        self.proj_colour = nn.Linear(COLOUR_W, d)
        self.proj_card = nn.Linear(CARD_W, d)
        self.proj_noble = nn.Linear(NOBLE_W, d)
        # One embedding per *kind* of token, not per token: the twelve board
        # slots deliberately share theirs, which is the slot symmetry made
        # structural rather than merely trained.
        self.kind_embed = nn.Parameter(torch.zeros(4, d))
        nn.init.normal_(self.kind_embed, std=0.02)

        self.token_blocks = nn.ModuleList(
            TokenBlock(d, self.cfg.heads) for _ in range(self.cfg.token_blocks))
        self.token_norm = nn.LayerNorm(d)

        # Pooled tokens plus the raw globals, into the dense trunk.
        self.stem = nn.Linear(4 * d + GLOBAL_W, w)
        self.blocks = nn.ModuleList(DenseBlock(w) for _ in range(self.cfg.blocks))
        self.trunk_norm = nn.LayerNorm(w)

        self.queries = nn.Linear(w, N_QUERIES * d)
        self.query_bias = nn.Parameter(torch.zeros(N_QUERIES))
        self.head_global = nn.Linear(w, TIERS + 1)      # reserve from deck, pass

        self.value_head = nn.Sequential(
            nn.Linear(w, self.cfg.value_hidden), nn.ReLU(inplace=True),
            nn.Linear(self.cfg.value_hidden, self.cfg.seats))
        self.aux_head = nn.Sequential(
            nn.Linear(w, self.cfg.value_hidden), nn.ReLU(inplace=True),
            nn.Linear(self.cfg.value_hidden, self.AUX_OUTPUTS))
        nn.init.zeros_(self.aux_head[-1].weight)
        nn.init.zeros_(self.aux_head[-1].bias)

        self.register_buffer("trio_idx",
                             torch.tensor(np.asarray(TAKE3), dtype=torch.long),
                             persistent=False)
        self.register_buffer("pair_idx",
                             torch.tensor(np.asarray(TAKE2D), dtype=torch.long),
                             persistent=False)

    # ------------------------------------------------------------------ trunk
    def encode_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        raw_global = x[:, :GLOBAL_W]
        g = self.proj_global(raw_global).unsqueeze(1) + self.kind_embed[0]
        c = self.proj_colour(
            x[:, O_COLOUR:O_CARDS].view(n, COLOUR_TOKENS, COLOUR_W)
        ) + self.kind_embed[1]
        k = self.proj_card(
            x[:, O_CARDS:O_NOBLES].view(n, CARD_TOKENS, CARD_W)
        ) + self.kind_embed[2]
        b = self.proj_noble(
            x[:, O_NOBLES:].view(n, NOBLE_TOKENS, NOBLE_W)) + self.kind_embed[3]

        h = torch.cat((g, c, k, b), dim=1)
        for block in self.token_blocks:
            h = block(h)
        h = self.token_norm(h)

        i0 = 1
        i1 = i0 + COLOUR_TOKENS
        i2 = i1 + CARD_TOKENS
        pooled = torch.cat((h[:, 0], h[:, i0:i1].mean(1), h[:, i1:i2].mean(1),
                            h[:, i2:].mean(1), raw_global), dim=1)
        return h, pooled

    def trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, pooled = self.encode_tokens(x)
        z = self.stem(pooled)
        for block in self.blocks:
            z = block(z)
        return tokens, self.trunk_norm(z)

    # ----------------------------------------------------------------- policy
    def policy_from(self, tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Assemble the 72 logits as dot products against the right token."""
        n = z.shape[0]
        d = self.cfg.token_width
        q = self.queries(z).view(n, N_QUERIES, d)

        i0 = 1
        i1 = i0 + COLOUR_TOKENS
        i2 = i1 + CARD_TOKENS
        col = tokens[:, i0:i1]                        # (n, 6, d)
        card = tokens[:, i1:i2]                       # (n, 24, d)
        noble = tokens[:, i2:]                        # (n, 5, d)

        # (n, queries, tokens): every pointer logit in three matmuls.
        col_l = torch.bmm(q, col.transpose(1, 2))
        card_l = torch.bmm(q, card.transpose(1, 2))
        noble_l = torch.bmm(q, noble.transpose(1, 2))
        bias = self.query_bias

        take1 = col_l[:, Q_TAKE1, :GEMS] + bias[Q_TAKE1]
        take2s = col_l[:, Q_TAKE2S, :GEMS] + bias[Q_TAKE2S]
        discard = col_l[:, Q_DISCARD, :] + bias[Q_DISCARD]
        # A three-colour take is the mean of its colours' tokens against one
        # query, which is the same as the mean of their dot products.
        trio = col_l[:, Q_TRIO][:, self.trio_idx].mean(2) + bias[Q_TRIO]
        pair = col_l[:, Q_PAIR][:, self.pair_idx].mean(2) + bias[Q_PAIR]

        buy = card_l[:, Q_BUY, :BOARD_SLOTS] + bias[Q_BUY]
        reserve = card_l[:, Q_RESERVE, :BOARD_SLOTS] + bias[Q_RESERVE]
        # Seat-relative index 0 is the mover, so its reserves lead the block.
        buyres = (card_l[:, Q_BUYRES, BOARD_SLOTS:BOARD_SLOTS + MAX_RESERVED]
                  + bias[Q_BUYRES])
        nobles = noble_l[:, Q_NOBLE] + bias[Q_NOBLE]
        glob = self.head_global(z)

        return torch.cat((trio, pair, take1, take2s, buy, buyres, reserve,
                          glob[:, :TIERS], discard, nobles, glob[:, TIERS:]),
                         dim=1)

    # --------------------------------------------------------------- forwards
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, z = self.trunk(x)
        return self.policy_from(tokens, z), torch.tanh(self.value_head(z))

    def forward_all(self, x: torch.Tensor):
        """``(policy, value, aux)`` -- the trainer's view of the same pass."""
        tokens, z = self.trunk(x)
        return (self.policy_from(tokens, z), torch.tanh(self.value_head(z)),
                self.aux_head(z))


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def default_config(width: int = 256, blocks: int = 3, token_width: int = 32,
                   token_blocks: int = 1) -> NetConfig:
    return NetConfig(width=width, blocks=blocks, token_width=token_width,
                     token_blocks=token_blocks)


def new_net(width: int = 256, blocks: int = 3, token_width: int = 32,
            token_blocks: int = 1) -> SplendorNet:
    return SplendorNet(default_config(width, blocks, token_width, token_blocks))


def save_checkpoint(path, net: SplendorNet, extra: dict | None = None) -> None:
    torch.save({"cfg": net.cfg.to_dict(), "state_dict": net.state_dict(),
                "extra": extra or {}}, path)


def load_checkpoint(path, map_location="cpu") -> tuple[SplendorNet, dict]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = NetConfig(**blob["cfg"])
    if cfg.features != FEATURES:
        raise ValueError(
            f"checkpoint was trained on feature set {cfg.features!r}, not "
            f"{FEATURES!r}: it would play badly rather than fail, so it is "
            f"refused here instead")
    net = SplendorNet(cfg)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob.get("extra", {})
