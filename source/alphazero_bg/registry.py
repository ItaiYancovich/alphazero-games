"""Picklable agent descriptions for backgammon worker processes.

Same shape as the other two registries, and the same ``kind`` strings where the
games have something in common (``random``, ``net``).  What differs is that the
strength dial is *plies*, not simulations: there is no MCTS budget to turn up,
only the choice between evaluating this roll's moves and averaging the
opponent's reply over all 21 of theirs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agents.base import Agent, RandomAgent
from .agents.heuristic import HeuristicAgent
from .agents.net_agent import NetAgent
from .evaluator import Evaluator
from .net import load_checkpoint

_EVALUATOR_CACHE: dict[str, Evaluator] = {}


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    label: str
    params: tuple = ()

    def as_dict(self) -> dict:
        return dict(self.params)


def spec(kind: str, label: str, **params) -> AgentSpec:
    return AgentSpec(kind=kind, label=label, params=tuple(sorted(params.items())))


def evaluator(ckpt: str) -> Evaluator:
    """The evaluator for a checkpoint, loaded at most once per process."""
    hit = _EVALUATOR_CACHE.get(ckpt)
    if hit is None:
        net, _ = load_checkpoint(ckpt)
        hit = Evaluator(net)
        _EVALUATOR_CACHE[ckpt] = hit
    return hit


def build_agent(s: AgentSpec, seed: int) -> Agent:
    p = s.as_dict()
    if s.kind == "random":
        return RandomAgent(seed=seed, name=s.label)
    if s.kind == "heuristic":
        return HeuristicAgent(seed=seed, noise=p.get("noise", 0.3), name=s.label)
    if s.kind == "net":
        return NetAgent(evaluator(p["ckpt"]), plies=p.get("plies", 1),
                        explore=p.get("explore", 0.0),
                        candidates=p.get("candidates", 6),
                        seed=seed, name=s.label)
    raise ValueError(f"unknown agent kind {s.kind!r}")


def default_field(ckpt: str) -> list[AgentSpec]:
    return [
        spec("random", "random"),
        spec("heuristic", "heuristic (pips + blots + points)", noise=0.3),
        spec("net", "network, 1-ply", ckpt=ckpt, plies=1),
        spec("net", "network, 2-ply", ckpt=ckpt, plies=2),
    ]
