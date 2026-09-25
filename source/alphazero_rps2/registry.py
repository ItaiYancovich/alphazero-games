"""Picklable agent descriptions, so worker processes can build their own.

Agents hold networks, RNGs and search trees, which are awkward to ship between
processes.  A tournament therefore passes around small ``AgentSpec`` records and
each worker instantiates from them, caching anything expensive (the loaded
network) per process.

Same shape as the Connect Four registry, and deliberately the same ``kind``
strings -- ``random``, ``rule``, ``minimax``, ``rollout``, ``policy``, ``az`` --
so the rating keys, the GUI's opponent ids and the ladder scripts read
identically for every game here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .agents.az_agent import AlphaZeroAgent, PolicyOnlyAgent
from .agents.base import Agent, RandomAgent
from .agents.mcts_rollout import RolloutMCTSAgent
from .agents.minimax import MinimaxAgent
from .agents.rule_based import RuleBasedAgent
from .evaluator import BatchEvaluator
from .net import load_checkpoint
from .rps2_game import N

_EVALUATOR_CACHE: dict[tuple, BatchEvaluator] = {}


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    label: str
    params: tuple = ()  # (key, value) pairs; a tuple so the spec stays hashable

    def as_dict(self) -> dict:
        return dict(self.params)


def spec(kind: str, label: str, **params) -> AgentSpec:
    return AgentSpec(kind=kind, label=label, params=tuple(sorted(params.items())))


def evaluator(ckpt: str, rows: int = N, cols: int = N, device: str = "cpu") -> BatchEvaluator:
    """The evaluator for a checkpoint, loaded and JIT-traced at most once.

    Shared process-wide: a second copy would mean a second trace and a second
    set of weights for no benefit.
    """
    try:
        stamp = os.path.getmtime(ckpt)
    except OSError:
        stamp = None  # let load_checkpoint raise the useful error
    key = (ckpt, rows, cols, stamp, device.lower())
    hit = _EVALUATOR_CACHE.get(key)
    if hit is None:
        net, _ = load_checkpoint(ckpt)
        hit = BatchEvaluator(net, rows=rows, cols=cols, device=device)
        for stale in [k for k in _EVALUATOR_CACHE if k[:3] == key[:3]]:
            del _EVALUATOR_CACHE[stale]
        _EVALUATOR_CACHE[key] = hit
    return hit


def build_agent(s: AgentSpec, seed: int, rows: int = N, cols: int = N) -> Agent:
    p = s.as_dict()
    if s.kind == "random":
        return RandomAgent(seed=seed, name=s.label)
    if s.kind == "rule":
        return RuleBasedAgent(seed=seed, noise=p.get("noise", 0.05),
                              defence_weight=p.get("defence_weight", 1.0), name=s.label)
    if s.kind == "minimax":
        return MinimaxAgent(time_budget=p.get("time_budget", 1.0),
                            max_depth=p.get("max_depth", 40), seed=seed, name=s.label)
    if s.kind == "rollout":
        return RolloutMCTSAgent(simulations=p.get("simulations", 5000),
                                time_budget=p.get("time_budget"),
                                c_uct=p.get("c_uct", 1.0), seed=seed, name=s.label)
    device = p.get("device", "cpu")
    if s.kind == "az":
        return AlphaZeroAgent(evaluator(p["ckpt"], rows, cols, device=device),
                              simulations=p.get("simulations", 400),
                              c_puct=p.get("c_puct", 1.6),
                              explore_moves=p.get("explore_moves", 0),
                              explore_temperature=p.get("explore_temperature", 0.6),
                              batch_size=p.get("batch_size", 16),
                              seed=seed, name=s.label)
    if s.kind == "policy":
        return PolicyOnlyAgent(evaluator(p["ckpt"], rows, cols, device=device),
                               temperature=p.get("temperature", 0.0),
                               seed=seed, name=s.label)
    raise ValueError(f"unknown agent kind {s.kind!r}")


def default_field(ckpt: str) -> list[AgentSpec]:
    """The tournament field: the trained agent against each classical family."""
    return [
        spec("random", "random"),
        spec("rule", "rule-based (matchup material + base defence)", noise=0.05),
        spec("minimax", "brute-force alpha-beta (1.0s/move)", time_budget=1.0),
        spec("rollout", "classic MCTS rollouts (1.0s/move)",
             simulations=200_000, time_budget=1.0),
        spec("policy", "AlphaZero policy only (no search)", ckpt=ckpt),
        spec("az", "AlphaZero (400 sims)", ckpt=ckpt, simulations=400),
    ]
