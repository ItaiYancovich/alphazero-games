"""Picklable agent descriptions, so worker processes can build their own.

Agents hold RNGs and search trees, which are awkward to ship between processes.
A tournament therefore passes around small ``AgentSpec`` records and each worker
instantiates from them.

Same shape as the other games' registries, and deliberately the same ``kind``
strings -- ``random``, ``rule``, ``minimax``, ``rollout`` -- so the rating keys,
the GUI's opponent ids and the ladder script read identically across the
project.  What is missing is the other half of every other registry here: there
are no ``policy`` and ``az`` kinds, because Ultimate XX has no trained network,
and so this module imports neither torch nor an evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agents.base import Agent, RandomAgent
from .agents.mcts_rollout import RolloutMCTSAgent
from .agents.minimax import MinimaxAgent
from .agents.rule_based import RuleBasedAgent
from .uxx_game import N


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    label: str
    params: tuple = ()  # (key, value) pairs; a tuple so the spec stays hashable

    def as_dict(self) -> dict:
        return dict(self.params)


def spec(kind: str, label: str, **params) -> AgentSpec:
    return AgentSpec(kind=kind, label=label, params=tuple(sorted(params.items())))


def build_agent(s: AgentSpec, seed: int, rows: int = N, cols: int = N) -> Agent:
    p = s.as_dict()
    if s.kind == "random":
        return RandomAgent(seed=seed, name=s.label)
    if s.kind == "rule":
        return RuleBasedAgent(seed=seed, noise=p.get("noise", 0.05),
                              defence_weight=p.get("defence_weight", 1.0), name=s.label)
    if s.kind == "minimax":
        return MinimaxAgent(time_budget=p.get("time_budget", 1.0),
                            max_depth=p.get("max_depth", 81), seed=seed, name=s.label)
    if s.kind == "rollout":
        return RolloutMCTSAgent(simulations=p.get("simulations", 5000),
                                time_budget=p.get("time_budget"),
                                c_uct=p.get("c_uct", 1.0), seed=seed, name=s.label)
    raise ValueError(f"unknown agent kind {s.kind!r}")


def default_field() -> list[AgentSpec]:
    """The tournament field: one entrant per classical family."""
    return [
        spec("random", "random"),
        spec("rule", "rule-based (minefields + free-choice penalty)", noise=0.05),
        spec("minimax", "brute-force alpha-beta (1.0s/move)", time_budget=1.0),
        spec("rollout", "classic MCTS rollouts (1.0s/move)",
             simulations=200_000, time_budget=1.0),
    ]
