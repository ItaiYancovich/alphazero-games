"""Picklable agent descriptions, so worker processes can build their own.

The same shape as the other three games' registries, and deliberately the same
``kind`` strings where the agent is the same idea -- ``random``, ``policy``,
``az`` -- so the rating keys, the GUI's opponent ids and the ladder scripts read
the same across games.  ``heuristic`` is this game's name for what Hex and
Connect Four call ``rule``: there is no two-distance here, and calling it
``rule`` would suggest the Hex agent had been ported.

There is no ``minimax`` and no ``rollout``.  Alpha-beta prunes on the promise
that a branch cannot change the result, which needs two players and a zero-sum
score -- at three or four seats there is no such cut to make.  And a random
playout wants the playout to be evidence, which in a game where the deck decides
much of what is available it mostly is not.  What takes their place is
``planner``: a beam-limited max^n lookahead over a stronger hand-written
evaluation, which is this game's answer to the "smart brute force" rung the
other two board games fill with alpha-beta.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agents.az_agent import AlphaZeroAgent, PolicyOnlyAgent
from .agents.base import Agent, RandomAgent
from .agents.heuristic import HeuristicAgent
from .agents.planner import PlannerAgent
from .evaluator import BatchEvaluator
from .net import load_checkpoint
from .splendor_game import NACTIONS

_EVALUATOR_CACHE: dict[str, BatchEvaluator] = {}


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    label: str
    params: tuple = ()  # (key, value) pairs; a tuple so the spec stays hashable

    def as_dict(self) -> dict:
        return dict(self.params)


def spec(kind: str, label: str, **params) -> AgentSpec:
    return AgentSpec(kind=kind, label=label, params=tuple(sorted(params.items())))


def evaluator(ckpt: str, device: str = "cpu") -> BatchEvaluator:
    """The evaluator for a checkpoint, loaded and JIT-traced at most once.

    Shared process-wide: a second copy would mean a second trace and a second
    set of weights for no benefit.
    """
    key = (ckpt, device.lower())
    hit = _EVALUATOR_CACHE.get(key)
    if hit is None:
        net, _ = load_checkpoint(ckpt)
        hit = BatchEvaluator(net, device=device)
        _EVALUATOR_CACHE[key] = hit
    return hit


def build_agent(s: AgentSpec, seed: int) -> Agent:
    p = s.as_dict()
    if s.kind in ("az2", "policy2"):
        # The v2 network, wrapped so it takes the positions this package deals
        # in.  It lives here rather than in a parallel tournament because a
        # rating is only meaningful against a field: the whole point of the
        # number is that v2 and v1 appear on one leaderboard, having actually
        # played each other.
        from alphazero_splendor2.registry import build_agent as _build_v2
        from alphazero_splendor2.registry import spec as _spec_v2

        return _build_v2(_spec_v2("az" if s.kind == "az2" else "policy",
                                  s.label, **p), seed)
    if s.kind == "random":
        return RandomAgent(seed=seed, name=s.label)
    if s.kind == "heuristic":
        return HeuristicAgent(seed=seed, noise=p.get("noise", 0.15), name=s.label)
    if s.kind == "planner":
        return PlannerAgent(depth=p.get("depth", 1), beam=p.get("beam", 8),
                            samples=p.get("samples", 2),
                            noise=p.get("noise", 0.03), seed=seed, name=s.label)
    device = p.get("device", "cpu")
    if s.kind == "az":
        return AlphaZeroAgent(evaluator(p["ckpt"], device=device), NACTIONS,
                              simulations=p.get("simulations", 200),
                              c_puct=p.get("c_puct", 1.6),
                              explore_moves=p.get("explore_moves", 0),
                              explore_temperature=p.get("explore_temperature", 0.6),
                              batch_size=p.get("batch_size", 16),
                              seed=seed, name=s.label)
    if s.kind == "policy":
        return PolicyOnlyAgent(evaluator(p["ckpt"], device=device),
                               temperature=p.get("temperature", 0.0),
                               seed=seed, name=s.label)
    raise ValueError(f"unknown agent kind {s.kind!r}")


def default_field(ckpt: str) -> list[AgentSpec]:
    """The tournament field: the trained agent against what it has to beat."""
    return [
        spec("random", "random"),
        spec("heuristic", "heuristic (greedy expert)", noise=0.15),
        spec("planner", "planner (plans its purchases)", depth=1),
        spec("policy", "AlphaZero policy only (no search)", ckpt=ckpt),
        spec("az", "AlphaZero (200 sims)", ckpt=ckpt, simulations=200),
    ]
