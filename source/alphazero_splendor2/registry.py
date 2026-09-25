"""Picklable agent descriptions, so worker processes can build their own.

The same shape as the other games' registries, and the same ``kind`` strings
where the agent is the same idea, so the rating keys and the GUI's opponent ids
read the same across games.

The one thing this registry does that the others do not is **adapt**: the GUI,
the ladder and the tournament all hold a v1
:class:`~alphazero_splendor.splendor_game.SplendorGame`, so every agent built
here is wrapped in a :class:`V1Adapter` that converts the position on the way in.
The two implementations agree on the rules and on the 72-action numbering, so a
move index handed back means the same thing on either side.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agents import Agent, MCTSAgent, PlannerAgent, PolicyAgent, RandomAgent
from .bridge import from_v1
from .evaluator import BatchEvaluator
from .net import load_checkpoint

_EVALUATOR_CACHE: dict[str, BatchEvaluator] = {}


class V1Adapter:
    """A v2 agent that accepts a v1 position.

    The conversion is a few hundred nanoseconds and happens once per move, which
    is nothing beside the search it feeds.  ``last_value`` and ``name`` pass
    through, because that is what the GUI reads off an agent.
    """

    def __init__(self, agent: Agent):
        self.agent = agent
        self.name = agent.name

    @property
    def last_value(self) -> float:
        return self.agent.last_value

    def reset(self) -> None:
        self.agent.reset()

    def select_move(self, board, last_move=None) -> int:
        return int(self.agent.select_move(from_v1(board), last_move))


@dataclass(frozen=True)
class AgentSpec:
    kind: str
    label: str
    params: tuple = ()  # (key, value) pairs; a tuple so the spec stays hashable

    def as_dict(self) -> dict:
        return dict(self.params)


def spec(kind: str, label: str, **params) -> AgentSpec:
    return AgentSpec(kind=kind, label=label, params=tuple(sorted(params.items())))


def evaluator(ckpt: str) -> BatchEvaluator:
    """The evaluator for a checkpoint, loaded and traced at most once."""
    hit = _EVALUATOR_CACHE.get(ckpt)
    if hit is None:
        net, _ = load_checkpoint(ckpt)
        hit = BatchEvaluator(net)
        _EVALUATOR_CACHE[ckpt] = hit
    return hit


def build_native(s: AgentSpec, seed: int) -> Agent:
    """The agent itself, taking v2 positions."""
    p = s.as_dict()
    if s.kind == "random":
        return RandomAgent(seed=seed, name=s.label)
    if s.kind == "planner":
        return PlannerAgent(samples=p.get("samples", 2),
                            noise=p.get("noise", 0.05), seed=seed, name=s.label)
    if s.kind == "az":
        return MCTSAgent(evaluator(p["ckpt"]), simulations=p.get("simulations", 200),
                         c_puct=p.get("c_puct", 1.5),
                         explore_moves=p.get("explore_moves", 0),
                         explore_temperature=p.get("explore_temperature", 0.6),
                         batch_size=p.get("batch_size", 16), seed=seed,
                         name=s.label)
    if s.kind == "policy":
        return PolicyAgent(evaluator(p["ckpt"]),
                           temperature=p.get("temperature", 0.0), seed=seed,
                           name=s.label)
    raise ValueError(f"unknown agent kind {s.kind!r}")


def build_agent(s: AgentSpec, seed: int) -> V1Adapter:
    """The same, wrapped to accept the v1 positions the rest of the app holds."""
    return V1Adapter(build_native(s, seed))


def default_field(ckpt: str) -> list[AgentSpec]:
    """The tournament field: the trained agent against what it has to beat."""
    return [
        spec("random", "random"),
        spec("planner", "planner (plans its purchases)", samples=2),
        spec("policy", "AlphaZero v2 policy only (no search)", ckpt=ckpt),
        spec("az", "AlphaZero v2 (200 sims)", ckpt=ckpt, simulations=200),
    ]
