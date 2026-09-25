"""Picklable agent descriptions, so worker processes can build their own.

The same shape as the other Splendor registries and the same ``kind`` strings,
so rating keys and the GUI's opponent ids read the same across versions.  Like
v2's, every agent built here is wrapped in a :class:`V1Adapter`: the GUI and the
ladder hold a v1 position, and converting it on the way in costs a few hundred
nanoseconds against a search that costs milliseconds.

The one v3-specific knob is ``gumbel_scale``.  The Gumbel search has no
Dirichlet noise and no temperature -- its exploration *is* the root sample -- so
"vary the play between games", which the GUI offers so a bot does not replay one
line for ever, is a non-zero scale rather than a separate mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

from .agents import Agent, GumbelAgent, PlannerAgent, PolicyAgent, RandomAgent
from .bridge import from_v1
from .evaluator import BatchEvaluator
from .net import load_checkpoint

_EVALUATOR_CACHE: dict[str, BatchEvaluator] = {}


class V1Adapter:
    """A v3 agent that accepts a v1 position."""

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
    """The evaluator for a checkpoint, loaded at most once."""
    hit = _EVALUATOR_CACHE.get(ckpt)
    if hit is None:
        net, _ = load_checkpoint(ckpt)
        hit = BatchEvaluator(net)
        _EVALUATOR_CACHE[ckpt] = hit
    return hit


def build_native(s: AgentSpec, seed: int) -> Agent:
    """The agent itself, taking v3 positions."""
    p = s.as_dict()
    if s.kind == "random":
        return RandomAgent(seed=seed, name=s.label)
    if s.kind == "planner":
        return PlannerAgent(samples=p.get("samples", 2),
                            noise=p.get("noise", 0.05), seed=seed, name=s.label)
    if s.kind == "az":
        return GumbelAgent(evaluator(p["ckpt"]),
                           simulations=p.get("simulations", 128),
                           max_considered=p.get("max_considered", 16),
                           gumbel_scale=p.get("gumbel_scale", 0.0),
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
        spec("policy", "AlphaZero v3 policy only (no search)", ckpt=ckpt),
        spec("az", "AlphaZero v3 (128 sims)", ckpt=ckpt, simulations=128),
    ]
