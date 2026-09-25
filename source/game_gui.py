#!/usr/bin/env python3
"""A browser GUI for playing any of the games here against any agent here.

    python3 game_gui.py                       # opens http://127.0.0.1:7333
    python3 game_gui.py --game c4             # start on Connect Four
    python3 game_gui.py --game uttt           # ...or Ultimate Tic-Tac-Toe
    python3 game_gui.py --port 8080 --no-browser

The server is stdlib-only: it holds one game session in memory and serves a
single page that polls it.  Bot moves run on a worker thread so the board stays
responsive while a search is thinking, and an optional analysis thread keeps a
second AlphaZero tree growing on the current position so the evaluation on
screen sharpens as the search widens.

The game lives on the *server*, so reloading the page re-attaches to whatever
is in progress rather than starting a new one.

``torch`` is imported lazily, and only when an AlphaZero opponent or the live
evaluation is switched on -- the classical agents (random, rule-based,
alpha-beta, rollout MCTS) need nothing but numpy.

Two games, one server
---------------------
Everything the server does that depends on *which* game is being played goes
through a :class:`GameAdapter`: how to make a board, what the opponent menu is,
where the checkpoints and ratings live, how to name a move, which cells won.
The session, the analysis thread, the review and the HTTP layer never mention
Hex or Connect Four.  Switching game is just starting a new game with a
different adapter, so a session is always exactly one game of one of them.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import threading
import time
import traceback
import webbrowser
from collections import deque
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from alphazero_core.mcts import MCTSConfig, Search
from alphazero_hex.agents.base import Agent

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

# Slider stops.  Powers of two read better than a linear ramp: the interesting
# range of AlphaZero strength is multiplicative.  0 simulations means the raw
# policy head with no search at all.
SIM_STOPS = [0, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]
# The analysis ceiling goes far past what a move needs because it is not a
# move: left on a position, the tree keeps sharpening for as long as you let
# it.  A node costs roughly a kilobyte, so the top stop is about a gigabyte of
# tree -- fine on a desktop, and the reason the ramp stops there.
EVAL_STOPS = [500, 1000, 2000, 5000, 10_000, 25_000, 50_000, 100_000,
              250_000, 500_000, 1_000_000]
# Splendor's analysis drives itself instead of reading the panel's ceiling
# (each position gets its own tree, see `analysis_stream`), so it keeps the
# ceiling the panel used to have rather than following the stops above.
STREAM_EVAL_SIMS = 100_000
DEFAULT_SIMS = 400
# What a draw is worth to the v3 match bot.  0 is the truth; a small negative
# number is contempt, which makes it play for a win rather than accept a draw --
# the point of the exercise is beating a human, not scoring Elo on draws.  Set
# from the A/B match in PLAN_uttt_v3.md Phase 4.
V3_DRAW_VALUE = 0.0
DEFAULT_EVAL_SIMS = 2000
DEVICE = "auto"

# Compiled networks, kept for the life of the server and shared by every game.
# Building one means converting the net and compiling it for the GPU, which
# costs a couple of seconds, a thread pool and ~60 MB -- none of which used to
# be given back when a game ended, so starting games slowly ground the server
# down.  Keyed by checkpoint and backend; the evaluator serialises its own
# inference, so several agents may share one.
_V3_EVALUATORS: dict[tuple[str, str], object] = {}


def _v3_evaluator(ckpt: str, backend: str):
    key = (ckpt, backend)
    if key not in _V3_EVALUATORS:
        from alphazero_uttt.rs_agent import NetBoardEvaluator
        from alphazero_uttt.v3net import load_v3

        net, _ = load_v3(ckpt)
        _V3_EVALUATORS[key] = NetBoardEvaluator(net, net.cfg.in_planes, backend=backend)
    return _V3_EVALUATORS[key]

# A varied AlphaZero explores its opening the way self-play does: root
# Dirichlet noise plus temperature sampling for this many plies, then full
# strength. Off (0) reproduces exactly what the ladder rated.
VARY_EXPLORE_MOVES = 6
VARY_TEMPERATURE = 0.6

# How much value a move may throw away before it earns a label.  Values live on
# a -1..+1 scale, so 0.5 is a swing from level to lost.
REVIEW_BANDS = ((0.5, "blunder"), (0.25, "mistake"), (0.12, "inaccuracy"))
REVIEW_STOPS = [200, 400, 800, 1600, 3200, 6400, 12_800, 25_600]
DEFAULT_REVIEW_SIMS = 800

# The seats, as the engines number them.  Every game here numbers its players
# from 1; three of the four stop at two, and Splendor keeps going to four.
FIRST = 1
SECOND = 2


# --------------------------------------------------------------- rating table
class _RatingTable:
    """A tournament's rating file, re-read whenever it changes on disk."""

    def __init__(self, ratings_module) -> None:
        self.R = ratings_module
        self.lock = threading.Lock()
        self._stamp: float | None = None
        self._data: dict = {}

    def get(self) -> dict:
        try:
            stamp = self.R.RATINGS_PATH.stat().st_mtime
        except OSError:
            stamp = None
        with self.lock:
            if stamp != self._stamp or not self._data:
                self._stamp = stamp
                self._data = self.R.load_ratings()
            return self._data


# ------------------------------------------------------------ training log
class _TrainingLog:
    """A run's ``log.jsonl``, re-read whenever it changes on disk.

    The trainer appends one JSON object per iteration to its run directory and
    knows nothing about the GUI; this reads that file, so watching a run costs
    the run nothing and works whether the trainer is a child of this process,
    a separate terminal, or the unattended supervisor.

    One entry per iteration is small -- a few hundred bytes -- so even a
    finished run is a couple of hundred kilobytes and is simply re-read whole
    when the mtime moves.  Cached by run directory rather than by game, since
    that is what actually identifies a run.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._cache: dict[Path, tuple[float | None, dict]] = {}

    def get(self, adapter: "GameAdapter") -> dict:
        path = adapter.ckpt_dir / "log.jsonl"
        try:
            stamp = path.stat().st_mtime
        except OSError:
            stamp = None
        with self.lock:
            hit = self._cache.get(path)
            if hit is not None and hit[0] == stamp:
                return hit[1]
            blob = self._read(adapter, path, stamp)
            self._cache[path] = (stamp, blob)
            return blob

    @staticmethod
    def _read(adapter: "GameAdapter", path: Path, stamp: float | None) -> dict:
        # Every return below has the same keys: the page reads them without
        # checking, and a run that has not started yet is the common case.
        empty = dict(game=adapter.key, run_dir=str(adapter.ckpt_dir), running=False,
                     updated=None, config=None, planned=None, offset=0,
                     iterations=[])
        if stamp is None:
            return empty
        config = None
        offset = 0
        rows: list[dict] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # The trainer may be part-way through a write; the next
                        # poll will see the finished line.
                        continue
                    if record.get("event") and record.get("event") != "start":
                        # A record that is not an iteration.  Intransitive's
                        # warm start is one: it writes losses like an iteration
                        # but its "games" is a few thousand teacher games at
                        # once, which on the same axis as a 300-game iteration
                        # flattens the whole chart.  It stays in the log; it is
                        # simply not a point on a per-iteration curve.
                        continue
                    if record.get("event") == "start":
                        config = record.get("config")
                        # A resumed run keeps counting iterations from where it
                        # stopped, but its stage list starts again from the top
                        # -- so the schedule is shifted by however much was
                        # already done.  Without this the progress bar and the
                        # ETA both read against boundaries the run is not
                        # actually following.
                        offset = rows[-1]["iteration"] if rows else 0
                        continue
                    rows.append(record)
        except OSError as exc:
            return dict(empty, error=str(exc))

        # Only the fields the page plots, so a long run stays a small response.
        keep = ("iteration", "stage", "games", "new_positions", "buffer", "avg_plies",
                "draw_rate", "black_win_rate", "games_per_sec", "policy_loss",
                "value_loss", "aux_loss", "policy_top1_agreement", "train_steps",
                "selfplay_sec", "train_sec", "time",
                "vs_random", "vs_rule_based", "vs_start_checkpoint",
                # The two that keep measuring after the hand-written opponents
                # saturate -- a frozen network, and the run's own past.  The
                # Intransitive run beat random and rule-based 100% of the time
                # from iteration 5, so without these the panel's rate chart
                # shows three flat lines at 1.0 and says nothing.
                "vs_reference", "vs_past",
                # The same matches counting only decisive games.  Two close
                # networks draw most of theirs, which pulls every score towards
                # 0.5 and hides real progress.
                "vs_past_decisive", "vs_reference_decisive")
        iterations = [{k: r[k] for k in keep if k in r} for r in rows]
        # A run is "running" if its log moved recently.  There is no handshake
        # with the trainer -- it is a different process, possibly on its own
        # schedule -- so recency is the only honest signal, and one iteration
        # here is tens of minutes, hence the generous window.
        running = stamp is not None and (time.time() - stamp) < 3600
        return dict(
            game=adapter.key,
            run_dir=str(adapter.ckpt_dir),
            running=running,
            updated=stamp,
            config=config,
            offset=offset,
            planned=_planned_iterations(config, offset),
            iterations=iterations,
        )


def _planned_iterations(config: dict | None, offset: int = 0) -> int | None:
    """The iteration this run's schedule finishes on.

    ``offset`` is what had already been done when the trainer last started; a
    resumed run replays its stage list from the top, so it finishes that many
    iterations later than the schedule alone would suggest.
    """
    if not config:
        return None
    stages = config.get("stages") or []
    total = sum(int(s.get("iterations", 0)) for s in stages)
    return (total + offset) or None


TRAINING = _TrainingLog()


# ------------------------------------------------------------- game adapters
class GameAdapter:
    """Everything the server needs to know about one game.

    Implementations are thin: each method is a call into that game's package.
    The point is that the session below can be written once.
    """

    key = ""
    label = ""
    # Display names for the two sides, first player first.  The engines call
    # them BLACK and WHITE; nobody playing has ever wanted to see that.
    side_names = ("Red", "Blue")
    # How many seats this game can be played by.  Three of the four are duels
    # and take the defaults; Splendor seats two to four and overrides them,
    # along with ``seat_ids`` and ``seat_names`` below.
    min_players = 2
    max_players = 2
    sizes: list[dict] = []
    default_size = ""
    draws = False
    ckpt_dir: Path = ROOT
    # The strength dial the page shows for the trained opponent.  Simulations
    # for the two search games; backgammon overrides it with search depth.
    strength_stops = SIM_STOPS
    strength_label = "sims"
    default_strength = DEFAULT_SIMS
    # The opponent a new game starts against, and what to fall back to when
    # there is no trained network yet.  Catalog ids differ between games, so a
    # session that carried one game's id into another would silently end up
    # with no bot at all.
    default_opponent = "az"
    fallback_opponent = "rule"
    ladder_script = "run_ladder.py"
    # What a move is called when the page has to refuse one.  A cell in the two
    # board games; backgammon plays sequences of hops, which are not cells.
    move_noun = "cell"
    # Least time a bot's move may take, in seconds, so that a watcher can see it
    # happen.  A cheap agent answers in microseconds, and a board that changes
    # between two frames is a board nobody can follow -- which matters most
    # where a move is not a piece appearing on a square.  Multiplied by the
    # session's pace, so it can be turned off.
    bot_pause = 0.0

    # --- what kind of game this is -------------------------------------
    # Hex and Connect Four are deterministic, a move is a cell, and the shared
    # PUCT search analyses them.  A dice game is none of those things, so the
    # defaults below describe the cell games and backgammon overrides them.
    dice = False          # does a turn begin with a roll nobody chooses?
    uses_search = True    # can the shared MCTS analyse and review this game?
    analysable = True     # is the live evaluation offered at all?
    reviewable = True     # is match review offered at all?
    # Why match review is unavailable, when it is.  Games that have no engine
    # for one reason say so in their own words rather than in backgammon's.
    review_note: str | None = None

    # ------------------------------------------------------------- the seats
    def seat_ids(self, board) -> list[int]:
        """The engine's number for each seat, in turn order.

        The session keys everything it holds per player -- the agent, the
        opponent choice, the strength, the network -- by these, and reads the
        seat on turn straight off ``board.to_move``.  So a game with four
        players needs nothing from the session beyond returning four of them.
        """
        return [FIRST, SECOND]

    def seat_names(self, board) -> list[str]:
        """What to call each seat on screen, in the same order."""
        return list(self.side_names)

    def seat_count(self, size_id: str) -> int:
        """How many seats a game of this size has, before there is a board."""
        return 2

    def needs_roll(self, board) -> bool:
        """Is the position waiting for a roll before anyone can move?"""
        return False

    def roll(self, board, rng) -> list[int]:
        """Roll for the side to move, in place.  Returns the dice."""
        return []

    def select(self, agent, board, last_move):
        """Ask an agent for its move.  Cell games also pass the last move."""
        return agent.select_move(board, last_move)

    def is_legal(self, board, move) -> bool:
        return board.is_legal(move)

    def apply(self, board, move) -> None:
        board.play(move)

    def parse_move(self, payload: dict, board):
        """Turn a request body into a move, or ``None`` if it is not one."""
        cell = payload.get("cell")
        return None if cell is None else int(cell)

    def move_entry(self, move, before, after, by: str) -> dict:
        """One row of the move list, as the page will render it."""
        return dict(cell=int(move), player=before.to_move, by=by,
                    label=self.move_label(int(move), after))

    def replay_move(self, board, entry: dict) -> None:
        """Re-apply a stored move entry when rebuilding after a take-back."""
        board.play(int(entry["cell"]))

    def restore_roll(self, board, entry: dict) -> bool:
        """Put back the roll that was in hand before ``entry`` was played.

        A take-back that rerolled would be a free second chance at the dice,
        which is not a take-back.  Cell games have nothing to restore.
        """
        return False

    def board_fields(self, board) -> dict:
        """Game-specific parts of the state the page renders."""
        rows, cols = self.shape(board)
        return dict(cells=[int(v) for v in board.array().reshape(-1)],
                    rows=rows, cols=cols, n=rows)

    def analysis_stream(self, board, ckpt):
        """Successive readouts for a non-search game; see ``uses_search``."""
        raise NotImplementedError

    # ------------------------------------------------------------- checkpoints
    def checkpoints(self) -> list[str]:
        """Available network snapshots, best-known first."""
        if not self.ckpt_dir.is_dir():
            return []
        # ``trainer_*.pt`` is the run's own resumable state -- the optimiser
        # moments -- and not a network.  It lives beside the checkpoints and
        # matches the same glob, so it has to be named out here, or it reaches
        # the menu as an opponent and fails to load the moment it is picked.
        files = sorted(p.name for p in self.ckpt_dir.glob("*.pt")
                       if not p.name.startswith("trainer_"))
        preferred = [n for n in ("final.pt", "latest.pt") if n in files]
        return preferred + [n for n in files if n not in preferred]

    def ckpt_path(self, ckpt: str) -> str:
        """Where a name from :meth:`checkpoints` actually lives.

        A hook rather than ``ckpt_dir / ckpt`` spelled out at each use, because
        a game whose snapshots are spread over more than one run directory has
        to answer this for itself -- and the match review resolves checkpoints
        for whichever game it was handed, so it cannot special-case any of them.
        """
        return str(self.ckpt_dir / ckpt)

    def analysis_checkpoint(self) -> str:
        """The network that judges positions, whoever you happen to be playing.

        Deliberately *not* the opponent's network: the evaluation is a fixed
        yardstick, so picking a weak curriculum snapshot as an opponent must not
        also make the eval bar weaker.  Always the strongest network available.

        Which one that is cannot be inferred from the filenames, so it is named
        in ``best.txt``.  Without it this falls through to ``checkpoints()[0]``,
        which is ``final.pt`` -- fine while every snapshot is of similar
        strength, and badly wrong once a network hundreds of Elo stronger
        exists alongside it.
        """
        pointer = self.ckpt_dir / "best.txt"
        try:
            named = pointer.read_text(encoding="utf-8").strip()
            if named and (self.ckpt_dir / named).is_file():
                return named
        except OSError:
            pass
        return (self.checkpoints() or ["final.pt"])[0]

    # ------------------------------------------------------------------ board
    def new_board(self, size_id: str, seed: int | None = None):
        """A fresh position.

        ``seed`` fixes anything the setup shuffles.  Three of the four games
        have nothing to shuffle and ignore it; Splendor deals three decks from
        it, and *must* get the same one again when a take-back rebuilds the
        position by replaying the moves -- a take-back that redealt would be a
        different game, not an earlier one.
        """
        raise NotImplementedError

    def size_id(self, board) -> str:
        raise NotImplementedError

    def rating_board_size(self, board):
        """What the ratings file records as the board this field was rated on."""
        return self.rating_size_for(self.size_id(board))

    def rating_size_for(self, size_id: str):
        """The same, from a size id rather than a live board."""
        return size_id

    def shape(self, board) -> tuple[int, int]:
        raise NotImplementedError

    def position_key(self, board) -> bytes:
        """Identity of a position, for noticing that the board has changed."""
        return board.canonical_board().tobytes()

    def format_value(self, value: float) -> str:
        """A value the way this game says it.

        The two board games report the root value of a search, which is an
        expected result on a -1..+1 scale and reads fine as a bare number.
        Backgammon reports equity, which has a unit -- points a game -- and is
        carried divided by three, so a bare number there is off by a factor of
        three from everything else on the page.
        """
        return f"{value:+.2f}"

    def move_label(self, move: int, board) -> str:
        raise NotImplementedError

    def win_path(self, board) -> list[int]:
        """The cells that won, for highlighting; empty if nobody won."""
        raise NotImplementedError

    def score_for(self, board, colour: int) -> float:
        """1 win, 0.5 draw, 0 loss, for a finished game."""
        if board.winner == 0:
            return 0.5
        return 1.0 if board.winner == colour else 0.0

    # ----------------------------------------------------------------- agents
    def catalog(self) -> list[dict]:
        raise NotImplementedError

    def valid_strength(self, value) -> int:
        """``value`` if it is one of this game's stops, else its default."""
        try:
            value = int(value)
        except (TypeError, ValueError):
            return self.default_strength
        return value if value in self.strength_stops else self.default_strength

    def valid_choice(self, choice: str, default: str) -> str:
        """``choice`` if this game offers it, otherwise something it does.

        Opponent ids are per game -- ``az`` in the search games, ``net`` here --
        and a session that switched games would otherwise keep the old id and
        fail to build any agent for it.
        """
        kind = str(choice or "").partition(":")[0]
        known = {entry["id"].partition(":")[0] for entry in self.catalog()}
        if kind in known:
            return choice
        if not self.checkpoints() and default != self.fallback_opponent:
            return self.fallback_opponent
        return default

    def build_agent(self, choice: str, seed: int, board, ckpt: str,
                    sims: int = DEFAULT_SIMS, vary: bool = False) -> Agent | None:
        raise NotImplementedError

    def evaluator(self, ckpt: str, board):
        """The cached evaluator for a checkpoint name (needs torch)."""
        raise NotImplementedError

    # --------------------------------------------------------------- analysis
    def readout(self, search, board) -> dict:
        raise NotImplementedError

    def terminal_readout(self, board) -> dict:
        raise NotImplementedError

    def review_tasks(self, moves: list[int], size_id: str, ckpt_path: str,
                     sims: int) -> list[tuple]:
        raise NotImplementedError

    @property
    def evaluate_position(self):
        """The picklable worker function a review process pool calls."""
        raise NotImplementedError

    # ---------------------------------------------------------------- ratings
    @property
    def R(self):
        raise NotImplementedError

    def ratings(self) -> dict:
        raise NotImplementedError


class HexAdapter(GameAdapter):
    key = "hex"
    label = "Hex"
    side_names = ("Red", "Blue")
    sizes = [{"id": str(n), "label": f"{n} x {n}"} for n in (5, 7, 9, 11, 13)]
    default_size = "11"
    draws = False
    ckpt_dir = ROOT / "runs" / "az_hex"
    v3_dir = ROOT / "runs" / "az_v3_treatment"

    def checkpoints(self) -> list[str]:
        v2_files = sorted(p.name for p in self.ckpt_dir.glob("*.pt")
                          if not p.name.startswith("trainer_"))
        v3_files = []
        if self.v3_dir.is_dir():
            v3_files = sorted((f"v3/{p.name}" for p in self.v3_dir.glob("*.pt")
                               if not p.name.startswith("selfplay") and not p.name.startswith("optimizer")),
                              reverse=True)
            if "v3/latest.pt" in v3_files:
                v3_files.remove("v3/latest.pt")
                v3_files.insert(0, "v3/latest.pt")
        preferred = [n for n in ("v3/latest.pt", "v2_final.pt", "final.pt", "latest.pt")
                     if n in v3_files or n in v2_files]
        rest = [n for n in (v3_files + v2_files) if n not in preferred]
        return preferred + rest

    def ckpt_path(self, ckpt: str) -> str:
        if ckpt.startswith("v3/"):
            return str(self.v3_dir / ckpt[3:])
        return str(self.ckpt_dir / ckpt)

    _MINIMAX_BLURB = ("Negamax with alpha-beta, iterative deepening, a transposition "
                      "table, and beam pruning to the best cells by two-distance.")
    _ROLLOUT_BLURB = ("UCT with uniformly random playouts: the strongest approach to "
                      "Hex before neural networks.")

    def new_board(self, size_id: str, seed: int | None = None):
        from alphazero_hex.hex_game import HexBoard

        return HexBoard(int(size_id))

    def size_id(self, board) -> str:
        return str(board.n)

    def rating_size_for(self, size_id: str) -> int:
        # An int, because that is what every ratings.json written so far holds.
        return int(size_id)

    def shape(self, board) -> tuple[int, int]:
        return board.n, board.n

    def move_label(self, move: int, board) -> str:
        from alphazero_hex.hex_game import move_to_str

        return move_to_str(move, board.n)

    def win_path(self, board) -> list[int]:
        return hex_winning_path(board)

    def catalog(self) -> list[dict]:
        entries = [
            dict(id="human", group="Human", label="You (mouse)", torch=False, sims=False,
                 blurb="Click a cell to play."),
            dict(id="random", group="Classical", label="Random", torch=False, sims=False,
                 blurb="Uniform legal move -- the floor of the rating scale."),
            dict(id="rule", group="Classical", label="Rule-based (two-distance + bridges)",
                 torch=False, sims=False,
                 blurb="Anshelevich two-distance, plus win-now / block / bridge-restore. "
                       "No search at all."),
        ]
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"minimax:{budget}", group="Classical",
                                label=f"Alpha-beta brute force — {budget:g}s",
                                torch=False, sims=False, blurb=self._MINIMAX_BLURB))
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"rollout:{budget}", group="Classical",
                                label=f"Classic MCTS rollouts — {budget:g}s",
                                torch=False, sims=False, blurb=self._ROLLOUT_BLURB))
        entries.append(dict(id="az", group="AlphaZero", label="AlphaZero (network + search)",
                            torch=True, sims=True,
                            blurb="PUCT search guided by the trained network. The slider "
                                  "sets simulations per move; at 0 it is the bare policy "
                                  "head."))
        return entries

    def build_agent(self, choice: str, seed: int, board, ckpt: str,
                    sims: int = DEFAULT_SIMS, vary: bool = False) -> Agent | None:
        from alphazero_hex.agents.base import RandomAgent
        from alphazero_hex.agents.mcts_rollout import RolloutMCTSAgent
        from alphazero_hex.agents.minimax import MinimaxAgent
        from alphazero_hex.agents.rule_based import RuleBasedAgent

        kind, _, arg = choice.partition(":")
        if kind == "human":
            return None
        if kind == "random":
            return RandomAgent(seed=seed, name="random")
        if kind == "rule":
            return RuleBasedAgent(seed=seed, noise=0.05, name="rule-based")
        if kind == "minimax":
            return MinimaxAgent(time_budget=float(arg), beam=8, seed=seed,
                                name=f"alpha-beta ({float(arg):g}s)")
        if kind == "rollout":
            # A simulation cap far above what the budget allows: the clock decides.
            return RolloutMCTSAgent(simulations=200_000, time_budget=float(arg), seed=seed,
                                    name=f"mcts-rollout ({float(arg):g}s)")
        if kind == "az":
            from alphazero_hex.registry import build_agent as from_spec  # needs torch
            from alphazero_hex.registry import spec

            path = self.ckpt_path(ckpt)
            dev = getattr(self, "device", DEVICE)
            if int(sims) <= 0:
                s = spec("policy", "alphazero policy (no search)", ckpt=path,
                         temperature=VARY_TEMPERATURE if vary else 0.0, device=dev)
            else:
                s = spec("az", f"alphazero ({sims} sims)", ckpt=path, simulations=int(sims),
                         explore_moves=VARY_EXPLORE_MOVES if vary else 0,
                         explore_temperature=VARY_TEMPERATURE, device=dev)
            return from_spec(s, seed, board.n)
        raise ValueError(f"unknown opponent {choice!r}")

    def evaluator(self, ckpt: str, board):
        from alphazero_hex.registry import evaluator  # needs torch

        dev = getattr(self, "device", DEVICE)
        return evaluator(self.ckpt_path(ckpt), board.n, device=dev)

    def readout(self, search, board) -> dict:
        from alphazero_hex.review import readout

        return readout(search, board)

    def terminal_readout(self, board) -> dict:
        from alphazero_hex.review import terminal_readout

        return terminal_readout(board)

    def review_tasks(self, moves, size_id, ckpt_path, sims) -> list[tuple]:
        return [(moves[:i], int(size_id), ckpt_path, sims, i)
                for i in range(len(moves) + 1)]

    @property
    def evaluate_position(self):
        from alphazero_hex.review import evaluate_position

        return evaluate_position

    @property
    def R(self):
        from alphazero_hex import ratings as R

        return R

    def ratings(self) -> dict:
        return HEX_RATINGS.get()


class ConnectFourAdapter(GameAdapter):
    key = "c4"
    label = "Connect Four"
    side_names = ("Red", "Yellow")
    sizes = [{"id": "6x7", "label": "6 x 7 (standard)"},
             {"id": "5x6", "label": "5 x 6"},
             {"id": "7x8", "label": "7 x 8"},
             {"id": "6x9", "label": "6 x 9"}]
    default_size = "6x7"
    ladder_script = "run_ladder_c4.py"
    draws = True
    ckpt_dir = ROOT / "runs" / "az_c4"

    _MINIMAX_BLURB = ("Negamax with alpha-beta, iterative deepening, a transposition "
                      "table and centre-first move ordering -- the classical way to "
                      "play this game, and a strong one.")
    _ROLLOUT_BLURB = ("UCT with random playouts that take a win and block a loss: "
                      "search with no knowledge beyond the rules.")

    @staticmethod
    def _dims(size_id: str) -> tuple[int, int]:
        rows, _, cols = size_id.partition("x")
        return int(rows), int(cols)

    def new_board(self, size_id: str, seed: int | None = None):
        from alphazero_c4.c4_game import Connect4Board

        rows, cols = self._dims(size_id)
        return Connect4Board(rows, cols)

    def size_id(self, board) -> str:
        return f"{board.rows}x{board.cols}"

    def shape(self, board) -> tuple[int, int]:
        return board.rows, board.cols

    def move_label(self, move: int, board) -> str:
        from alphazero_c4.c4_game import move_to_str

        return move_to_str(move, board.cols)

    def win_path(self, board) -> list[int]:
        return list(board.winning_cells())

    def catalog(self) -> list[dict]:
        entries = [
            dict(id="human", group="Human", label="You (mouse)", torch=False, sims=False,
                 blurb="Click a column to drop a disc."),
            dict(id="random", group="Classical", label="Random", torch=False, sims=False,
                 blurb="Uniform legal column -- the floor of the rating scale."),
            dict(id="rule", group="Classical", label="Rule-based (windows + centre)",
                 torch=False, sims=False,
                 blurb="Win now, block a win, otherwise the best cell by counting "
                       "four-in-a-row windows. Never volunteers the square above. "
                       "No search at all."),
        ]
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"minimax:{budget}", group="Classical",
                                label=f"Alpha-beta brute force — {budget:g}s",
                                torch=False, sims=False, blurb=self._MINIMAX_BLURB))
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"rollout:{budget}", group="Classical",
                                label=f"Classic MCTS rollouts — {budget:g}s",
                                torch=False, sims=False, blurb=self._ROLLOUT_BLURB))
        entries.append(dict(id="az", group="AlphaZero", label="AlphaZero (network + search)",
                            torch=True, sims=True,
                            blurb="PUCT search guided by the trained network. The slider "
                                  "sets simulations per move; at 0 it is the bare policy "
                                  "head."))
        return entries

    def build_agent(self, choice: str, seed: int, board, ckpt: str,
                    sims: int = DEFAULT_SIMS, vary: bool = False) -> Agent | None:
        from alphazero_c4.agents.base import RandomAgent
        from alphazero_c4.agents.mcts_rollout import RolloutMCTSAgent
        from alphazero_c4.agents.minimax import MinimaxAgent
        from alphazero_c4.agents.rule_based import RuleBasedAgent

        kind, _, arg = choice.partition(":")
        if kind == "human":
            return None
        if kind == "random":
            return RandomAgent(seed=seed, name="random")
        if kind == "rule":
            return RuleBasedAgent(seed=seed, noise=0.05, name="rule-based")
        if kind == "minimax":
            return MinimaxAgent(time_budget=float(arg), seed=seed,
                                name=f"alpha-beta ({float(arg):g}s)")
        if kind == "rollout":
            return RolloutMCTSAgent(simulations=200_000, time_budget=float(arg), seed=seed,
                                    name=f"mcts-rollout ({float(arg):g}s)")
        if kind == "az":
            from alphazero_c4.registry import build_agent as from_spec  # needs torch
            from alphazero_c4.registry import spec

            path = str(self.ckpt_dir / ckpt)
            dev = getattr(self, "device", DEVICE)
            if int(sims) <= 0:
                s = spec("policy", "alphazero policy (no search)", ckpt=path,
                         temperature=VARY_TEMPERATURE if vary else 0.0, device=dev)
            else:
                s = spec("az", f"alphazero ({sims} sims)", ckpt=path, simulations=int(sims),
                         explore_moves=VARY_EXPLORE_MOVES if vary else 0,
                         explore_temperature=VARY_TEMPERATURE, device=dev)
            return from_spec(s, seed, board.rows, board.cols)
        raise ValueError(f"unknown opponent {choice!r}")

    def evaluator(self, ckpt: str, board):
        from alphazero_c4.registry import evaluator  # needs torch

        dev = getattr(self, "device", DEVICE)
        return evaluator(str(self.ckpt_dir / ckpt), board.rows, board.cols, device=dev)

    def readout(self, search, board) -> dict:
        from alphazero_c4.review import readout

        return readout(search, board)

    def terminal_readout(self, board) -> dict:
        from alphazero_c4.review import terminal_readout

        return terminal_readout(board)

    def review_tasks(self, moves, size_id, ckpt_path, sims) -> list[tuple]:
        shape = self._dims(size_id)
        return [(moves[:i], shape, ckpt_path, sims, i) for i in range(len(moves) + 1)]

    @property
    def evaluate_position(self):
        from alphazero_c4.review import evaluate_position

        return evaluate_position

    @property
    def R(self):
        from alphazero_c4 import ratings as R

        return R

    def ratings(self) -> dict:
        return C4_RATINGS.get()



class UltimateAdapter(GameAdapter):
    """Ultimate Tic-Tac-Toe: one 9x9 grid, nine small games, one big one.

    A cell game like the first two, so almost everything the base class assumes
    holds.  The one thing it does not is that a position is *not* determined by
    the marks: a move sends the opponent to the small board of the same index,
    so which cells are playable follows from the last move.  Two consequences
    reach this file -- ``board_fields`` ships the legal set and the forced board
    to the page, because the page cannot derive them from the cells, and
    ``position_key`` has to include them, because two positions with identical
    marks and different constraints are different positions and the analysis
    thread must notice when one becomes the other.
    """

    key = "uttt"
    label = "Ultimate Tic-Tac-Toe"
    side_names = ("X", "O")
    sizes = [{"id": "9x9", "label": "9 x 9 (standard)"}]
    default_size = "9x9"
    ladder_script = "run_ladder_uttt.py"
    draws = True
    # The current run.  Pointing here rather than at ``az_uttt`` also aims the
    # live training panel at the run that is actually training.
    ckpt_dir = ROOT / "runs" / "az_uttt_v2"
    # The first run's network, kept on the menu as the thing to measure against.
    # It is a different architecture -- 7 input planes and a scalar value head,
    # against 23 and a three-way one -- which is exactly why it is worth having:
    # a checkpoint carries its own ``NetConfig``, so both load and play side by
    # side with nothing to configure.
    legacy_ckpt_dir = ROOT / "runs" / "az_uttt"
    LEGACY_TAG = "prev:"

    _MINIMAX_BLURB = ("Negamax with alpha-beta, iterative deepening and a "
                      "transposition table, over a two-level static evaluation. "
                      "The classical approach; the tree is deep here, so it sees "
                      "less far than it does at Connect Four.")
    _ROLLOUT_BLURB = ("UCT with playouts that finish a game when they can, block "
                      "when they must and take a small board when one is going "
                      "free: search with almost no knowledge beyond the rules.")

    def new_board(self, size_id: str, seed: int | None = None):
        from alphazero_uttt.uttt_game import UltimateBoard

        return UltimateBoard()

    def size_id(self, board) -> str:
        return "9x9"

    def shape(self, board) -> tuple[int, int]:
        return board.rows, board.cols

    def position_key(self, board) -> bytes:
        # The canonical board already encodes the constraint -- a reachable
        # empty reads 0 and an unreachable one 3 -- so the inherited
        # implementation would in fact be correct.  Spelled out because the
        # analysis thread's correctness depends on it and it is not obvious.
        return board.canonical_board().tobytes()

    def move_label(self, move: int, board) -> str:
        from alphazero_uttt.uttt_game import move_to_str

        return move_to_str(move)

    def win_path(self, board) -> list[int]:
        return list(board.winning_cells())

    def board_fields(self, board) -> dict:
        rows, cols = self.shape(board)
        return dict(
            cells=[int(v) for v in board.array().reshape(-1)],
            rows=rows, cols=cols, n=rows,
            # The three things the marks do not say.  ``legal`` is what the
            # page shades and accepts clicks on, ``active`` is the board the
            # last move pointed at (null when it pointed at a decided one and
            # the choice is free), and ``small`` is each small board's status:
            # 0 open, 1/2 won, 3 full and drawn.
            legal=[int(m) for m in board.legal_moves()],
            active=None if board.active is None else int(board.active),
            small=[board.board_state(b) for b in range(9)],
            win_boards=list(board.winning_boards()),
        )

    def catalog(self) -> list[dict]:
        entries = [
            dict(id="human", group="Human", label="You (mouse)", torch=False, sims=False,
                 blurb="Click a highlighted square."),
            dict(id="random", group="Classical", label="Random", torch=False, sims=False,
                 blurb="Uniform legal square -- the floor of the rating scale."),
            dict(id="rule", group="Classical", label="Rule-based (two-level lines)",
                 torch=False, sims=False,
                 blurb="Win now, block a win, otherwise the best square by counting "
                       "lines of three inside every small board and across the grid "
                       "of small boards -- and by refusing to send you somewhere you "
                       "may choose freely. No search at all."),
        ]
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"minimax:{budget}", group="Classical",
                                label=f"Alpha-beta brute force — {budget:g}s",
                                torch=False, sims=False, blurb=self._MINIMAX_BLURB))
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"rollout:{budget}", group="Classical",
                                label=f"Classic MCTS rollouts — {budget:g}s",
                                torch=False, sims=False, blurb=self._ROLLOUT_BLURB))
        entries.append(dict(id="az", group="AlphaZero", label="AlphaZero (network + search)",
                            torch=True, sims=True,
                            blurb="PUCT search guided by the trained network. The slider "
                                  "sets simulations per move; at 0 it is the bare policy "
                                  "head."))
        # The v3 match bot: our own network in the native search, thinking on a
        # clock and on your time.  Never uses any imported network.
        if (ROOT / "runs" / "uttt_v3" / "best.pt").is_file():
            for secs in (3.0, 10.0, 20.0):
                entries.append(dict(
                    id=f"v3:{secs:g}", group="v3 match bot",
                    label=f"v3 - {secs:g}s a move", torch=True, sims=False,
                    # Outside the Elo ladder on purpose: the ladder rates bots
                    # by simulation count against one another, and this one is
                    # on a clock and far above the top rung.  A rated game
                    # against it is refused, with a reason, by _rating_problem.
                    rated=False,
                    blurb="The v3 network in the native search: bitboards, proven wins and "
                          "losses, an exact endgame solver, a shared evaluation cache, the "
                          "tree kept between moves, and thinking while you think."))
        return entries

    def build_agent(self, choice: str, seed: int, board, ckpt: str,
                    sims: int = DEFAULT_SIMS, vary: bool = False) -> Agent | None:
        from alphazero_uttt.agents.base import RandomAgent
        from alphazero_uttt.agents.mcts_rollout import RolloutMCTSAgent
        from alphazero_uttt.agents.minimax import MinimaxAgent
        from alphazero_uttt.agents.rule_based import RuleBasedAgent

        kind, _, arg = choice.partition(":")
        if kind == "human":
            return None
        if kind == "random":
            return RandomAgent(seed=seed, name="random")
        if kind == "rule":
            return RuleBasedAgent(seed=seed, noise=0.05, name="rule-based")
        if kind == "minimax":
            return MinimaxAgent(time_budget=float(arg), seed=seed,
                                name=f"alpha-beta ({float(arg):g}s)")
        if kind == "rollout":
            return RolloutMCTSAgent(simulations=200_000, time_budget=float(arg), seed=seed,
                                    name=f"mcts-rollout ({float(arg):g}s)")
        if kind == "v3":
            from alphazero_uttt.rs_agent import PonderingAgent

            dev = getattr(self, "device", DEVICE)
            backend = "ov-gpu-f16" if dev.lower() in ("auto", "gpu", "ov-gpu") else "torch"
            ev = _v3_evaluator(str(ROOT / "runs" / "uttt_v3" / "best.pt"), backend)
            book = ROOT / "runs" / "uttt_v3" / "book.json"
            return PonderingAgent(ev, time_budget=float(arg), batch=128, seed=seed,
                                  draw_value=V3_DRAW_VALUE,
                                  book=str(book) if book.is_file() else None,
                                  name=f"v3 ({float(arg):g}s)")
        if kind == "az":
            from alphazero_uttt.registry import build_agent as from_spec  # needs torch
            from alphazero_uttt.registry import spec

            path = self.ckpt_path(ckpt)
            dev = getattr(self, "device", DEVICE)
            if int(sims) <= 0:
                s = spec("policy", "alphazero policy (no search)", ckpt=path,
                         temperature=VARY_TEMPERATURE if vary else 0.0, device=dev)
            else:
                s = spec("az", f"alphazero ({sims} sims)", ckpt=path, simulations=int(sims),
                         explore_moves=VARY_EXPLORE_MOVES if vary else 0,
                         explore_temperature=VARY_TEMPERATURE, device=dev)
            return from_spec(s, seed, board.rows, board.cols)
        raise ValueError(f"unknown opponent {choice!r}")

    def evaluator(self, ckpt: str, board):
        from alphazero_uttt.registry import evaluator  # needs torch

        dev = getattr(self, "device", DEVICE)
        return evaluator(self.ckpt_path(ckpt), board.rows, board.cols, device=dev)

    def checkpoints(self) -> list[str]:
        """This run's snapshots, then the previous run's network to beat."""
        names = super().checkpoints()
        if self.legacy_ckpt_dir.is_dir():
            names += [self.LEGACY_TAG + name for name in ("final.pt", "latest.pt")
                      if (self.legacy_ckpt_dir / name).is_file()]
        return names

    def ckpt_path(self, ckpt: str) -> str:
        if ckpt.startswith(self.LEGACY_TAG):
            return str(self.legacy_ckpt_dir / ckpt[len(self.LEGACY_TAG):])
        return str(self.ckpt_dir / ckpt)

    def readout(self, search, board) -> dict:
        from alphazero_uttt.review import readout

        return readout(search, board)

    def terminal_readout(self, board) -> dict:
        from alphazero_uttt.review import terminal_readout

        return terminal_readout(board)

    def review_tasks(self, moves, size_id, ckpt_path, sims) -> list[tuple]:
        shape = (9, 9)
        return [(moves[:i], shape, ckpt_path, sims, i) for i in range(len(moves) + 1)]

    @property
    def evaluate_position(self):
        from alphazero_uttt.review import evaluate_position

        return evaluate_position

    @property
    def R(self):
        from alphazero_uttt import ratings as R

        return R

    def ratings(self) -> dict:
        return UTTT_RATINGS.get()


class UltimateXXAdapter(GameAdapter):
    """Ultimate XX: the same nine boards, one mark, and winning loses.

    Everything geometric is Ultimate Tic-Tac-Toe's, so the sending rule, the
    forced board and the free choice all reach the page the same way.  Three
    things do not, and each is a consequence of there being only one mark:

    * ``small`` is who *claimed* each small board, which -- unlike in Ultimate
      Tic-Tac-Toe -- the page could not work out from the cells if it tried: a
      line of three belongs to whoever closed it, and that is a fact about the
      order the marks were played in.  The move list carries the player of each
      move, so the review scrubber replays them and tracks the claims itself.
    * ``hot`` and ``fatal`` name the squares that take a small board and the
      squares that lose the game outright.  Both are derivable by the player
      and both are the whole content of a turn, so the board shows them rather
      than letting a first-time player lose to a rule they have not met yet.
    * ``win_path`` is the *losing* line: the nine marks of the three small
      boards whoever lost was made to take.

    There is no trained network for this game and none of the machinery for one,
    so the two panels that need one -- live evaluation and match review -- are
    switched off rather than offered and then failing.
    """

    key = "uxx"
    label = "Ultimate XX"
    side_names = ("Red", "Blue")
    sizes = [{"id": "9x9", "label": "9 x 9 (standard)"}]
    default_size = "9x9"
    ladder_script = "run_ladder_uxx.py"
    draws = True
    # Classical agents only: no checkpoints to pick, nothing to analyse with,
    # and nothing to review a finished game with.
    analysable = False
    uses_search = False
    reviewable = False
    review_note = ("Not available for Ultimate XX: reviewing a game means "
                   "re-searching each position with a trained network, and this "
                   "game ships with its classical opponents only.")
    default_opponent = "rule"
    fallback_opponent = "rule"

    _MINIMAX_BLURB = ("Negamax with alpha-beta, iterative deepening and a "
                      "transposition table, over a static evaluation built from "
                      "how near each side is to a third small board. The tactics "
                      "here are sharp and the game is short, which suits it.")
    _ROLLOUT_BLURB = ("UCT with playouts that refuse to step on their own losing "
                      "square unless every square is one: search with almost no "
                      "knowledge beyond the rules.")

    def new_board(self, size_id: str, seed: int | None = None):
        from alphazero_uxx.uxx_game import UltimateXXBoard

        return UltimateXXBoard()

    def size_id(self, board) -> str:
        return "9x9"

    def shape(self, board) -> tuple[int, int]:
        return board.rows, board.cols

    def move_label(self, move: int, board) -> str:
        from alphazero_uxx.uxx_game import move_to_str

        return move_to_str(move)

    def win_path(self, board) -> list[int]:
        # The marks that *lost* it.  The page colours them accordingly.
        return list(board.losing_cells())

    def score_for(self, board, colour: int) -> float:
        if board.winner == 0:
            return 0.5
        return 1.0 if board.winner == colour else 0.0

    def board_fields(self, board) -> dict:
        rows, cols = self.shape(board)
        legal = [int(m) for m in board.legal_moves()]
        mover = board.to_move
        return dict(
            cells=[int(v) for v in board.array().reshape(-1)],
            rows=rows, cols=cols, n=rows,
            legal=legal,
            active=None if board.active is None else int(board.active),
            # 0 open, 1/2 claimed by that player.  No drawn value: a small board
            # is always claimed before it can fill.
            small=[board.board_state(b) for b in range(9)],
            # The squares that close a line, and so hand their small board to
            # whoever steps on one; and the subset of those that would complete
            # the mover's own line of three boards and lose the game.
            hot=[m for m in legal if board.claims_small(m)],
            fatal=[m for m in legal if board.would_lose(mover, m)],
            # Open boards with no safe square left in them: whoever is sent
            # there takes the board.
            poisoned=[b for b in range(9) if board.is_poisoned(b)],
            lose_boards=list(board.losing_boards()),
            loser=int(board.loser),
        )

    def catalog(self) -> list[dict]:
        entries = [
            dict(id="human", group="Human", label="You (mouse)", torch=False, sims=False,
                 blurb="Click a highlighted square."),
            dict(id="random", group="Classical", label="Random", torch=False, sims=False,
                 blurb="Uniform legal square -- and since a random player steps on "
                       "its own losing square as readily as anywhere else, this is "
                       "the floor of the rating scale and then some."),
            dict(id="rule", group="Classical", label="Rule-based (minefields)",
                 torch=False, sims=False,
                 blurb="Win when the opponent has nothing safe to play, never "
                       "complete its own line of three boards, and otherwise pick "
                       "the square that leaves them the least room -- counting how "
                       "near each side is to a third board, how many harmless "
                       "squares they will have, and whether they end up free to "
                       "choose. No search at all."),
        ]
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"minimax:{budget}", group="Classical",
                                label=f"Alpha-beta brute force — {budget:g}s",
                                torch=False, sims=False, blurb=self._MINIMAX_BLURB))
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"rollout:{budget}", group="Classical",
                                label=f"Classic MCTS rollouts — {budget:g}s",
                                torch=False, sims=False, blurb=self._ROLLOUT_BLURB))
        return entries

    def build_agent(self, choice: str, seed: int, board, ckpt: str,
                    sims: int = DEFAULT_SIMS, vary: bool = False) -> Agent | None:
        from alphazero_uxx.agents.base import RandomAgent
        from alphazero_uxx.agents.mcts_rollout import RolloutMCTSAgent
        from alphazero_uxx.agents.minimax import MinimaxAgent
        from alphazero_uxx.agents.rule_based import RuleBasedAgent

        kind, _, arg = choice.partition(":")
        if kind == "human":
            return None
        if kind == "random":
            return RandomAgent(seed=seed, name="random")
        if kind == "rule":
            return RuleBasedAgent(seed=seed, noise=0.05, name="rule-based")
        if kind == "minimax":
            return MinimaxAgent(time_budget=float(arg), seed=seed,
                                name=f"alpha-beta ({float(arg):g}s)")
        if kind == "rollout":
            return RolloutMCTSAgent(simulations=200_000, time_budget=float(arg), seed=seed,
                                    name=f"mcts-rollout ({float(arg):g}s)")
        raise ValueError(f"unknown opponent {choice!r}")

    def checkpoints(self) -> list[str]:
        return []

    def analysis_checkpoint(self) -> str:
        return ""

    @property
    def R(self):
        from alphazero_uxx import ratings as R

        return R

    def ratings(self) -> dict:
        return UXX_RATINGS.get()


class IntransitiveAdapter(GameAdapter):
    """Intransitive (RPS2): chess kings on a 9x9 board, capture by rock-paper-scissors.

    The first game here whose move is not a square.  A move is a piece and one
    of eight directions, which reaches this file in four places and is the only
    thing about the adapter that is not the Ultimate one with different imports:

    * ``board_fields`` ships the legal moves as a *map from origin square to
      destinations*, because the page has to know what a half-made move can
      finish with before the server sees it.  It also ships the origin and
      destination of the last move, which the page needs to draw an arrow and
      cannot recover from the cells: a piece that moved leaves no mark behind.
    * ``parse_move`` takes ``{from, to}`` from the page and finds the direction,
      so the page never has to know the action encoding.
    * ``move_entry`` labels the move against the position *before* it, because
      whether a move is written ``e2-e3`` or ``e2xe3`` depends on what was
      standing on the destination -- and by the time the base class labels it,
      the capturing piece is standing there instead.
    * ``bot_pause`` is non-zero.  In every other game here a bot's move is a
      mark appearing; here it is a piece that was in one place and is now in
      another, and a board that changes between two frames is one nobody can
      follow.

    Draws are real -- 200 half-moves with no capture -- so the half point in
    the arena, the proven-draw state in the search and the three-way value head
    all apply as they do in Connect Four.
    """

    key = "rps2"
    label = "Intransitive"
    # Blue moves first, and the engine's FIRST/SECOND are its two sides.
    side_names = ("Blue", "Red")
    sizes = [{"id": "9x9", "label": "9 x 9 (standard)"}]
    default_size = "9x9"
    ladder_script = "run_ladder_rps2.py"
    draws = True
    ckpt_dir = ROOT / "runs" / "az_rps2"
    move_noun = "move"
    # A piece that moves needs to be seen moving.
    bot_pause = 0.35

    _MINIMAX_BLURB = ("Negamax with alpha-beta, iterative deepening and a "
                      "transposition table, over a static evaluation of matchup "
                      "material and the race to the two corners. The branching "
                      "factor is near forty, so it sees a good deal less far "
                      "here than it does at Connect Four.")
    _ROLLOUT_BLURB = ("UCT with playouts that win when they can, defend their "
                      "own base when they must, and take a free piece most of "
                      "the time -- cut off after forty plies so a shuffling "
                      "playout does not report a draw it never played for.")

    def new_board(self, size_id: str, seed: int | None = None):
        from alphazero_rps2.rps2_game import RPS2Board

        return RPS2Board()

    def size_id(self, board) -> str:
        return "9x9"

    def shape(self, board) -> tuple[int, int]:
        return board.rows, board.cols

    def position_key(self, board) -> bytes:
        # The stagnation counter is part of the position -- two identical
        # arrangements a hundred quiet plies apart are not the same thing -- and
        # the canonical board does not carry it, so the analysis thread would
        # not otherwise notice a take-back that changed only the counter.
        return (board.canonical_board().tobytes()
                + bytes([board.since_capture & 0xFF, board.since_capture >> 8]))

    def move_label(self, move: int, board) -> str:
        from alphazero_rps2.rps2_game import move_to_str

        return move_to_str(move, board)

    def move_entry(self, move, before, after, by: str) -> dict:
        from alphazero_rps2.rps2_game import action_squares

        src, dst = action_squares(int(move))
        entry = dict(cell=int(move), player=before.to_move, by=by,
                     # Labelled against the position the move was played in, so
                     # a capture is written with an "x".
                     label=self.move_label(int(move), before),
                     captured=int(before.cells[dst]))
        entry["from"] = src
        entry["to"] = dst
        return entry

    def parse_move(self, payload: dict, board):
        """``{from, to}`` from the page -> an action index."""
        from alphazero_rps2.rps2_game import NCELLS, NDIRS, STEP_LIST

        src, dst = payload.get("from"), payload.get("to")
        if src is None or dst is None:
            # A bare ``cell`` is still accepted, so an action index posted by a
            # script or replayed from a saved game works without translation.
            cell = payload.get("cell")
            return None if cell is None else int(cell)
        src, dst = int(src), int(dst)
        if not (0 <= src < NCELLS and 0 <= dst < NCELLS):
            return None
        for d in range(NDIRS):
            if STEP_LIST[src][d] == dst:
                return d * NCELLS + src
        return None

    def win_path(self, board) -> list[int]:
        return list(board.win_path())

    def board_fields(self, board) -> dict:
        from alphazero_rps2.rps2_game import (BLUE_BASE, RED_BASE,
                                              STAGNATION_LIMIT, action_squares)

        rows, cols = self.shape(board)
        # Origin -> the squares that piece may move to.  The page needs this to
        # light up destinations on the first click of a two-click move, and it
        # cannot derive it: legality depends on the capture cycle, not on
        # emptiness.
        # Named ``legal_from`` and not ``moves``: the session already puts the
        # move *history* on the state under that name, and the later key wins.
        legal_from: dict[str, list[int]] = {}
        for m in board.legal_moves():
            src, dst = action_squares(int(m))
            legal_from.setdefault(str(src), []).append(dst)
        last_from = last_to = None
        if board.last_move is not None:
            last_from, last_to = action_squares(int(board.last_move))
        return dict(
            # 0 empty, 1-3 blue rock/paper/scissors, 4-6 red's.
            cells=[int(v) for v in board.array().reshape(-1)],
            rows=rows, cols=cols, n=rows,
            legal_from=legal_from,
            bases={"first": BLUE_BASE, "second": RED_BASE},
            last_from=last_from, last_to=last_to,
            # The no-capture draw, and how close it is.  The board shows a
            # warning once it is within thirty moves, which is what the original
            # client does and what makes the rule discoverable at all.
            quiet=int(board.since_capture),
            # How many times this exact position has occurred; three is a draw,
            # and a player cannot count that off the board any more than they
            # can count the quiet plies.
            repetitions=int(board.repetitions()),
            stagnation_in=int(board.moves_to_stagnation()),
            stagnation_limit=STAGNATION_LIMIT,
            end_reason=board.end_reason,
        )

    def catalog(self) -> list[dict]:
        entries = [
            dict(id="human", group="Human", label="You (mouse)", torch=False, sims=False,
                 blurb="Click one of your pieces, then a highlighted square."),
            dict(id="random", group="Classical", label="Random", torch=False, sims=False,
                 blurb="Uniform legal move -- the floor of the rating scale."),
            dict(id="rule", group="Classical", label="Rule-based (matchup + base)",
                 torch=False, sims=False,
                 blurb="Run into their base when it is reachable, and stop you "
                       "doing the same -- by taking the piece next to your base, "
                       "or by standing on the base with something that piece "
                       "cannot capture. Otherwise the best move by matchup-priced "
                       "material, progress towards the far corner, and refusing "
                       "squares where something that beats it is waiting. "
                       "No search at all."),
        ]
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"minimax:{budget}", group="Classical",
                                label=f"Alpha-beta brute force \u2014 {budget:g}s",
                                torch=False, sims=False, blurb=self._MINIMAX_BLURB))
        for budget in (0.5, 1.0, 3.0):
            entries.append(dict(id=f"rollout:{budget}", group="Classical",
                                label=f"Classic MCTS rollouts \u2014 {budget:g}s",
                                torch=False, sims=False, blurb=self._ROLLOUT_BLURB))
        entries.append(dict(id="az", group="AlphaZero", label="AlphaZero (network + search)",
                            torch=True, sims=True,
                            blurb="PUCT search guided by the trained network. The "
                                  "slider sets simulations per move; at 0 it is the "
                                  "bare policy head."))
        return entries

    def build_agent(self, choice: str, seed: int, board, ckpt: str,
                    sims: int = DEFAULT_SIMS, vary: bool = False) -> Agent | None:
        from alphazero_rps2.agents.base import RandomAgent
        from alphazero_rps2.agents.mcts_rollout import RolloutMCTSAgent
        from alphazero_rps2.agents.minimax import MinimaxAgent
        from alphazero_rps2.agents.rule_based import RuleBasedAgent

        kind, _, arg = choice.partition(":")
        if kind == "human":
            return None
        if kind == "random":
            return RandomAgent(seed=seed, name="random")
        if kind == "rule":
            return RuleBasedAgent(seed=seed, noise=0.05, name="rule-based")
        if kind == "minimax":
            return MinimaxAgent(time_budget=float(arg), seed=seed,
                                name=f"alpha-beta ({float(arg):g}s)")
        if kind == "rollout":
            return RolloutMCTSAgent(simulations=200_000, time_budget=float(arg), seed=seed,
                                    name=f"mcts-rollout ({float(arg):g}s)")
        if kind == "az":
            from alphazero_rps2.registry import build_agent as from_spec  # needs torch
            from alphazero_rps2.registry import spec

            path = self.ckpt_path(ckpt)
            dev = getattr(self, "device", DEVICE)
            if int(sims) <= 0:
                s = spec("policy", "alphazero policy (no search)", ckpt=path,
                         temperature=VARY_TEMPERATURE if vary else 0.0, device=dev)
            else:
                s = spec("az", f"alphazero ({sims} sims)", ckpt=path, simulations=int(sims),
                         explore_moves=VARY_EXPLORE_MOVES if vary else 0,
                         explore_temperature=VARY_TEMPERATURE, device=dev)
            return from_spec(s, seed, board.rows, board.cols)
        raise ValueError(f"unknown opponent {choice!r}")

    def evaluator(self, ckpt: str, board):
        from alphazero_rps2.registry import evaluator  # needs torch

        dev = getattr(self, "device", DEVICE)
        return evaluator(self.ckpt_path(ckpt), board.rows, board.cols, device=dev)

    def readout(self, search, board) -> dict:
        from alphazero_rps2.review import readout

        return readout(search, board)

    def terminal_readout(self, board) -> dict:
        from alphazero_rps2.review import terminal_readout

        return terminal_readout(board)

    def review_tasks(self, moves, size_id, ckpt_path, sims) -> list[tuple]:
        shape = (9, 9)
        return [(moves[:i], shape, ckpt_path, sims, i) for i in range(len(moves) + 1)]

    @property
    def evaluate_position(self):
        from alphazero_rps2.review import evaluate_position

        return evaluate_position

    @property
    def R(self):
        from alphazero_rps2 import ratings as R

        return R

    def ratings(self) -> dict:
        return RPS2_RATINGS.get()


class BackgammonAdapter(GameAdapter):
    """Backgammon: dice, move sequences, and no tree to search.

    Three of the base class's assumptions are false here and each override says
    so: a turn starts with a roll the server makes, a move is a tuple of hops
    rather than a cell, and the analysis is afterstate evaluation instead of
    MCTS.  Match review is switched off -- it is built around re-searching a
    position, and re-searching a *roll* is a different thing that would need its
    own equity-loss accounting.
    """

    key = "bg"
    label = "Backgammon"
    side_names = ("Red", "White")
    sizes = [{"id": "standard", "label": "Standard (15 checkers)"}]
    default_size = "standard"
    ladder_script = "run_ladder_bg.py"
    move_noun = "move"
    draws = False
    dice = True
    uses_search = False
    reviewable = False
    ckpt_dir = ROOT / "runs" / "az_bg"
    # The strength dial is search depth, not simulations: one ply reads this
    # roll's moves, two averages the reply over all 21 of the opponent's.
    strength_stops = [1, 2]
    strength_label = "ply"
    default_strength = 1
    default_opponent = "net"
    fallback_opponent = "heuristic"

    _HEURISTIC_BLURB = ("Pip count, blot exposure, points made and checkers off -- "
                        "the terms a beginner is taught, and no search past this roll.")
    _NET_BLURB = ("The trained value network. At one ply it scores the positions this "
                  "roll can reach; at two it also averages the opponent's best reply "
                  "over all 21 rolls, which is much stronger and much slower.")

    # ------------------------------------------------------------------ board
    def new_board(self, size_id: str, seed: int | None = None):
        from alphazero_bg.bg_game import Backgammon

        return Backgammon()

    def size_id(self, board) -> str:
        return "standard"

    def shape(self, board) -> tuple[int, int]:
        # Nothing renders a grid for this game; the page draws points, a bar and
        # two trays.  Reported so the field exists for anything generic.
        return 2, 12

    def position_key(self, board) -> bytes:
        return board.signature() + bytes(board.dice)

    def move_label(self, move, board) -> str:
        from alphazero_bg.bg_game import move_to_str

        return move_to_str(tuple(move), board.to_move)

    def win_path(self, board) -> list[int]:
        return []  # a backgammon win is fifteen checkers off, not a line

    def format_value(self, value: float) -> str:
        # Equity is carried as points/3 -- a plain win is 1/3 -- and shown in
        # points, which is the unit the game is actually scored in.
        return f"{value * 3:+.2f} points"

    def score_for(self, board, colour: int) -> float:
        return 1.0 if board.winner == colour else 0.0

    # ------------------------------------------------------------- the dice
    def needs_roll(self, board) -> bool:
        return board.needs_roll()

    def roll(self, board, rng) -> list[int]:
        return list(board.roll_dice(rng))

    # -------------------------------------------------------------- moving
    def select(self, agent, board, last_move):
        return agent.select_move(board)

    def is_legal(self, board, move) -> bool:
        if move is None:
            return False
        return tuple(move) in {tuple(m) for m in board.legal_moves()}

    def apply(self, board, move) -> None:
        board.play(tuple(move))

    def parse_move(self, payload: dict, board):
        """Validate a sequence of hops the page built up, click by click.

        The page lets you play the hops in whatever order feels natural, which
        need not be the order the engine happens to store, and two different
        orders can even use different intermediate points to reach the same
        place.  So rather than matching the sequence, this replays it -- checking
        every hop against the rules and the dice still in hand -- and then asks
        whether the position it reached is one of the legal moves' positions.
        """
        raw = payload.get("hops")
        if raw is None:
            return None
        try:
            hops = [(int(a), int(b)) for a, b in raw]
        except (TypeError, ValueError):
            return None

        state = board.copy()
        remaining = list(board.dice)
        for hop in hops:
            # Usually the hop names its die: the distance it covers.  Bearing
            # off is the exception -- a checker closer in than the die shown
            # still comes off on it, so a hop whose exact die is not in hand
            # takes the smallest one that is large enough.
            wanted = state.hop_die(hop)
            die = wanted if wanted in remaining else min(
                (d for d in remaining if d > wanted), default=None)
            if die is None:
                return None
            if hop not in state._hops_for(die, state.to_move):
                return None
            state.apply_hop(hop)
            remaining.remove(die)

        target = state.signature()
        for move in board.legal_moves():
            if board.after(move).signature() == target:
                return move
        return None

    def move_entry(self, move, before, after, by: str) -> dict:
        from alphazero_bg.bg_game import move_to_str

        # The raw pair is stored, not the expanded four-of-a-kind, because that
        # is what re-rolling the same turn on a take-back needs.
        dice = list(before.dice)
        roll = [dice[0], dice[0]] if len(dice) == 4 else dice[:2]
        return dict(cell=None, hops=[list(h) for h in move], roll=roll,
                    dice=dice, player=before.to_move, by=by,
                    label=move_to_str(tuple(move), before.to_move))

    def replay_move(self, board, entry: dict) -> None:
        roll = entry.get("roll") or [1, 1]
        board.set_dice(int(roll[0]), int(roll[1]))
        board.play(tuple(tuple(h) for h in entry["hops"]))

    def restore_roll(self, board, entry: dict) -> bool:
        roll = entry.get("roll")
        if not roll or board.is_terminal():
            return False
        board.set_dice(int(roll[0]), int(roll[1]))
        return True

    # ------------------------------------------------------------ the state
    def board_fields(self, board) -> dict:
        from alphazero_bg.bg_game import BLACK, POINTS, WHITE, move_to_str

        options = []
        if not board.is_terminal() and board.dice:
            for index, move in enumerate(board.legal_moves()):
                options.append(dict(index=index, hops=[list(h) for h in move],
                                    label=move_to_str(move, board.to_move)))
        return dict(
            rows=2, cols=12, n=2,
            # Positive counts are the first player's, negative the second's --
            # the page reads the sign, exactly as the engine does.
            points=[int(v) for v in board.points],
            bar=[board.bar[BLACK], board.bar[WHITE]],
            off=[board.off[BLACK], board.off[WHITE]],
            dice=list(board.dice),
            pips=[board.pip_count(BLACK), board.pip_count(WHITE)],
            options=options,
            points_won=board.points_won(),
            cells=[],  # the cell games' field, empty so the page can tell them apart
        )

    # ----------------------------------------------------------------- agents
    def catalog(self) -> list[dict]:
        return [
            dict(id="human", group="Human", label="You (mouse)", torch=False,
                 sims=False, blurb="Click a checker, then where it should go."),
            dict(id="random", group="Classical", label="Random", torch=False,
                 sims=False, blurb="A legal move at random -- the floor of the scale."),
            dict(id="heuristic", group="Classical", label="Heuristic (pips + blots)",
                 torch=False, sims=False, blurb=self._HEURISTIC_BLURB),
            dict(id="net", group="Trained", label="Network (afterstate evaluation)",
                 torch=True, sims=True, blurb=self._NET_BLURB),
        ]

    def build_agent(self, choice: str, seed: int, board, ckpt: str,
                    sims: int = 1, vary: bool = False):
        from alphazero_bg.agents.base import RandomAgent
        from alphazero_bg.agents.heuristic import HeuristicAgent

        kind, _, _arg = choice.partition(":")
        if kind == "human":
            return None
        if kind == "random":
            return RandomAgent(seed=seed, name="random")
        if kind == "heuristic":
            return HeuristicAgent(seed=seed, noise=0.3, name="heuristic")
        if kind == "net":
            from alphazero_bg.registry import build_agent as from_spec  # needs torch
            from alphazero_bg.registry import spec

            plies = 2 if int(sims) >= 2 else 1
            path = str(self.ckpt_dir / ckpt)
            s = spec("net", f"network ({plies}-ply)", ckpt=path, plies=plies,
                     explore=VARY_TEMPERATURE if vary else 0.0)
            return from_spec(s, seed)
        raise ValueError(f"unknown opponent {choice!r}")

    def evaluator(self, ckpt: str, board):
        from alphazero_bg.registry import evaluator  # needs torch

        return evaluator(str(self.ckpt_dir / ckpt))

    # --------------------------------------------------------------- analysis
    def analysis_stream(self, board, ckpt):
        """One pass at one ply, then one at two, and then nothing more.

        There is no tree to grow: with the roll already known the candidate set
        is fixed, so "thinking longer" means looking a ply deeper rather than
        widening a search.  Two passes is all there is.
        """
        from alphazero_bg.bg_game import BLACK

        if board.is_terminal():
            yield dict(plies=board.move_count, sims=0, nodes=0, depth=0,
                       value_black=board.result_for(BLACK), to_move=board.to_move,
                       top=[], proven=True, terminal=True)
            return
        if not board.dice:
            return
        engine = self.evaluator(ckpt, board)
        moves, equities = engine.move_equities(board)
        yield self._readout(board, moves, equities, depth=1)
        if len(moves) > 1:
            moves, equities = engine.move_equities_two_ply(board, candidates=8)
            yield self._readout(board, moves, equities, depth=2)

    def _readout(self, board, moves, equities, depth: int) -> dict:
        from alphazero_bg.bg_game import BLACK, move_to_str

        order = np.argsort(-equities)[:8]
        best = float(equities[order[0]]) if len(order) else 0.0
        top = [dict(cell=int(i), label=move_to_str(moves[int(i)], board.to_move),
                    q=float(equities[int(i)]), visits=None, share=None,
                    hops=[list(h) for h in moves[int(i)]], proven=False)
               for i in order]
        return dict(
            plies=board.move_count,
            sims=depth, nodes=len(moves), depth=depth,
            value_black=best if board.to_move == BLACK else -best,
            to_move=board.to_move,
            top=top, proven=False, terminal=False,
        )

    # ---------------------------------------------------------------- ratings
    @property
    def R(self):
        from alphazero_bg import ratings as R

        return R

    def ratings(self) -> dict:
        return BG_RATINGS.get()

class SplendorAdapter(GameAdapter):
    """Splendor: two to four seats, shuffled decks, and a fixed action list.

    Four of the base class's assumptions are false here.  A table can seat more
    than two, so ``seat_ids`` returns as many seats as the size says and the
    session keys everything by them.  A move is an *action index* rather than a
    cell.  The setup shuffles, so ``new_board`` takes the deal seed the session
    holds -- without it a take-back would redeal.  And the analysis is a search
    that cannot be re-rooted between plies (the deck may have turned up anything
    in between), so it runs as a stream that deepens in place, the same hook
    backgammon uses for a different reason.

    Match review is off.  It is built around re-searching each position of a
    finished game and pricing what a move threw away; re-searching a position
    whose deck was shuffled differently is a different question, and one that
    would need its own accounting to answer honestly.
    """

    key = "splendor"
    label = "Splendor"
    side_names = ("Player 1", "Player 2", "Player 3", "Player 4")
    min_players = 2
    max_players = 4
    sizes = [{"id": str(n), "label": f"{n} players"} for n in (2, 3, 4)]
    default_size = "2"
    ladder_script = "run_ladder_splendor.py"
    move_noun = "move"
    draws = True          # a tie on prestige *and* cards is a real result
    dice = False
    uses_search = False   # ...but only because the shared tree is not this one
    reviewable = False
    ckpt_dir = ROOT / "runs" / "az_splendor"
    # The v2 network keeps its own directory, because its checkpoints are not
    # loadable by the v1 code: it is a different encoding, and a network fed the
    # wrong feature set does not fail, it plays badly.  The checkpoint dropdown
    # applies to the v1 opponent; the v2 one always uses the strongest network
    # it has, which is the one the training gate promoted.
    v2_ckpt_dir = ROOT / "runs" / "az_splendor2"
    # ...and v3 its own again, for the same reason: a different feature set
    # and a different search.  Its checkpoint is picked by iteration rather
    # than by name, because that run's gate stopped promoting while the
    # network kept improving, so ``best.pt`` is not its best network.
    v3_ckpt_dir = ROOT / "runs" / "az_splendor3"
    # Simulations, as in the two board games -- but a Splendor game is ~60
    # plies at two seats and ~115 at four, and every simulation re-samples the
    # deck, so the ceiling is lower than theirs.
    strength_stops = [0, 25, 50, 100, 200, 400, 800, 1600]
    strength_label = "sims"
    default_strength = 200
    fallback_opponent = "planner"
    # A Splendor move rearranges several parts of the board at once -- a card
    # leaves, another turns over, tokens move, a noble may arrive -- and the
    # cheap agents answer in well under a millisecond.  Without a floor the
    # whole thing happens between two frames and nobody can follow it.
    bot_pause = 0.9

    _HEURISTIC_BLURB = ("Buys whatever is worth most right now: prestige, the "
                        "discount a bonus gives, and how close it lands you to a "
                        "noble. One move deep, no search.")
    _PLANNER_BLURB = ("Knows what it is saving for: it scores a position by how "
                      "many turns of collecting stand between it and each card "
                      "worth buying, so its gems are the ones it can spend. "
                      "About 120 Elo above the greedy agent, and no network.")
    _AZ_BLURB = ("The trained network with max^n PUCT search. Every simulation "
                 "re-shuffles the cards it has not been shown, so it plans "
                 "against the deck rather than around it.")
    _AZ3_BLURB = ("The v3 network. Cards and colours are tokens rather than "
                  "fixed slots, so what it knows about a card follows the card "
                  "around the board; it searches with Gumbel sequential "
                  "halving instead of PUCT, and it beats v2 98 games in 100.")
    _AZ2_BLURB = ("The v2 network. It reads the board the way a player does -- "
                  "which colours each card is short of, how many turns away it "
                  "is, what an opponent could take from you -- on a game core "
                  "four times quicker, and it starts from the planner rather "
                  "than from nothing.")

    # ------------------------------------------------------- the v2 network
    @property
    def default_opponent(self) -> str:
        """The v2 network once one has been trained, the v1 one until then.

        A property rather than a constant because the answer depends on what is
        on disk: naming an opponent the catalog does not offer would leave a
        fresh checkout unable to build any agent at all.
        """
        return "az2" if self.v2_checkpoint() else "az"

    def v3_checkpoint(self) -> str | None:
        """Path to the strongest v3 network on disk, by iteration, or ``None``.

        Deliberately *not* ``best.pt`` first.  The v3 run's gate asked "is this
        50 Elo better than the current best?" every ten iterations while the
        real steps were worth about fifteen, so it stopped promoting at
        iteration 150 and the network went on improving for another fifty --
        ``latest.pt`` measured 73 elo above ``best.pt`` over 140 games.  So the
        checkpoints are read and the latest *iteration* wins, whatever it is
        called.
        """
        if not self.v3_ckpt_dir.is_dir():
            return None
        import torch

        best, best_it = None, -1
        for path in sorted(self.v3_ckpt_dir.glob("*.pt")):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
                it = int(blob.get("extra", {}).get("iteration", 0))
            except Exception:
                continue
            if it > best_it:
                best, best_it = str(path), it
        return best

    def v2_checkpoint(self) -> str | None:
        """Path to the best v2 network on disk, or ``None`` if there is none.

        ``best.pt`` first, because that is the one that actually beat its
        predecessor in a match rather than merely being saved last.
        """
        if not self.v2_ckpt_dir.is_dir():
            return None
        for name in ("best.pt", "final.pt", "latest.pt"):
            path = self.v2_ckpt_dir / name
            if path.is_file():
                return str(path)
        return None

    # ------------------------------------------------------------- the seats
    def seat_ids(self, board) -> list[int]:
        return list(range(1, board.players + 1))

    def seat_names(self, board) -> list[str]:
        return [f"Player {i}" for i in range(1, board.players + 1)]

    def seat_count(self, size_id: str) -> int:
        try:
            return max(2, min(4, int(size_id)))
        except (TypeError, ValueError):
            return 2

    # ------------------------------------------------------------------ board
    def new_board(self, size_id: str, seed: int | None = None):
        import numpy as _np

        from alphazero_splendor.splendor_game import SplendorGame

        return SplendorGame(self.seat_count(size_id), _np.random.default_rng(seed))

    def size_id(self, board) -> str:
        return str(board.players)

    def rating_size_for(self, size_id: str):
        return f"{self.seat_count(size_id)} players"

    def shape(self, board) -> tuple[int, int]:
        # The face-up display: three tiers of four.  Nothing draws a grid of
        # cells, but the field exists for anything generic.
        return 3, 4

    def position_key(self, board) -> bytes:
        return board.signature()

    def win_path(self, board) -> list[int]:
        return []  # a Splendor win is fifteen prestige, not a line

    def score_for(self, board, colour: int) -> float:
        return board.score_for(colour)

    # -------------------------------------------------------------- moving
    def select(self, agent, board, last_move):
        return agent.select_move(board, last_move)

    def is_legal(self, board, move) -> bool:
        return move is not None and board.is_legal(move)

    def apply(self, board, move) -> None:
        board.play(int(move))

    def parse_move(self, payload: dict, board):
        """The page sends an action index; ``cell`` is the field it already had."""
        raw = payload.get("action", payload.get("cell"))
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def move_label(self, move: int, board) -> str:
        return board.move_label(int(move))

    def move_entry(self, move, before, after, by: str) -> dict:
        # Labelled from the position *before* the move: "buy T2 green" needs to
        # know which card was in the slot, and by now it has been replaced.
        return dict(cell=int(move), player=before.to_move, by=by,
                    label=before.move_label(int(move)),
                    motion=self.motion(before, int(move)))

    def motion(self, before, move: int) -> dict:
        """What this move *moves*, for the page to animate.

        Described here rather than worked out in the browser for the same reason
        every clickable thing carries its own action id: the action space lives
        on this side, and a second copy of its layout in JavaScript is the sort
        of duplication that goes quietly wrong the first time the space changes.
        The page gets "these gems, from the bank, to seat 2" and animates that
        without knowing what action 17 is.
        """
        from alphazero_splendor.cards import COST, GOLD
        from alphazero_splendor.splendor_game import (A_BUY, A_BUY_RESERVED,
                                                      A_DISCARD, A_NOBLE, A_PASS,
                                                      A_RESERVE, A_RESERVE_DECK,
                                                      A_TAKE1, A_TAKE2D, A_TAKE2S,
                                                      A_TAKE3, TAKE2D, TAKE3)
        import numpy as _np

        seat = before.to_move

        def paid(card_id: int) -> dict:
            """Which tokens leave the buyer's hand for the bank."""
            need = _np.maximum(COST[card_id] - before.bonus[seat], 0)
            spend = _np.minimum(need, before.hand[seat, :5])
            return dict(gems=[int(v) for v in spend],
                        gold=int((need - spend).sum()))

        if A_TAKE3 <= move < A_TAKE2D:
            return dict(kind="take", seat=seat, gems=list(TAKE3[move - A_TAKE3]))
        if A_TAKE2D <= move < A_TAKE1:
            return dict(kind="take", seat=seat, gems=list(TAKE2D[move - A_TAKE2D]))
        if A_TAKE1 <= move < A_TAKE2S:
            return dict(kind="take", seat=seat, gems=[move - A_TAKE1])
        if A_TAKE2S <= move < A_BUY:
            colour = move - A_TAKE2S
            return dict(kind="take", seat=seat, gems=[colour, colour])
        if A_BUY <= move < A_BUY_RESERVED:
            slot = move - A_BUY
            card = int(before.board[slot])
            return dict(kind="buy", seat=seat, slot=slot, card=card, **paid(card))
        if A_BUY_RESERVED <= move < A_RESERVE:
            index = move - A_BUY_RESERVED
            held = before.reserved[seat]
            card = int(held[index]) if index < len(held) else -1
            return dict(kind="buy_reserved", seat=seat, index=index, card=card,
                        **(paid(card) if card >= 0 else dict(gems=[], gold=0)))
        if A_RESERVE <= move < A_RESERVE_DECK:
            slot = move - A_RESERVE
            return dict(kind="reserve", seat=seat, slot=slot,
                        card=int(before.board[slot]),
                        gold=1 if before.bank[GOLD] > 0 else 0)
        if A_RESERVE_DECK <= move < A_DISCARD:
            tier = move - A_RESERVE_DECK
            return dict(kind="reserve_deck", seat=seat, tier=tier,
                        gold=1 if before.bank[GOLD] > 0 else 0)
        if A_DISCARD <= move < A_NOBLE:
            return dict(kind="discard", seat=seat, gems=[move - A_DISCARD])
        if A_NOBLE <= move < A_PASS:
            index = move - A_NOBLE
            nobles = before.nobles
            return dict(kind="noble", seat=seat, index=index,
                        noble=int(nobles[index]) if index < len(nobles) else -1)
        return dict(kind="pass", seat=seat)

    def replay_move(self, board, entry: dict) -> None:
        board.play(int(entry["cell"]))

    # ------------------------------------------------------------ the state
    def board_fields(self, board) -> dict:
        from alphazero_splendor.cards import CARDS, GOLD, NOBLES
        from alphazero_splendor.splendor_game import (A_BUY, A_BUY_RESERVED,
                                                      A_DISCARD, A_NOBLE, A_PASS,
                                                      A_RESERVE, A_RESERVE_DECK,
                                                      A_TAKE1, A_TAKE2D, A_TAKE2S,
                                                      A_TAKE3, MAX_RESERVED,
                                                      PHASE_MAIN, ROW, TAKE2D,
                                                      TAKE3, TIERS)

        mover = board.to_move or 1
        legal_mask = board.legal_mask()

        def act(action: int):
            """The action id if it is legal here, else ``None``.

            Every clickable thing on the page carries its own action this way,
            so the browser never has to know how the action space is laid out --
            which is the sort of duplication that goes wrong silently the first
            time the space changes.
            """
            return int(action) if legal_mask[action] else None

        def card_blob(card_id: int, owner: int | None = None) -> dict | None:
            """One card as the page draws it, or ``None`` for an empty slot.

            ``owner`` names whose purse to price it against -- the player to
            move for a card on the table, its holder for a reserved one, since
            what matters about an opponent's reserve is whether *they* can pay.
            """
            if card_id is None or card_id < 0:
                return None
            card = CARDS[card_id]
            seat = mover if owner is None else owner
            short = board.token_shortfall(seat, card_id)
            gold = int(board.hand[seat][GOLD])
            # Per colour, how many of that gem are still missing after bonuses
            # and tokens.  The page paints a cost pip as covered or not from
            # this, which is what makes "can I buy this, and what for" readable
            # off the card itself rather than by counting.
            need = np.maximum(np.array(card.cost) - board.bonus[seat], 0)
            lack = np.maximum(need - board.hand[seat][:len(need)], 0)
            return dict(id=int(card_id), tier=card.tier, gem=card.gem,
                        points=card.points, cost=list(card.cost),
                        # What this actually costs *you*, bonuses taken off.
                        need=[int(v) for v in need],
                        short=[int(v) for v in lack],
                        affordable=short <= gold,
                        # Bought outright, or only by spending gold on the gap.
                        with_gold=0 < short <= gold,
                        # What the page shows on the card: the gold this would
                        # eat if bought now, or how many tokens are still
                        # missing when it cannot be.
                        gold_needed=min(short, gold),
                        missing=max(short - gold, 0))

        # Named ``seat_state`` rather than ``seats``: the snapshot already has a
        # ``seats`` -- who is *playing* each seat -- and this is what each seat
        # owns.  The two are merged into one object on the page.
        seat_state = []
        for seat in range(1, board.players + 1):
            reserved = []
            for i in range(MAX_RESERVED):
                if i >= len(board.reserved[seat]):
                    continue
                card_id = int(board.reserved[seat][i])
                secret = bool(board.hidden[seat][i])
                # A card reserved off a deck is the owner's business.  The page
                # is served to whoever is sitting at the machine, so it is shown
                # only when that seat is the one on turn -- the same rule the
                # network plays under.
                visible = (seat == mover) or not secret
                blob = card_blob(card_id if visible else -1, owner=seat)
                reserved.append(dict(index=i, hidden=secret, card=blob,
                                     tier=CARDS[card_id].tier,
                                     buy=act(A_BUY_RESERVED + i) if seat == mover
                                     else None))
            seat_state.append(dict(
                seat=seat,
                tokens=[int(v) for v in board.hand[seat]],
                bonus=[int(v) for v in board.bonus[seat]],
                points=int(board.points[seat]),
                cards=int(board.ncards[seat]),
                reserved=reserved,
                # Which of this seat's tokens can be handed back, while it is
                # over the ten-token limit.
                discard=[act(A_DISCARD + g) if seat == mover else None
                         for g in range(len(board.hand[seat]))],
            ))

        rows_of_cards = []
        for tier in range(TIERS):
            row = []
            for i in range(ROW):
                slot = tier * ROW + i
                blob = card_blob(int(board.board[slot]))
                if blob is not None:
                    blob["buy"] = act(A_BUY + slot)
                    blob["reserve"] = act(A_RESERVE + slot)
                row.append(blob)
            rows_of_cards.append(row)

        # Every legal way to take tokens, as the set of colours it takes.  The
        # page collects a set of clicks and looks it up, rather than working out
        # which of the four kinds of take it has assembled.
        takes = []
        for i, trio in enumerate(TAKE3):
            if legal_mask[A_TAKE3 + i]:
                takes.append(dict(gems=list(trio), action=A_TAKE3 + i))
        for i, pair in enumerate(TAKE2D):
            if legal_mask[A_TAKE2D + i]:
                takes.append(dict(gems=list(pair), action=A_TAKE2D + i))
        for g in range(len(board.bank) - 1):
            if legal_mask[A_TAKE1 + g]:
                takes.append(dict(gems=[g], action=A_TAKE1 + g))
            if legal_mask[A_TAKE2S + g]:
                takes.append(dict(gems=[g, g], action=A_TAKE2S + g))

        legal = [int(m) for m in board.legal_moves()]
        return dict(
            rows=TIERS, cols=ROW, n=TIERS,
            players=board.players,
            bank=[int(v) for v in board.bank],
            rows_of_cards=rows_of_cards,
            decks=[dict(tier=t, size=len(board.decks[t]),
                        reserve=act(A_RESERVE_DECK + t)) for t in range(TIERS)],
            nobles=[dict(id=int(n), need=list(NOBLES[n].need), points=3,
                         take=act(A_NOBLE + i))
                    for i, n in enumerate(board.nobles)],
            seat_state=seat_state,
            phase=board.phase,
            pending_nobles=[int(n) for n in board.pending],
            triggered=bool(board.triggered),
            standings=board.standings() if board.is_terminal() else [],
            takes=takes,
            pass_action=act(A_PASS),
            legal=legal,
            # Every legal move with its name, so the page can offer the ones
            # that are not a click on a card (taking tokens, passing) and can
            # explain why a card is greyed out.
            options=[dict(action=m, label=board.move_label(m)) for m in legal],
            main_phase=board.phase == PHASE_MAIN,
            cells=[],  # the cell games' field, empty so the page can tell them apart
        )

    # ----------------------------------------------------------------- agents
    def catalog(self) -> list[dict]:
        out = [
            dict(id="human", group="Human", label="You (mouse)", torch=False,
                 sims=False, blurb="Click a card to buy or reserve it, or the "
                                   "tokens you want."),
            dict(id="random", group="Classical", label="Random", torch=False,
                 sims=False, blurb="A legal move at random -- the floor of the scale."),
            dict(id="heuristic", group="Classical", label="Heuristic (greedy expert)",
                 torch=False, sims=False, blurb=self._HEURISTIC_BLURB),
            dict(id="planner", group="Classical", label="Planner (plans purchases)",
                 torch=False, sims=False, blurb=self._PLANNER_BLURB),
            dict(id="az", group="Trained", label="AlphaZero v1 (network + search)",
                 torch=True, sims=True, blurb=self._AZ_BLURB),
        ]
        if self.v2_checkpoint():
            out.append(dict(id="az2", group="Trained",
                            label="AlphaZero v2 (network + search)",
                            torch=True, sims=True, blurb=self._AZ2_BLURB))
        if self.v3_checkpoint():
            out.append(dict(id="az3", group="Trained",
                            label="AlphaZero v3 (network + search)",
                            torch=True, sims=True, blurb=self._AZ3_BLURB))
        return out

    def build_agent(self, choice: str, seed: int, board, ckpt: str,
                    sims: int = 200, vary: bool = False):
        from alphazero_splendor.agents.base import RandomAgent
        from alphazero_splendor.agents.heuristic import HeuristicAgent

        kind, _, _arg = choice.partition(":")
        if kind == "human":
            return None
        if kind == "random":
            return RandomAgent(seed=seed, name="random")
        if kind == "heuristic":
            return HeuristicAgent(seed=seed, noise=0.15, name="heuristic")
        if kind == "planner":
            from alphazero_splendor.agents.planner import PlannerAgent

            return PlannerAgent(depth=1, beam=8, samples=2, seed=seed,
                                name="planner")
        if kind == "az3":
            from alphazero_splendor3.registry import build_agent as from_spec3
            from alphazero_splendor3.registry import spec as spec3

            path = self.v3_checkpoint()
            if path is None:
                raise ValueError("no v3 checkpoint has been trained yet")
            if int(sims) <= 0:
                s = spec3("policy", "v3 policy (no search)", ckpt=path,
                          temperature=VARY_TEMPERATURE if vary else 0.0)
            else:
                # The Gumbel search has no Dirichlet noise and no temperature:
                # its exploration is the root sample, so "vary" is a non-zero
                # gumbel scale rather than a separate mechanism.
                s = spec3("az", f"alphazero v3 ({int(sims)} sims)", ckpt=path,
                          simulations=int(sims),
                          gumbel_scale=1.0 if vary else 0.0)
            return from_spec3(s, seed)
        if kind == "az2":
            from alphazero_splendor2.registry import build_agent as from_spec2
            from alphazero_splendor2.registry import spec as spec2

            path = self.v2_checkpoint()
            if path is None:
                raise ValueError("no v2 checkpoint has been trained yet")
            if int(sims) <= 0:
                s = spec2("policy", "v2 policy (no search)", ckpt=path,
                          temperature=VARY_TEMPERATURE if vary else 0.0)
            else:
                s = spec2("az", f"alphazero v2 ({int(sims)} sims)", ckpt=path,
                          simulations=int(sims),
                          explore_moves=VARY_EXPLORE_MOVES if vary else 0,
                          explore_temperature=VARY_TEMPERATURE)
            return from_spec2(s, seed)
        if kind == "az":
            from alphazero_splendor.registry import build_agent as from_spec  # torch
            from alphazero_splendor.registry import spec

            path = str(self.ckpt_dir / ckpt)
            dev = getattr(self, "device", DEVICE)
            if int(sims) <= 0:
                s = spec("policy", "policy (no search)", ckpt=path,
                         temperature=VARY_TEMPERATURE if vary else 0.0, device=dev)
            else:
                s = spec("az", f"alphazero ({int(sims)} sims)", ckpt=path,
                         simulations=int(sims),
                         explore_moves=VARY_EXPLORE_MOVES if vary else 0,
                         explore_temperature=VARY_TEMPERATURE, device=dev)
            return from_spec(s, seed)
        raise ValueError(f"unknown opponent {choice!r}")

    def evaluator(self, ckpt: str, board):
        from alphazero_splendor.registry import evaluator  # needs torch

        dev = getattr(self, "device", DEVICE)
        return evaluator(str(self.ckpt_dir / ckpt), device=dev)

    # --------------------------------------------------------------- analysis
    def analysis_stream(self, board, ckpt):
        """One max^n tree, deepened in place, reporting as it goes.

        Not the shared search path, for a reason worth stating: that one
        re-roots the previous ply's tree at the move played, and here the
        position one ply on depends on a card that had not been turned over
        when the tree was built.  So each position gets its own tree, grown in
        batches until it hits the ceiling and then left alone.
        """
        from alphazero_core.vecmcts import VecMCTSConfig, VecSearch

        from alphazero_splendor.splendor_game import NACTIONS

        if not board.is_terminal() and self.v2_checkpoint():
            yield from self._analysis_stream_v2(board)
            return
        if board.is_terminal():
            values = board.result_vector()
            yield dict(plies=board.move_count, sims=0, nodes=0, depth=0,
                       value_black=float(values[1]),
                       seat_values=[float(v) for v in values[1:]],
                       to_move=board.to_move, top=[], proven=True, terminal=True)
            return
        engine = self.evaluator(ckpt, board)
        cfg = VecMCTSConfig(simulations=STREAM_EVAL_SIMS, add_noise=False)
        search = VecSearch(board, cfg, np.random.default_rng(), NACTIONS)
        step = 64
        while search.sims_done < cfg.simulations:
            target = min(search.sims_done + step, cfg.simulations)
            while search.sims_done < target:
                states = search.next_leaf_batch(16)
                if not states:
                    break
                priors, values = engine.evaluate(states)
                search.expand_batch(priors, values)
            yield self._readout(board, search)
            if search.sims_done < target:
                return  # the tree ran out of anything to expand
            step = min(step * 2, 512)

    def _analysis_stream_v2(self, board):
        """The same, on the v2 network and the v2 search.

        The readout below is written against the search's interface rather than
        against either implementation, so both feed it unchanged.  What differs
        is that this one converts the position on the way in and is several
        times quicker per simulation, so the bar settles sooner.
        """
        import random as _random

        from alphazero_splendor2.bridge import from_v1
        from alphazero_splendor2.registry import evaluator as v2_evaluator
        from alphazero_splendor2.search import Search, SearchConfig

        engine = v2_evaluator(self.v2_checkpoint())
        cfg = SearchConfig(simulations=STREAM_EVAL_SIMS, add_noise=False)
        search = Search(from_v1(board), cfg, _random.Random())
        step = 64
        while search.sims_done < cfg.simulations:
            target = min(search.sims_done + step, cfg.simulations)
            while search.sims_done < target:
                states = search.next_leaf_batch(32)
                if not states:
                    break
                priors, values = engine.evaluate(states)
                search.expand_batch(priors, values)
            yield self._readout(board, search)
            if search.sims_done < target:
                return  # the tree ran out of anything to expand
            step = min(step * 2, 512)

    def _readout(self, board, search) -> dict:
        values = search.root_values()
        visits = search.root_visit_counts()
        scores = search.root_child_scores()
        total = float(visits.sum())
        order = np.argsort(-visits)[:8]
        top = [dict(cell=int(i), label=board.move_label(int(i)),
                    q=float(scores[int(i)]), visits=int(visits[int(i)]),
                    share=(float(visits[int(i)]) / total) if total > 0 else None,
                    proven=False)
               for i in order if visits[int(i)] > 0]
        return dict(
            plies=board.move_count,
            sims=int(search.sims_done), nodes=int(search.n_nodes),
            depth=None,
            # The bar is drawn from the first player's point of view, as in
            # every other game here.
            value_black=float(values[1]),
            seat_values=[float(v) for v in values[1:]],
            to_move=board.to_move,
            top=top, proven=False, terminal=False, error=None,
        )

    # ---------------------------------------------------------------- ratings
    @property
    def R(self):
        from alphazero_splendor import ratings as R

        return R

    def ratings(self) -> dict:
        return SPLENDOR_RATINGS.get()


def _hex_ratings_module():
    from alphazero_hex import ratings as R

    return R


def _c4_ratings_module():
    from alphazero_c4 import ratings as R

    return R


def _uttt_ratings_module():
    from alphazero_uttt import ratings as R

    return R


def _rps2_ratings_module():
    from alphazero_rps2 import ratings as R

    return R


def _uxx_ratings_module():
    from alphazero_uxx import ratings as R

    return R


def _bg_ratings_module():
    from alphazero_bg import ratings as R

    return R


def _splendor_ratings_module():
    from alphazero_splendor import ratings as R

    return R


HEX_RATINGS = _RatingTable(_hex_ratings_module())
C4_RATINGS = _RatingTable(_c4_ratings_module())
UTTT_RATINGS = _RatingTable(_uttt_ratings_module())
UXX_RATINGS = _RatingTable(_uxx_ratings_module())
RPS2_RATINGS = _RatingTable(_rps2_ratings_module())
BG_RATINGS = _RatingTable(_bg_ratings_module())
SPLENDOR_RATINGS = _RatingTable(_splendor_ratings_module())

GAMES: dict[str, GameAdapter] = {"hex": HexAdapter(), "c4": ConnectFourAdapter(),
                                 "uttt": UltimateAdapter(),
                                 "uxx": UltimateXXAdapter(),
                                 "rps2": IntransitiveAdapter(),
                                 "bg": BackgammonAdapter(),
                                 "splendor": SplendorAdapter()}
DEFAULT_GAME = "hex"


def game_for(key: str | None) -> GameAdapter:
    return GAMES.get(str(key or DEFAULT_GAME), GAMES[DEFAULT_GAME])


# ------------------------------------------------------------------ Hex extras
def hex_winning_path(board) -> list[int]:
    """One connecting chain for the winner, for highlighting on the board."""
    from alphazero_hex.hex_game import BLACK, WHITE, neighbour_table

    who = board.winner
    if not who:
        return []
    n, cells = board.n, board.board
    nei = neighbour_table(n)
    if who == BLACK:
        starts = [c for c in range(n) if cells[c] == BLACK]
        at_goal = lambda i: i >= (n - 1) * n  # noqa: E731
    else:
        starts = [r * n for r in range(n) if cells[r * n] == WHITE]
        at_goal = lambda i: i % n == n - 1  # noqa: E731
    prev: dict[int, int | None] = {s: None for s in starts}
    queue = deque(starts)
    while queue:
        cur = queue.popleft()
        if at_goal(cur):
            path = []
            node: int | None = cur
            while node is not None:
                path.append(node)
                node = prev[node]
            return path
        for nb in nei[cur]:
            if nb not in prev and cells[nb] == who:
                prev[nb] = cur
                queue.append(nb)
    return []


def other_side(colour: int) -> int:
    return SECOND if colour == FIRST else FIRST


# ----------------------------------------------------------------- live analysis
class Analysis:
    """A second AlphaZero tree, grown continuously on the displayed position.

    This is deliberately *not* the opponent's search: it keeps running while you
    think, on whatever position is on the board, so the evaluation on screen
    sharpens the longer you look at it -- up to a simulation ceiling you set.
    """

    BATCH = 24  # leaves per network call; the same virtual-loss trick self-play uses

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.enabled = False
        self.max_sims = DEFAULT_EVAL_SIMS
        self._wanted: tuple | None = None
        self._key: tuple | None = None
        self._search: Search | None = None
        self._board = None  # what _search is rooted at, for reuse
        self._game: GameAdapter | None = None
        self._result: dict = {}
        self._ckpt: str | None = None
        self._stream = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------- API
    def configure(self, enabled: bool, max_sims: int) -> None:
        with self.lock:
            was_off = not self.enabled
            self.enabled = enabled
            self.max_sims = max(1, int(max_sims))
            if was_off and enabled:
                self._result = {}
        if enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def set_position(self, game: GameAdapter, board, ckpt: str) -> None:
        key = (game.key, game.position_key(board), board.to_move,
               game.size_id(board), ckpt)
        with self.lock:
            self._wanted = (board.copy(), key, ckpt, game)
            self._ckpt = ckpt

    def snapshot(self) -> dict:
        with self.lock:
            # ckpt is reported so the page can say which network the bar
            # reflects: it is *not* the opponent's, and that is easy to
            # forget when the two differ.
            return dict(self._result, enabled=self.enabled, max_sims=self.max_sims,
                        ckpt=self._ckpt)

    # ---------------------------------------------------------------- worker
    def _run(self) -> None:
        while True:
            with self.lock:
                enabled, ceiling, wanted = self.enabled, self.max_sims, self._wanted
            if not enabled or wanted is None:
                time.sleep(0.1)
                continue
            board, key, ckpt, game = wanted
            try:
                if not game.analysable:
                    # Nothing to evaluate this position with.  The page hides
                    # the switch for such a game, but the switch is global and
                    # may have been left on by the game before this one, so the
                    # thread has to know as well -- otherwise it would spend the
                    # session publishing the same failure.
                    if self._key is not None:
                        self._key = None
                        self._search = None
                        self._store({})
                    time.sleep(0.2)
                    continue
                if not game.uses_search:
                    self._step_stream(game, board, key, ckpt)
                    continue
                if key != self._key:
                    self._start(game, board, key, ceiling)
                if self._search is None:
                    time.sleep(0.1)
                    continue
                self._search.cfg.simulations = ceiling
                if self._search.sims_done >= ceiling:
                    time.sleep(0.12)  # ceiling reached; wait for a new position
                    continue
                self._step(game, board, key, ckpt)
            except ImportError:
                self._publish_error("Live evaluation needs PyTorch, which is not installed.")
                time.sleep(2.0)
            except Exception as exc:
                traceback.print_exc()
                self._publish_error(f"analysis stopped: {exc}")
                time.sleep(2.0)

    def _start(self, game: GameAdapter, board, key: tuple, ceiling: int) -> None:
        previous, previous_board = self._search, self._board
        same_game = self._game is not None and self._game.key == game.key
        self._key = key
        self._board = board.copy()
        self._game = game
        if board.is_terminal():
            self._search = None
            self._publish_terminal(game, board)
            return
        cfg = MCTSConfig(simulations=ceiling, c_puct=1.6, add_noise=False)
        # One move on from the last position: keep everything already searched
        # below the move that was played rather than starting over.
        self._search = reuse_or_new(previous if same_game else None,
                                    previous_board if same_game else None,
                                    board, cfg, ceiling)
        self._publish(game, board, key)

    def _step_stream(self, game: GameAdapter, board, key: tuple, ckpt: str) -> None:
        """Drive a game that has its own analysis instead of a search tree.

        Backgammon has no tree to deepen: the roll is already known, so the
        candidates are simply this roll's moves, evaluated once at one ply and
        then again at two.  The stream yields a readout per pass and then stops,
        which is why exhausting it means "wait for a new position" rather than
        "keep going".
        """
        if key != self._key:
            self._key = key
            self._board = board.copy()
            self._game = game
            self._search = None
            self._stream = game.analysis_stream(board, ckpt)
        stream = getattr(self, "_stream", None)
        if stream is None:
            time.sleep(0.1)
            return
        try:
            self._store(dict(next(stream), error=None))
        except StopIteration:
            self._stream = None
            time.sleep(0.15)

    def _step(self, game: GameAdapter, board, key: tuple, ckpt: str) -> None:
        engine = game.evaluator(ckpt, board)
        states = self._search.next_leaf_batch(self.BATCH)
        if not states:
            time.sleep(0.05)
            return
        priors, values = engine.evaluate(states)
        self._search.expand_batch(priors, values)
        self._publish(game, board, key)

    # -------------------------------------------------------------- readouts
    def _publish(self, game: GameAdapter, board, key: tuple) -> None:
        search = self._search
        if not search.is_expanded(0):
            # Published once before the first batch lands, so the panel can show
            # the new position immediately instead of the previous one's numbers.
            self._store(dict(plies=board.move_count, sims=0, nodes=0, value_black=0.0,
                             to_move=board.to_move, top=[], terminal=False, error=None))
            return
        moves, visits = search.moves[0], search.N[0]
        # Plies below the root that each branch has actually been read to --
        # the tree is far deeper down the line it believes in than down the
        # ones it is only checking, and that is worth seeing.
        depths = search.child_depth
        top: list[dict] = []
        if visits.sum() > 0:
            scores = search.root_child_scores()
            proven = set(int(j) for j in search.root_proven_moves())
            total = float(visits.sum())
            for i in np.argsort(-visits)[:8]:
                if visits[i] <= 0:
                    break
                move = int(moves[i])
                top.append(dict(cell=move, label=game.move_label(move, board),
                                visits=int(visits[i]), share=float(visits[i] / total),
                                q=float(scores[i]), prior=float(search.P[0][i]),
                                depth=None if depths is None else int(depths[i]),
                                proven=int(i) in proven))
        # Exact once the tree has proven the result; the average otherwise.
        value = search.root_score()
        self._store(dict(
            plies=board.move_count,
            sims=int(search.sims_done),
            nodes=int(search.n_nodes),
            depth=int(search.depth_reached),
            # Reported for the first player throughout, so the bar does not flip
            # every ply.
            value_black=value if board.to_move == FIRST else -value,
            to_move=board.to_move,
            top=top,
            proven=search.solved[0] != 0,
            terminal=False,
            error=None,
        ))

    def _publish_terminal(self, game: GameAdapter, board) -> None:
        entry = game.terminal_readout(board)
        self._store(dict(plies=board.move_count, sims=0, nodes=0,
                         value_black=entry["value_black"],
                         to_move=board.to_move, top=[], proven=True,
                         terminal=True, error=None))

    def _publish_error(self, message: str) -> None:
        self._store(dict(plies=None, sims=0, nodes=0, value_black=0.0,
                         to_move=FIRST, top=[], proven=False,
                         terminal=False, error=message))

    def _store(self, result: dict) -> None:
        with self.lock:
            self._result = result


# ------------------------------------------------------------------ match review
def classify(loss: float, played_best: bool) -> str:
    """Name for a move, given how much it cost the player who made it."""
    for threshold, label in REVIEW_BANDS:
        if loss >= threshold:
            return label
    return "best" if played_best else "good"


def advanced_by(old, new) -> int | None:
    """The single move that turns ``old`` into ``new``, if that is what happened."""
    if old is None or new.move_count != old.move_count + 1:
        return None
    if old.ncells != new.ncells:
        return None
    before, after = old.array().reshape(-1), new.array().reshape(-1)
    diff = np.flatnonzero(before != after)
    return int(diff[0]) if len(diff) == 1 else None


# Re-rooting keeps the work already done below the move that was played, but it
# has to copy the reachable subtree first.  Measured on 11x11 Hex, that subtree
# carries only about 10% of the budget -- PUCT spends its visits on the line it
# prefers, and the move actually played is often not that line -- so the copy is
# worth it only once the budget is large: 1.4x *slower* at 800 simulations,
# ~1.1x faster from ~3000 up.  Below this, starting fresh wins.
REUSE_MIN_SIMS = 2000


def reuse_or_new(previous: Search | None, previous_board, board,
                 cfg: MCTSConfig, target: int) -> Search:
    """Continue the existing tree if that is cheaper than starting over."""
    if target >= REUSE_MIN_SIMS:
        move = advanced_by(previous_board, board)
        if previous is not None and move is not None:
            sub = previous.child_search(move)
            if sub is not None:
                sub.cfg = cfg
                return sub
    return Search(board.copy(), cfg, np.random.default_rng(0))


# Spinning up workers costs a checkpoint load and a JIT trace each, so the pool
# only earns its keep once there is enough work to spread.  Below this, the
# single-process path (which also gets to reuse its tree) finishes first.
PARALLEL_MIN_WORK = 20_000  # positions x simulations
MAX_REVIEW_WORKERS = 10


def review_workers(positions: int, sims: int) -> int:
    """How many processes to review with; 1 means do it in this thread."""
    if positions * sims < PARALLEL_MIN_WORK or positions < 4:
        return 1
    cores = os.cpu_count() or 1
    return max(1, min(cores - 1, positions, MAX_REVIEW_WORKERS))


class MatchReview:
    """Evaluates every position of a finished game, for replaying it afterwards.

    Each position gets a fixed-size search; a move's cost is then the drop in
    value across it, in the mover's favour.  Positions are independent, so the
    work goes to a process pool when there is enough of it to be worth the
    workers' start-up, and otherwise runs here where it can reuse its tree.
    """

    BATCH = 24

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._state: dict = self._blank()
        self._token = 0

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self._state, plies=list(self._state["plies"]))

    def clear(self) -> None:
        with self.lock:
            self._token += 1
            self._state = self._blank()

    @staticmethod
    def _blank() -> dict:
        return {"running": False, "done": False, "plies": [], "total": 0,
                "completed": 0, "workers": 1, "sims": DEFAULT_REVIEW_SIMS,
                "error": None}

    def start(self, game: GameAdapter, moves: list[int], size_id: str, ckpt: str,
              sims: int) -> str | None:
        # The page disables the button for a game with nothing to review with;
        # the endpoint says so too, so a stale tab cannot start a run that would
        # only fail inside a worker.
        if not game.reviewable:
            return f"match review is not available for {game.label}"
        if len(moves) < 2:
            return "nothing to analyse yet"
        with self.lock:
            if self._state["running"]:
                return "already analysing this game"
            self._token += 1
            token = self._token
            self._state = dict(self._blank(), running=True, total=len(moves) + 1,
                               sims=int(sims), ckpt=ckpt,
                               workers=review_workers(len(moves) + 1, int(sims)))
        threading.Thread(target=self._run, daemon=True,
                         args=(token, game, list(moves), size_id, ckpt, int(sims))).start()
        return None

    def _run(self, token: int, game: GameAdapter, moves: list[int], size_id: str,
             ckpt: str, sims: int) -> None:
        workers = self.snapshot()["workers"]
        try:
            if workers > 1:
                plies = self._run_parallel(token, game, moves, size_id, ckpt, sims, workers)
            else:
                plies = self._run_serial(token, game, moves, size_id, ckpt, sims)
            if plies is None:
                return  # superseded
            for index, entry in enumerate(plies):
                entry["played"] = int(moves[index]) if index < len(moves) else None
            with self.lock:
                if token != self._token:
                    return
                self._state["plies"] = plies
            self._finish(token)
        except ImportError:
            self._fail(token, "Match analysis needs PyTorch, which is not installed.")
        except Exception as exc:
            traceback.print_exc()
            self._fail(token, f"analysis failed: {exc}")

    def _run_serial(self, token: int, game: GameAdapter, moves: list[int], size_id: str,
                    ckpt: str, sims: int) -> list[dict] | None:
        """One position after another, re-rooting the tree as the game advances."""
        from alphazero_core.mcts import run_search

        board = game.new_board(size_id)
        engine = game.evaluator(ckpt, board)
        cfg = MCTSConfig(simulations=sims, c_puct=1.6, add_noise=False)
        search: Search | None = None
        previous_board = None
        plies: list[dict] = []
        for index in range(len(moves) + 1):
            if not self._alive(token):
                return None
            if board.is_terminal():
                entry = game.terminal_readout(board)
                search = None
            else:
                # Walking the game forwards means each position is the last one
                # plus a move, so the tree can simply be re-rooted.
                search = reuse_or_new(search, previous_board, board, cfg, sims)
                search.cfg.simulations = sims
                run_search(search, engine, sims, self.BATCH)
                entry = game.readout(search, board)
                previous_board = board.copy()
            entry["ply"] = index
            plies.append(entry)
            self._advance(token)
            if index < len(moves):
                board.play(moves[index])
        return plies

    def _run_parallel(self, token: int, game: GameAdapter, moves: list[int], size_id: str,
                      ckpt: str, sims: int, workers: int) -> list[dict] | None:
        """Every position at once, one process each.

        Positions are independent, so this scales with cores.  It gives up tree
        reuse -- a worker has no idea what the previous position found -- which
        is a good trade: reuse was worth about a tenth of the budget.
        """
        import concurrent.futures as cf
        import multiprocessing as mp

        path = game.ckpt_path(ckpt)
        # The device rides along at the end: a spawned worker never sees DEVICE.
        dev = getattr(game, "device", DEVICE)
        tasks = [task + (dev,) for task in game.review_tasks(moves, size_id, path, sims)]
        worker = game.evaluate_position
        results: list[dict | None] = [None] * len(tasks)
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork" if "fork" in methods else "spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(worker, task): task[4] for task in tasks}
            for future in cf.as_completed(futures):
                if not self._alive(token):
                    pool.shutdown(cancel_futures=True)
                    return None
                results[futures[future]] = future.result()
                self._advance(token)
        return [entry for entry in results if entry is not None]

    def _advance(self, token: int) -> None:
        with self.lock:
            if token == self._token:
                self._state["completed"] += 1

    def _alive(self, token: int) -> bool:
        with self.lock:
            return token == self._token

    def _finish(self, token: int) -> None:
        with self.lock:
            if token != self._token:
                return
            plies = self._state["plies"]
            # A move's cost is the value it gave away, in its own player's favour.
            for i, entry in enumerate(plies[:-1]):
                sign = 1.0 if entry["to_move"] == FIRST else -1.0
                loss = (entry["value_black"] - plies[i + 1]["value_black"]) * sign
                entry["loss"] = round(max(0.0, loss), 4)
                entry["quality"] = classify(entry["loss"], entry["played"] == entry["best"])
            self._state["running"] = False
            self._state["done"] = True

    def _fail(self, token: int, message: str) -> None:
        with self.lock:
            if token != self._token:
                return
            self._state["running"] = False
            self._state["error"] = message


# ------------------------------------------------------------------- game state
class Session:
    """One game, plus the worker thread that plays the bots' moves.

    Every mutation bumps ``generation``; a bot thread checks it before writing
    its move back, so changing the opponent or starting a new game while a
    search is running simply makes that search's result irrelevant instead of
    corrupting the position.  ``revision`` bumps on every observable change and
    lets the page discard a poll answer that overtook a newer one.

    Which game is being played is part of the session: switching from Hex to
    Connect Four is a new game with a different adapter, and everything below
    reads the board through that adapter rather than knowing either game.
    """

    def __init__(self, game: str = DEFAULT_GAME) -> None:
        self.lock = threading.RLock()
        self.rng = np.random.default_rng()   # the dice, for games that have them
        self.analysis = Analysis()
        self.review = MatchReview()
        self.generation = 0
        self.revision = 0
        self.started = False           # no game yet: a fresh page may start one
        self.game = game_for(game)
        self.size = self.game.default_size
        self.deal = 0                  # what this game's decks were shuffled from
        self.board = self.game.new_board(self.size, self.deal)
        # Everything held per player is keyed by the engine's seat number, and
        # ``seats`` is the turn order.  Three of the four games return exactly
        # (1, 2) here for ever; Splendor returns two, three or four.
        self.seats = self.game.seat_ids(self.board)
        self.choices = {FIRST: "human", SECOND: "az"}
        self.sims = {FIRST: DEFAULT_SIMS, SECOND: DEFAULT_SIMS}
        self.vary = {FIRST: False, SECOND: False}
        self.agents: dict[int, Agent | None] = {FIRST: None, SECOND: None}
        self.ckpt = (self.game.checkpoints() or ["final.pt"])[0]
        # Per seat, so two AlphaZero networks can be played against each
        # other.  self.ckpt stays as the default for whichever seat does
        # not name one, and is what the UI shows as the shared selection.
        self.ckpts = {FIRST: self.ckpt, SECOND: self.ckpt}
        self.seed = 0
        self.moves: list[dict] = []
        self.thinking = False
        self.error: str | None = None
        self.info: dict[str, str] = {}
        self.user: str | None = None
        # How long to dwell on a bot's move, as a multiple of the game's own
        # ``bot_pause``.  0 plays as fast as the machine can, which is what the
        # tests and an impatient watcher both want.
        self.pace = 1.0
        self.rated = False
        self.scored = False
        self.rating_result: dict | None = None

    # ------------------------------------------------------------- new / undo
    def new_game(self, black: str, white: str, size: str = "", ckpt: str = "",
                 seed: int = 0, black_sims: int = DEFAULT_SIMS,
                 white_sims: int = DEFAULT_SIMS, black_vary: bool = False,
                 white_vary: bool = False, rated: bool = False,
                 black_ckpt: str | None = None, white_ckpt: str | None = None,
                 game: str | None = None, players: list | None = None) -> str | None:
        """Start a new game.

        ``players`` is the general form: one entry per seat, in turn order, each
        either an opponent id or a dict of ``choice``/``sims``/``vary``/``ckpt``.
        The ``black_*`` / ``white_*`` arguments are the two-seat spelling of the
        same thing and are what the two-player games' callers (and every test
        written before Splendor existed) use; when ``players`` is absent they
        are assembled into it.
        """
        adapter = game_for(game) if game else self.game
        size = str(size or adapter.default_size)
        if size not in {s["id"] for s in adapter.sizes}:
            size = adapter.default_size
        available = adapter.checkpoints() or ["final.pt"]
        ckpt = ckpt or available[0]
        switching = adapter.key != self.game.key
        # Switching game changes which checkpoints and ratings exist, so a
        # network named for the other game's directory cannot carry over.
        if switching:
            ckpt = ckpt if ckpt in available else available[0]
            black_ckpt = white_ckpt = None
            # Profiles are per game.  The same name usually exists in both, and
            # keeping it is what a player expects; a name this game has never
            # heard of would otherwise sit in the picker unrated.
            if self.user and self.user not in adapter.R.load_users():
                self.user = None

        if players is None:
            players = [
                {"choice": black, "sims": black_sims, "vary": black_vary,
                 "ckpt": black_ckpt},
                {"choice": white, "sims": white_sims, "vary": white_vary,
                 "ckpt": white_ckpt},
            ]
        # The deal, for the games that shuffle something.  Fixed here rather
        # than per board so that rebuilding the position after a take-back
        # deals the same cards; a fresh one every game so two games in a row
        # are not identical.
        deal = int(self.rng.integers(1 << 30))
        seats = adapter.seat_ids(adapter.new_board(size, deal))
        entries = self._seat_entries(adapter, players, seats, ckpt, available,
                                     drop_ckpt=switching)

        choices = {seat: e["choice"] for seat, e in zip(seats, entries)}
        sims = {seat: e["sims"] for seat, e in zip(seats, entries)}
        # A varied opponent is not exactly what the ladder rated, so a rated
        # game always gets the deterministic, full-strength agent.
        vary = {seat: e["vary"] and not rated for seat, e in zip(seats, entries)}
        ckpts = {seat: e["ckpt"] for seat, e in zip(seats, entries)}
        problem = None
        if rated:
            problem = self._rating_problem(adapter, seats, choices, sims, size, ckpts)
        self.review.clear()  # the previous game's analysis is not about this one
        with self.lock:
            self.generation += 1
            self.started = True
            self.game = adapter
            self.size = size
            self.deal = deal
            self.board = adapter.new_board(size, deal)
            self.seats = list(seats)
            self.choices = choices
            self.sims = sims
            self.vary = vary
            self.ckpt = ckpt
            self.ckpts = ckpts
            self.seed = seed
            self.moves = []
            self.error = None
            self.info = {}
            self.thinking = False
            self.rated = rated and problem is None
            self.scored = False
            self.rating_result = None
            self._rebuild()
            self._roll_if_needed()
            self._touch()
        self.analysis.set_position(self.game, self.board, self.game.analysis_checkpoint())
        self._wake_bot()
        return problem

    @staticmethod
    def _seat_entries(adapter: GameAdapter, players: list, seats: list[int],
                      ckpt: str, available: list[str],
                      drop_ckpt: bool = False) -> list[dict]:
        """Normalise ``players`` into one validated dict per seat.

        Short lists are padded with the game's default opponent and long ones
        truncated, so a page that still thinks the table seats four cannot start
        a two-player game with two extra bots in it.
        """
        entries: list[dict] = []
        for i, seat in enumerate(seats):
            raw = players[i] if i < len(players) else adapter.default_opponent
            if not isinstance(raw, dict):
                raw = {"choice": raw}
            choice = adapter.valid_choice(str(raw.get("choice") or
                                              adapter.default_opponent),
                                          adapter.default_opponent)
            # The strength dial means different things in different games -- 400
            # simulations, or 2 plies -- so a value from another game's slider
            # is not just wrong, it is a different unit.
            strength = adapter.valid_strength(raw.get("sims", adapter.default_strength))
            net = None if drop_ckpt else (raw.get("ckpt") or None)
            entries.append({
                "choice": choice,
                "sims": int(strength),
                "vary": bool(raw.get("vary", False)),
                "ckpt": net if net in available else ckpt,
            })
        return entries

    def _roll_if_needed(self) -> None:
        """Roll for whoever is on turn, if this game rolls at all.

        The dice belong to the *server*: a page that rolled for itself could
        reroll a bad one, and a bot needs them before it can be asked to move.
        Caller holds the lock.
        """
        if self.game.needs_roll(self.board) and not self.board.is_terminal():
            self.game.roll(self.board, self.rng)

    def _rating_problem(self, adapter: GameAdapter, seats: list[int],
                        choices: dict, sims: dict, size: str,
                        ckpts: dict | None = None) -> str | None:
        """Why this pairing cannot be played for rating, if it cannot."""
        if not self.user:
            return "pick a player first — a rated game needs someone to rate"
        # Incremental Elo is a statement about one player against one opponent
        # of known strength.  Finishing third of four is not that, so a rated
        # game is a duel whatever the game can otherwise seat.
        if len(seats) != 2:
            return "a rated game is a two-player game — set the table to two seats"
        picks = [choices[seat] for seat in seats]
        if picks.count("human") != 1:
            return "a rated game is one human against one bot"
        black, white = picks
        black_sims, white_sims = (sims[seats[0]], sims[seats[1]])
        table = adapter.ratings()
        ladder = adapter.ladder_script
        if not table.get("bots"):
            return (f"no bot ratings yet for {adapter.label} — run the tournament "
                    f"first (python3 {ladder})")
        want = adapter.rating_size_for(size)
        if table.get("board_size") not in (None, want):
            return (f"the bots were rated on {table['board_size']}; "
                    f"play that size for a rated game")
        choice, sims = (white, white_sims) if black == "human" else (black, black_sims)
        key = adapter.R.bot_key(choice, sims)
        if key not in table["bots"]:
            return f"{adapter.R.bot_label(key)} has no rating yet — pick a rated opponent"
        # The ladder rated one specific network.  An AlphaZero opponent
        # running different weights is a different strength entirely, so
        # scoring it against that rating would corrupt the human rating.
        if choice == "az" and ckpts is not None:
            rated_net = adapter.analysis_checkpoint()
            bot_net = ckpts[seats[1]] if black == "human" else ckpts[seats[0]]
            if bot_net != rated_net:
                return (f"the ratings are for {rated_net}; pick that network "
                        f"for a rated game, or play unrated")
        return None

    def set_opponent(self, colour: int, choice: str, sims: int, vary: bool = False,
                     ckpt: str | None = None) -> str | None:
        """Swap one side's player mid-game, keeping the position."""
        with self.lock:
            self.generation += 1  # any search in flight is now irrelevant
            self.thinking = False
            self.choices[colour] = choice
            self.sims[colour] = int(sims)
            self.vary[colour] = bool(vary)
            if ckpt:
                self.ckpts[colour] = ckpt
            self.info = {}
            if self.rated:
                # Changing opponent mid-game would make the result meaningless.
                self.rated = False
                self.error = "changing the opponent made this game unrated"
            self._rebuild()
            problem = self.error
            self._touch()
        self._wake_bot()
        return problem

    def _release_agents(self) -> None:
        """Let go of the outgoing agents before their replacements are built.

        An agent that thinks on the opponent's time runs a background thread.
        Dropping the reference is not enough: the thread keeps the agent -- and
        the compiled network behind it -- alive until it times out by itself, so
        starting game after game piled up threads and memory until the whole
        server crawled.  Asking each one to stop makes that immediate.
        """
        for agent in getattr(self, "agents", {}).values():
            stop = getattr(agent, "stop_ponder", None) or getattr(agent, "close", None)
            if stop is None:
                continue
            try:
                stop()
            except Exception:       # a broken agent must not block the new game
                traceback.print_exc()
        self.agents = {}

    def _rebuild(self) -> None:
        """(Re)create every seat's agent from the current choices. Caller holds the lock."""
        self._release_agents()
        try:
            self.agents = {
                seat: self.game.build_agent(self.choices[seat], self.seed + seat,
                                            self.board, self.ckpts[seat],
                                            self.sims[seat], self.vary[seat])
                for seat in self.seats
            }
        except ImportError:
            self.agents = {seat: None for seat in self.seats}
            self.error = ("That opponent needs PyTorch, which is not installed. "
                          "Install it with:  pip install torch --index-url "
                          "https://download.pytorch.org/whl/cpu")
        except Exception as exc:  # a missing checkpoint, a bad spec, ...
            self.agents = {seat: None for seat in self.seats}
            self.error = f"could not create that opponent: {exc}"

    def undo(self) -> str | None:
        """Take back to the human's previous turn (both plies of an exchange)."""
        with self.lock:
            if self.rated:
                return "no take-backs in a rated game"
            if self.thinking:
                return "wait for the bot to finish its move"
            if not self.moves:
                return "nothing to take back"
            humans = [c for c, a in self.agents.items() if a is None]
            self.generation += 1
            popped = [self.moves.pop()]
            if humans:
                # Keep popping until a human is on turn again.  Whoever played
                # the move just removed is the seat now on turn, which is the
                # only way to know it in a game whose turns are not a strict
                # alternation: a Splendor turn can be two or three moves long.
                while self.moves and popped[-1]["player"] not in humans:
                    popped.append(self.moves.pop())
            self._replay(restore=popped[-1])
            self.info = {}
            self._touch()
        self.review.clear()  # it described a game that no longer exists
        self.analysis.set_position(self.game, self.board, self.game.analysis_checkpoint())
        self._wake_bot()  # a no-op unless both sides are bots
        return None

    def _replay(self, restore: dict | None = None) -> None:
        # The same deal as the game being rebuilt, so a take-back reaches the
        # earlier position of *this* game rather than a fresh one.
        board = self.game.new_board(self.size, self.deal)
        for entry in self.moves:
            self.game.replay_move(board, entry)
        self.board = board
        # A take-back hands back the roll that was in hand, not a fresh one.
        if restore is None or not self.game.restore_roll(board, restore):
            self._roll_if_needed()
        for agent in self.agents.values():
            if agent is not None:
                agent.reset()

    def _touch(self) -> None:
        self.revision += 1

    # ------------------------------------------------------------- human move
    def play_human(self, cell) -> str | None:
        with self.lock:
            if cell is None:
                return f"that {self.game.move_noun} is not available"
            if self.thinking:
                return "the bot is still thinking"
            if self.board.is_terminal():
                return "the game is over"
            if self.agents[self.board.to_move] is not None:
                return "it is not your turn"
            if not self.game.is_legal(self.board, cell):
                return f"that {self.game.move_noun} is not available"
            self._record(cell, "you")
        self._wake_bot()
        return None

    def _record(self, move, by: str) -> None:
        before = self.board.copy()
        self.game.apply(self.board, move)
        self.moves.append(self.game.move_entry(move, before, self.board, by))
        self._roll_if_needed()
        self._touch()
        self.analysis.set_position(self.game, self.board, self.game.analysis_checkpoint())
        if self.board.is_terminal():
            self._score_game()

    # ------------------------------------------------------------ rated games
    def _score_game(self) -> None:
        """Apply a finished rated game to the human's Elo. Caller holds the lock."""
        if not self.rated or self.scored or not self.user:
            return
        # A rated game is a duel; ``_rating_problem`` refused anything else.
        if len(self.seats) != 2:
            self.rated = False
            return
        self.scored = True
        first, second = self.seats
        human = first if self.agents[first] is None else second
        bot = second if human == first else first
        adapter = self.game
        R = adapter.R
        table = adapter.ratings()
        key = R.bot_key(self.choices[bot], self.sims[bot])
        entry = table["bots"].get(key)
        if entry is None:
            self.rated = False
            return
        before = R.load_users().get(self.user, {}).get("rating", R.START_RATING)
        score = adapter.score_for(self.board, human)
        profile = self._record_rated(adapter, key, float(entry["elo"]), score,
                                     human == first,
                                     float(table.get("first_player_advantage", 0.0)))
        if profile is None:
            self.rated = False
            return
        self.rating_result = {
            "opponent": R.bot_label(key),
            "opponent_rating": round(float(entry["elo"])),
            "won": score > 0.75,
            "result": "win" if score > 0.75 else ("draw" if score >= 0.25 else "loss"),
            "before": round(float(before)),
            "after": round(float(profile["rating"])),
            "delta": round(float(profile["rating"]) - float(before), 1),
        }

    def _record_rated(self, adapter: GameAdapter, key: str, opponent_rating: float,
                      score: float, played_first: bool, advantage: float):
        """Write the result through whichever signature the game's ratings use.

        Hex cannot be drawn and its ``record_result`` takes a bool, which is
        also what every stored Hex history was written with; Connect Four takes
        a score, because half a point is a real outcome there.
        """
        R = adapter.R
        common = dict(name=self.user, opponent_key=key, opponent_rating=opponent_rating,
                      played_first=played_first, advantage=advantage,
                      plies=self.board.move_count,
                      board_size=adapter.rating_board_size(self.board))
        if adapter.draws:
            return R.record_result(score=score, **common)
        return R.record_result(won=score > 0.5, **common)

    def set_pace(self, pace: float) -> None:
        """How long a bot dwells on its move; 0 is as fast as it can."""
        with self.lock:
            self.pace = max(0.0, min(4.0, float(pace)))
            self._touch()

    def select_user(self, name: str | None) -> None:
        with self.lock:
            self.user = name or None
            if self.rated and not self.user:
                self.rated = False
            self._touch()

    # --------------------------------------------------------------- bot turn
    def _wake_bot(self) -> None:
        with self.lock:
            if self.thinking or self.board.is_terminal():
                return
            if self.agents[self.board.to_move] is None:
                return
            self.thinking = True
            gen = self.generation
            self._touch()
        threading.Thread(target=self._bot_loop, args=(gen,), daemon=True).start()

    def _bot_loop(self, gen: int) -> None:
        """Play bot moves until a human is on turn (or the game ends)."""
        try:
            while True:
                with self.lock:
                    if gen != self.generation or self.board.is_terminal():
                        return
                    agent = self.agents[self.board.to_move]
                    if agent is None:
                        return
                    position = self.board.copy()
                    last = self.moves[-1].get("cell") if self.moves else None
                    game = self.game
                # Searching outside the lock keeps /api/state answering instantly.
                started = time.time()
                try:
                    cell = game.select(agent, position, last)
                except Exception:
                    traceback.print_exc()
                    with self.lock:
                        if gen == self.generation:
                            self.error = "the bot crashed while thinking; see the console"
                            self._touch()
                    return
                # Hold a move that came back too fast, so that a watcher can
                # see it played rather than having the board change under them.
                # Outside the lock, and generation-checked afterwards, so a new
                # game started during the pause still cancels this move.
                slack = game.bot_pause * self.pace - (time.time() - started)
                if slack > 0:
                    time.sleep(slack)
                with self.lock:
                    if gen != self.generation:
                        return  # a new game started under us; drop this move
                    self._record(cell, agent.name)
                    self.info = self._agent_info(agent)
        finally:
            with self.lock:
                if gen == self.generation:
                    self.thinking = False
                    self._touch()

    def _agent_info(self, agent: Agent) -> dict[str, str]:
        out: dict[str, str] = {}
        value = getattr(agent, "last_value", None)
        if value is not None:
            out["its own estimate"] = self.game.format_value(float(value))
        depth = getattr(agent, "last_depth", None)
        if depth:
            out["depth reached"] = str(depth)
        nodes = getattr(agent, "nodes", None)
        if nodes:
            out["nodes searched"] = f"{nodes:,}"
        return out

    # ------------------------------------------------------------------ state
    def snapshot(self) -> dict:
        with self.lock:
            board = self.board
            game = self.game
            table = game.ratings()
            seats = [self._side(seat) for seat in self.seats]
            return dict(
                revision=self.revision,
                started=self.started,
                game=game.key,
                game_label=game.label,
                side_names=game.seat_names(board),
                draws=game.draws,
                reviewable=game.reviewable,
                analysable=game.analysable,
                review_note=game.review_note,
                size=self.size,
                **game.board_fields(board),
                to_move=board.to_move,
                winner=board.winner,
                drawn=board.is_terminal() and board.winner == 0,
                move_count=board.move_count,
                thinking=self.thinking,
                last_move=self.moves[-1]["cell"] if self.moves else None,
                moves=[dict(m) for m in self.moves],
                win_path=game.win_path(board),
                ckpt=self.ckpt,
                error=self.error,
                info=dict(self.info),
                pace=self.pace,
                paced=game.bot_pause > 0,
                # The general form, one entry per seat in turn order.
                seats=seats,
                seat_ids=list(self.seats),
                # ...and the two-seat spelling of the same thing, which the
                # three duels' half of the page and every test written before
                # Splendor still read.  Absent when there is no such thing.
                black=seats[0] if len(seats) == 2 else None,
                white=seats[1] if len(seats) == 2 else None,
                analysis=self.analysis.snapshot(),
                review=self.review.snapshot(),
                rated=self.rated,
                rating_result=self.rating_result,
                user=self._user_blob(),
                ratings=table,
                your_turn=(not self.thinking and not board.is_terminal()
                           and self.agents[board.to_move] is None),
            )

    def _side(self, colour: int) -> dict:
        agent = self.agents[colour]
        game = self.game
        # A comparison network has its own rating key; the rated network keeps
        # the bare one, so an ordinary game still finds the rating it was given.
        net = self.ckpts[colour]
        key = game.R.bot_key(self.choices[colour], self.sims[colour],
                             None if net == game.analysis_checkpoint() else net)
        entry = game.ratings()["bots"].get(key)
        return dict(choice=self.choices[colour], sims=self.sims[colour], key=key,
                    vary=self.vary[colour], ckpt=self.ckpts[colour],
                    elo=round(float(entry["elo"])) if entry else None,
                    name="you" if agent is None else agent.name,
                    human=agent is None, seat=int(colour))

    def _user_blob(self) -> dict | None:
        if not self.user:
            return None
        R = self.game.R
        profile = R.load_users().get(self.user)
        if profile is None:
            return None
        return dict(name=profile["name"], rating=profile["rating"], games=profile["games"],
                    wins=profile["wins"], losses=profile["losses"],
                    draws=profile.get("draws", 0),
                    provisional=profile["games"] < R.PROVISIONAL_GAMES,
                    history=profile["history"][-8:])


SESSION = Session()

# Shared secret required on every request when the server is reachable from
# outside this machine.  A tunnelled URL is public: it is unlisted, not
# private, and scanners do find them.  None means "localhost only, no gate".
ACCESS_TOKEN: str | None = None


def game_blob(adapter: GameAdapter) -> dict:
    """Everything the page needs in order to offer one game in its picker."""
    R = adapter.R
    return dict(
        key=adapter.key,
        label=adapter.label,
        sides=list(adapter.side_names),
        min_players=adapter.min_players,
        max_players=adapter.max_players,
        draws=adapter.draws,
        bots=adapter.catalog(),
        checkpoints=adapter.checkpoints(),
        sizes=adapter.sizes,
        default_size=adapter.default_size,
        default_opponent=adapter.default_opponent,
        fallback_opponent=adapter.fallback_opponent,
        sim_stops=list(adapter.strength_stops),
        sim_label=adapter.strength_label,
        default_sims=adapter.default_strength,
        dice=adapter.dice,
        reviewable=adapter.reviewable,
        analysable=adapter.analysable,
        # Which script fits an Elo to this game's bots, for the empty ratings
        # table to name.
        ladder=adapter.ladder_script,
        analysis_ckpt=adapter.analysis_checkpoint(),
        users=sorted(R.load_users()),
    )


# ----------------------------------------------------------------- http server
class Handler(BaseHTTPRequestHandler):
    server_version = "GameGUI/1.0"

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        pass

    # ------------------------------------------------------------------ auth
    def _token_ok(self) -> bool:
        """Token from the query string, a header, or the cookie we set on entry."""
        if not ACCESS_TOKEN:
            return True
        if parse_qs(urlparse(self.path).query).get("t", [None])[0] == ACCESS_TOKEN:
            return True
        if self.headers.get("X-Hex-Token") == ACCESS_TOKEN:
            return True
        raw = self.headers.get("Cookie")
        if raw:
            jar = SimpleCookie(raw)
            if "hex_token" in jar and jar["hex_token"].value == ACCESS_TOKEN:
                return True
        return False

    def _deny(self) -> None:
        """Refuse a request that arrived without the access token.

        The page reads JSON and nothing else, so an HTML refusal to one of its
        own fetches is indistinguishable from the server having died -- which
        is exactly how a share link with a stale token used to present: a page
        stuck on "Loading..." with nothing to go on.
        """
        if urlparse(self.path).path.startswith("/api/"):
            return self._send_json(
                dict(error="This link needs its access token. Open the full URL "
                           "printed by share_hex.py."), 401)
        body = (b"<!doctype html><meta name=viewport content='width=device-width'>"
                b"<body style='font:16px system-ui;padding:2rem'>"
                b"<h1>Hex &amp; Connect Four</h1><p>This link needs its access token. "
                b"Open the full URL printed by <code>share_hex.py</code>.</p>")
        self._send_bytes(body, "text/html; charset=utf-8", 401)

    # ------------------------------------------------------------------ verbs
    def do_GET(self) -> None:
        self._guard(self._get)

    def do_POST(self) -> None:
        self._guard(self._post)

    def _guard(self, verb) -> None:
        """Answer the request even when the handler raises.

        An exception reaching the socket layer closes the connection without a
        reply.  The page's ``fetch`` then rejects, the control that made the
        request does nothing whatsoever, and the only trace is a stack trace on
        a console nobody is watching -- every button in the page becomes a dead
        button for as long as the bug lasts.  Answering with the failure lets
        the banner say what happened.
        """
        try:
            verb()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            return  # The remote client disconnected; nothing to send back
        except Exception as exc:  # noqa: BLE001 -- last line of defence
            traceback.print_exc()
            if getattr(self, "_answered", False):
                return  # too late to say anything; the reply is already out
            detail = f"{type(exc).__name__}: {exc}"
            try:
                if urlparse(self.path).path.startswith("/api/"):
                    self._send_json(
                        dict(error=f"the server failed on this request -- {detail}"), 500)
                else:
                    self._send_bytes(
                        b"<!doctype html><body style='font:16px system-ui;padding:2rem'>"
                        + detail.encode("utf-8", "replace"),
                        "text/html; charset=utf-8", 500)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                pass

    def _get(self) -> None:
        path = urlparse(self.path).path
        if not self._token_ok():
            return self._deny()
        if path in ("/", "/index.html"):
            # Carry the token into a cookie so in-page fetches (and any link the
            # phone follows later) do not each need it in the query string.
            if ACCESS_TOKEN:
                self._set_cookie = ACCESS_TOKEN
            return self._send_file(WEB / "index.html", "text/html; charset=utf-8")
        if path == "/api/bots":
            current = SESSION.game
            return self._send_json(dict(
                games=[game_blob(GAMES[k])
                       for k in ("hex", "c4", "uttt", "uxx", "rps2", "bg", "splendor")],
                game=current.key,
                # The current game's menu, flat, as the page has always read it.
                bots=current.catalog(), checkpoints=current.checkpoints(),
                sizes=current.sizes,
                sim_stops=SIM_STOPS, eval_stops=EVAL_STOPS,
                review_stops=REVIEW_STOPS, default_review_sims=DEFAULT_REVIEW_SIMS,
                analysis_ckpt=current.analysis_checkpoint(),
                default_sims=DEFAULT_SIMS, default_eval_sims=DEFAULT_EVAL_SIMS,
                users=sorted(current.R.load_users()),
                provisional_games=current.R.PROVISIONAL_GAMES))
        if path == "/api/state":
            return self._send_json(SESSION.snapshot())
        if path == "/api/training":
            key = parse_qs(urlparse(self.path).query).get("game", [None])[0]
            return self._send_json(TRAINING.get(game_for(key or SESSION.game.key)))
        return self._not_found()

    def _post(self) -> None:
        path = urlparse(self.path).path
        if not self._token_ok():
            return self._deny()
        try:
            body = json.loads(self._read_body() or "{}")
        except json.JSONDecodeError:
            return self._send_json(dict(error="malformed request"), 400)

        if path == "/api/new":
            problem = SESSION.new_game(
                game=body.get("game") or None,
                black=str(body.get("black", "human")),
                white=str(body.get("white", "rule")),
                size=str(body.get("size") or ""),
                ckpt=str(body.get("ckpt") or ""),
                seed=int(body.get("seed", 0)),
                black_sims=int(body.get("black_sims", DEFAULT_SIMS)),
                white_sims=int(body.get("white_sims", DEFAULT_SIMS)),
                black_vary=bool(body.get("black_vary", False)),
                white_vary=bool(body.get("white_vary", False)),
                black_ckpt=body.get("black_ckpt") or None,
                white_ckpt=body.get("white_ckpt") or None,
                # One entry per seat, for the games that seat more than two.
                # Absent for the duels, which send black/white above.
                players=body.get("players") or None,
                rated=bool(body.get("rated", False)))
            return self._send_json(SESSION.snapshot(), 200, problem)
        if path == "/api/opponent":
            # ``seat`` is the general form -- an index into the turn order --
            # and ``colour`` is the two-seat spelling the duels' page sends.
            if body.get("seat") is not None:
                index = max(0, min(int(body["seat"]), len(SESSION.seats) - 1))
                seat = SESSION.seats[index]
            else:
                seat = SESSION.seats[0 if str(body.get("colour")) == "black" else 1]
            problem = SESSION.set_opponent(seat, str(body.get("choice", "human")),
                                           int(body.get("sims", DEFAULT_SIMS)),
                                           bool(body.get("vary", False)),
                                           body.get("ckpt") or None)
            return self._send_json(SESSION.snapshot(), 200, problem)
        if path == "/api/review":
            with SESSION.lock:
                game = SESSION.game
                moves = [m["cell"] for m in SESSION.moves]
                size = SESSION.size
            problem = SESSION.review.start(game, moves, size, game.analysis_checkpoint(),
                                           int(body.get("sims", DEFAULT_REVIEW_SIMS)))
            return self._send_json(SESSION.snapshot(), 200, problem)
        if path == "/api/pace":
            SESSION.set_pace(float(body.get("pace", 1.0)))
            return self._send_json(SESSION.snapshot())
        if path == "/api/analysis":
            SESSION.analysis.configure(bool(body.get("enabled", False)),
                                       int(body.get("max_sims", DEFAULT_EVAL_SIMS)))
            SESSION.analysis.set_position(SESSION.game, SESSION.board,
                                          SESSION.game.analysis_checkpoint())
            return self._send_json(SESSION.snapshot())
        if path == "/api/user":
            R = SESSION.game.R
            problem = None
            if body.get("action") == "create":
                _, problem = R.create_user(str(body.get("name", "")))
            if problem is None:
                SESSION.select_user(str(body.get("name", "")).strip() or None)
            payload = dict(SESSION.snapshot(), users=sorted(R.load_users()))
            return self._send_json(payload, 200, problem)
        if path == "/api/move":
            problem = SESSION.play_human(SESSION.game.parse_move(body, SESSION.board))
            return self._send_json(SESSION.snapshot(), 200 if problem is None else 409,
                                   problem)
        if path == "/api/undo":
            problem = SESSION.undo()
            return self._send_json(SESSION.snapshot(), 200 if problem is None else 409,
                                   problem)
        return self._not_found()

    # ---------------------------------------------------------------- helpers
    def _not_found(self) -> None:
        """404, in the language the caller was speaking."""
        if urlparse(self.path).path.startswith("/api/"):
            return self._send_json(dict(error="no such endpoint"), 404)
        self._answered = True
        self.send_error(404)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send_json(self, payload: dict, status: int = 200,
                   problem: str | None = None) -> None:
        if problem:
            payload = dict(payload, problem=problem)
        self._send_bytes(json.dumps(payload).encode("utf-8"),
                         "application/json; charset=utf-8", status)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            return self.send_error(404, f"missing {path.name}")
        self._send_bytes(data, content_type)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        # Once this runs the reply is committed, and `_guard` can no longer
        # turn a later failure into an answer.
        self._answered = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        token = getattr(self, "_set_cookie", None)
        if token:
            self.send_header("Set-Cookie",
                             f"hex_token={token}; Path=/; Max-Age=2592000; SameSite=Lax")
        self.end_headers()
        self.wfile.write(data)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7333)
    ap.add_argument("--game", choices=sorted(GAMES), default=DEFAULT_GAME,
                    help="which game the page opens on (either can be picked in the UI)")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser window")
    ap.add_argument("--token", default=None,
                    help="require this token on every request; implied (and generated) "
                         "whenever --host is not loopback")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                    help="inference device for AlphaZero models (auto prefers GPU with CPU fallback)")
    args = ap.parse_args(argv)

    global ACCESS_TOKEN, SESSION, DEVICE
    DEVICE = args.device
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if args.token:
        ACCESS_TOKEN = args.token
    elif not loopback:
        # Binding beyond loopback without a gate would leave the machine open to
        # anyone who can reach the port, so refuse to do it silently.
        ACCESS_TOKEN = secrets.token_urlsafe(12)
    if args.game != SESSION.game.key:
        SESSION = Session(args.game)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "localhost" if loopback else args.host
    url = f"http://{shown}:{server.server_address[1]}/"
    if ACCESS_TOKEN:
        url += f"?t={ACCESS_TOKEN}"
        print(f"access token: {ACCESS_TOKEN}")
    print(f"Hex + Connect Four GUI on {url}   (ctrl-c to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
