"""MoHex (Benzene) over GTP, as an ordinary Hex agent.

MoHex is the University of Alberta's Monte-Carlo Hex engine -- Computer
Olympiad champion for years, with a virtual-connection engine and inferior-cell
analysis behind its search.  It is here as a *yardstick*: every other opponent
in this project is at least 800 Elo below the network, so the ladder can no
longer say how strong the network actually is.  A fixed external engine at a
fixed time per move can.

It runs inside WSL (Benzene is a Linux build), one engine process per agent,
single-threaded so it gets the same one core per game that a network worker
does.

The build needs ``tools/benzene_patches.diff``.  Two of its three fixes are
real engine bugs that a modern toolchain exposes, both in positions MoHex has
already proven won: ``Resistance::ComputeScores`` segfaulted once a colour's
edges were joined by fill-in, and with that fixed, ``genmove`` returned an
invalid move when fill-in covered every empty cell.  An unpatched MoHex
crashes out of a typical 11x11 match within a few games.

Coordinates match: Benzene names a cell by column letter and row number, and
its Black joins north to south -- the same as :func:`hex_game.move_to_str` and
this project's Black.  ``tests/test_mohex.py`` checks that rather than assuming
it.

The swap is where the two differ, and only in bookkeeping.  Benzene's
``swap-pieces`` leaves the stone where it is and **exchanges the players'
colours**: the swapper now plays Black, and it is White's turn.  This project
mirrors the stone across the long diagonal and changes its colour, so the same
colours keep the same edges and Black moves again.  Transposing the board and
exchanging the colours maps one onto the other exactly, so after a swap this
bridge sends every move transposed and colour-flipped, and reads every reply
back the same way.
"""

from __future__ import annotations

import shutil
import subprocess

from ..hex_game import BLACK, EMPTY, WHITE, HexBoard, move_to_str, str_to_move
from .base import Agent

DEFAULT_BINARY = "/opt/hex/benzene-vanilla-cmake/build/src/mohex/mohex"


class GtpError(RuntimeError):
    pass


class GtpProcess:
    """A GTP engine on the other end of a pipe."""

    def __init__(self, argv: list[str], stderr_path: str | None = None):
        # The engine's log goes nowhere unless asked for: it is chatty, and a
        # pipe nobody reads would eventually fill and stall the engine.  When
        # an engine dies, pass a path and the reason is in that file.
        self._stderr = open(stderr_path, "a") if stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._stderr, text=True, bufsize=1)

    transcript: list[str] | None = None  # set to a list to record every command

    def send(self, command: str) -> str:
        assert self.proc.stdin and self.proc.stdout
        if self.transcript is not None:
            self.transcript.append(command)
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise GtpError(f"engine exited while answering {command!r}")
            line = line.rstrip("\r\n")
            if line == "" and lines:
                break
            if line or lines:
                lines.append(line)
        reply = "\n".join(lines)
        if reply.startswith("?"):
            raise GtpError(f"{command!r} -> {reply}")
        return reply[1:].strip()

    def close(self) -> None:
        try:
            self.send("quit")
        except Exception:
            pass
        self.proc.kill()


def wsl_argv(binary: str = DEFAULT_BINARY, distro: str = "Ubuntu") -> list[str]:
    """Command line that starts ``binary`` in WSL, or directly on Linux."""
    if shutil.which("wsl.exe") or shutil.which("wsl"):
        return ["wsl.exe", "-d", distro, "--", binary, "--quiet"]
    return [binary, "--quiet"]


class MoHexAgent(Agent):
    """MoHex at a fixed time per move (or a fixed number of search games)."""

    def __init__(self, seconds: float = 1.0, max_games: int | None = None,
                 threads: int = 1, binary: str = DEFAULT_BINARY,
                 name: str | None = None, swap: bool = False,
                 stderr_path: str | None = None):
        self.seconds = seconds
        self.max_games = max_games
        self.threads = threads
        self.swap = swap
        self.name = name or (f"mohex@{max_games}g" if max_games else f"mohex@{seconds:g}s")
        self.engine = GtpProcess(wsl_argv(binary), stderr_path)
        self.size: int | None = None
        self.sent = bytearray()  # this project's board as last sent, in its own frame
        self.flipped = False     # has a swap made Benzene's frame the transpose of ours?

    def _configure(self, n: int) -> None:
        e = self.engine
        e.send(f"boardsize {n} {n}")
        e.send(f"param_mohex num_threads {self.threads}")
        e.send("param_mohex ponder 0")
        e.send("param_mohex reuse_subtree 1")
        e.send("param_mohex use_time_management 0")
        e.send(f"param_mohex max_time {self.seconds}")
        e.send(f"param_mohex max_games {self.max_games or 99999999}")
        e.send(f"param_game allow_swap {1 if self.swap else 0}")
        self.size = n

    def reset(self) -> None:
        if self.size is not None:
            self.engine.send("clear_board")
        self.sent = bytearray()
        self.flipped = False

    # ---------------------------------------------------- frame translation
    def _to_engine_cell(self, cell: int, n: int) -> str:
        if self.flipped:
            r, c = divmod(cell, n)
            cell = c * n + r
        return move_to_str(cell, n)

    def _from_engine_cell(self, text: str, n: int) -> int:
        cell = str_to_move(text, n)
        if self.flipped:
            r, c = divmod(cell, n)
            cell = c * n + r
        return cell

    def _to_engine_colour(self, colour: int) -> str:
        black = colour == BLACK
        return "b" if black != self.flipped else "w"

    def _record_swap(self, n: int) -> None:
        """Our frame after a swap: the opening stone mirrored and turned White."""
        x = self.sent.index(BLACK)
        r, c = divmod(x, n)
        self.sent[x] = EMPTY
        self.sent[c * n + r] = WHITE
        self.flipped = True

    # ---------------------------------------------------------------- play
    def _sync(self, board: HexBoard) -> None:
        """Bring the engine's board up to date with ``board``.

        Hex positions do not depend on move order, so sending the new stones
        with explicit colours is exact.  The one move that is not a new stone
        is the swap, recognised by the opening Black stone no longer being
        Black -- gone, or turned White where the diagonal maps it onto itself.
        """
        n = board.n
        stones = [c for c in range(n * n) if self.sent[c] != EMPTY]
        if (board.swap_rule and not self.flipped and len(stones) == 1
                and self.sent[stones[0]] == BLACK and board.board[stones[0]] != BLACK):
            self.engine.send("play w swap-pieces")
            self._record_swap(n)
        for cell in range(n * n):
            stone = board.board[cell]
            if stone != self.sent[cell]:
                if self.sent[cell] != EMPTY:
                    raise GtpError("board went backwards; call reset() between games")
                self.engine.send(f"play {self._to_engine_colour(stone)} "
                                 f"{self._to_engine_cell(cell, n)}")
                self.sent[cell] = stone

    def select_move(self, board: HexBoard, last_move: int | None = None) -> int:
        n = board.n
        if self.size != n:
            self._configure(n)
            self.engine.send("clear_board")
            self.sent = bytearray()
            self.flipped = False
        if len(self.sent) != n * n:
            self.sent = bytearray(n * n)
        self._sync(board)
        colour = self._to_engine_colour(board.to_move)
        reply = self.engine.send(f"genmove {colour}").lower()
        if reply == "swap-pieces":
            self._record_swap(n)
            return board.SWAP
        if reply in ("resign", "pass"):
            # A resigning MoHex has seen a proven loss; any legal cell will do.
            move = int(board.legal_moves()[0])
            self.engine.send(f"play {colour} {self._to_engine_cell(move, n)}")
        else:
            move = self._from_engine_cell(reply, n)
        self.sent[move] = board.to_move
        return move

    def close(self) -> None:
        self.engine.close()

    def __del__(self):
        try:
            self.engine.proc.kill()
        except Exception:
            pass
