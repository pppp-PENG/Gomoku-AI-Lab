from __future__ import annotations

import numpy as np

RULE_VERSION = "freestyle-v1"


class Game:
    """Black=1, white=-1. to_play flips even after a terminal move."""

    def __init__(self, size: int = 9, connect: int = 5, moves=None):
        if not 5 <= size <= 19 or not 3 <= connect <= size:
            raise ValueError("size must be 5..19 and connect 3..size")
        self.size, self.connect = size, connect
        self.board = np.zeros((size, size), dtype=np.int8)
        self.moves: list[int] = []
        self.to_play = 1
        self.winner: int | None = None
        for action in moves or []:
            self.play(action)

    def clone(self):
        other = Game(self.size, self.connect)
        other.board = self.board.copy()
        other.moves = self.moves.copy()
        other.to_play, other.winner = self.to_play, self.winner
        return other

    @property
    def terminal(self):
        return self.winner is not None

    def legal_mask(self):
        return (self.board.ravel() == 0) & (not self.terminal)

    def play(self, action: int):
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        if self.terminal or not 0 <= action < self.size**2 or self.board.ravel()[action]:
            raise ValueError("illegal move or finished game")
        r, c = divmod(int(action), self.size)
        player = self.to_play
        self.board[r, c] = player
        self.moves.append(int(action))
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                rr, cc = r + dr * sign, c + dc * sign
                while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == player:
                    count += 1
                    rr, cc = rr + dr * sign, cc + dc * sign
            if count >= self.connect:
                self.winner = player
                break
        if self.winner is None and len(self.moves) == self.size**2:
            self.winner = 0
        self.to_play = -player
        return self

    def undo(self, count=1):
        if count < 0:
            raise ValueError("negative undo")
        keep = self.moves[:max(0, len(self.moves) - count)]
        self.__init__(self.size, self.connect, keep)

    def encode(self):
        return np.stack((self.board == self.to_play, self.board == -self.to_play,
                         np.ones_like(self.board)), axis=0).astype(np.float32)

    def terminal_wdl(self):
        if not self.terminal:
            raise ValueError("not terminal")
        return np.array([self.winner == self.to_play, self.winner == 0,
                         self.winner == -self.to_play], dtype=np.float32)

    def as_dict(self):
        return dict(size=self.size, connect=self.connect, rules=RULE_VERSION,
                    moves=self.moves, board=self.board.tolist(), to_play=self.to_play,
                    winner=self.winner, terminal=self.terminal)


def flip_wdl(wdl):
    return np.asarray(wdl)[..., [2, 1, 0]].copy()


def augment(state, policy, symmetry: int):
    """Apply the same D4 transform to channels and action labels."""
    if not 0 <= symmetry < 8:
        raise ValueError("symmetry must be 0..7")
    n = state.shape[-1]
    x, p = state.copy(), policy.reshape(n, n).copy()
    if symmetry >= 4:
        x, p = np.flip(x, -1), np.flip(p, -1)
    return (np.rot90(x, symmetry % 4, axes=(-2, -1)).copy(),
            np.rot90(p, symmetry % 4).copy().ravel())


def coordinate(action, size):
    r, c = divmod(int(action), size)
    return f"{chr(65+c)}{r+1}"
