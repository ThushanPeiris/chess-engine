"""
Trivial baseline: plays a uniformly random legal move every time. This
is not meant to be competitive - it exists purely as a sanity check.
If your real engine can't consistently and heavily beat this, something
is genuinely broken; if it does, that's expected and not itself
evidence of strength (any correct engine beats random easily).
"""

import chess
import random


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    return random.choice(list(board.legal_moves)).uci()