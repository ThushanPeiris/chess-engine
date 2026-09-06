"""
Competition entry point. The published interface requires exactly one
function:

    def get_move(fen: str, time_left_ms: int) -> str

One process serves one game, started fresh each time. Module-level state
(the transposition table and killer-move table imported below) persists
in memory across every get_move() call WITHIN a single game - the docs
confirm this explicitly - but starts empty again for each new game,
since a new process is launched per game.

IMPORTANT CORRECTION from earlier in this project: the docs state the
process is fully SUSPENDED while the opponent is thinking. A background
"pondering" thread - discussed earlier as a plausible optimization -
would never actually get CPU time in this environment and was based on
an outdated read of the rules. Nothing here attempts it.
"""

import chess
from search_engine import iterative_deepening_search, allocate_time

# Fixed by the competition's published time control (120s + 0.5s/move).
# If this ever changes, it's the one constant to update.
INCREMENT_SECONDS = 0.5


def _estimate_ply_count(fen: str) -> int:
    """
    The wire protocol only ever gives us the current FEN and our
    remaining time - no move history. Standard FEN already encodes the
    side to move and the fullmove number though, which is enough to
    derive an approximate ply count for allocate_time's "how many moves
    are probably left" estimate, with no need to track state across
    calls ourselves.
    """
    parts = fen.split()
    side_to_move = parts[1]        # 'w' or 'b'
    fullmove_number = int(parts[5])
    return 2 * (fullmove_number - 1) + (0 if side_to_move == "w" else 1)


def get_move(fen: str, time_left_ms: int) -> str:
    """
    Required entry point. MUST always return a legal move in UCI
    notation - a malformed or illegal return, or an uncaught exception
    here, both count as an immediate loss per the competition's failure
    reference (same as running out of time). Every path below is
    wrapped so we always return *something* legal, even if the search
    itself fails in some unexpected way.
    """
    board = chess.Board(fen)

    try:
        remaining_seconds = time_left_ms / 1000.0
        ply_count = _estimate_ply_count(fen)
        time_budget = allocate_time(remaining_seconds, INCREMENT_SECONDS, ply_count)

        result = iterative_deepening_search(board, time_budget)

        if result is not None and result.best_move is not None:
            return result.best_move.uci()
    except Exception:
        # Any unexpected failure in search must NOT crash the process -
        # a crash is an instant loss, exactly like an illegal move or a
        # flag. Fall through to the safe fallback below instead of
        # letting this propagate.
        pass

    # Safe fallback: if the search failed or somehow returned nothing,
    # play literally any legal move rather than risk returning nothing
    # or a malformed string. A mediocre move can lose a game eventually;
    # a crash or illegal-move forfeit loses it immediately.
    legal_moves = list(board.legal_moves)
    return legal_moves[0].uci()