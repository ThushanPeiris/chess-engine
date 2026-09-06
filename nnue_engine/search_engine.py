import chess
import chess.polyglot
import numpy as np
import onnxruntime as ort
import math
import time

# ============================================================
# TRAINED NETWORK (replaces the piece-square-table evaluation)
# ============================================================
# Loaded ONCE at module import, not inside evaluate() - matches the
# competition's documented model, since import time is covered by the
# 90s init budget, separate from the per-move match clock. Loading it
# inside evaluate() would reload the model on every single call, which
# would be both slow and wrong.
#
# onnxruntime is used for inference rather than torch directly: it's
# generally faster for pure inference and has a much smaller runtime
# footprint, which matters when every millisecond of the per-move
# time budget counts.

NNUE_MODEL_PATH = "value_network.onnx"

try:
    _nnue_session = ort.InferenceSession(NNUE_MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"Failed to load NNUE model from '{NNUE_MODEL_PATH}' - this must "
        f"succeed at import time, since a failure here would otherwise "
        f"only surface on the first real move of a match. Original "
        f"error: {e}"
    )

# Must exactly match the encoding used in train_network.py's
# fen_to_features - any mismatch here would silently feed the network
# garbage input it was never trained on, without necessarily crashing.
_PIECE_TO_INDEX = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

# Engineered features (material balance, PST score) - must exactly match
# train_network.py's PIECE_VALUES and piece-square tables, since these
# feed the network as extra inputs alongside raw board placement.
_PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

_PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
_BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
_ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
_QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
_KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
_PIECE_SQUARE_TABLES = {
    chess.PAWN: _PAWN_TABLE, chess.KNIGHT: _KNIGHT_TABLE, chess.BISHOP: _BISHOP_TABLE,
    chess.ROOK: _ROOK_TABLE, chess.QUEEN: _QUEEN_TABLE, chess.KING: _KING_TABLE,
}


def _pst_value(piece_type: int, square: int, is_white: bool) -> int:
    table = _PIECE_SQUARE_TABLES[piece_type]
    if is_white:
        row = 7 - (square // 8)
        col = square % 8
    else:
        row = square // 8
        col = square % 8
    return table[row * 8 + col]


def _compute_engineered_features(board: chess.Board):
    """Must exactly match train_network.py's compute_engineered_features."""
    material = 0
    pst = 0
    for piece_type, value in _PIECE_VALUES.items():
        for square in board.pieces(piece_type, chess.WHITE):
            material += value
            pst += _pst_value(piece_type, square, is_white=True)
        for square in board.pieces(piece_type, chess.BLACK):
            material -= value
            pst -= _pst_value(piece_type, square, is_white=False)

    if board.turn == chess.BLACK:
        material = -material
        pst = -pst

    return material / 3000.0, pst / 200.0


def _board_to_nnue_features(board: chess.Board) -> np.ndarray:
    """
    771 features: 12 piece types x 64 squares (one-hot) + 1 side-to-move
    flag + 2 engineered features (material balance, PST score). Must be
    identical to train_network.py's encoding - this is duplicated rather
    than imported because the competition environment won't have
    train_network.py available, only whatever ships in agent.zip.
    """
    features = np.zeros(771, dtype=np.float32)
    for square, piece in board.piece_map().items():
        idx = _PIECE_TO_INDEX[(piece.piece_type, piece.color)]
        features[idx * 64 + square] = 1.0
    features[768] = 1.0 if board.turn == chess.WHITE else 0.0

    material_normalized, pst_normalized = _compute_engineered_features(board)
    features[769] = material_normalized
    features[770] = pst_normalized

    return features


def evaluate(board: chess.Board) -> int:
    """
    Trained-network evaluation, replacing the piece-square-table version.
    Returns a centipawn-EQUIVALENT score from the perspective of the side
    to move - same convention as before, so nothing else in the search
    (alpha-beta bounds, mate sentinels, quiescence stand-pat) needs to
    change.

    The network outputs a WDL (win probability) in [0, 1], since that's
    what it was trained to predict (see train_network.py). We convert
    that back to a centipawn-equivalent number via the inverse of the
    same sigmoid transform used to build the training targets, purely so
    the output stays on a familiar, debuggable scale and stays safely
    far below the -999999 checkmate sentinel used elsewhere in the
    search.

    Terminal positions (checkmate/stalemate/insufficient material) are
    still special-cased directly rather than left to the network - this
    matches standard practice: exact game-over conditions are cheap and
    unambiguous to compute directly, and there's no reason to trust a
    learned approximation for something known with certainty.
    """
    if board.is_checkmate():
        return -999999

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    features = _board_to_nnue_features(board).reshape(1, -1)
    wdl = _nnue_session.run(None, {"board_features": features})[0][0][0]

    # Clip away from the exact 0/1 boundary before the inverse-sigmoid
    # log, which would otherwise divide by zero or take log(0) at the
    # extremes.
    wdl = min(max(float(wdl), 1e-6), 1 - 1e-6)
    cp_equivalent = -400.0 * math.log10(1.0 / wdl - 1.0)

    return int(cp_equivalent)


# Still needed for MVV-LVA move ordering (captures ranked by piece
# value), even though it's no longer used inside evaluate() itself.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# ============================================================
# MOVE ORDERING
# ============================================================
# Alpha-beta pruning only prunes effectively if good moves are tried
# first - if the best move is checked last, you get none of the benefit.
# Three techniques combined here, tried in this priority order:
#
# 1. Transposition table move: if we've searched this exact position
#    before (possibly via a different move order reaching the same
#    position - a "transposition"), we already know what looked best.
# 2. MVV-LVA (Most Valuable Victim - Least Valuable Aggressor): for
#    captures, prefer capturing a valuable piece with a cheap one (e.g.
#    pawn takes queen) over the reverse - these are typically the
#    strongest tactical moves.
# 3. Killer moves: quiet (non-capture) moves that caused a beta-cutoff
#    in a SIBLING branch at the same depth are tried early here too,
#    since a move that refuted one line often refutes similar ones.

MAX_KILLER_DEPTH = 64
killer_moves = [[None, None] for _ in range(MAX_KILLER_DEPTH)]


def _mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    if victim is None or attacker is None:
        return 0
    # Multiply victim value up so it always dominates attacker value in
    # the comparison - we want "victim value" to be the primary sort key
    # and "cheaper attacker is better" to be the tiebreaker.
    return PIECE_VALUES[victim.piece_type] * 100 - PIECE_VALUES[attacker.piece_type]


def order_moves(board: chess.Board, moves, tt_move, depth: int):
    def move_priority(move):
        if tt_move is not None and move == tt_move:
            return 1_000_000  # always try the TT's remembered best move first
        if board.is_capture(move):
            return 100_000 + _mvv_lva_score(board, move)
        if depth < MAX_KILLER_DEPTH and move in killer_moves[depth]:
            return 50_000
        return 0

    return sorted(moves, key=move_priority, reverse=True)


# ============================================================
# TRANSPOSITION TABLE
# ============================================================
# Caches search results keyed by a Zobrist hash of the position (a fast,
# near-collision-free fingerprint of the board state - python-chess
# provides this built in via chess.polyglot.zobrist_hash, so we don't
# need to implement our own hashing scheme). If the same position is
# reached again - very common, since different move orders often lead
# to identical positions - we can reuse the earlier result instead of
# re-searching the whole subtree.
#
# Each entry stores: depth searched, the resulting score, a bound type
# (EXACT / LOWERBOUND / UPPERBOUND - needed because alpha-beta cutoffs
# mean a stored score isn't always the exact value, only a bound on it),
# and the best move found, which also feeds move ordering above.

TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

transposition_table = {}


# ============================================================
# TIME MANAGEMENT
# ============================================================
# Two separate problems, both handled here:
#
# 1. HOW MUCH time to spend on this move, given the actual clock state
#    (allocate_time, below) - not just "search for N seconds" as a fixed
#    constant, which either wastes time in easy positions or risks
#    running out of clock in a long game.
# 2. HOW TO STOP MID-SEARCH the instant that budget is used up, rather
#    than only checking between depths. A single iterative-deepening
#    depth can take dramatically longer than the previous one (branching
#    factor growth), so if you only check the clock between depths, one
#    "one depth too many" attempt can blow through your entire remaining
#    time and lose the game on time forfeit - regardless of how good the
#    moves found were. This is a hard correctness requirement, not an
#    optimization.

class SearchTimeout(Exception):
    """Raised to unwind the search immediately once the deadline passes."""
    pass


# How often (in nodes) to check the clock. Even 256 wasn't tight enough -
# a self-play test still lost one game on time in a long (67-ply) game,
# from small overshoot compounding across many moves rather than one
# single catastrophic overshoot. Positions with more checks/tactics cost
# more per node than quiet positions (board.gives_check(move) internally
# does a hypothetical push/pop, which isn't cheap), so worst-case
# per-interval cost is higher than the "clean opening position" numbers
# it was originally tuned against. 64 trades a bit more checking overhead
# for meaningfully tighter safety.
TIME_CHECK_INTERVAL = 64


def allocate_time(remaining_seconds: float, increment_seconds: float, ply_count: int) -> float:
    """
    Decides how much time to spend on THIS move, given the actual clock
    state - not a fixed constant. The core idea: divide your remaining
    time by roughly how many moves you expect to still need to play, so
    you spend time roughly evenly across the whole game rather than
    exhausting your clock early or hoarding it needlessly late.

    ply_count is the number of half-moves played so far in the game -
    used to estimate how many moves are likely left (games tend to run
    somewhere around 40-80 total moves; this is a simple, commonly used
    heuristic, not a precise prediction).
    """
    # Panic mode: when very little time is left, abandon the normal
    # formula entirely. A self-play test showed that even careful
    # per-move budgeting can accumulate small overshoot across many
    # moves in a long game - by the time the clock is this low, the
    # priority is guaranteed not to flag, not finding the best possible
    # move. This forces a very shallow, very fast search rather than
    # trusting the formula below to behave safely at extremes it wasn't
    # really designed for.
    if remaining_seconds < 1.0:
        return max(0.01, remaining_seconds * 0.1)

    # Assume the game has roughly 40 total moves (80 plies); as more
    # moves are played, fewer are assumed to remain. Floor of 10 moves
    # left keeps this sane in unusually long games rather than assuming
    # almost no time is needed near some fixed cutoff.
    estimated_moves_left = max(10, 40 - ply_count // 2)

    base_time = remaining_seconds / estimated_moves_left

    # We get the increment back every move regardless of how much of it
    # we use, so it's close to "free" time - but reserve a small fraction
    # of it as a safety margin against network/processing overhead beyond
    # pure search time (writing the move, engine communication, etc.).
    time_for_move = base_time + increment_seconds * 0.8

    # Hard safety cap: never commit more than 25% of remaining time to a
    # single move, no matter what the formula above suggests. Protects
    # against the estimate being badly wrong early in a long game.
    max_allowed = remaining_seconds * 0.25
    time_for_move = min(time_for_move, max_allowed)

    # Reserve an absolute buffer so the search always returns with a
    # little real time left over, rather than cutting it exactly to zero.
    safety_buffer = 0.1
    time_for_move = max(0.05, time_for_move - safety_buffer)

    return time_for_move


# ============================================================
# QUIESCENCE SEARCH
# ============================================================

QUIESCENCE_MAX_DEPTH = 8


def quiescence(board: chess.Board, alpha: int, beta: int, nodes_counter: list, deadline: float, depth: int = 0) -> int:
    nodes_counter[0] += 1

    if nodes_counter[0] % TIME_CHECK_INTERVAL == 0 and time.time() >= deadline:
        raise SearchTimeout()

    if board.is_checkmate():
        return -999999 - (QUIESCENCE_MAX_DEPTH - depth)

    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    if depth >= QUIESCENCE_MAX_DEPTH:
        return stand_pat

    noisy_moves = [
        m for m in board.legal_moves
        if board.is_capture(m) or m.promotion or board.gives_check(m)
    ]
    noisy_moves = order_moves(board, noisy_moves, tt_move=None, depth=depth)

    for move in noisy_moves:
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, nodes_counter, deadline, depth + 1)
        finally:
            board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


# ============================================================
# MAIN SEARCH
# ============================================================

class SearchResult:
    def __init__(self, best_move, score, nodes_searched, depth_reached):
        self.best_move = best_move
        self.score = score
        self.nodes_searched = nodes_searched
        self.depth_reached = depth_reached


def negamax(board: chess.Board, depth: int, alpha: int, beta: int, nodes_counter: list, deadline: float) -> int:
    nodes_counter[0] += 1

    if nodes_counter[0] % TIME_CHECK_INTERVAL == 0 and time.time() >= deadline:
        raise SearchTimeout()

    original_alpha = alpha

    zobrist_key = chess.polyglot.zobrist_hash(board)
    tt_entry = transposition_table.get(zobrist_key)
    tt_move = None
    if tt_entry is not None:
        tt_move = tt_entry.get("move")
        if tt_entry["depth"] >= depth:
            if tt_entry["flag"] == TT_EXACT:
                return tt_entry["score"]
            elif tt_entry["flag"] == TT_LOWERBOUND:
                alpha = max(alpha, tt_entry["score"])
            elif tt_entry["flag"] == TT_UPPERBOUND:
                beta = min(beta, tt_entry["score"])
            if alpha >= beta:
                return tt_entry["score"]

    if board.is_checkmate():
        return -999999 - depth

    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
        return 0

    if depth == 0:
        return quiescence(board, alpha, beta, nodes_counter, deadline)


    best_score = float("-inf")
    best_move_here = None

    moves = order_moves(board, list(board.legal_moves), tt_move, depth)

    for move in moves:
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, nodes_counter, deadline)
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move_here = move
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            # This move caused a cutoff. If it's a quiet (non-capture)
            # move, remember it as a killer for this depth - it's likely
            # to be strong in sibling positions too.
            if not board.is_capture(move) and depth < MAX_KILLER_DEPTH:
                if move != killer_moves[depth][0]:
                    killer_moves[depth][1] = killer_moves[depth][0]
                    killer_moves[depth][0] = move
            break

    # Store this result in the transposition table for future lookups.
    if best_score <= original_alpha:
        flag = TT_UPPERBOUND
    elif best_score >= beta:
        flag = TT_LOWERBOUND
    else:
        flag = TT_EXACT
    transposition_table[zobrist_key] = {
        "depth": depth,
        "score": best_score,
        "flag": flag,
        "move": best_move_here,
    }

    return best_score


def find_best_move(board: chess.Board, depth: int, deadline: float) -> SearchResult:
    """
    Searches at a fixed depth. Note this can raise SearchTimeout partway
    through - the caller (iterative_deepening_search) is responsible for
    deciding what to do with a result that got interrupted before every
    root move was considered.
    """
    nodes_counter = [0]
    best_move = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    zobrist_key = chess.polyglot.zobrist_hash(board)
    tt_entry = transposition_table.get(zobrist_key)
    tt_move = tt_entry["move"] if tt_entry else None

    moves = order_moves(board, list(board.legal_moves), tt_move, depth)

    for move in moves:
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, nodes_counter, deadline)
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    return SearchResult(best_move, best_score, nodes_counter[0], depth)


def iterative_deepening_search(board: chess.Board, time_budget_seconds: float) -> SearchResult:
    """
    Searches at depth 1, 2, 3, ... until time_budget_seconds is used up.

    IMPORTANT: if a depth gets interrupted partway through by
    SearchTimeout, its result is DISCARDED, not returned. An interrupted
    depth means some root moves were fully evaluated and others weren't
    even tried - comparing "the best of the moves I got to" against
    moves that were skipped entirely isn't a fair or safe comparison, so
    we fall back to the last FULLY completed depth's result instead. This
    is the standard, safe way to combine iterative deepening with a hard
    time cutoff.
    """
    start_time = time.time()
    deadline = start_time + time_budget_seconds

    last_completed_result = None
    depth = 1

    while True:
        if time.time() >= deadline:
            break

        try:
            result = find_best_move(board, depth, deadline)
        except SearchTimeout:
            print(f"  depth {depth}: timed out mid-search - "
                  f"keeping depth {depth - 1}'s result")
            break

        last_completed_result = result

        elapsed = time.time() - start_time
        print(f"  depth {depth}: best={result.best_move}, "
              f"score={result.score}, nodes={result.nodes_searched}, "
              f"time={elapsed:.2f}s")

        if abs(result.score) > 999000:
            break  # found a forced mate - no point searching deeper

        depth += 1

    return last_completed_result


if __name__ == "__main__":
    # Demo 1: fixed time budget, as before - useful for quick manual testing.
    board = chess.Board()
    print("Searching starting position with a fixed 5s budget...")
    result = iterative_deepening_search(board, time_budget_seconds=5.0)
    print(f"\nFinal choice: {result.best_move} "
          f"(score={result.score}, depth={result.depth_reached}, "
          f"nodes={result.nodes_searched})")

    # Demo 2: realistic clock-based allocation, as it would actually be
    # used in a game - e.g. 90 seconds left on the clock, 0.5s increment,
    # 12 plies (6 full moves) played so far.
    print("\n--- Simulating clock-based time allocation ---")
    remaining_clock = 90.0
    increment = 0.5
    ply_count = 12
    budget = allocate_time(remaining_clock, increment, ply_count)
    print(f"Clock: {remaining_clock}s remaining, {increment}s increment, "
          f"ply {ply_count} -> allocated {budget:.2f}s for this move")

    transposition_table.clear()  # fresh position, avoid stale demo-1 entries
    result = iterative_deepening_search(board, time_budget_seconds=budget)
    print(f"\nFinal choice: {result.best_move} "
          f"(score={result.score}, depth={result.depth_reached}, "
          f"nodes={result.nodes_searched})")