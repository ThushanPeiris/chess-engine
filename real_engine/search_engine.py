import chess
import chess.polyglot
import time

# ============================================================
# EVALUATION
# ============================================================
# Two upgrades from the pure-material version:
#
# 1. Piece-square tables (PST): a bonus/penalty added to each piece's
#    value depending on which square it sits on. This is what actually
#    fixes the "Nh3" problem you saw - pure material counting can't tell
#    a good developing move from a bad one (no material changes either
#    way), but PSTs directly encode "knights are worth more in the
#    center, less on the rim" as a numeric bonus. This is standard,
#    well-established chess programming knowledge (the widely-used
#    "Simplified Evaluation Function" tables, originally by Tomasz
#    Michniewski, documented on the Chess Programming Wiki) - these are
#    plain numeric lookup tables, not anyone's creative writing, and
#    they're explicitly published for engine builders to use directly.
#
# 2. This is STILL a placeholder for the eventual NNUE net - PSTs are a
#    well-known intermediate step, better than pure material but far
#    less accurate than a trained network. Once your net is ready, this
#    whole evaluate() function gets replaced by a call into the model;
#    nothing else in the search needs to change.

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Each table is written from White's point of view, with index 0 = a8
# (top-left when White is at the bottom) and index 63 = h1, matching
# python-chess's square numbering (a1=0 ... h8=63) when read bottom-to-top.
# To keep this readable, tables below are written top-to-bottom as you'd
# see the board from White's side, then flipped in code.

PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}


def _pst_value(piece_type: int, square: int, is_white: bool) -> int:
    """
    Looks up a piece-square bonus. The tables above are written from
    White's perspective (rank 8 first). python-chess squares are
    numbered a1=0 through h8=63, so for White we need to read the table
    "upside down" relative to how it's written, and for Black we mirror
    the square vertically (rank 1 becomes rank 8, etc.) since the tables
    are symmetric between the two sides.
    """
    table = PIECE_SQUARE_TABLES[piece_type]
    if is_white:
        # square 0 (a1) should read the table's LAST row (index 56-63),
        # square 63 (h8) should read the table's FIRST row (index 0-7).
        row = 7 - (square // 8)
        col = square % 8
    else:
        row = square // 8
        col = square % 8
    return table[row * 8 + col]


def evaluate(board: chess.Board) -> int:
    """
    Material + piece-square tables, from the perspective of the side to
    move (negamax convention - see the search section below).
    """
    if board.is_checkmate():
        return -999999

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for piece_type, value in PIECE_VALUES.items():
        for square in board.pieces(piece_type, chess.WHITE):
            score += value + _pst_value(piece_type, square, is_white=True)
        for square in board.pieces(piece_type, chess.BLACK):
            score -= value + _pst_value(piece_type, square, is_white=False)

    return score if board.turn == chess.WHITE else -score


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

    # ============================================================
    # NULL-MOVE PRUNING
    # ============================================================
    # Idea: if we "pass" (give the opponent a free move) and we're STILL
    # doing fine even after that, our actual position must be strong
    # enough that we don't need to search it deeply - we can cut here.
    # This is a real, well-established Elo gain, but it has one classic
    # failure mode: in zugzwang positions (mainly certain endgames),
    # having to move is actually a disadvantage - "passing" would be
    # BETTER than any real move available, so the null-move assumption
    # (passing tells us about our worst case) is backwards. The standard
    # safeguard is to disable null-move pruning whenever the side to
    # move has only pawns and a king left (no minor/major pieces) -
    # exactly the material profile where zugzwang is common - and to
    # never use it while in check (a "free" move can't be given there
    # regardless).
    NULL_MOVE_REDUCTION = 2  # search the resulting position 2 ply shallower
    has_non_pawn_material = any(
        board.pieces(pt, board.turn)
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    if (depth >= 3 and not board.is_check() and has_non_pawn_material
            and beta < 999000):  # skip near mate scores, where pruning is unsafe
        board.push(chess.Move.null())
        try:
            null_score = -negamax(board, depth - 1 - NULL_MOVE_REDUCTION,
                                   -beta, -beta + 1, nodes_counter, deadline)
        finally:
            board.pop()
        if null_score >= beta:
            return beta  # even giving the opponent a free move, we're still fine here

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