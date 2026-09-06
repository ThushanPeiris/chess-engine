import chess
import chess.engine

# Start up the Stockfish engine as a subprocess and open a UCI connection to it.
# This path is relative to the project root (~/Downloads/chess-engine) -
# it assumes you run this script from there, with the extracted "stockfish"
# folder sitting alongside it.
engine = chess.engine.SimpleEngine.popen_uci(
    "stockfish/stockfish-macos-x86-64-bmi2"
)

# A small, varied set of test positions:
# - the standard starting position
# - a common opening line (1.e4 e5)
# - a simple developed middlegame-ish position
# - a bare king+pawn endgame
# Testing across different position "types" here (not just one FEN) helps
# catch bugs that only show up in certain phases of the game - e.g. sign
# errors, or the engine choking on sparse endgame positions.
test_fens = [
    chess.STARTING_FEN,
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # after 1.e4 e5
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",  # Italian-ish
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",  # simple king+pawn endgame
]

# Loop over each test position, evaluate it with Stockfish, and print the result.
for fen in test_fens:
    board = chess.Board(fen)

    # Skip positions where the side to move is in check.
    # This mirrors the filtering you'll want in the real data pipeline:
    # positions "in check" tend to have volatile, less stable evaluations,
    # which makes for noisier training labels later on.
    if board.is_check():
        continue

    # Ask Stockfish to search this position to a fixed depth (not a fixed
    # time) - depth 14 here is just a starting point for testing; the real
    # data-generation run will need to balance depth (label quality) against
    # how many positions you can label in the time you have.
    info = engine.analyse(board, chess.engine.Limit(depth=14))

    # info["score"] is a PovScore - i.e. it's relative to whichever side
    # is to move in that position. Calling .pov(chess.WHITE) normalizes it
    # so every printed score is always "from White's perspective," which
    # is the consistent sign convention you want before this data ever
    # touches a training script - a subtle sign bug here would silently
    # corrupt every label you generate downstream.
    score = info["score"].pov(chess.WHITE)

    # info["pv"] is Stockfish's principal variation - the sequence of moves
    # it considers best. pv[0] is just the best move for this position.
    best_move = info["pv"][0] if info.get("pv") else "none"

    print(fen)
    print("  eval:", score)
    print("  best move:", best_move)

# Always close the engine process cleanly when done, so it doesn't linger
# as an orphaned subprocess.
engine.quit()