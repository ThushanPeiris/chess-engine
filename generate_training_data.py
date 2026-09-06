import zstandard
import chess
import chess.pgn
import chess.engine
import io
import csv
import os
import time
import multiprocessing as mp

# ---- Configuration ----
# These are the "optimal for your timeline" values discussed: depth 13
# balances label quality against how many positions/hour you can generate;
# skipping the first 10 plies avoids wasting labels on pure opening theory;
# sampling every 6th ply keeps positions varied rather than near-duplicates.
INPUT_PGN_ZST = "lichess_db_standard_rated_2015-05.pgn.zst"
OUTPUT_CSV = "training_positions.csv"
STOCKFISH_PATH = "stockfish/stockfish-macos-x86-64-bmi2"
STOCKFISH_DEPTH = 13
SKIP_FIRST_PLIES = 10
SAMPLE_EVERY_N_PLIES = 6
MIN_TIME_CONTROL_SECONDS = 180  # excludes bullet, keeps blitz/rapid/classical
TARGET_POSITIONS = 1_000_000    # overnight-scale real run
NUM_WORKERS = mp.cpu_count()    # use every core available on your machine
                                 # (the competition's 1-core limit only applies
                                 # to the *submitted agent*, not to this
                                 # offline data-generation step)


def is_acceptable_time_control(game):
    """
    Filter out bullet games. TimeControl is "initial+increment" in seconds
    (e.g. "300+5"). We keep games with at least MIN_TIME_CONTROL_SECONDS
    on the clock, which favours more considered play and a broader mix
    of position types than bullet games alone would give us.
    """
    tc = game.headers.get("TimeControl", "")
    if tc in ("-", "?"):
        return False
    try:
        initial_seconds = int(tc.split("+")[0])
    except (ValueError, IndexError):
        return False
    return initial_seconds >= MIN_TIME_CONTROL_SECONDS


def position_generator():
    """
    Streams the compressed PGN file game by game (never writing a full
    decompressed .pgn to disk), and yields sampled FENs from accepted
    games. This runs in the main process - it's inherently sequential,
    since reading the PGN stream one game at a time is how python-chess
    works - the parallelism happens later, in labeling.
    """
    with open(INPUT_PGN_ZST, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    return  # end of file

                if not is_acceptable_time_control(game):
                    continue

                board = game.board()
                for ply_index, move in enumerate(game.mainline_moves()):
                    board.push(move)

                    if ply_index < SKIP_FIRST_PLIES:
                        continue
                    if ply_index % SAMPLE_EVERY_N_PLIES != 0:
                        continue
                    if board.is_check():
                        continue

                    yield board.fen()


# ---- Worker process setup ----
# Each worker process opens its OWN Stockfish engine once, and reuses it
# for every position it labels. Opening a fresh engine per position would
# add huge overhead (each launch has its own startup cost) - this is why
# we use a pool initializer rather than starting Stockfish inside the
# labeling function itself.
_worker_engine = None


def _init_worker():
    global _worker_engine
    _worker_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


def _label_position(fen):
    """
    Runs in a worker process. Analyses one position at the configured
    depth and returns (fen, eval) as a plain string - PovScore objects
    aren't easily written to CSV, so we normalize to White's perspective
    here and convert to a simple string/number before returning.
    """
    board = chess.Board(fen)
    info = _worker_engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
    score = info["score"].pov(chess.WHITE)

    # Centipawn scores and mate scores need different handling - a mate
    # score has no meaningful centipawn number, so we encode it distinctly
    # (e.g. "#5" for mate in 5) rather than forcing it into a cp value.
    if score.is_mate():
        eval_str = f"#{score.mate()}"
    else:
        eval_str = str(score.score())

    return fen, eval_str


def count_existing_positions():
    """
    If OUTPUT_CSV already exists from a previous (possibly interrupted)
    run, count how many positions are already in it (excluding the header
    row). This lets us resume rather than starting from zero after a
    crash, sleep interruption, or accidental terminal close during a
    long overnight run.
    """
    if not os.path.exists(OUTPUT_CSV):
        return 0
    with open(OUTPUT_CSV, "r") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 1)  # subtract the header row


def resumable_position_generator(already_done):
    """
    Wraps position_generator(), skipping the first `already_done`
    positions it would yield. Since the underlying PGN stream is read
    in a fixed, deterministic order (same file, same filtering config),
    re-running this from the start and skipping the first N positions
    reliably lands us back where a previous run left off - no need to
    track file offsets or game indices separately.
    """
    gen = position_generator()
    for _ in range(already_done):
        next(gen, None)
    yield from gen


def main():
    already_done = count_existing_positions()
    if already_done > 0:
        print(f"Found {already_done} existing positions in {OUTPUT_CSV} - resuming.")

    remaining_target = TARGET_POSITIONS - already_done
    if remaining_target <= 0:
        print(f"Already have {already_done} positions, target of "
              f"{TARGET_POSITIONS} already met. Nothing to do.")
        return

    print(f"Using {NUM_WORKERS} worker processes")
    print(f"Target: {TARGET_POSITIONS} total labeled positions at depth {STOCKFISH_DEPTH} "
          f"({remaining_target} remaining)")

    start_time = time.time()
    labeled_count = 0

    # Append mode (not "w") so a resumed run adds to the existing file
    # rather than overwriting the positions already labeled. Only write
    # the header if we're starting fresh - a resumed file already has one.
    file_mode = "a" if already_done > 0 else "w"
    with open(OUTPUT_CSV, file_mode, newline="") as csv_file:
        writer = csv.writer(csv_file)
        if already_done == 0:
            writer.writerow(["fen", "eval"])

        # A pool of persistent worker processes, each with its own
        # Stockfish engine (via _init_worker). imap preserves order and
        # streams results back as they complete, so we can write to disk
        # incrementally - if this run gets interrupted partway through,
        # everything written so far is still on disk and usable, and the
        # resume logic above picks up from there on the next run.
        with mp.Pool(NUM_WORKERS, initializer=_init_worker) as pool:
            positions = resumable_position_generator(already_done)
            for fen, eval_str in pool.imap(_label_position, positions, chunksize=8):
                writer.writerow([fen, eval_str])
                labeled_count += 1

                if labeled_count % 500 == 0:
                    csv_file.flush()  # ensure progress is actually saved to disk
                    elapsed = time.time() - start_time
                    rate = labeled_count / elapsed
                    total_so_far = already_done + labeled_count
                    print(f"{total_so_far}/{TARGET_POSITIONS} total positions "
                          f"({rate:.1f}/sec this run, {elapsed:.0f}s elapsed)")

                if labeled_count >= remaining_target:
                    break

    elapsed = time.time() - start_time
    total_so_far = already_done + labeled_count
    print(f"\nDone this run. {labeled_count} new positions in {elapsed:.0f}s "
          f"({labeled_count / elapsed:.1f}/sec). "
          f"Total in file: {total_so_far}/{TARGET_POSITIONS}")


if __name__ == "__main__":
    main()