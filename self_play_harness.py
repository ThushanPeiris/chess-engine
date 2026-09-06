"""
Self-play testing harness.

Plays N full games between two agents (each in its own subprocess,
communicating over the documented JSON wire protocol - see runner.py),
alternating which agent plays White each game, and reports a summary:
wins/draws/losses for each agent, plus a rough Elo-difference estimate.

This is what lets you actually PROVE a change helped, rather than
guessing from a handful of manual test positions - e.g. once the NNUE
net is trained, run it here against the current piece-square-table
version and look at the actual score, not just "it looks smarter."

Usage example (see the bottom of this file):
    run_match("agent_a_dir", "agent_b_dir", num_games=10,
              time_control_ms=10_000, increment_ms=100)

A short time_control_ms/increment_ms (e.g. 10s+0.1s) is deliberately
useful during development - fast enough to run many games quickly.
Before trusting a real conclusion about strength, re-run at least a
sample of games at the ACTUAL competition clock (120000ms + 500ms), since
time pressure can change which engine performs better.
"""

import subprocess
import json
import time
import sys
import threading
import collections
import chess

MAX_PLIES = 300  # matches the competition's adjudication rule


class GameResult:
    def __init__(self, winner, reason, final_board, move_history=None):
        self.winner = winner  # "white", "black", or None (draw)
        self.reason = reason  # e.g. "checkmate", "illegal", "flag", "crash", "adjudication"
        self.final_board = final_board
        self.move_history = move_history or []  # SAN moves, in order


def _start_agent_process(agent_dir):
    """
    Starts an agent subprocess AND a background thread that continuously
    drains its stderr into a small rolling buffer.

    This matters more than it looks: the agent's search prints
    depth-by-depth progress on every move (redirected to stderr by the
    wire-protocol runner). subprocess.PIPE has a fixed OS buffer size
    (~64KB) - if nothing ever reads from it, that buffer fills up over
    a long game's worth of debug output, and the CHILD PROCESS BLOCKS
    trying to write to a full pipe. This is a classic subprocess
    deadlock, and it's exactly what caused a real-clock game to hang
    indefinitely during testing on this project - not slowness, an
    actual stall. Continuously draining stderr in the background (and
    keeping only the last N lines, since that's all that's useful for
    diagnostics) prevents this entirely.
    """
    process = subprocess.Popen(
        [sys.executable, "runner.py", agent_dir],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    stderr_buffer = collections.deque(maxlen=200)

    def _drain_stderr():
        try:
            for line in process.stderr:
                stderr_buffer.append(line)
        except Exception:
            pass  # process closed/died - nothing more to drain

    drain_thread = threading.Thread(target=_drain_stderr, daemon=True)
    drain_thread.start()

    process._stderr_buffer = stderr_buffer  # stash for diagnostics later
    return process


def _request_move(process, fen, time_left_ms, hard_timeout_seconds):
    """
    Sends one request and waits for one response line, with a hard
    wall-clock timeout as a safety net - if a subprocess hangs entirely
    (rather than just running slow), we don't want the harness to hang
    forever with it. Returns None on any failure (timeout, crash,
    malformed response) - the caller treats None as a loss for this side,
    matching the competition's actual failure handling.

    On any failure, this also prints the subprocess's stderr - without
    this, every failure just looks like a generic "crash_or_malformed"
    with no way to tell what actually went wrong. That's a real gap:
    debugging a real failure without seeing the actual traceback is far
    harder than it needs to be.
    """
    try:
        request = json.dumps({"fen": fen, "time_left_ms": time_left_ms})
        process.stdin.write(request + "\n")
        process.stdin.flush()
    except (BrokenPipeError, OSError):
        _print_stderr_diagnostic(process, "stdin write failed - process already dead")
        return None  # the process has already died

    start = time.time()
    while True:
        if process.poll() is not None:
            _print_stderr_diagnostic(process, "process exited unexpectedly")
            return None  # process exited unexpectedly - a crash

        if time.time() - start > hard_timeout_seconds:
            _print_stderr_diagnostic(process, "hard timeout - process appears hung")
            return None  # hung - treat like a flag/crash

        line = process.stdout.readline()
        if line:
            try:
                response = json.loads(line)
                return response.get("move")
            except json.JSONDecodeError:
                _print_stderr_diagnostic(process, f"malformed response: {line!r}")
                return None  # malformed output counts as an illegal move
        # if no line yet, loop and keep checking (simple polling approach -
        # fine at this scale, not worth a more complex async setup here)


def _print_stderr_diagnostic(process, reason):
    """
    Surfaces the subprocess's recent stderr when something goes wrong,
    reading from the rolling buffer that _start_agent_process's
    background thread has been continuously filling - NOT via
    communicate(), since that thread is already consuming the pipe
    (calling communicate() here would race with it and typically find
    nothing left to read).
    """
    print(f"\n[DIAGNOSTIC] Move request failed: {reason}")
    try:
        if process.poll() is None:
            process.terminate()
        buffer = getattr(process, "_stderr_buffer", None)
        if buffer:
            print(f"[DIAGNOSTIC] Subprocess stderr (last {len(buffer)} lines):")
            print("".join(buffer))
        else:
            print("[DIAGNOSTIC] Subprocess produced no stderr output.")
    except Exception as e:
        print(f"[DIAGNOSTIC] Could not retrieve stderr: {e}")
    print()


INIT_BUDGET_SECONDS = 90  # matches the competition's documented init budget


def _warm_up_process(process, side_name):
    """
    Sends one throwaway request to a freshly-started process, using the
    competition's actual init budget (90s) as the timeout, and discards
    the timing entirely - it is NOT charged against that side's match
    clock. This mirrors the real competition's separation between init
    time (covered separately, before the clock starts) and match time.

    Without this, any agent with real import-time cost - e.g. loading an
    ONNX model - gets that cost silently charged against its FIRST
    move's time budget in this harness, which doesn't happen in the real
    environment. That's exactly what caused a real, reproducible "flag
    on move 1" loss for an NNUE-based agent during testing on this
    project, purely from model-load time eating a fast test clock -  not
    from anything wrong with the agent's actual chess logic.
    """
    dummy_board = chess.Board()
    move = _request_move(process, dummy_board.fen(), 0, INIT_BUDGET_SECONDS)
    if move is None:
        print(f"[WARNING] {side_name} failed to respond even during warm-up "
              f"(within the {INIT_BUDGET_SECONDS}s init budget) - this agent "
              f"would likely fail to start at all in the real competition.")


def play_one_game(agent_a_dir, agent_b_dir, a_plays_white,
                   time_control_ms, increment_ms):
    """
    Plays one full game. a_plays_white controls which physical agent is
    White this game - callers alternate this across a match so neither
    agent gets an unfair colour advantage over the whole run.
    """
    white_dir, black_dir = (agent_a_dir, agent_b_dir) if a_plays_white else (agent_b_dir, agent_a_dir)

    white_process = _start_agent_process(white_dir)
    black_process = _start_agent_process(black_dir)

    # Warm-up round: forces each process to finish importing and loading
    # any model BEFORE the real, timed game begins - this time is not
    # charged against either side's clock, matching the real
    # competition's separate init budget.
    _warm_up_process(white_process, "White")
    _warm_up_process(black_process, "Black")

    board = chess.Board()
    clocks_ms = {"white": time_control_ms, "black": time_control_ms}
    move_history = []

    try:
        while True:
            if board.is_checkmate():
                winner = "black" if board.turn == chess.WHITE else "white"
                return GameResult(winner, "checkmate", board, move_history)

            if board.is_stalemate() or board.is_insufficient_material() or \
               board.can_claim_fifty_moves() or board.is_repetition(3):
                return GameResult(None, "draw", board, move_history)

            if board.ply() >= MAX_PLIES:
                # Adjudicate on material, per the competition's own rule.
                material = sum(
                    len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
                    for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
                )
                if material > 0:
                    return GameResult("white", "adjudication", board, move_history)
                elif material < 0:
                    return GameResult("black", "adjudication", board, move_history)
                else:
                    return GameResult(None, "adjudication", board, move_history)

            side = "white" if board.turn == chess.WHITE else "black"
            process = white_process if side == "white" else black_process

            # Give a generous wall-clock safety margin beyond the side's
            # own remaining time - this catches a genuinely hung process
            # without being so tight that normal search near the buzzer
            # gets falsely flagged as a hang.
            hard_timeout = (clocks_ms[side] / 1000.0) + 5.0

            move_start = time.time()
            move_uci = _request_move(process, board.fen(), clocks_ms[side], hard_timeout)
            elapsed_ms = (time.time() - move_start) * 1000

            clocks_ms[side] -= elapsed_ms
            if clocks_ms[side] <= 0:
                winner = "black" if side == "white" else "white"
                return GameResult(winner, "flag", board, move_history)

            if move_uci is None:
                winner = "black" if side == "white" else "white"
                return GameResult(winner, "crash_or_malformed", board, move_history)

            try:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    raise ValueError("illegal move")
            except (ValueError, chess.InvalidMoveError):
                winner = "black" if side == "white" else "white"
                return GameResult(winner, "illegal", board, move_history)

            move_history.append(board.san(move))
            board.push(move)
            clocks_ms[side] += increment_ms  # increment lands after moving, per the docs

    finally:
        for p in (white_process, black_process):
            try:
                p.stdin.close()
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                p.kill()


def run_match(agent_a_dir, agent_b_dir, num_games=10,
              time_control_ms=10_000, increment_ms=100):
    """
    Runs a full match, alternating colours each game, and prints a
    running log plus a final summary. Returns (a_wins, b_wins, draws)
    for further use (e.g. computing your own Elo estimate downstream).
    """
    a_wins = 0
    b_wins = 0
    draws = 0

    for game_num in range(num_games):
        a_plays_white = (game_num % 2 == 0)
        result = play_one_game(agent_a_dir, agent_b_dir, a_plays_white,
                                time_control_ms, increment_ms)

        if result.winner is None:
            draws += 1
            outcome = "draw"
        else:
            a_won = (result.winner == "white") == a_plays_white
            if a_won:
                a_wins += 1
                outcome = "A wins"
            else:
                b_wins += 1
                outcome = "B wins"

        colour_note = "A=White" if a_plays_white else "A=Black"
        print(f"Game {game_num + 1}/{num_games} ({colour_note}): "
              f"{outcome} ({result.reason}, {result.final_board.ply()} plies)")

        # Auto-print the full transcript for suspiciously short games -
        # a real checkmate almost never happens this fast from a normal
        # opening, so this is worth seeing immediately rather than
        # discovering later that a fast, repeated loss happened without
        # ever capturing what actually occurred.
        SUSPICIOUSLY_SHORT_PLIES = 15
        if result.reason == "checkmate" and result.final_board.ply() < SUSPICIOUSLY_SHORT_PLIES:
            print(f"  [SHORT GAME - full move list]: {' '.join(result.move_history)}")

    total = a_wins + b_wins + draws
    print(f"\n--- Match summary over {total} games ---")
    print(f"A: {a_wins} wins, B: {b_wins} wins, {draws} draws")

    # Standard score-percentage-to-Elo-difference conversion. Treat this
    # as a rough indicator, not a precise rating - with only a handful of
    # games the statistical noise is large; see the note below about
    # running more games (or a proper SPRT) before trusting small
    # differences.
    score_fraction = (a_wins + 0.5 * draws) / total if total > 0 else 0.5
    if 0 < score_fraction < 1:
        import math
        elo_diff = -400 * math.log10(1 / score_fraction - 1)
        print(f"A's score: {score_fraction:.1%} "
              f"(~{elo_diff:+.0f} Elo vs B, rough estimate)")
    else:
        print("A's score: 100% or 0% - Elo estimate undefined at the extremes; "
              "run more games for a meaningful number.")

    print("\nNote: this is a simple win/loss tally, not a statistically "
          "rigorous test. With few games, a real Elo gain and pure chance "
          "can look similar. For anything you're about to commit to, run "
          "more games than this default, and consider implementing a "
          "proper SPRT stopping rule rather than eyeballing a fixed count.")

    return a_wins, b_wins, draws


if __name__ == "__main__":
    # Sanity check: the real engine (agent.py) against a random-move
    # baseline (random_agent.py). If this ISN'T a near-total wipeout,
    # something in the search or agent wiring is genuinely broken -
    # this is the first thing to check before trusting any other result
    # from this harness.
    print("Sanity check: real engine vs random-move baseline")
    print("(expect A to win almost every game)\n")
    run_match(
        agent_a_dir="real_engine",
        agent_b_dir="random_baseline",
        num_games=6,
        time_control_ms=5_000,   # short clock for fast iteration
        increment_ms=50,
    )