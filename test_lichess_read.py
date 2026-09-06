import zstandard
import chess.pgn
import io

def is_acceptable_time_control(game):
    """
    Skip bullet games (very short time controls) since they tend to
    have less considered play and a narrower position distribution.
    TimeControl is formatted as "initial+increment" in seconds, e.g.
    "60+0" for 1-minute bullet, "300+3" for 5-minute blitz.
    We keep games with at least a Blitz-level time budget or slower.
    """
    tc = game.headers.get("TimeControl", "")
    if tc == "-" or tc == "?":
        return False  # unknown/correspondence, skip
    try:
        initial_seconds = int(tc.split("+")[0])
    except (ValueError, IndexError):
        return False
    return initial_seconds >= 180  # 3+ minutes: excludes bullet, keeps blitz/rapid/classical

with open("lichess_db_standard_rated_2015-05.pgn.zst", "rb") as compressed_file:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(compressed_file) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        accepted = 0
        skipped = 0
        while accepted < 5:  # just testing on a handful first
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break  # end of file
            if is_acceptable_time_control(game):
                accepted += 1
                print(game.headers["Event"], game.headers["TimeControl"])
            else:
                skipped += 1

        print(f"\nAccepted: {accepted}, Skipped: {skipped}")