"""
Trains the value network that replaces search_engine.py's evaluate()
function in nnue_engine/.

- Input: 768 binary features (12 piece types x 64 squares, one-hot) plus
  a side-to-move flag, PLUS two hand-engineered features (material
  balance and piece-square-table score) - 771 inputs total.

  The two engineered features were added after a real self-play test
  showed the pure-raw-placement network (769 inputs) came out roughly
  even with, or slightly behind, the existing piece-square-table
  evaluator (45% score over 10 games). That version has to learn "the
  center matters" and "material matters" entirely from scratch, from a
  single pass over moderate-depth labels - concepts the PST version
  already encodes directly. Feeding the network the PST/material scores
  as extra inputs gives it that positional prior for free, rather than
  asking it to rediscover well-known chess knowledge unaided.

- Output: a single WDL (win/draw/loss) probability via sigmoid, not raw
  centipawns. Centipawn evals have huge outliers that would otherwise
  dominate the loss and distort training in the -200..+200 range where
  real decisions actually happen. Passing eval through a sigmoid
  compresses those outliers while preserving resolution where it matters.

- Small network, sized to comfortably fit the 50MB submission cap and
  run fast enough for real-time search on one CPU core with no GPU.

IMPORTANT: the feature encoding here must exactly match the
inference-side encoding in nnue_engine/search_engine.py. They're
duplicated across both files rather than shared, since the competition
environment only ships whatever's in agent.zip - keep any change to one
in sync with the other.
"""

import csv
import time
import chess
import torch
import torch.nn as nn
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# FEATURE ENCODING
# ============================================================

PIECE_TO_INDEX = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}

# Reused from the original piece-square-table evaluator - these are the
# standard, publicly documented "Simplified Evaluation Function" tables
# (Tomasz Michniewski, Chess Programming Wiki), used here as an
# engineered input feature rather than as the evaluation itself.
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

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
    chess.PAWN: PAWN_TABLE, chess.KNIGHT: KNIGHT_TABLE, chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE, chess.QUEEN: QUEEN_TABLE, chess.KING: KING_TABLE,
}


def _pst_value(piece_type: int, square: int, is_white: bool) -> int:
    table = PIECE_SQUARE_TABLES[piece_type]
    if is_white:
        row = 7 - (square // 8)
        col = square % 8
    else:
        row = square // 8
        col = square % 8
    return table[row * 8 + col]


def compute_engineered_features(board: chess.Board):
    """
    Material balance and PST score, each computed from White's
    perspective then flipped to the side-to-move's perspective - same
    convention as the WDL target itself. Normalized to roughly [-1, 1]
    for typical positions so they sit on a comparable scale to the 0/1
    one-hot placement features.
    """
    material = 0
    pst = 0
    for piece_type, value in PIECE_VALUES.items():
        for square in board.pieces(piece_type, chess.WHITE):
            material += value
            pst += _pst_value(piece_type, square, is_white=True)
        for square in board.pieces(piece_type, chess.BLACK):
            material -= value
            pst -= _pst_value(piece_type, square, is_white=False)

    if board.turn == chess.BLACK:
        material = -material
        pst = -pst

    material_normalized = material / 3000.0  # roughly a queen+rook lead at 1.0
    pst_normalized = pst / 200.0             # typical PST swings are tens to ~100

    return material_normalized, pst_normalized


def fen_to_features_from_board(board: chess.Board) -> torch.Tensor:
    """
    771 inputs: 768 one-hot piece-placement features (12 piece types x
    64 squares) + 1 side-to-move flag + 2 engineered features (material
    balance, PST score). Takes an already-parsed Board (rather than a
    FEN string) so callers that already have a Board don't pay for
    parsing it twice.
    """
    features = torch.zeros(771, dtype=torch.float32)

    for square, piece in board.piece_map().items():
        idx = PIECE_TO_INDEX[(piece.piece_type, piece.color)]
        features[idx * 64 + square] = 1.0

    features[768] = 1.0 if board.turn == chess.WHITE else 0.0

    material_normalized, pst_normalized = compute_engineered_features(board)
    features[769] = material_normalized
    features[770] = pst_normalized

    return features


def fen_to_features(fen: str) -> torch.Tensor:
    """Convenience wrapper when you only have a FEN string, not a Board."""
    return fen_to_features_from_board(chess.Board(fen))


def eval_to_wdl_target(eval_str: str, side_to_move_is_white: bool) -> float:
    """
    Converts a stored eval (centipawns, always from White's perspective
    per the data-generation script) into a WDL-style probability target
    from the perspective of the side to move - matching evaluate()'s
    existing convention in search_engine.py, so this network is a
    genuine drop-in replacement with no sign-convention surprises.

    Mate scores ("#5", "#-3") are treated as near-certain win/loss rather
    than converted through the centipawn formula, since mate distance
    isn't a centipawn-comparable quantity.
    """
    if eval_str.startswith("#"):
        mate_in = int(eval_str[1:])
        white_wdl = 0.999 if mate_in > 0 else 0.001
    else:
        cp = int(eval_str)
        white_wdl = 1.0 / (1.0 + 10 ** (-cp / 400.0))

    if side_to_move_is_white:
        return white_wdl
    else:
        return 1.0 - white_wdl


def _precompute_one(row):
    """
    Runs in a worker process (see PositionDataset below). Parses a FEN
    exactly ONCE and produces both the feature vector and target in one
    pass, rather than the two redundant chess.Board() parses per sample
    an earlier version did.
    """
    fen, eval_str = row
    board = chess.Board(fen)
    features = fen_to_features_from_board(board)
    target = eval_to_wdl_target(eval_str, board.turn == chess.WHITE)
    return features, target


class PositionDataset(Dataset):
    """
    NOTE - a known, deliberate simplification: this splits train/val by
    ROW, not by GAME. Positions from the same game are correlated (they
    share a lot of board structure), so a stricter split would group by
    game and hold out whole games for validation. The current CSV format
    from generate_training_data.py doesn't retain a game identifier, only
    FEN + eval per row, so a proper by-game split isn't available without
    regenerating data with an added game-id column. Flagging this
    honestly rather than silently accepting a validation number that's a
    bit more optimistic than a true held-out-game split would give.

    PERFORMANCE NOTE - an earlier version re-parsed each FEN into a
    chess.Board from scratch inside __getitem__, TWICE per sample, and
    since DataLoader re-accesses every sample every epoch, that meant the
    expensive parsing work was redone in full for all positions, every
    single epoch. On a 1,000,000-position dataset over 20 epochs, that
    turned into many hours of pure parsing overhead - by far the
    dominant cost, completely swamping the actual (tiny) cost of
    training this small a network. This version precomputes every
    feature vector and target ONCE, in parallel across CPU cores, before
    training starts - after that, __getitem__ is just a tensor index
    lookup, and epochs run in a small fraction of the previous time.
    """
    def __init__(self, csv_path, num_workers=None):
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((row["fen"], row["eval"]))

        print(f"Precomputing features for {len(rows)} positions "
              f"(this happens once, not every epoch)...")
        start = time.time()

        num_workers = num_workers or mp.cpu_count()
        with mp.Pool(num_workers) as pool:
            results = pool.map(_precompute_one, rows, chunksize=256)

        self.features = torch.stack([r[0] for r in results])
        self.targets = torch.tensor([[r[1]] for r in results], dtype=torch.float32)

        elapsed = time.time() - start
        print(f"Done in {elapsed:.1f}s ({len(rows) / elapsed:.0f} positions/sec)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ============================================================
# MODEL
# ============================================================

class ValueNetwork(nn.Module):
    """
    Small enough to comfortably fit the 50MB submission cap and run fast
    on one CPU core with no GPU.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(771, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # output is a WDL probability in [0, 1]
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# TRAINING
# ============================================================

def train(csv_path, output_path="value_network.pt", epochs=20, batch_size=256, learning_rate=1e-3):
    dataset = PositionDataset(csv_path)
    print(f"Loaded {len(dataset)} positions from {csv_path}")

    val_fraction = 0.1
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = ValueNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()  # target is a probability in [0,1], BCE is the standard fit

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * features.size(0)
        train_loss = train_loss_total / len(train_set)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                predictions = model(features)
                loss = loss_fn(predictions, targets)
                val_loss_total += loss.item() * features.size(0)
        val_loss = val_loss_total / len(val_set)

        # Save the checkpoint with the BEST validation loss seen so far,
        # not just whatever the final epoch happens to produce - training
        # loss can (and often does) keep dropping past the point where
        # the model is actually still improving on unseen data.
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)

        marker = " <- best so far, saved" if improved else ""
        print(f"epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")

    print(f"\nBest val_loss: {best_val_loss:.4f} - saved to {output_path}")

    # Reload the BEST checkpoint before returning, so export_to_onnx
    # exports the version that actually generalizes best.
    model.load_state_dict(torch.load(output_path))
    return model


def export_to_onnx(model, output_path="value_network.onnx"):
    """
    Exports to ONNX so the agent can run inference via onnxruntime.
    Uses the classic dynamic_axes argument rather than the newer
    dynamic_shapes API - dynamic_axes works across a much wider range of
    torch versions (dynamic_shapes requires a fairly recent torch and
    isn't available at all in older ones, like 2.2.x). On very new torch
    versions this may print a suggestion to use dynamic_shapes instead -
    that's just a suggestion, not an error, safe to ignore here.
    """
    model.eval()
    dummy_input = torch.zeros(1, 771, dtype=torch.float32)
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["board_features"],
        output_names=["win_probability"],
        dynamic_axes={"board_features": {0: "batch_size"}, "win_probability": {0: "batch_size"}},
    )
    print(f"Exported to {output_path}")


if __name__ == "__main__":
    model = train("training_positions.csv", epochs=20)
    export_to_onnx(model)