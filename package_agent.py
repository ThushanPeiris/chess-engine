"""
Packages the final submission and verifies it actually works from a
clean extraction - not just "it works in my dev folder where other
files happen to also be sitting around." This catches a whole class of
bug where the agent works for you locally but fails for the real
runner, because it was accidentally relying on some file outside what
actually ships in the zip.

Run this from the project root, with nnue_engine/agent.py,
nnue_engine/search_engine.py, and value_network_int8.onnx (produced by
quantize_model.py) all present.
"""

import os
import shutil
import zipfile
import subprocess
import sys
import tempfile

SOURCE_AGENT_DIR = "nnue_engine"
QUANTIZED_MODEL = "value_network_int8.onnx"
OUTPUT_ZIP = "agent.zip"
MAX_UNZIPPED_MB = 50


def build_zip():
    """
    Stages exactly the files that should ship, at the zip's ROOT (not
    inside a subfolder) - the competition's documented requirement.
    Uses the quantized model, renamed to value_network.onnx so
    search_engine.py's NNUE_MODEL_PATH constant doesn't need to change
    between dev testing and the final submission.
    """
    with tempfile.TemporaryDirectory() as staging_dir:
        shutil.copy(os.path.join(SOURCE_AGENT_DIR, "agent.py"), staging_dir)
        shutil.copy(os.path.join(SOURCE_AGENT_DIR, "search_engine.py"), staging_dir)
        shutil.copy(QUANTIZED_MODEL, os.path.join(staging_dir, "value_network.onnx"))

        staged_files = os.listdir(staging_dir)
        print(f"Staging: {staged_files}")

        if os.path.exists(OUTPUT_ZIP):
            os.remove(OUTPUT_ZIP)

        with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in staged_files:
                zf.write(os.path.join(staging_dir, filename), arcname=filename)

    print(f"Created {OUTPUT_ZIP}")


def check_size():
    with zipfile.ZipFile(OUTPUT_ZIP) as zf:
        total_unzipped = sum(info.file_size for info in zf.infolist())
    total_mb = total_unzipped / (1024 * 1024)
    print(f"Unzipped size: {total_mb:.2f} MB (cap: {MAX_UNZIPPED_MB} MB)")
    if total_mb > MAX_UNZIPPED_MB:
        raise RuntimeError(
            f"agent.zip exceeds the {MAX_UNZIPPED_MB}MB unzipped cap - "
            f"this WILL be rejected by the competition."
        )
    print("Within size limit.")


def verify_clean_extraction():
    """
    The real test: extract agent.zip into a completely fresh, empty
    directory - nothing else present, no dev-folder files it could be
    accidentally leaning on - and run an actual move through it exactly
    as the real runner would. If this works, the zip is genuinely
    self-contained.
    """
    with tempfile.TemporaryDirectory() as clean_dir:
        with zipfile.ZipFile(OUTPUT_ZIP) as zf:
            zf.extractall(clean_dir)

        print(f"Extracted to clean directory: {os.listdir(clean_dir)}")

        test_script = (
            "import chess\n"
            "from agent import get_move\n"
            "board = chess.Board()\n"
            "move = get_move(board.fen(), 120000)\n"
            "assert chess.Move.from_uci(move) in board.legal_moves, 'ILLEGAL MOVE'\n"
            "print('OK - move returned:', move)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            cwd=clean_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print("VERIFICATION FAILED. stderr:")
            print(result.stderr)
            raise RuntimeError(
                "agent.zip does not work from a clean extraction - "
                "do not submit this until fixed."
            )

        print(result.stdout.strip())
        print("\nVerification passed: agent.zip is self-contained and works "
              "from a completely clean extraction.")


if __name__ == "__main__":
    build_zip()
    check_size()
    verify_clean_extraction()