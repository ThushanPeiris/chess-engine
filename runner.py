"""
Generic wire-protocol wrapper for any agent.

Usage: python3 runner.py /path/to/agent_directory

Loads get_move from agent.py in the given directory, then loops reading
one JSON request per line from stdin and writing one JSON response per
line to stdout - exactly the protocol documented by the competition:

    request:  {"fen": "...", "time_left_ms": 87500}
    response: {"move": "e2e4"}

CRITICAL: before importing the agent, this redirects file descriptor 1
(stdout) to stderr, and keeps a private duplicate of the real stdout for
writing protocol responses. This mirrors the real competition runner's
documented behaviour exactly: "The runner moves the protocol onto a
private handle and points file descriptor 1 at stderr before importing
your agent, so print is safe." Without this, any print() call inside the
agent or the search engine (e.g. the depth-by-depth debug output in
search_engine.py) lands on the same stream as the JSON protocol and
corrupts it - which is exactly what happened the first time this harness
was run, before this fix was added.

Running each agent as its own subprocess (rather than importing both
agents into one Python process) also avoids a separate real bug risk:
if two agents both imported the same search_engine module in one
process, they'd silently share the same global transposition table and
killer-move tables (Python caches module imports), corrupting both
agents' search results without any visible error. Separate processes
have separate memory, so this can't happen.
"""

import sys
import os
import json

agent_dir = sys.argv[1]

# Preserve the real stdout on a private file descriptor for protocol use,
# then redirect fd 1 to stderr so any print() calls inside the agent (or
# anything it imports) can never corrupt the JSON wire protocol.
_protocol_out = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)

sys.path.insert(0, agent_dir)  # so `import agent` finds this specific copy
from agent import get_move  # noqa: E402 (import after fd redirection is intentional)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    request = json.loads(line)
    fen = request["fen"]
    time_left_ms = request["time_left_ms"]

    move = get_move(fen, time_left_ms)

    response = json.dumps({"move": move})
    _protocol_out.write(response + "\n")
    _protocol_out.flush()