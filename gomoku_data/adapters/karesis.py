"""Signed, ZERO-based y*15+x sequences. JSON only; never unpickle downloads."""
import json
from ..core import InvalidGame, move, replay


def parse(sequence, game_id, provenance):
    if not isinstance(sequence, list):
        raise InvalidGame("invalid_sequence")
    moves = []
    for ply, value in enumerate(sequence):
        if type(value) is not int or abs(value) >= 225:
            raise InvalidGame("invalid_signed_action")
        # Signed zero cannot encode colour; do not silently repair an ambiguous record.
        if value == 0:
            raise InvalidGame("ambiguous_signed_zero")
        color = "black" if value > 0 else "white"
        y, x = divmod(abs(value), 15)
        moves.append(move(ply, x, y, color))
    winner = replay(moves)
    return dict(schema_version=1, source="karesis", game_id=game_id, board_size=15,
                ruleset="freestyle", license="MIT", moves=moves, winner=winner,
                termination="board", result_origin="inferred_by_terminal_replay",
                provenance=provenance, metadata={"original_signed_moves": sequence,
                "ruleset_basis": "WinePy five-or-more semantics; dataset card names Gomoku, not a formal ruleset"})


def records(path, provenance):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise InvalidGame("expected_game_list")
    for i, seq in enumerate(data):
        yield str(i), seq, lambda seq=seq, i=i: parse(seq, str(i), provenance)
