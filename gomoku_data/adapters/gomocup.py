"""Provisional local PSQ parser; acquisition/integration remains license-blocked.

Only the simple Piskvorky x,y,time linear variant is supported. Caller must supply
verified coordinate base, ruleset, result and league metadata. Never guess from
file names or confuse starting AI with Black. No historical archive was acquired.
"""
import re
from ..core import InvalidGame, move


def parse_psq(text, *, coordinate_base, ruleset, game_id, provenance, winner, termination="board"):
    lines = text.splitlines()
    header = re.fullmatch(r"Piskvorky (\d+)x(\d+),.*", lines[0] if lines else "")
    if not header or header.groups() != ("15", "15"):
        raise InvalidGame("board_size_or_psq_header_mismatch")
    if coordinate_base not in (0, 1) or type(coordinate_base) is not int:
        raise InvalidGame("explicit_coordinate_base_required")
    moves = []; footer = False
    for line in lines[1:]:
        if line == "-":
            footer = True; break
        match = re.fullmatch(r"(\d+),(\d+),(\d+)", line)
        if not match:
            raise InvalidGame("unsupported_psq_line")
        x, y, _time = map(int, match.groups())
        moves.append(move(len(moves), x-coordinate_base, y-coordinate_base, "black" if len(moves)%2==0 else "white"))
    if not footer:
        raise InvalidGame("missing_psq_footer")
    return dict(schema_version=1, source="gomocup", game_id=game_id, board_size=15, ruleset=ruleset,
                license="LicenseRef-Gomocup-permission-required", moves=moves, winner=winner,
                termination=termination, result_origin="verified_league_result", provenance=provenance, metadata={})


def records(path, provenance):
    raise PermissionError("Gomocup training/redistribution permission not established; adapter disabled")
