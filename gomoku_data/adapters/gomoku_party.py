"""Strict linear SGF FF4 reader, including escaped values and multi-value properties."""
import gzip
import re
from ..core import InvalidGame, move


def nodes(text):
    pos, length = 0, len(text)
    def ws():
        nonlocal pos
        while pos < length and text[pos].isspace():
            pos += 1
    ws()
    if pos >= length or text[pos] != "(":
        raise InvalidGame("invalid_sgf_tree")
    pos += 1
    out = []
    while True:
        ws()
        if pos >= length:
            raise InvalidGame("truncated_sgf_tree")
        if text[pos] == ")":
            pos += 1
            ws()
            if pos != length or not out:
                raise InvalidGame("trailing_sgf_data")
            return out
        if text[pos] != ";":
            raise InvalidGame("variation_or_invalid_node")
        pos += 1
        props = {}
        while True:
            ws()
            if pos >= length or text[pos] in ";)(":
                break
            m = re.match(r"[A-Z]+", text[pos:])
            if not m:
                raise InvalidGame("invalid_property")
            name = m.group(); pos += len(name)
            if name in props:
                raise InvalidGame("duplicate_property")
            values = []
            ws()
            while pos < length and text[pos] == "[":
                pos += 1; value = []
                while pos < length and text[pos] != "]":
                    c = text[pos]; pos += 1
                    if c == "\\":
                        if pos >= length:
                            raise InvalidGame("truncated_escape")
                        c = text[pos]; pos += 1
                        if c in "\r\n":
                            if c == "\r" and pos < length and text[pos] == "\n":
                                pos += 1
                            continue
                    value.append(c)
                if pos >= length:
                    raise InvalidGame("truncated_value")
                pos += 1; values.append("".join(value)); ws()
            if not values:
                raise InvalidGame("missing_property_value")
            props[name] = values
        out.append(props)


def scalar(props, key):
    value = props.get(key)
    if not value or len(value) != 1:
        raise InvalidGame("missing_or_multivalued:" + key)
    return value[0]


RULES = {"FREESTYLE": ("freestyle", "Gomoku Freestyle"), "RENJU": ("renju", "Gomoku Renju"),
         "SWAP2": ("swap2", "Gomoku Swap2"), "PENTE": ("pente", "Pente"),
         "NINUKI_RENJU": ("ninuki_renju", "Ninuki Renju"), "CARO": ("caro", "Caro")}


def parse(text, provenance):
    tree = nodes(text); root = tree[0]
    for key, expected in {"FF":"4", "GM":"4", "SZ":"15", "CA":"UTF-8", "CP":"CC0-1.0", "XGP":"gomoku-party-sgf-2"}.items():
        if scalar(root, key) != expected:
            raise InvalidGame("profile_mismatch:" + key)
    xr = scalar(root, "XR")
    if xr not in RULES or scalar(root, "RU") != RULES[xr][1]:
        raise InvalidGame("ruleset_metadata_mismatch")
    # Unsupported opening/capture semantics are quarantined before alternating replay.
    if xr != "FREESTYLE":
        raise InvalidGame("unsupported_ruleset:" + RULES[xr][0])
    if any(k in node for node in tree for k in ("AB", "AW", "AE", "PL", "XCP")):
        raise InvalidGame("setup_or_capture_not_supported")
    if "B" in root or "W" in root:
        raise InvalidGame("move_in_root")
    moves = []
    for ply, node in enumerate(tree[1:]):
        colors = [k for k in ("B", "W") if k in node]
        if len(colors) != 1:
            raise InvalidGame("missing_or_multiple_move")
        color = colors[0]; coord = scalar(node, color)
        if not re.fullmatch(r"[a-o]{2}", coord):
            raise InvalidGame("invalid_sgf_coordinate_or_pass")
        moves.append(move(ply, ord(coord[0])-97, ord(coord[1])-97, "black" if color == "B" else "white"))
    try:
        if int(scalar(root, "XMC")) != len(moves) or len(moves) < 5:
            raise InvalidGame("move_count_mismatch")
    except ValueError as exc:
        raise InvalidGame("move_count_mismatch") from exc
    result = scalar(root, "RE"); reason = scalar(root, "XER")
    termination = {"FIVE_IN_ROW":"board", "RESIGNATION":"resignation", "TIMEOUT":"timeout", "DRAW":"agreed_draw"}.get(reason)
    suffix = {"FIVE_IN_ROW":"", "RESIGNATION":"R", "TIMEOUT":"T"}.get(reason)
    if reason == "DRAW" and result == "0":
        winner = "draw"
    elif suffix is not None and result in ("B+"+suffix, "W+"+suffix):
        winner = "black" if result.startswith("B") else "white"
    else:
        raise InvalidGame("result_reason_mismatch")
    return dict(schema_version=1, source="gomoku_party", game_id=scalar(root, "GN"), board_size=15,
                ruleset="freestyle", license="CC0-1.0", moves=moves, winner=winner,
                termination=termination, result_origin="source_record", provenance=provenance,
                metadata={"sgf_root":root})


def records(path, provenance):
    # The versioned publishing profile guarantees one complete tree per line.
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                yield str(line), text.strip(), lambda text=text: parse(text, provenance)
