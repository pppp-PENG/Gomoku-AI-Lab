import hashlib
import json


class InvalidGame(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def replay(moves, size=15, ruleset="freestyle"):
    """Return board adjudication; never infer a winner from the last player's colour."""
    if size != 15:
        raise InvalidGame("board_size_mismatch")
    if ruleset not in ("freestyle", "standard"):
        raise InvalidGame("unsupported_ruleset:" + ruleset)
    if not isinstance(moves, list) or not 1 <= len(moves) <= size * size:
        raise InvalidGame("invalid_move_count")
    board, winner = {}, None
    for ply, move in enumerate(moves):
        if winner is not None:
            raise InvalidGame("moves_after_terminal")
        if not isinstance(move, dict):
            raise InvalidGame("invalid_move")
        x, y, colour = (move.get(k) for k in ("x", "y", "color"))
        if type(x) is not int or type(y) is not int or not 0 <= x < size or not 0 <= y < size:
            raise InvalidGame("coordinate_out_of_bounds")
        if colour != ("black" if ply % 2 == 0 else "white"):
            raise InvalidGame("colour_turn_mismatch")
        if (x, y) in board:
            raise InvalidGame("occupied_intersection")
        if move.get("ply") != ply or move.get("action") != y * size + x:
            raise InvalidGame("inconsistent_move_encoding")
        board[x, y] = colour
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                xx, yy = x + dx * sign, y + dy * sign
                while board.get((xx, yy)) == colour:
                    count += 1
                    xx, yy = xx + dx * sign, yy + dy * sign
            if count >= 5 if ruleset == "freestyle" else count == 5:
                winner = colour
        if winner is None and len(board) == size * size:
            winner = "draw"
    return winner


def move(ply, x, y, color):
    return dict(ply=ply, x=x, y=y, action=y * 15 + x, color=color)


def validate(game):
    if game.get("schema_version") != 1 or not game.get("source") or not game.get("game_id"):
        raise InvalidGame("missing_identity")
    if not game.get("license") or not game.get("provenance"):
        raise InvalidGame("missing_provenance")
    board_winner = replay(game["moves"], game["board_size"], game["ruleset"])
    winner, termination = game.get("winner"), game.get("termination")
    if winner not in ("black", "white", "draw"):
        raise InvalidGame("unknown_result")
    if termination == "board":
        if board_winner is None:
            raise InvalidGame("incomplete_game_no_terminal_result")
        if winner != board_winner:
            raise InvalidGame("winner_mismatch")
    elif termination in ("resignation", "timeout", "agreed_draw"):
        if board_winner is not None and board_winner != winner:
            raise InvalidGame("winner_mismatch")
        if termination == "agreed_draw" and winner != "draw":
            raise InvalidGame("draw_result_mismatch")
        if termination != "agreed_draw" and winner == "draw":
            raise InvalidGame("decisive_result_required")
    else:
        raise InvalidGame("unknown_termination")
    return game


def canonical_moves(game):
    """D4 canonicalisation preserves move order and stone colours."""
    variants = []
    for mirror in (False, True):
        for rotation in range(4):
            seq = []
            for m in game["moves"]:
                x, y = m["x"], m["y"]
                if mirror:
                    x = 14 - x
                for _ in range(rotation):
                    x, y = 14 - y, x
                seq.append((m["color"], y * 15 + x))
            variants.append(tuple(seq))
    return min(variants)


def identity(game):
    # Outcome intentionally absent: conflicting duplicate labels must be detected.
    return digest([game["board_size"], game["ruleset"], canonical_moves(game)])


def split_for(game_hash, seed=42, fractions=(0.8, 0.1, 0.1)):
    if len(fractions) != 3 or any(type(x) not in (int, float) or not 0 <= x <= 1 for x in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("split fractions must be three numbers summing to one")
    n = int(digest([seed, game_hash])[:16], 16) / 2**64
    return "train" if n < fractions[0] else "validation" if n < sum(fractions[:2]) else "test"


def sft_samples(game):
    """History is the board before target_action; WDL is from the mover's view."""
    history = []
    for m in game["moves"]:
        w = game["winner"]
        yield dict(schema_version=1, game_hash=game["game_hash"], split=game["split"],
                   source=game["source"], game_id=game["game_id"], board_size=15,
                   ruleset=game["ruleset"], license=game["license"], ply=m["ply"],
                   history=history.copy(), to_play=m["color"], target_action=m["action"],
                   wdl=[int(w == m["color"]), int(w == "draw"), int(w not in (m["color"], "draw"))])
        history.append(m["action"])
