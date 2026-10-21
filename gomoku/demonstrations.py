"""Human demonstrations, validated by the same engine as self-play."""
from .game import Game, RULE_VERSION


def validate_demonstration(size, moves, name="", rules=RULE_VERSION, terminal=True):
    if rules != RULE_VERSION:
        raise ValueError("棋谱规则版本不兼容")
    if isinstance(size, bool) or not isinstance(size, int) or not 5 <= size <= 19:
        raise ValueError("棋盘大小必须为 5..19")
    if not isinstance(moves, list) or len(moves) > size**2:
        raise ValueError("落子序列无效")
    if not isinstance(name, str) or len(name) > 120:
        raise ValueError("棋谱名称最多 120 字")
    game = Game(size, moves=moves)
    if terminal and not game.terminal:
        raise ValueError("请完成棋局后再决定是否收录")
    # Never trust client supplied winners, boards, or training labels.
    return dict(schema=1, name=name.strip(), source="human", **game.as_dict())
