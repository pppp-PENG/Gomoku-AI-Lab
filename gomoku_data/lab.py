"""Read-only bridge to Gomoku AI Lab. Test games never enter SFT training."""
from collections import Counter
from pathlib import Path
import os

from .core import identity
from .pipeline import load_games, read_json


def corpus_path():
    return Path(os.environ.get("GOMOKU_EXTERNAL_DATA", str(Path(__file__).resolve().parents[1]/"data/external_gomoku/normalized")))


def catalog():
    labels = {"karesis":"Karesis / WinePy", "gomoku_party":"Gomoku Party",
              "gomocup":"Gomocup", "renjunet":"RenjuNet"}
    root = Path(__file__).resolve().parents[1]
    lock = read_json(root/"datasets/gomoku/sources.lock.json")
    result = []
    for source, policy in lock["sources"].items():
        row = dict(source=source, label=labels[source], license=policy["license"],
                   status=policy["status"], blocker=policy.get("blocker", ""),
                   board_size=15 if source in ("karesis", "gomoku_party") else None,
                   ruleset="freestyle" if source in ("karesis", "gomoku_party") else None,
                   games=0, train=0, validation=0, test=0)
        if source == "gomocup":
            row["blocker"] = "已找到公开档案入口，但尚未确认训练和再分发许可；暂不下载或接入。"
        elif source == "renjunet":
            row["blocker"] = "仅限非商业离线数据库，禁止网站或在线系统使用；本次不下载或接入。"
        if policy["status"] == "permitted":
            try:
                games = list(load_games(corpus_path(), [source]))
                row.update(Counter(g["split"] for g in games)); row["games"] = len(games)
                row["status"] = "ready"
            except (OSError, ValueError, KeyError) as exc:
                row.update(status="unavailable", blocker=str(exc))
        result.append(row)
    return result


def selected_for_sft(sources, size):
    if size != 15:
        raise ValueError("外部棋谱仅支持 15×15；请切换棋谱分类或选择对应尺寸权重")
    games = []
    for game in load_games(corpus_path(), sources):
        if game["split"] == "test":
            continue
        games.append(dict(id="external:"+game["game_hash"], size=15,
            moves=[m["action"] for m in game["moves"]],
            winner={"black":1,"white":-1,"draw":0}[game["winner"]], rules="freestyle-v1",
            name=game["source"]+":"+game["game_id"], source=game["source"],
            external_split=game["split"], game_hash=game["game_hash"],
            provenance=game["provenance"], aliases=game["aliases"], license=game["license"],
            termination=game["termination"]))
    if not any(g["external_split"] == "train" for g in games):
        raise ValueError("所选外部数据没有可用训练棋局")
    return games


def deduplicate_human(games, external):
    """Prevent collected copies of an external validation/test game leaking into train."""
    lookup = {}
    for game in external:
        lookup[game["game_hash"]] = game
    # Include held-out test games in the exclusion map.
    sources = sorted({a["source"] for g in external for a in g["aliases"]})
    for game in load_games(corpus_path(), sources, "test"):
        lookup[game["game_hash"]] = dict(winner={"black":1,"white":-1,"draw":0}[game["winner"]])
    keep = []
    for game in games:
        normalized = dict(board_size=game["size"], ruleset="freestyle", moves=[
            dict(x=a%15,y=a//15,color="black" if ply%2==0 else "white") for ply,a in enumerate(game["moves"])])
        match = lookup.get(identity(normalized))
        if match is None:
            keep.append(game)
        elif match["winner"] != game["winner"]:
            raise ValueError("采集棋谱与外部数据的重复棋局胜负冲突")
    return keep
