from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from .adapters import ADAPTERS
from .core import InvalidGame, identity, split_for, sft_samples, validate


ALLOWED = {"karesis": "MIT", "gomoku_party": "CC0-1.0"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def selected(lock, sources):
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("choose unique, nonempty sources")
    for source in sources:
        if source not in lock["sources"]:
            raise ValueError("unknown source: " + source)
        policy = lock["sources"][source]
        if source not in ALLOWED or policy["status"] != "permitted" or policy["license"] != ALLOWED[source]:
            raise PermissionError(source + ": " + policy.get("blocker", "permission not verified"))
    return sorted(sources)


def raw_path(root, relative):
    root = Path(root).resolve(); path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("unsafe manifest path")
    return path


def verified_files(root, lock, sources):
    for source in selected(lock, sources):
        for asset in lock["sources"][source]["files"]:
            path = raw_path(root, asset["path"])
            if not path.is_file() or sha256(path) != asset["sha256"] or path.stat().st_size != asset["bytes"]:
                raise ValueError("missing/corrupt source asset: " + str(path))
            yield source, asset, path


def acquire(root, lock, sources, offline=False):
    sources = selected(lock, sources)  # permission preflight before any network or mutation
    receipts = []
    for source in sources:
        for asset in lock["sources"][source]["files"]:
            path = raw_path(root, asset["path"])
            cached = path.is_file() and sha256(path) == asset["sha256"] and path.stat().st_size == asset["bytes"]
            if not cached:
                if offline:
                    raise ValueError("offline cache missing/corrupt: " + str(path))
                if not asset["url"].startswith("https://"):
                    raise ValueError("HTTPS required")
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix+".part")
                for attempt in range(3):
                    try:
                        request = urllib.request.Request(asset["url"], headers={"User-Agent":"GomokuDatasetPipeline/1.0"})
                        with urllib.request.urlopen(request, timeout=40) as response, temp.open("wb") as out:
                            remaining = asset["bytes"]
                            while block := response.read(min(1024*1024, remaining+1)):
                                out.write(block); remaining -= len(block)
                                if remaining < 0:
                                    raise ValueError("download exceeds locked size")
                        if temp.stat().st_size != asset["bytes"] or sha256(temp) != asset["sha256"]:
                            raise ValueError("download size/checksum mismatch")
                        temp.replace(path)
                        break
                    except (OSError, ValueError):
                        temp.unlink(missing_ok=True)
                        if attempt == 2:
                            raise
                        time.sleep(attempt+1)
            receipts.append(dict(source=source, **asset, verified=True, cached=cached))
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(), files=receipts)
    write_json(Path(root)/"acquisition.json", report)
    return report


def build(root, lock, config, out):
    sources = selected(lock, config["sources"])
    if config.get("board_size") != 15 or config.get("rulesets") != ["freestyle"]:
        raise ValueError("this SFT engine supports only size=15, rulesets=['freestyle']")
    fractions = config.get("splits", [0.8, 0.1, 0.1]); seed = config.get("seed", 42)
    split_for("preflight", seed, fractions)
    files = list(verified_files(root, lock, sources))
    accepted, quarantine, groups, counts = [], [], defaultdict(list), Counter()
    for source, asset, path in files:
        provenance = dict(source_url=asset["url"], raw_path=asset["path"], sha256=asset["sha256"],
                          revision=lock["sources"][source].get("revision"),
                          license_url=lock["sources"][source]["license_url"])
        for record_id, raw, parser in ADAPTERS[source].records(path, provenance):
            counts[source+".input"] += 1
            try:
                game = validate(parser())
                if game["source"] != source or game["license"] != ALLOWED[source]:
                    raise InvalidGame("source_provenance_mismatch")
                game["game_hash"] = identity(game)
                groups[game["game_hash"]].append(game)
            except (InvalidGame, KeyError, TypeError, ValueError) as exc:
                quarantine.append(dict(source=source, record_id=record_id, provenance=provenance,
                                       reason=str(exc), raw=raw))
                counts[source+".quarantined"] += 1
    duplicates = []
    for key, games in sorted(groups.items()):
        if len({g["winner"] for g in games}) > 1:
            for game in games:
                quarantine.append(dict(source=game["source"], record_id=game["game_id"],
                                       reason="conflicting_duplicate_results", raw=game))
                counts[game["source"]+".quarantined"] += 1
            continue
        game = games[0]
        game["aliases"] = [dict(source=g["source"], game_id=g["game_id"], license=g["license"],
                                 provenance=g["provenance"]) for g in games]
        game["split"] = split_for(key, seed, fractions)
        accepted.append(game)
        counts[game["source"]+".unique"] += 1
        for extra in games[1:]:
            counts[extra["source"]+".duplicate"] += 1
            duplicates.append(dict(game_hash=key, source=extra["source"], game_id=extra["game_id"]))
    out = Path(out)
    # Do not silently mix a new selection with stale artifacts from a previous run.
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be a new or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out/"games.jsonl", accepted)
    write_jsonl(out/"quarantine.jsonl", quarantine)
    write_jsonl(out/"duplicates.jsonl", duplicates)
    for split in ("train", "validation", "test"):
        games = [g for g in accepted if g["split"] == split]
        write_jsonl(out/(split+".games.jsonl"), games)
        write_jsonl(out/(split+".sft.jsonl"), (row for g in games for row in sft_samples(g)))
    artifacts = {p.name:dict(sha256=sha256(p), bytes=p.stat().st_size) for p in sorted(out.glob("*.jsonl"))}
    report = dict(schema_version=1, config=config, source_lock_sha256=hashlib.sha256(
                  json.dumps(lock, sort_keys=True).encode()).hexdigest(), sources=sources,
                  counts=dict(sorted(counts.items())), unique_games=len(accepted),
                  samples=sum(len(g["moves"]) for g in accepted), quarantine=len(quarantine),
                  quarantine_reasons=dict(Counter(q["reason"] for q in quarantine)),
                  split_games=dict(Counter(g["split"] for g in accepted)),
                  split_samples=dict((s,sum(len(g["moves"]) for g in accepted if g["split"]==s)) for s in ("train","validation","test")),
                  artifacts=artifacts)
    write_json(out/"manifest.json", report)
    return report


def load_games(out, sources, split=None):
    """Select any permitted source combination, retaining global dedup/split assignments.

    Membership checks aliases so a deduped cross-source game remains selectable by
    either source. These files are derived from verified downloads, not arbitrary uploads.
    """
    if not sources or any(s not in ALLOWED for s in sources):
        raise PermissionError("select permitted sources only")
    out = Path(out); manifest = read_json(out/"manifest.json")
    if not set(sources) <= set(manifest["sources"]):
        raise ValueError("selected source has not been built")
    expected = manifest["artifacts"]["games.jsonl"]["sha256"]
    if sha256(out/"games.jsonl") != expected:
        raise ValueError("normalized corpus checksum mismatch")
    with (out/"games.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            game = json.loads(line)
            if {a["source"] for a in game["aliases"]} & set(sources) and (split is None or game["split"] == split):
                validate(game)
                if game["ruleset"] != "freestyle" or game["board_size"] != 15:
                    raise ValueError("SFT rule/size mismatch")
                if identity(game) != game["game_hash"] or game["split"] != split_for(game["game_hash"], manifest["config"].get("seed",42), manifest["config"].get("splits",[.8,.1,.1])):
                    raise ValueError("identity/split mismatch")
                yield game
