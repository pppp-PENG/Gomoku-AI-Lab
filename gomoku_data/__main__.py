import argparse
import json
from pathlib import Path
import sys
from .pipeline import acquire, build, load_games, read_json, write_jsonl
from .core import sft_samples


def main():
    parser = argparse.ArgumentParser(description="Verified 15x15 Gomoku datasets; blocked sources fail closed")
    parser.add_argument("--root", type=Path, default=Path("data/external_gomoku"))
    parser.add_argument("--lock", type=Path, default=Path("datasets/gomoku/sources.lock.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--sources", nargs="+", default=["karesis","gomoku_party"])
    download.add_argument("--offline", action="store_true")
    make = sub.add_parser("build")
    make.add_argument("--config", type=Path, default=Path("datasets/gomoku/config.json"))
    make.add_argument("--sources", nargs="+", help="override config sources")
    make.add_argument("--out", type=Path)
    sub.add_parser("status")
    export = sub.add_parser("export")
    export.add_argument("--sources", nargs="+", required=True)
    export.add_argument("--split", choices=["train","validation","test"], required=True)
    export.add_argument("--corpus", type=Path)
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock = read_json(args.lock)
        if args.command == "status":
            result = lock
        elif args.command == "download":
            result = acquire(args.root, lock, args.sources, args.offline)
        elif args.command == "build":
            config = read_json(args.config)
            if args.sources:
                config["sources"] = args.sources
            result = build(args.root, lock, config, args.out or args.root/"normalized")
        else:
            if args.out.exists():
                raise ValueError("export output already exists")
            games = load_games(args.corpus or args.root/"normalized", args.sources, args.split)
            args.out.parent.mkdir(parents=True,exist_ok=True)
            write_jsonl(args.out, (row for game in games for row in sft_samples(game)))
            result = dict(output=str(args.out))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
