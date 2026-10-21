import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="gomoku")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", default="configs/cnn.json")
    p.add_argument("--resume")
    p.add_argument("--games", type=int)
    p = sub.add_parser("sft")
    p.add_argument("--config", required=True)
    p.add_argument("--resume")
    p = sub.add_parser("worker")
    p.add_argument("--experiment", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--resume")
    p = sub.add_parser("evaluate")
    p.add_argument("checkpoint")
    p.add_argument("--opponent", default="heuristic")
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--simulations", type=int, default=16)
    p.add_argument("--seeds", default="42,43,44")
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("gomoku.api:app", host="127.0.0.1", port=args.port)
        return
    from .storage import Store
    from .training import TrainConfig, train, load_checkpoint
    if args.command == "worker":
        store = Store(args.data)
        raw = store.experiment(args.experiment)["config"]
        if raw.get("kind") == "sft":
            from .sft import SFTConfig, train_sft
            train_sft(SFTConfig(**raw), store, args.experiment, args.resume)
        else:
            train(TrainConfig(**raw), store, args.experiment, args.resume)
    elif args.command == "sft":
        from .sft import SFTConfig, train_sft
        store = Store()
        config = SFTConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
        eid = train_sft(config, store, resume=args.resume)
        print(json.dumps(dict(experiment=eid, checkpoints=[c["id"] for c in store.checkpoints() if c["experiment"] == eid])))
    elif args.command == "train":
        store = Store()
        config = TrainConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
        if args.games:
            config.games = args.games
            config.__post_init__()
        eid = train(config, store, resume=args.resume)
        print(json.dumps(dict(experiment=eid, checkpoints=[c["id"] for c in store.checkpoints() if c["experiment"] == eid])))
    else:
        import numpy as np
        import torch
        from .search import Evaluator
        from .evaluation import evaluate, tactics, benchmark
        torch.set_num_threads(4)
        store = Store()
        checkpoint = store.get_checkpoint(args.checkpoint)
        model, _ = load_checkpoint(checkpoint["path"])
        evaluator = Evaluator(model)
        opponent = args.opponent
        if opponent not in ("random", "heuristic"):
            other, _ = load_checkpoint(store.get_checkpoint(opponent)["path"])
            if other.config["size"] != model.config["size"]:
                raise ValueError("board mismatch")
            opponent = Evaluator(other)
        reports = []
        for seed in map(int, args.seeds.split(",")):
            report = evaluate(evaluator, opponent, model.config["size"], args.pairs, args.simulations, seed)
            report.update(checkpoint=args.checkpoint, opponent_id=args.opponent,
                          experiment=checkpoint["experiment"], games_trained=checkpoint["games"],
                          training_seconds=checkpoint.get("training_seconds"),
                          tactics=tactics(evaluator, model.config["size"], args.simulations),
                          benchmark=benchmark(model, model.config["size"]))
            store.save_evaluation(report)
            reports.append(report)
        print(json.dumps(dict(reports=reports, score_mean=float(np.mean([r["score"] for r in reports])),
                              score_std=float(np.std([r["score"] for r in reports]))), indent=2))


if __name__ == "__main__":
    main()
