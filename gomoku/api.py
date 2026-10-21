from collections import OrderedDict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
import uuid
import numpy as np
import psutil
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from .game import Game, RULE_VERSION
from .manager import TrainingManager
from .models import ModelConfig, build_model, estimated_flops
from .search import Evaluator, MCTS, analyze
from .storage import Store
from .training import TrainConfig, load_checkpoint


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: str
    moves: list[StrictInt] = Field(default_factory=list, max_length=361)
    size: int = Field(default=9, ge=5, le=19)
    simulations: int = Field(default=0, ge=0, le=256)


class NewGame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint: str
    human: int = 1
    simulations: int = Field(default=16, ge=0, le=256)
    size: int = Field(default=9, ge=5, le=19)


class Move(BaseModel):
    action: StrictInt
    revision: int = Field(ge=0)


class Revision(BaseModel):
    revision: int = Field(ge=0)


class Branch(BaseModel):
    step: int = Field(ge=0, le=361)
    checkpoint: str | None = None
    human: int = 1
    simulations: int = Field(default=16, ge=0, le=256)


class TrainingRequest(BaseModel):
    config: dict
    resume: str | None = None


class DemonstrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: StrictInt = Field(default=9, ge=5, le=19)
    moves: list[StrictInt] = Field(default_factory=list, max_length=361)
    name: str = Field(default="", max_length=120)
    rules: str = RULE_VERSION


class EvaluationRequest(BaseModel):
    checkpoint: str
    opponent: str = "heuristic"
    pairs: int = Field(default=2, ge=1, le=20)
    simulations: int = Field(default=16, ge=0, le=256)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=10)


def create_app(store=None):
    store = store or Store()
    torch.set_num_threads(2)
    manager = TrainingManager(store)
    cache, cache_lock = OrderedDict(), threading.Lock()
    inference_lock, game_lock = threading.Lock(), threading.Lock()
    jobs, jobs_lock = {}, threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evaluation")

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.shutdown()
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Gomoku AI Lab", version="0.1.0", lifespan=lifespan)
    app.state.store, app.state.manager = store, manager

    @app.exception_handler(ValueError)
    async def bad_value(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "record not found"})

    def evaluator(cid, size):
        with cache_lock:
            ck = store.get_checkpoint(cid)
            if ck["model_config"]["size"] != size or ck["rules"] != RULE_VERSION:
                raise ValueError("checkpoint board/rules mismatch")
            if cid not in cache:
                model, _ = load_checkpoint(ck["path"])
                cache[cid] = Evaluator(model)
                if len(cache) > 2:
                    cache.popitem(last=False)
            cache.move_to_end(cid)
            return cache[cid]

    def state(payload):
        return Game(payload["size"], moves=payload["moves"])

    def validate_revision(game, revision):
        if game["revision"] != revision:
            raise HTTPException(409, "game changed; reload before retry")

    def persist(payload, gid=None):
        game = state(payload)
        payload.update(game.as_dict())
        payload.pop("id", None)
        payload.pop("created", None)
        gid = store.save_game(payload, gid)
        return store.game(gid)

    @app.get("/api/health")
    def health():
        manager.reconcile()
        return dict(status="ok", cuda=torch.cuda.is_available(), cpu_percent=psutil.cpu_percent(),
                    memory_percent=psutil.virtual_memory().percent,
                    server_rss_mb=psutil.Process().memory_info().rss/1024**2,
                    inference_device="cpu", schema=1)

    @app.get("/api/architectures")
    def architectures():
        rows = []
        with torch.random.fork_rng():
            for arch in ("cnn", "vit"):
                c = ModelConfig(architecture=arch)
                m = build_model(c)
                rows.append(dict(config=asdict(c), parameters=sum(p.numel() for p in m.parameters()),
                                 estimated_flops=estimated_flops(c)))
        return rows

    @app.get("/api/checkpoints")
    def checkpoints():
        return [{k: v for k, v in c.items() if k != "path"} for c in store.checkpoints()]

    @app.get("/api/checkpoints/{cid}/deletion-preview")
    def deletion_preview(cid: str):
        checkpoint = store.get_checkpoint(cid)
        path = Path(checkpoint["path"])
        return dict(id=cid, references=store.checkpoint_references(cid),
                    bytes=path.stat().st_size if path.exists() else 0)

    @app.delete("/api/checkpoints/{cid}")
    def delete_checkpoint(cid: str):
        with manager.lock, jobs_lock:
            manager.reconcile()
            if any(e["status"] in ("running", "pausing", "stopping") for e in store.experiments()):
                raise HTTPException(409, "请先暂停或停止训练，再删除 Checkpoint")
            if any(j["status"] in ("queued", "running") for j in jobs.values()):
                raise HTTPException(409, "请等待评估结束后再删除 Checkpoint")
            with inference_lock, cache_lock:
                result = store.delete_checkpoint(cid)
                cache.pop(cid, None)
                return result

    @app.post("/api/analyze")
    def analysis(p: Position):
        game = Game(p.size, moves=p.moves)
        begin = time.monotonic()
        with inference_lock:
            result = analyze(game, evaluator(p.checkpoint, p.size), p.simulations)
        return dict(**result, elapsed_ms=(time.monotonic()-begin)*1000, checkpoint=p.checkpoint)

    @app.post("/api/games")
    def new_game(p: NewGame):
        if p.human not in (-1, 1):
            raise ValueError("human must be -1 or 1")
        evaluator(p.checkpoint, p.size)
        return persist(dict(size=p.size, moves=[], checkpoint=p.checkpoint, human=p.human,
                            simulations=p.simulations, revision=0, parent=None))

    @app.get("/api/games")
    def games():
        return store.games()

    @app.get("/api/games/{gid}")
    def get_game(gid: str):
        return store.game(gid)

    @app.post("/api/games/{gid}/move")
    def human_move(gid: str, p: Move):
        with game_lock:
            payload = store.game(gid)
            validate_revision(payload, p.revision)
            game = state(payload)
            if game.to_play != payload["human"]:
                raise ValueError("AI turn")
            game.play(p.action)
            payload.update(moves=game.moves, revision=payload["revision"]+1)
            return persist(payload, gid)

    @app.post("/api/games/{gid}/ai")
    def ai_move(gid: str, p: Revision):
        payload = store.game(gid)
        validate_revision(payload, p.revision)
        game = state(payload)
        if game.to_play == payload["human"] or game.terminal:
            raise ValueError("not an AI turn")
        with inference_lock:
            net = evaluator(payload["checkpoint"], game.size)
            if payload["simulations"]:
                policy, _, _ = MCTS(net, payload["simulations"]).run(game)
            else:
                policy, _ = net(game)
        game.play(int(np.argmax(policy)))
        with game_lock:
            validate_revision(store.game(gid), p.revision)
            payload.update(moves=game.moves, revision=payload["revision"]+1)
            return persist(payload, gid)

    @app.post("/api/games/{gid}/undo")
    def undo(gid: str, p: Revision):
        with game_lock:
            payload = store.game(gid)
            validate_revision(payload, p.revision)
            game = state(payload)
            game.undo(1)
            if game.to_play != payload["human"] and game.moves:
                game.undo(1)
            payload.update(moves=game.moves, revision=payload["revision"]+1)
            return persist(payload, gid)

    @app.post("/api/games/{gid}/branch")
    def branch(gid: str, p: Branch):
        original = store.game(gid)
        if p.step > len(original["moves"]) or p.human not in (-1, 1):
            raise ValueError("invalid branch step/player")
        checkpoint = p.checkpoint or original["checkpoint"]
        evaluator(checkpoint, original["size"])
        prefix = original["moves"][:p.step]
        if Game(original["size"], moves=prefix).terminal:
            raise ValueError("cannot continue a terminal position")
        return persist(dict(size=original["size"], moves=prefix, checkpoint=checkpoint,
                            human=p.human, simulations=p.simulations, revision=0,
                            parent=dict(game=gid, step=p.step)))

    @app.get("/api/games/{gid}/export")
    def export_game(gid: str):
        payload = store.game(gid)
        return Response(json.dumps(dict(schema=1, game=payload), ensure_ascii=False), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="gomoku-{gid}.json"'})

    @app.post("/api/games/import")
    def import_game(payload: dict):
        if payload.get("schema") != 1 or not isinstance(payload.get("game"), dict):
            raise ValueError("unsupported game schema")
        source = payload["game"]
        if source.get("rules") != RULE_VERSION or source.get("connect", 5) != 5:
            raise ValueError("unsupported game rules")
        spec = NewGame(checkpoint=source["checkpoint"], size=source["size"],
                       human=source["human"], simulations=source.get("simulations", 16))
        if spec.human not in (-1, 1) or not isinstance(source["moves"], list) or len(source["moves"]) > spec.size**2:
            raise ValueError("invalid game")
        return persist(dict(size=spec.size, checkpoint=spec.checkpoint, human=spec.human,
                            simulations=spec.simulations, moves=source["moves"], revision=0, parent=source.get("parent")))

    @app.get("/api/experiments")
    def experiments():
        manager.reconcile()
        return store.experiments()

    @app.post("/api/demonstrations/position")
    def demonstration_position(p: DemonstrationRequest):
        from .demonstrations import validate_demonstration
        return validate_demonstration(p.size, p.moves, p.name, p.rules, terminal=False)

    @app.get("/api/demonstrations")
    def demonstrations(size: int | None = Query(default=None, ge=5, le=19)):
        return store.demonstrations(size)

    @app.post("/api/demonstrations")
    def save_demonstration(p: DemonstrationRequest):
        from .demonstrations import validate_demonstration
        return store.save_demonstration(validate_demonstration(p.size, p.moves, p.name, p.rules))

    @app.get("/api/demonstrations/export")
    def export_demonstrations():
        return Response(json.dumps(dict(schema=1, rules=RULE_VERSION, demonstrations=store.demonstrations()),
                                   ensure_ascii=False), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="gomoku-demonstrations.json"'})

    @app.get("/api/sft/sources")
    def sft_sources():
        from gomoku_data.lab import catalog
        return catalog()

    @app.post("/api/sft/experiments")
    def start_sft(p: TrainingRequest):
        from .sft import SFTConfig
        try:
            config = SFTConfig(**p.config)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        return dict(id=manager.start_sft(config, p.resume))

    @app.post("/api/experiments")
    def start_experiment(p: TrainingRequest):
        try:
            config = TrainConfig(**p.config)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        return dict(id=manager.start(config, p.resume))

    @app.post("/api/experiments/{eid}/{action}")
    def control(eid: str, action: str):
        if action not in ("pause", "stop"):
            raise ValueError("unknown action")
        manager.control(eid, action)
        return dict(status="requested")

    @app.get("/api/experiments/{eid}/metrics")
    def metrics(eid: str):
        store.experiment(eid)
        return store.metrics(eid)

    @app.get("/api/experiments/{eid}/export")
    def export_experiment(eid: str):
        return dict(schema=1, experiment=store.experiment(eid), metrics=store.metrics(eid, limit=None),
                    evaluations=[e for e in store.evaluations() if e.get("experiment") == eid])

    @app.get("/api/experiments/{eid}/live")
    def live(eid: str):
        store.experiment(eid)
        p = store.root / f"live-{eid}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @app.get("/api/evaluations")
    def evaluations():
        return store.evaluations()

    @app.post("/api/evaluations")
    def start_evaluation(p: EvaluationRequest):
        store.get_checkpoint(p.checkpoint)
        if p.opponent not in ("random", "heuristic"):
            store.get_checkpoint(p.opponent)
        with jobs_lock:
            if any(j["status"] in ("queued", "running") for j in jobs.values()):
                raise ValueError("one evaluation at a time")
            jid = uuid.uuid4().hex
            jobs[jid] = dict(status="queued")

        def work():
            jobs[jid] = dict(status="running")
            try:
                from .evaluation import evaluate, tactics, benchmark
                ck = store.get_checkpoint(p.checkpoint)
                n = ck["model_config"]["size"]
                with inference_lock:
                    net = evaluator(p.checkpoint, n)
                    opponent = p.opponent if p.opponent in ("random", "heuristic") else evaluator(p.opponent, n)
                    reports = []
                    for seed in p.seeds:
                        r = evaluate(net, opponent, n, p.pairs, p.simulations, seed)
                        r.update(checkpoint=p.checkpoint, experiment=ck["experiment"], games_trained=ck["games"],
                                 opponent_id=p.opponent, training_seconds=ck.get("training_seconds"),
                                 tactics=tactics(net, n, p.simulations),
                                 benchmark=benchmark(net.model, n))
                        r["id"] = store.save_evaluation(r)
                        reports.append(r)
                jobs[jid] = dict(status="completed", reports=reports,
                                 score_mean=float(np.mean([r["score"] for r in reports])),
                                 score_std=float(np.std([r["score"] for r in reports])))
            except Exception as exc:
                jobs[jid] = dict(status="failed", error=str(exc))
        executor.submit(work)
        return dict(id=jid)

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        if jid not in jobs:
            raise KeyError(jid)
        return jobs[jid]

    dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"

    @app.get("/{path:path}")
    def frontend(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        file = (dist / path).resolve()
        if dist.resolve() not in file.parents and file != dist.resolve():
            raise HTTPException(404)
        if file.is_file():
            return FileResponse(file)
        if (dist / "index.html").exists():
            return FileResponse(dist / "index.html")
        raise HTTPException(503, "Build frontend first: npm run build")

    return app


app = create_app()
