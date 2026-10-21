from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import psutil
import torch
from .game import Game, RULE_VERSION, augment
from .models import ModelConfig, build_model, estimated_flops
from .search import Evaluator, choose, search_many
from .storage import Store


@dataclass
class TrainConfig:
    model: dict = field(default_factory=lambda: asdict(ModelConfig()))
    seed: int = 42
    games: int = 200
    simulations: int = 32
    parallel_games: int = 4
    batch_size: int = 64
    replay_size: int = 20000
    updates_per_game: int = 4
    learning_rate: float = .001
    weight_decay: float = .0001
    checkpoint_every: int = 20
    eval_every: int = 40
    eval_pairs: int = 2
    device: str = "auto"
    threads: int = 4
    temperature_moves: int = 12
    mode: str = "independent"
    opponent: str | None = None
    opponent_model: dict | None = None
    initial_checkpoint: str | None = None
    opponent_initial_checkpoint: str | None = None
    max_hours: float = 2.
    disk_budget_gb: float = 4.

    def __post_init__(self):
        self.model = asdict(ModelConfig(**self.model))
        for source in (self.initial_checkpoint, self.opponent_initial_checkpoint):
            if source is not None and (not isinstance(source, str) or not source.strip() or len(source) > 128):
                raise ValueError("初始 Checkpoint 必须是有效的 ID")
        if self.opponent_initial_checkpoint and self.mode != "paired":
            raise ValueError("只有双模型训练可以设置模型 B 的初始 Checkpoint")
        limits = {"games": (1, 1000000), "simulations": (1, 1024), "parallel_games": (1, 16),
                  "seed": (0, 4294967295),
                  "batch_size": (1, 1024), "replay_size": (32, 200000), "updates_per_game": (1, 100),
                  "checkpoint_every": (1, 10000), "threads": (1, 16), "eval_every": (0, 100000),
                  "eval_pairs": (1, 20), "temperature_moves": (0, 361)}
        for key, (low, high) in limits.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{key} must be {low}..{high}")
        if not 0 < self.learning_rate <= .1 or not 0 <= self.weight_decay <= 1 or not 0 < self.max_hours <= 168 or not .01 <= self.disk_budget_gb <= 100:
            raise ValueError("invalid learning rate/time/disk budget")
        if self.mode not in ("independent", "history", "mixed", "paired"):
            raise ValueError("unknown training mode")
        if self.opponent_model is not None:
            self.opponent_model = asdict(ModelConfig(**self.opponent_model))
        if self.mode == "paired":
            if self.opponent_model is None and not self.opponent_initial_checkpoint:
                raise ValueError("双模型训练需要配置模型 B")
            if (self.opponent_model is not None and not self.initial_checkpoint and
                    not self.opponent_initial_checkpoint and self.opponent_model["size"] != self.model["size"]):
                raise ValueError("两个模型的棋盘大小必须一致")
        if self.mode == "mixed" and not self.opponent:
            raise ValueError("mixed mode requires a fixed opponent checkpoint")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto/cpu/cuda")


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(path, device="cpu"):
    # Only application-owned checkpoints are accepted through the API.
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("schema") != 1 or ck.get("rules") != RULE_VERSION:
        raise ValueError("incompatible checkpoint schema/rules")
    model = build_model(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


def resolve_initial_models(config: TrainConfig, store: Store):
    """Use each selected checkpoint's own model, irrespective of its previous A/B role."""
    models = [config.model, config.opponent_model]
    for index, cid in enumerate((config.initial_checkpoint, config.opponent_initial_checkpoint)):
        if not cid:
            continue
        checkpoint = store.get_checkpoint(cid)
        if checkpoint.get("rules") != RULE_VERSION:
            raise ValueError(f"模型 {'AB'[index]} 的初始 Checkpoint 规则版本不兼容")
        if not Path(checkpoint["path"]).is_file():
            raise ValueError(f"模型 {'AB'[index]} 的初始 Checkpoint 文件不存在")
        models[index] = asdict(ModelConfig(**checkpoint["model_config"]))
    if config.mode == "paired" and (models[1] is None or models[0]["size"] != models[1]["size"]):
        raise ValueError("模型 A 和 B 的棋盘大小必须一致，请更换 Checkpoint 或调整新模型尺寸")
    return replace(config, model=models[0], opponent_model=models[1])


def resolve_resume_models(config: TrainConfig, store: Store, resume: str):
    """An original warm-start JSON can recover without reopening its source artifacts."""
    saved = store.get_checkpoint(resume)["config"]
    overrides = {}
    for model_key, source_key in (("model", "initial_checkpoint"),
                                  ("opponent_model", "opponent_initial_checkpoint")):
        source = getattr(config, source_key)
        if source and source == saved.get(source_key):
            overrides[model_key] = saved[model_key]
    return replace(config, **overrides) if overrides else config


def initialize_weights(config, store, model, partner_model=None):
    """Warm starts copy weights only; never share parameters or import training state."""
    provenance, loaded = {}, {}
    slots = [("A", model, config.initial_checkpoint)]
    if partner_model is not None:
        slots.append(("B", partner_model, config.opponent_initial_checkpoint))
    for role, destination, cid in slots:
        if not cid:
            provenance[role] = dict(kind="random", model_config=destination.config, seed=config.seed)
            continue
        if cid not in loaded:
            # Loading constructs a temporary CPU module; keep the new task's RNG unchanged.
            with torch.random.fork_rng(devices=[]):
                source, artifact = load_checkpoint(store.get_checkpoint(cid)["path"], "cpu")
            loaded[cid] = (source.config, source.state_dict(),
                           dict(kind="checkpoint", checkpoint=cid, source_agent=artifact.get("agent"),
                                source_games=artifact["games"], source_updates=artifact["updates"],
                                model_config=source.config))
            del source, artifact
        architecture, weights, metadata = loaded[cid]
        if architecture != destination.config:
            raise ValueError(f"模型 {role} 的 Checkpoint 文件结构与登记信息不一致")
        destination.load_state_dict(weights)
        provenance[role] = metadata.copy()
    return provenance


def save_checkpoint(store, eid, model, optimizer, scheduler, replay, rng, progress, config,
                    partner=None, agent=None, pair_id=None, extra_metadata=None, extra_state=None):
    cid = uuid.uuid4().hex
    if partner is None:
        progress.setdefault("opponent_pool", []).append(cid)
    directory = store.root / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"{cid}.pt"
    metadata = dict(model_config=model.config, rules=RULE_VERSION, config=asdict(config),
                    games=progress["games"], updates=progress["updates_b"] if agent == "B" else progress["updates"],
                    training_seconds=progress["elapsed"], agent=agent, pair_id=pair_id,
                    initialization=progress.get("initialization"),
                    parameters=sum(p.numel() for p in model.parameters()),
                    estimated_flops=estimated_flops(ModelConfig(**model.config)))
    metadata.update(training_kind=getattr(config, "kind", "rl"))
    if extra_metadata:
        metadata.update(extra_metadata)
    ck = dict(schema=1, experiment=eid, **metadata, model=model.state_dict(),
              optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
              replay=list(replay), progress=progress.copy(), rng=rng.bit_generator.state,
              python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
              cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    if partner is not None:
        ck["partner"] = partner
    if extra_state:
        ck.update(extra_state)
    temp = path.with_suffix(".tmp")
    torch.save(ck, temp)
    os.replace(temp, path)
    store.checkpoint(cid, eid, path, metadata)
    return cid


def self_play(config, evaluator, rng, count, opponent=None, offset=0, collect_both=False):
    size = config.model["size"]
    games = [Game(size) for _ in range(count)]
    records = [[] for _ in games]
    # Alternate learner colors when using an external opponent.
    while any(not g.terminal for g in games):
        indices = [i for i, g in enumerate(games) if not g.terminal]
        active = [games[i] for i in indices]
        nets = [opponent if opponent and games[i].to_play != (1 if (i+offset) % 2 == 0 else -1)
                else evaluator for i in indices]
        policies = search_many(active, nets, config.simulations, rng, noise=True)
        for i, p, net in zip(indices, policies, nets):
            game = games[i]
            # Extra modes learn only from decisions made by the learner.
            if net is evaluator or collect_both:
                records[i].append((game.encode(), p, game.to_play, net is evaluator))
            action = choose(p, 1. if len(game.moves) < config.temperature_moves else .1, rng)
            game.play(action)
    samples, opponent_samples = [], []
    for game, record in zip(games, records):
        for state, policy, player, is_learner in record:
            target = np.array([game.winner == player, game.winner == 0,
                               game.winner == -player], dtype=np.float32)
            (samples if is_learner else opponent_samples).append((state, policy, target))
    return ((samples, opponent_samples) if collect_both else samples), games


def update(model, optimizer, replay, config, rng, device):
    indices = rng.integers(0, len(replay), size=config.batch_size)
    states, policies, targets = [], [], []
    for i in indices:
        x, p, w = replay[int(i)]
        x, p = augment(x, p, int(rng.integers(8)))
        states.append(x)
        policies.append(p)
        targets.append(w)
    x = torch.from_numpy(np.stack(states)).to(device)
    p = torch.from_numpy(np.stack(policies)).to(device)
    w = torch.from_numpy(np.stack(targets)).to(device)
    model.train()
    logits, value = model(x)
    legal = (x[:, 0]+x[:, 1]).flatten(1) == 0
    # Finite sentinel avoids 0 * -inf in policy cross entropy.
    log_policy = logits.masked_fill(~legal, -1e9).log_softmax(-1)
    policy_loss = -(p*log_policy).sum(-1).mean()
    value_loss = -(w*value.log_softmax(-1)).sum(-1).mean()
    loss = policy_loss + value_loss
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optimizer.step()
    return dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                loss=float(loss.detach()), grad_norm=float(norm))


def _train(config: TrainConfig, store, eid, resume=None):
    seed_all(config.seed, config.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device == "auto":
        device = "cpu"
    model = build_model(ModelConfig(**config.model)).to(device)
    paired = config.mode == "paired"
    partner_model = build_model(ModelConfig(**config.opponent_model)).to(device) if paired else None
    initialization = initialize_weights(config, store, model, partner_model) if not resume else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9995)
    replay = deque(maxlen=config.replay_size)
    partner_replay = deque(maxlen=config.replay_size)
    rng = np.random.default_rng(config.seed)
    progress = dict(games=0, updates=0, elapsed=0., lineage=[eid], opponent_pool=[])
    if initialization is not None:
        progress["initialization"] = initialization
    if paired:
        progress["updates_b"] = 0
    partner_checkpoint = None
    if resume:
        model, ck = load_checkpoint(store.get_checkpoint(resume)["path"], device)
        own_checkpoint = ck
        if paired:
            if "partner" not in ck:
                raise ValueError("此 Checkpoint 不包含双模型训练状态")
            if ck.get("agent") == "B":
                partner_model = model
                own_checkpoint = ck["partner"]
                partner_checkpoint = ck
                model = build_model(ModelConfig(**own_checkpoint["model_config"])).to(device)
                model.load_state_dict(own_checkpoint["model"])
            else:
                partner_checkpoint = ck["partner"]
                partner_model.load_state_dict(partner_checkpoint["model"])
        if config.games <= ck["games"]:
            raise ValueError("resume cumulative games must exceed checkpoint games")
        if model.config != config.model:
            raise ValueError("resume model configuration mismatch")
        old = ck["config"]
        mutable = {"games", "max_hours", "disk_budget_gb", "checkpoint_every", "eval_every", "eval_pairs", "device"}
        if any(old.get(k) != v for k, v in asdict(config).items() if k not in mutable):
            raise ValueError("resume changes learning configuration; create a new experiment instead")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        optimizer.load_state_dict(own_checkpoint["optimizer"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9995)
        scheduler.load_state_dict(own_checkpoint["scheduler"])
        replay.extend(own_checkpoint["replay"])
        progress = ck["progress"].copy()
        progress["lineage"] = list(dict.fromkeys(progress.get("lineage", [ck["experiment"]]) + [eid]))
        progress.setdefault("opponent_pool", [c["id"] for c in reversed(store.checkpoints())
                                              if c["experiment"] == ck["experiment"] and c["games"] <= ck["games"]])
        rng.bit_generator.state = ck["rng"]
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["cuda_rng"]])
    partner_optimizer = partner_scheduler = None
    if paired:
        partner_optimizer = torch.optim.AdamW(partner_model.parameters(), lr=config.learning_rate,
                                               weight_decay=config.weight_decay)
        partner_scheduler = torch.optim.lr_scheduler.ExponentialLR(partner_optimizer, gamma=.9995)
        if partner_checkpoint:
            partner_optimizer.load_state_dict(partner_checkpoint["optimizer"])
            partner_scheduler.load_state_dict(partner_checkpoint["scheduler"])
            partner_replay.extend(partner_checkpoint["replay"])
    evaluator = Evaluator(model, device)
    opponent = Evaluator(partner_model, device) if paired else None

    def save_state():
        if not paired:
            return [save_checkpoint(store, eid, model, optimizer, scheduler, replay, rng, progress, config)]
        pair_id = uuid.uuid4().hex
        def bundle(net, opt, schedule, buffer, role):
            return dict(model_config=net.config, model=net.state_dict(), optimizer=opt.state_dict(),
                        scheduler=schedule.state_dict(), replay=list(buffer), agent=role)
        a = save_checkpoint(store, eid, model, optimizer, scheduler, replay, rng, progress, config,
                            partner=bundle(partner_model, partner_optimizer, partner_scheduler, partner_replay, "B"),
                            agent="A", pair_id=pair_id)
        b = save_checkpoint(store, eid, partner_model, partner_optimizer, partner_scheduler, partner_replay,
                            rng, progress, config, partner=bundle(model, optimizer, scheduler, replay, "A"),
                            agent="B", pair_id=pair_id)
        return [a, b]
    if config.mode == "mixed":
        with torch.random.fork_rng():
            other, _ = load_checkpoint(store.get_checkpoint(config.opponent)["path"], device)
        if other.config["size"] != config.model["size"]:
            raise ValueError("opponent board size mismatch")
        opponent = Evaluator(other, device)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(store.root / "runs" / eid), max_queue=10, flush_secs=15)
    except ImportError:
        pass
    if store.experiment(eid)["status"] not in ("pausing", "stopping"):
        store.status(eid, "running")
    start, prior_elapsed = time.monotonic(), progress["elapsed"]
    last_save = progress["games"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        while progress["games"] < config.games:
            status = store.experiment(eid)["status"]
            if status in ("pausing", "stopping", "interrupted"):
                store.status(eid, "paused" if status == "pausing" else "stopped")
                break
            if time.monotonic()-start >= config.max_hours*3600:
                store.status(eid, "budget_reached")
                break
            disk = sum(p.stat().st_size for p in store.root.rglob("*") if p.is_file())
            if disk >= config.disk_budget_gb*1024**3:
                store.status(eid, "disk_budget_reached")
                break
            count = min(config.parallel_games, config.games-progress["games"])
            missing_opponents = 0
            if config.mode == "history":
                available = {c["id"]: c for c in store.checkpoints()}
                pool = [available[cid] for cid in progress["opponent_pool"] if cid in available]
                missing_opponents = len(progress["opponent_pool"]) - len(pool)
                if pool:
                    selected = pool[int(rng.integers(len(pool)))]
                    with torch.random.fork_rng():
                        historical, _ = load_checkpoint(selected["path"], device)
                    opponent = Evaluator(historical, device)
            begin = time.monotonic()
            samples, games = self_play(config, evaluator, rng, count, opponent, progress["games"], paired)
            sample_seconds = time.monotonic()-begin
            if paired:
                samples, partner_samples = samples
                partner_replay.extend(partner_samples)
            replay.extend(samples)
            update_begin = time.monotonic()
            metrics = {}
            for _ in range(config.updates_per_game*count):
                metrics = update(model, optimizer, replay, config, rng, device)
                scheduler.step()
                progress["updates"] += 1
                if paired:
                    partner_metrics = update(partner_model, partner_optimizer, partner_replay, config, rng, device)
                    partner_scheduler.step()
                    progress["updates_b"] += 1
            model.eval()
            if paired:
                partner_model.eval()
                a_scores = [.5 if game.winner == 0 else float(game.winner == (1 if (i+progress["games"]) % 2 == 0 else -1))
                            for i, game in enumerate(games)]
                metrics.update(partner_policy_loss=partner_metrics["policy_loss"],
                               partner_value_loss=partner_metrics["value_loss"], partner_loss=partner_metrics["loss"],
                               partner_grad_norm=partner_metrics["grad_norm"], replay_b=len(partner_replay),
                               updates_b=progress["updates_b"], a_score=float(np.mean(a_scores)),
                               b_score=1-float(np.mean(a_scores)))
            progress["games"] += count
            progress["elapsed"] = prior_elapsed + time.monotonic()-start
            metrics.update(games=progress["games"], updates=progress["updates"], replay=len(replay),
                           average_length=float(np.mean([len(g.moves) for g in games])),
                           black_win_rate=float(np.mean([g.winner == 1 for g in games])),
                           draw_rate=float(np.mean([g.winner == 0 for g in games])),
                           seconds=progress["elapsed"], games_per_second=count/(time.monotonic()-begin),
                           rss_mb=psutil.Process().memory_info().rss/1024**2,
                           gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if device == "cuda" else 0.,
                           device=device, learning_rate=scheduler.get_last_lr()[0])
            metrics.update(selfplay_seconds=sample_seconds, update_seconds=time.monotonic()-update_begin)
            if config.mode == "history":
                metrics["missing_history_opponents"] = missing_opponents
            store.metric(eid, progress["games"], metrics)
            if writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, progress["games"])
            for game in games[:1]:
                # Retain only the latest observation in a small atomic file.
                p = store.root / f"live-{eid}.json"
                temp = p.with_suffix(".tmp")
                temp.write_text(json.dumps(game.as_dict()), encoding="utf-8")
                os.replace(temp, p)
            eval_due = bool(config.eval_every and progress["games"] // config.eval_every > (progress["games"]-count) // config.eval_every)
            if progress["games"]-last_save >= config.checkpoint_every or progress["games"] >= config.games or eval_due:
                checkpoint_ids = save_state()
                last_save = progress["games"]
                if eval_due:
                    from .evaluation import evaluate
                    for cid, net in zip(checkpoint_ids, [evaluator, opponent] if paired else [evaluator]):
                        report = evaluate(net, "heuristic", config.model["size"],
                                          config.eval_pairs, config.simulations, config.seed)
                        report.update(checkpoint=cid, experiment=eid, games_trained=progress["games"],
                                      training_seconds=progress["elapsed"], opponent_id="heuristic")
                        store.save_evaluation(report)
        else:
            store.status(eid, "completed")
        if last_save != progress["games"] or not any(c["experiment"] == eid for c in store.checkpoints()):
            save_state()
        return eid
    except Exception as exc:
        store.metric(eid, progress["games"], dict(error=f"{type(exc).__name__}: {exc}"))
        store.status(eid, "failed")
        raise
    finally:
        if writer:
            writer.close()


def train(config: TrainConfig, store=None, eid=None, resume=None):
    store = store or Store()
    # Complete recovery reads its own saved networks, even if original source weights were deleted.
    config = resolve_resume_models(config, store, resume) if resume else resolve_initial_models(config, store)
    eid = eid or store.create_experiment(asdict(config))
    try:
        return _train(config, store, eid, resume)
    except Exception as exc:
        if store.experiment(eid)["status"] != "failed":
            store.metric(eid, 0, dict(error=f"{type(exc).__name__}: {exc}"))
            store.status(eid, "failed")
        raise
