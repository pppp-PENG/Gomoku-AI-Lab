"""Supervised cold start: imitate BOTH players, with outcome WDL supervision.

Splits are made by complete game before augmentation. Saved artifacts include the
immutable selected games, split, permutation, cursor and all training RNG states.
"""
from dataclasses import asdict, dataclass, field, replace
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch

from .demonstrations import validate_demonstration
from .game import Game, RULE_VERSION, augment
from .models import ModelConfig, build_model
from .storage import Store
from .training import load_checkpoint, save_checkpoint, seed_all


@dataclass
class SFTConfig:
    kind: str = "sft"
    model: dict = field(default_factory=lambda: asdict(ModelConfig()))
    initial_checkpoint: str | None = None
    dataset_ids: list[str] = field(default_factory=list)
    external_sources: list[str] = field(default_factory=list)
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = .001
    weight_decay: float = .0001
    value_weight: float = 1.
    validation_fraction: float = .2
    augment: bool = True
    seed: int = 42
    checkpoint_every: int = 5
    device: str = "auto"
    threads: int = 4
    max_hours: float = 1.
    disk_budget_gb: float = 4.

    def __post_init__(self):
        self.model = asdict(ModelConfig(**self.model))
        if self.kind != "sft":
            raise ValueError("不是 SFT 配置")
        if (not isinstance(self.external_sources, list) or len(set(self.external_sources)) != len(self.external_sources)
                or any(s not in ("karesis", "gomoku_party") for s in self.external_sources)):
            raise ValueError("请选择已获许可的外部数据源")
        if (not isinstance(self.dataset_ids, list) or not 0 <= len(self.dataset_ids) <= 5000 or
                (not self.dataset_ids and not self.external_sources) or
                any(not isinstance(i, str) or not i or len(i) > 128 for i in self.dataset_ids) or
                len(set(self.dataset_ids)) != len(self.dataset_ids)):
            raise ValueError("请选择 1..5000 局不重复的采集棋谱")
        if self.initial_checkpoint is not None and (not isinstance(self.initial_checkpoint, str) or
                not self.initial_checkpoint.strip() or len(self.initial_checkpoint) > 128):
            raise ValueError("初始 Checkpoint ID 无效")
        for key, low, high in (("epochs", 1, 10000), ("batch_size", 1, 1024), ("seed", 0, 4294967295),
                               ("checkpoint_every", 1, 10000), ("threads", 1, 16)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{key} 必须为 {low}..{high} 的整数")
        for key, low, high in (("learning_rate", 1e-8, .1), ("weight_decay", 0, 1),
                               ("value_weight", 0, 10), ("validation_fraction", 0, .5),
                               ("max_hours", 1e-6, 168), ("disk_budget_gb", .01, 100)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} 超出资源范围")
        if not isinstance(self.augment, bool) or self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("设备或增强设置无效")


def prepare_sft(config, store, resume=None):
    if resume:
        ck = store.get_checkpoint(resume)
        if ck.get("training_kind") != "sft":
            raise ValueError("只能从 SFT Checkpoint 完整恢复；RL 权重请用作初始来源")
        config = replace(config, model=ck["model_config"])
        mutable = {"epochs", "device", "max_hours", "disk_budget_gb", "checkpoint_every"}
        if any(ck["config"].get(k, [] if k == "external_sources" else None) != v for k, v in asdict(config).items() if k not in mutable):
            raise ValueError("恢复配置与原 SFT 任务不一致，请新建任务")
        if config.epochs <= ck["sft"]["completed_epochs"]:
            raise ValueError("累计轮数必须大于已完成轮数")
    elif config.initial_checkpoint:
        ck = store.get_checkpoint(config.initial_checkpoint)
        if ck["rules"] != RULE_VERSION or not Path(ck["path"]).is_file():
            raise ValueError("初始权重文件不存在或规则不兼容")
        config = replace(config, model=ck["model_config"])
    if not resume:
        selected_games(config, store)  # fail before an experiment is created
    return config


def selected_games(config, store):
    games = []
    for did in config.dataset_ids:
        source = store.demonstration(did)
        game = validate_demonstration(source["size"], source["moves"], source.get("name", ""), source["rules"])
        if game["size"] != config.model["size"]:
            raise ValueError("所选棋谱与模型的棋盘大小不一致")
        games.append(dict(id=did, **{k: game[k] for k in ("size", "moves", "winner", "rules", "name")}))
    if config.external_sources:
        from gomoku_data.lab import selected_for_sft, deduplicate_human
        external = selected_for_sft(config.external_sources, config.model["size"])
        games = deduplicate_human(games, external) + external
    if sum(len(g["moves"]) for g in games) > 200000:
        raise ValueError("单任务最多 200000 个落子样本，请分批训练")
    return games


def split_games(games, fraction, rng):
    order = rng.permutation(len(games)).tolist()
    count = min(len(games)-1, max(1, round(len(games)*fraction))) if fraction and len(games) > 1 else 0
    return order[count:], order[:count]


def split_selected_games(games, fraction, rng):
    """External split is immutable; held-out test games never reach this function."""
    human = [i for i, g in enumerate(games) if "external_split" not in g]
    train, validation = split_games([games[i] for i in human], fraction, rng)
    train = [human[i] for i in train]
    validation = [human[i] for i in validation]
    for i, game in enumerate(games):
        if game.get("external_split") == "train":
            train.append(i)
        elif game.get("external_split") == "validation":
            validation.append(i)
        elif "external_split" in game:
            raise ValueError("测试集禁止进入 SFT 训练")
    if not train:
        raise ValueError("没有可用训练棋局")
    return train, validation


class MoveDataset:
    """Compact indices; reconstruct only a batch instead of retaining all float boards."""
    def __init__(self, games, indices):
        self.games = games
        self.indices = [(i, ply) for i in indices for ply in range(len(games[i]["moves"]))]

    def __len__(self):
        return len(self.indices)

    def sample(self, index, symmetry=0):
        i, ply = self.indices[index]
        source = self.games[i]
        game = Game(source["size"], moves=source["moves"][:ply])
        action = source["moves"][ply]
        policy = np.zeros(game.size**2, dtype=np.float32)
        policy[action] = 1.
        x, policy = augment(game.encode(), policy, symmetry)
        wdl = np.array([source["winner"] == game.to_play, source["winner"] == 0,
                        source["winner"] == -game.to_play], dtype=np.float32)
        return x, int(policy.argmax()), wdl

    def batch(self, indices, rng=None):
        rows = [self.sample(int(i), int(rng.integers(8)) if rng is not None else 0) for i in indices]
        x, action, wdl = zip(*rows)
        return torch.from_numpy(np.stack(x)), torch.tensor(action), torch.from_numpy(np.stack(wdl))


def supervised_loss(model, batch, device, value_weight):
    x, action, wdl = [t.to(device) for t in batch]
    logits, value = model(x)
    legal = (x[:, 0] + x[:, 1]).flatten(1) == 0
    logits = logits.masked_fill(~legal, -1e9)
    policy_loss = torch.nn.functional.cross_entropy(logits, action)
    value_loss = -(wdl * value.log_softmax(-1)).sum(-1).mean()
    loss = policy_loss + value_weight * value_loss
    if not torch.isfinite(loss):
        raise RuntimeError("SFT loss 非有限值")
    metrics = dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                   loss=float(loss.detach()), accuracy=float((logits.argmax(-1) == action).float().mean()))
    return loss, metrics


def validate(model, dataset, config, device):
    if not len(dataset):
        return {}
    sums = dict(policy_loss=0., value_loss=0., loss=0., accuracy=0.)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), config.batch_size):
            indices = range(start, min(start+config.batch_size, len(dataset)))
            _, metrics = supervised_loss(model, dataset.batch(indices), device, config.value_weight)
            for key in sums:
                sums[key] += metrics[key] * len(indices)
    return {"val_"+key: value/len(dataset) for key, value in sums.items()}


def _train_sft(config, store, eid, resume):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_all(config.seed, config.threads)
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device == "auto":
        device = "cpu"
    rng = np.random.default_rng(config.seed)
    if resume:
        model, ck = load_checkpoint(store.get_checkpoint(resume)["path"], device)
        state, progress = ck["sft_state"], ck["progress"]
        progress["lineage"] = list(dict.fromkeys(progress["lineage"] + [eid]))
    else:
        model = build_model(ModelConfig(**config.model)).to(device)
        initialization = dict(kind="random", seed=config.seed, model_config=model.config)
        if config.initial_checkpoint:
            with torch.random.fork_rng(devices=[]):
                source, origin = load_checkpoint(store.get_checkpoint(config.initial_checkpoint)["path"])
            if source.config != model.config:
                raise ValueError("初始权重结构与登记信息不一致")
            model.load_state_dict(source.state_dict())
            initialization = dict(kind="checkpoint", checkpoint=config.initial_checkpoint,
                                  source_agent=origin.get("agent"), source_games=origin["games"],
                                  source_updates=origin["updates"], model_config=model.config)
            del source, origin
        games = selected_games(config, store)
        train_indices, val_indices = split_selected_games(games, config.validation_fraction, rng)
        state = dict(games=games, train_indices=train_indices, val_indices=val_indices,
                     epoch=0, order=[], offset=0, sums={}, count=0, samples_seen=0)
        progress = dict(games=len(games), updates=0, elapsed=0., lineage=[eid],
                        initialization={"A": initialization}, opponent_pool=[])
    training = MoveDataset(state["games"], state["train_indices"])
    validation = MoveDataset(state["games"], state["val_indices"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9995)
    if resume:
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        rng.bit_generator.state = ck["rng"]
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["cuda_rng"]])
        del ck
    if store.experiment(eid)["status"] not in ("pausing", "stopping"):
        store.status(eid, "running")
    begin, elapsed = time.monotonic(), progress["elapsed"]
    prior_samples = state["samples_seen"]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(store.root / "runs" / eid), max_queue=10, flush_secs=15)
    except ImportError:
        pass
    last_saved, last_reported = None, None

    def save():
        nonlocal last_saved
        progress["elapsed"] = elapsed + time.monotonic()-begin
        # Disk budget covers this task's lineage; retain at least the latest complete artifact.
        total = sum(Path(c["path"]).stat().st_size for c in store.checkpoints()
                    if c["experiment"] in progress["lineage"] and Path(c["path"]).exists())
        summary = dict(completed_epochs=state["epoch"], train_games=len(state["train_indices"]),
                       validation_games=len(state["val_indices"]), train_samples=len(training),
                       validation_samples=len(validation), samples_seen=state["samples_seen"])
        cid = save_checkpoint(store, eid, model, optimizer, scheduler, [], rng, progress, config,
                              extra_metadata={"sft": summary}, extra_state={"sft_state": state})
        total += (store.root / "checkpoints" / f"{cid}.pt").stat().st_size
        if total / 1024**3 >= config.disk_budget_gb and state["epoch"] < config.epochs:
            store.metric(eid, progress["updates"], dict(warning="SFT 权重磁盘预算已达到，已保存并暂停"))
            store.status(eid, "pausing")
        last_saved = (state["epoch"], state["offset"], progress["updates"])

    try:
        while state["epoch"] < config.epochs:
            status = store.experiment(eid)["status"]
            if status in ("pausing", "stopping", "interrupted") or time.monotonic()-begin >= config.max_hours*3600:
                store.status(eid, "stopped" if status == "stopping" else "paused")
                break
            if not state["order"]:
                state["order"] = rng.permutation(len(training)).tolist()
            indices = state["order"][state["offset"]:state["offset"]+config.batch_size]
            model.train()
            batch = training.batch(indices, rng if config.augment else None)
            loss, metrics = supervised_loss(model, batch, device, config.value_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise RuntimeError("SFT 梯度非有限值")
            optimizer.step()
            scheduler.step()
            state["offset"] += len(indices)
            state["count"] += len(indices)
            state["samples_seen"] += len(indices)
            progress["updates"] += 1
            for key, value in metrics.items():
                state["sums"][key] = state["sums"].get(key, 0.) + value*len(indices)
            epoch_finished = state["offset"] == len(training)
            if epoch_finished:
                metrics = {k: v/state["count"] for k, v in state["sums"].items()}
                metrics.update(validate(model, validation, config, device))
                state.update(epoch=state["epoch"]+1, order=[], offset=0, sums={}, count=0)
            if epoch_finished or progress["updates"] % 10 == 0:
                progress["elapsed"] = elapsed + time.monotonic()-begin
                metrics.update(epoch=state["epoch"]+state["offset"]/len(training), updates=progress["updates"],
                               completed_epochs=state["epoch"], train_samples=len(training),
                               validation_samples=len(validation), samples_seen=state["samples_seen"],
                               seconds=progress["elapsed"], device=device,
                               learning_rate=scheduler.get_last_lr()[0],
                               samples_per_second=(state["samples_seen"]-prior_samples)/max(.001, time.monotonic()-begin),
                               gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if device == "cuda" else 0.,
                               rss_mb=psutil.Process().memory_info().rss/1024**2)
                store.metric(eid, progress["updates"], metrics)
                last_reported = progress["updates"]
                if writer:
                    for key, value in metrics.items():
                        if isinstance(value, (float, int)):
                            writer.add_scalar(key, value, progress["updates"])
            if epoch_finished and (state["epoch"] % config.checkpoint_every == 0 or state["epoch"] == config.epochs):
                save()
        else:
            store.status(eid, "completed")
        if last_reported != progress["updates"]:
            metrics = {k: v/state["count"] for k, v in state["sums"].items()} if state["count"] else {}
            metrics.update(epoch=state["epoch"]+state["offset"]/len(training), completed_epochs=state["epoch"],
                           updates=progress["updates"], train_samples=len(training), validation_samples=len(validation),
                           samples_seen=state["samples_seen"], seconds=elapsed+time.monotonic()-begin, device=device,
                           gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if device == "cuda" else 0.)
            store.metric(eid, progress["updates"], metrics)
        if last_saved != (state["epoch"], state["offset"], progress["updates"]):
            save()
        if store.experiment(eid)["status"] == "pausing":
            store.status(eid, "paused")
        return eid
    finally:
        if writer:
            writer.close()


def train_sft(config: SFTConfig, store=None, eid=None, resume=None):
    store = store or Store()
    config = prepare_sft(config, store, resume)
    eid = eid or store.create_experiment(asdict(config))
    try:
        return _train_sft(config, store, eid, resume)
    except Exception as exc:
        store.metric(eid, 0, dict(error=f"{type(exc).__name__}: {exc}"))
        store.status(eid, "failed")
        raise
