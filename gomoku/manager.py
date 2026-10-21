import subprocess
import sys
import threading
from dataclasses import asdict
from .storage import Store
from .training import TrainConfig, resolve_initial_models, resolve_resume_models


class TrainingManager:
    def __init__(self, store: Store):
        self.store = store
        self.process = None
        self.eid = None
        self.lock = threading.Lock()
        # A PID is deliberately not reused after a backend restart.
        for exp in store.experiments():
            if exp["status"] in ("running", "pausing", "stopping"):
                store.status(exp["id"], "interrupted")

    def start(self, config: TrainConfig, resume=None):
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("one training process at a time; pause or stop it first")
            config = resolve_resume_models(config, self.store, resume) if resume else resolve_initial_models(config, self.store)
            if resume:
                checkpoint = self.store.get_checkpoint(resume)
                if checkpoint.get("training_kind") == "sft":
                    raise ValueError("SFT 权重可用作 RL 初始模型；完整恢复请进入 SFT 页面")
                if config.games <= checkpoint["games"]:
                    raise ValueError("resume cumulative games must exceed checkpoint games")
                mutable = {"games", "max_hours", "disk_budget_gb", "checkpoint_every", "eval_every", "eval_pairs", "device"}
                expected_model = config.opponent_model if config.mode == "paired" and checkpoint.get("agent") == "B" else config.model
                if checkpoint["model_config"] != expected_model or any(
                    checkpoint["config"].get(k) != v for k, v in asdict(config).items() if k not in mutable
                ):
                    raise ValueError("resume learning configuration mismatch")
            if config.mode == "mixed":
                opponent = self.store.get_checkpoint(config.opponent)
                if opponent["model_config"]["size"] != config.model["size"]:
                    raise ValueError("opponent board size mismatch")
            return self._spawn(config, resume)

    def start_sft(self, config, resume=None):
        from .sft import prepare_sft
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("已有训练在运行，请先暂停或停止")
            return self._spawn(prepare_sft(config, self.store, resume), resume)

    def _spawn(self, config, resume):
        eid = self.store.create_experiment(asdict(config))
        command = [sys.executable, "-m", "gomoku.cli", "worker", "--experiment", eid,
                   "--data", str(self.store.root)]
        if resume:
            command.extend(["--resume", resume])
        logdir = self.store.root / "logs"
        logdir.mkdir(exist_ok=True)
        try:
            with open(logdir / f"{eid}.log", "w", encoding="utf-8") as log:
                self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            self.store.status(eid, "failed")
            raise
        self.eid = eid
        self.store.status(eid, "running")
        return eid

    def control(self, eid, action):
        with self.lock:
            if self.eid != eid or not self.process or self.process.poll() is not None:
                raise ValueError("no active training worker for this experiment")
            self.store.status(eid, "pausing" if action == "pause" else "stopping")

    def reconcile(self):
        if self.process and self.process.poll() is not None:
            status = self.store.experiment(self.eid)["status"]
            if status in ("running", "pausing", "stopping"):
                self.store.status(self.eid, "failed")

    def shutdown(self):
        if self.process and self.process.poll() is None:
            self.store.status(self.eid, "stopping")
