from dataclasses import asdict
from pathlib import Path
import threading
import time
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from gomoku.api import create_app
from gomoku.models import ModelConfig, build_model
from gomoku.search import Evaluator, analyze
from gomoku.game import Game
from gomoku.storage import Store
from gomoku.training import TrainConfig, train, load_checkpoint, seed_all, self_play


def paired_config(a="cnn", b="cnn", games=2):
    return TrainConfig(model=asdict(ModelConfig(architecture=a, channels=8, blocks=1, dim=8, layers=1, heads=2)),
                       opponent_model=asdict(ModelConfig(architecture=b, channels=12, blocks=2, dim=12, layers=2, heads=2)),
                       mode="paired",games=games,simulations=2,parallel_games=1,batch_size=4,replay_size=128,
                       updates_per_game=1,checkpoint_every=1,eval_every=0,device="cpu",threads=1,seed=13)


@pytest.mark.parametrize("a,b",[("cnn","cnn"),("cnn","vit"),("vit","vit")])
def test_both_configured_models_really_learn(tmp_path,a,b):
    store=Store(tmp_path)
    config=paired_config(a,b)
    seed_all(config.seed,1)
    initial={"A":build_model(ModelConfig(**config.model)),"B":build_model(ModelConfig(**config.opponent_model))}
    eid=train(config,store)
    checkpoints=[c for c in store.checkpoints() if c["experiment"]==eid and c["games"]==2]
    assert len(checkpoints)==2 and {c["agent"] for c in checkpoints}=={"A","B"}
    assert checkpoints[0]["pair_id"]==checkpoints[1]["pair_id"]
    for checkpoint in checkpoints:
        role=checkpoint["agent"]
        model,saved=load_checkpoint(checkpoint["path"])
        assert saved["updates"]==2 and len(saved["replay"])>0 and len(saved["partner"]["replay"])>0
        assert model.config==(config.model if role=="A" else config.opponent_model)
        assert any(not torch.equal(initial[role].state_dict()[k],v) for k,v in model.state_dict().items())
        assert len(analyze(Game(moves=[40]),Evaluator(model))["top5"])==5
    metrics=store.metrics(eid)[-1]
    assert metrics["updates"]==metrics["updates_b"]==2
    assert metrics["partner_loss"]>0 and metrics["a_score"]+metrics["b_score"]==1


@pytest.mark.parametrize("role",["A","B"])
def test_resume_either_role_restores_both_exactly(tmp_path,role):
    store=Store(tmp_path)
    cfg=paired_config("cnn","vit",3)
    eid=train(cfg,store)
    checkpoints=[c for c in store.checkpoints() if c["experiment"]==eid]
    source=next(c for c in checkpoints if c["games"]==1 and c["agent"]==role)
    counterpart=next(c for c in checkpoints if c["pair_id"]==source["pair_id"] and c["agent"]!=role)
    # Each artifact is a self-contained resume point, even if its companion row is deleted.
    store.delete_checkpoint(counterpart["id"])
    resumed=train(cfg,store,resume=source["id"])
    for agent in ("A","B"):
        final=next(c for c in checkpoints if c["games"]==3 and c["agent"]==agent)
        continued=next(c for c in store.checkpoints() if c["experiment"]==resumed and c["agent"]==agent and c["games"]==3)
        expected,_=load_checkpoint(final["path"])
        actual,_=load_checkpoint(continued["path"])
        assert all(torch.equal(x,y) for x,y in zip(expected.state_dict().values(),actual.state_dict().values()))


class Scripted:
    def batch(self,games):
        script=[0,80,1,78,2,76,3,74,4]
        results=[]
        for game in games:
            policy=np.zeros(81,dtype=np.float32)
            policy[script[len(game.moves)]]=1
            results.append((policy,np.array([.5,0,.5])))
        return results


@pytest.mark.parametrize("offset",[0,1])
def test_paired_samples_use_each_players_outcome(offset):
    cfg=paired_config()
    cfg.simulations=1
    (samples_a,samples_b),games=self_play(cfg,Scripted(),np.random.default_rng(7),1,Scripted(),offset,True)
    assert games[0].winner==1
    assert len(samples_a)==(5 if offset==0 else 4)
    assert len(samples_b)==(4 if offset==0 else 5)
    assert all(w.tolist()==([1,0,0] if offset==0 else [0,0,1]) for _,_,w in samples_a)
    assert all(w.tolist()==([0,0,1] if offset==0 else [1,0,0]) for _,_,w in samples_b)


def test_delete_weight_preserves_archive_and_clears_cached_model(tmp_path):
    store=Store(tmp_path)
    eid=train(paired_config(),store)
    ck=next(c for c in store.checkpoints() if c["experiment"]==eid)
    with TestClient(create_app(store)) as client:
        game=client.post("/api/games",json=dict(checkpoint=ck["id"],human=1,simulations=0)).json()
        store.save_evaluation(dict(checkpoint=ck["id"],score=.5))
        preview=client.get(f"/api/checkpoints/{ck['id']}/deletion-preview")
        assert preview.json()["references"]==dict(games=1,evaluations=1)
        assert preview.json()["bytes"]>0
        response=client.delete(f"/api/checkpoints/{ck['id']}")
        assert response.status_code==200,response.text
        assert not Path(ck["path"]).exists()
        assert ck["id"] not in {c["id"] for c in client.get("/api/checkpoints").json()}
        assert client.post("/api/analyze",json=dict(checkpoint=ck["id"],moves=[])).status_code==404
        assert client.get(f"/api/games/{game['id']}").json()["moves"]==[]
        assert client.get(f"/api/games/{game['id']}/export").status_code==200
        assert len(store.evaluations())==1
        assert client.delete(f"/api/checkpoints/{ck['id']}").status_code==404


def test_delete_blocks_active_training_and_evaluation(tmp_path,monkeypatch):
    store=Store(tmp_path)
    train(paired_config(),store)
    ck=store.checkpoints()[0]
    app=create_app(store)
    with TestClient(app) as client:
        eid=store.create_experiment({})
        store.status(eid,"running")
        assert client.delete(f"/api/checkpoints/{ck['id']}").status_code==409
        store.status(eid,"stopped")
        entered,release=threading.Event(),threading.Event()
        def blocked_evaluate(*args,**kwargs):
            entered.set()
            release.wait(10)
            return dict(score=.5)
        monkeypatch.setattr("gomoku.evaluation.evaluate",blocked_evaluate)
        job=client.post("/api/evaluations",json=dict(checkpoint=ck["id"],opponent="random",simulations=0,pairs=1)).json()
        try:
            assert entered.wait(5)
            assert client.delete(f"/api/checkpoints/{ck['id']}").status_code==409
        finally:
            release.set()
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            status=client.get("/api/jobs/"+job["id"]).json()["status"]
            if status in ("completed","failed"):
                break
            time.sleep(.02)
        assert status=="completed"
    assert Path(ck["path"]).exists()


def test_delete_rejects_external_paths_and_recovers_file_error(tmp_path,monkeypatch):
    store=Store(tmp_path/"data")
    eid=train(paired_config(games=1),store)
    checkpoint=store.checkpoints()[0]
    outside=tmp_path/"outside.pt"
    outside.write_bytes(b"keep")
    store.checkpoint("outside",eid,outside,{})
    with pytest.raises(ValueError,match="outside"):
        store.delete_checkpoint("outside")
    assert outside.read_bytes()==b"keep"
    def fail_unlink(self,**kwargs):
        raise PermissionError("locked")
    with monkeypatch.context() as context:
        context.setattr(Path,"unlink",fail_unlink)
        with pytest.raises(ValueError,match="locked"):
            store.delete_checkpoint(checkpoint["id"])
    assert store.get_checkpoint(checkpoint["id"])["id"]==checkpoint["id"]
    assert Path(checkpoint["path"]).exists()


def test_config_requires_compatible_pair():
    with pytest.raises(ValueError,match="模型 B"):
        TrainConfig(mode="paired")
    with pytest.raises(ValueError,match="棋盘大小"):
        TrainConfig(mode="paired",opponent_model=asdict(ModelConfig(size=15)))


def test_history_resume_skips_a_deleted_pool_member(tmp_path):
    store=Store(tmp_path)
    cfg=paired_config(games=3)
    cfg.mode="history"
    cfg.opponent_model=None
    eid=train(cfg,store)
    checkpoints=[c for c in store.checkpoints() if c["experiment"]==eid]
    first=next(c for c in checkpoints if c["games"]==1)
    second=next(c for c in checkpoints if c["games"]==2)
    store.delete_checkpoint(first["id"])
    resumed=train(cfg,store,resume=second["id"])
    assert store.experiment(resumed)["status"]=="completed"
    assert store.metrics(resumed)[-1]["missing_history_opponents"]==1
