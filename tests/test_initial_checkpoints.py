from dataclasses import asdict, replace
from pathlib import Path
import time
import hashlib
import pytest
import torch
from fastapi.testclient import TestClient
import gomoku.training as training
from gomoku.api import create_app
from gomoku.manager import TrainingManager
from gomoku.models import ModelConfig, build_model
from gomoku.storage import Store
from gomoku.training import TrainConfig, train, load_checkpoint, resolve_initial_models, seed_all


def config(**kwargs):
    values=dict(model=asdict(ModelConfig(channels=8,blocks=1,dim=8,layers=1,heads=2)),
                opponent_model=asdict(ModelConfig(architecture="vit",channels=12,blocks=2,dim=12,layers=1,heads=2)),
                mode="paired",games=1,simulations=1,parallel_games=1,batch_size=4,replay_size=128,
                updates_per_game=1,checkpoint_every=1,eval_every=0,device="cpu",threads=1,seed=19)
    values.update(kwargs)
    return TrainConfig(**values)


@pytest.fixture
def sources(tmp_path):
    store=Store(tmp_path)
    eid=train(config(games=2),store)
    return store,{c["agent"]:c for c in store.checkpoints() if c["experiment"]==eid and c["games"]==2}


@pytest.mark.parametrize("a,b",[(None,None),("A",None),(None,"B"),("A","B"),("B","A"),("A","A")])
def test_every_initial_source_combination_copies_real_weights_and_learns(sources,monkeypatch,a,b):
    store,source=sources
    hashes={key:hashlib.sha256(Path(c["path"]).read_bytes()).digest() for key,c in source.items()}
    cfg=config(initial_checkpoint=source[a]["id"] if a else None,
               opponent_initial_checkpoint=source[b]["id"] if b else None,learning_rate=.003)
    resolved=resolve_initial_models(cfg,store)
    seed_all(cfg.seed,1)
    expected={"A":build_model(ModelConfig(**resolved.model)),"B":build_model(ModelConfig(**resolved.opponent_model))}
    for role,name in (("A",a),("B",b)):
        if name:
            expected[role],_=load_checkpoint(source[name]["path"])
    captured,game_lengths={},[]
    original=training.self_play
    original_update=training.update
    optimizers=[]
    def inspect_self_play(configuration,net,rng,count,opponent,*args):
        for role,evaluator in (("A",net),("B",opponent)):
            captured[role]={k:v.detach().cpu().clone() for k,v in evaluator.model.state_dict().items()}
            assert all(torch.equal(v,expected[role].state_dict()[k]) for k,v in captured[role].items())
        assert next(net.model.parameters()).data_ptr()!=next(opponent.model.parameters()).data_ptr()
        result=original(configuration,net,rng,count,opponent,*args)
        game_lengths.extend(len(g.moves) for g in result[1])
        return result
    def inspect_update(model,optimizer,*args):
        assert len(optimizer.state)==0  # source Adam moments were not imported
        optimizers.append(optimizer)
        return original_update(model,optimizer,*args)
    monkeypatch.setattr(training,"self_play",inspect_self_play)
    monkeypatch.setattr(training,"update",inspect_update)
    eid=train(cfg,store)
    assert all(hashlib.sha256(Path(c["path"]).read_bytes()).digest()==hashes[key] for key,c in source.items())
    assert len(optimizers)==2 and optimizers[0] is not optimizers[1]
    saved_config=store.experiment(eid)["config"]
    assert saved_config["model"]==resolved.model and saved_config["opponent_model"]==resolved.opponent_model
    for checkpoint in (c for c in store.checkpoints() if c["experiment"]==eid):
        role=checkpoint["agent"]
        net,saved=load_checkpoint(checkpoint["path"])
        assert saved["games"]==saved["updates"]==1
        assert saved["optimizer"]["param_groups"][0]["lr"]==pytest.approx(.003*.9995)
        assert saved["scheduler"]["last_epoch"]==1
        assert len(saved["replay"])==((game_lengths[0]+1)//2 if role=="A" else game_lengths[0]//2)
        assert any(not torch.equal(v,captured[role][k]) for k,v in net.state_dict().items())
        for slot,name in (("A",a),("B",b)):
            info=saved["initialization"][slot]
            assert info["kind"]==("checkpoint" if name else "random")
            if name:
                assert info["checkpoint"]==source[name]["id"] and info["source_games"]==2


def test_checkpoint_ids_can_supply_both_architectures_without_manual_configs(sources):
    store,source=sources
    cfg=TrainConfig(mode="paired",initial_checkpoint=source["B"]["id"],
                    opponent_initial_checkpoint=source["A"]["id"],games=1,device="cpu",
                    simulations=1,threads=1,eval_every=0,updates_per_game=1,batch_size=4)
    manager=TrainingManager(store)
    eid=manager.start(cfg)
    try:
        deadline=time.monotonic()+30
        while manager.process.poll() is None and time.monotonic()<deadline:
            time.sleep(.05)
        assert manager.process.poll()==0
        assert store.experiment(eid)["status"]=="completed"
        actual=store.experiment(eid)["config"]
        assert actual["model"]==source["B"]["model_config"]
        assert actual["opponent_model"]==source["A"]["model_config"]
    finally:
        if manager.process.poll() is None:
            manager.process.terminate()


@pytest.mark.parametrize("role",["A","B"])
def test_complete_recovery_does_not_require_original_initial_sources(sources,role):
    store,source=sources
    cfg=config(initial_checkpoint=source["B"]["id"],opponent_initial_checkpoint=source["A"]["id"],
               opponent_model=None,games=3)
    eid=train(cfg,store)
    checkpoints=[c for c in store.checkpoints() if c["experiment"]==eid]
    point=next(c for c in checkpoints if c["games"]==1 and c["agent"]==role)
    store.delete_checkpoint(source["A"]["id"])
    store.delete_checkpoint(source["B"]["id"])
    if role=="A":
        resumed=train(cfg,store,resume=point["id"])
    else:
        manager=TrainingManager(store)
        resumed=manager.start(cfg,point["id"])
        try:
            deadline=time.monotonic()+30
            while manager.process.poll() is None and time.monotonic()<deadline:
                time.sleep(.05)
            assert manager.process.poll()==0
        finally:
            if manager.process.poll() is None:
                manager.process.terminate()
    assert store.experiment(resumed)["status"]=="completed"
    for slot in ("A","B"):
        expected=next(c for c in checkpoints if c["games"]==3 and c["agent"]==slot)
        actual=next(c for c in store.checkpoints() if c["experiment"]==resumed and c["games"]==3 and c["agent"]==slot)
        a,_=load_checkpoint(expected["path"])
        b,_=load_checkpoint(actual["path"])
        assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))


def test_single_model_can_also_start_from_a_paired_b_checkpoint(sources):
    store,source=sources
    eid=train(config(mode="independent",opponent_model=None,initial_checkpoint=source["B"]["id"]),store)
    checkpoint=next(c for c in store.checkpoints() if c["experiment"]==eid)
    _,saved=load_checkpoint(checkpoint["path"])
    assert saved["games"]==1 and saved["model_config"]==source["B"]["model_config"]
    assert "partner" not in saved and set(saved["initialization"])=={"A"}


def test_api_rejects_missing_or_incompatible_sources_before_creating_worker(sources):
    store,source=sources
    size15=asdict(ModelConfig(size=15,channels=8,blocks=1,dim=8,layers=1,heads=2))
    eid=train(config(mode="independent",model=size15,opponent_model=None),store)
    large=next(c for c in store.checkpoints() if c["experiment"]==eid)
    with TestClient(create_app(store)) as client:
        before=len(store.experiments())
        mismatch=asdict(config(initial_checkpoint=source["A"]["id"],opponent_initial_checkpoint=large["id"]))
        r=client.post('/api/experiments',json=dict(config=mismatch))
        assert r.status_code==400 and "棋盘大小" in r.json()["detail"]
        store.delete_checkpoint(source["A"]["id"])
        r=client.post('/api/experiments',json=dict(config=asdict(config(initial_checkpoint=source["A"]["id"]))))
        assert r.status_code==404
        Path(source["B"]["path"]).unlink()
        r=client.post('/api/experiments',json=dict(config=asdict(config(opponent_initial_checkpoint=source["B"]["id"]))))
        assert r.status_code==400 and "文件不存在" in r.json()["detail"]
        assert len(store.experiments())==before


def test_incompatible_rules_and_invalid_source_config_are_rejected(sources):
    store,source=sources
    import json
    with store.db() as db:
        row=db.execute('SELECT payload FROM checkpoints WHERE id=?',(source['A']['id'],)).fetchone()
        payload=json.loads(row[0]);payload['rules']='unsupported'
        db.execute('UPDATE checkpoints SET payload=? WHERE id=?',(json.dumps(payload),source['A']['id']))
    with pytest.raises(ValueError,match="规则版本"):
        resolve_initial_models(config(initial_checkpoint=source['A']['id']),store)
    with pytest.raises(ValueError,match="只有双模型"):
        TrainConfig(opponent_initial_checkpoint=source['B']['id'])
    with pytest.raises(ValueError,match="有效的 ID"):
        TrainConfig(initial_checkpoint=True)


def test_bad_source_file_is_reported_as_failed_experiment(sources):
    store,source=sources
    Path(source['A']['path']).write_bytes(b'not a checkpoint')
    before={e['id'] for e in store.experiments()}
    with pytest.raises(Exception):
        train(config(initial_checkpoint=source['A']['id']),store)
    eid=next(e['id'] for e in store.experiments() if e['id'] not in before)
    assert store.experiment(eid)['status']=='failed'
    assert store.metrics(eid)[-1]['error']
