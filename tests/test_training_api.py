from dataclasses import asdict
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from gomoku.models import ModelConfig,build_model
from gomoku.storage import Store
from gomoku.training import TrainConfig, train, load_checkpoint,seed_all
from gomoku.api import create_app


def tiny(architecture='cnn',games=2):
    return TrainConfig(model=asdict(ModelConfig(architecture=architecture,channels=8,blocks=1,dim=8,layers=1,heads=2)),
                       games=games, simulations=2, parallel_games=1, batch_size=4,
                       replay_size=128, updates_per_game=1, checkpoint_every=1,
                       eval_every=0,device='cpu',threads=1,seed=7)


@pytest.mark.parametrize('architecture',['cnn','vit'])
def test_actual_update_checkpoint_resume(tmp_path,architecture):
    store=Store(tmp_path/architecture)
    config=tiny(architecture)
    seed_all(config.seed,1)
    initial=build_model(ModelConfig(**config.model)).state_dict()
    eid=train(config,store)
    assert store.experiment(eid)['status']=='completed'
    ck=next(c for c in store.checkpoints() if c['experiment']==eid)
    model,saved=load_checkpoint(ck['path'])
    assert saved['games']==2 and saved['updates']==2 and len(saved['replay'])>0
    assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
    first=next(c for c in store.checkpoints() if c['experiment']==eid and c['games']==1)
    resumed=train(config,store,resume=first['id'])
    ck2=next(c for c in store.checkpoints() if c['experiment']==resumed)
    model2,_=load_checkpoint(ck2['path'])
    # CPU deterministic continuation, including replay and exploration RNG.
    assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),model2.state_dict().values()))
    assert store.metrics(eid)[-1]['loss']>0


def test_game_api_analysis_branch_import(tmp_path):
    store=Store(tmp_path)
    eid=train(tiny(games=1),store)
    ck=store.checkpoints()[0]['id']
    with TestClient(create_app(store)) as client:
        assert client.get('/api/health').status_code==200
        game=client.post('/api/games',json=dict(checkpoint=ck,human=1,simulations=2)).json()
        gid=game['id']
        assert client.post(f'/api/games/{gid}/move',json=dict(action=0,revision=0)).status_code==200
        assert client.post(f'/api/games/{gid}/ai',json=dict(revision=0)).status_code==409
        response=client.post(f'/api/games/{gid}/ai',json=dict(revision=1))
        assert response.status_code==200,response.text
        game=response.json()
        assert len(game['moves'])==2
        a=client.post('/api/analyze',json=dict(checkpoint=ck,moves=game['moves'],simulations=2))
        assert a.status_code==200,a.text
        result=a.json()
        assert len(result['top5'])==5
        assert all(c['action'] not in game['moves'] for c in result['top5'])
        assert sum(c['policy'] for c in result['candidates'])==pytest.approx(1.)
        branch=client.post(f'/api/games/{gid}/branch',json=dict(step=1,human=-1)).json()
        assert branch['moves']==[0] and branch['parent']==dict(game=gid,step=1)
        assert client.get(f'/api/games/{gid}').json()['moves']==game['moves']
        exported=client.get(f'/api/games/{gid}/export').json()
        imported=client.post('/api/games/import',json=exported)
        assert imported.status_code==200,imported.text
        assert imported.json()['moves']==game['moves']
        undo=client.post(f'/api/games/{gid}/undo',json=dict(revision=2)).json()
        assert undo['moves']==[]
        assert client.post('/api/analyze',json=dict(checkpoint=ck,moves=[0,0])).status_code==400
        assert client.post('/api/analyze',json=dict(checkpoint=ck,moves=[True])).status_code==422
        assert client.post('/api/analyze',json=dict(checkpoint=ck,moves=[],size=15)).status_code==400
        assert client.get('/api/unknown').status_code==404
        assert client.post('/api/games',json=dict(checkpoint=ck,human=0)).status_code==400


def test_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(simulations=0)
    with pytest.raises(ValueError):
        TrainConfig(parallel_games=100)
    with pytest.raises(ValueError):
        TrainConfig(mode='mixed')
