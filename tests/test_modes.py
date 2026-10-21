from dataclasses import asdict
import pytest
from gomoku.storage import Store
from gomoku.training import TrainConfig,train
from gomoku.training import load_checkpoint
import torch
from gomoku.models import ModelConfig


def config(architecture='cnn',**kwargs):
    defaults=dict(model=asdict(ModelConfig(architecture=architecture,channels=8,blocks=1,dim=8,layers=1,heads=2)),
                  games=2,simulations=2,parallel_games=1,batch_size=4,replay_size=128,
                  updates_per_game=1,checkpoint_every=100,eval_every=0,device='cpu',threads=1)
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def test_eval_interval_independent_of_save_interval(tmp_path):
    store=Store(tmp_path)
    eid=train(config(eval_every=1,eval_pairs=1),store)
    evaluations=[e for e in store.evaluations() if e['experiment']==eid]
    assert sorted(e['games_trained'] for e in evaluations)==[1,2]
    assert len([c for c in store.checkpoints() if c['experiment']==eid])==2


def test_history_pool_and_mixed(tmp_path):
    store=Store(tmp_path)
    eid=train(config(mode='history',checkpoint_every=1),store)
    opponent=next(c for c in store.checkpoints() if c['experiment']==eid)['id']
    mixed=train(config('vit',games=1,mode='mixed',opponent=opponent),store)
    assert store.experiment(mixed)['status']=='completed'
    assert store.metrics(mixed)[-1]['updates']==1
    with pytest.raises(ValueError,match='cumulative'):
        train(config(games=2,mode='history',checkpoint_every=1),store,resume=opponent)


def test_history_resume_cannot_see_future_checkpoints(tmp_path):
    store=Store(tmp_path)
    cfg=config(mode='history',checkpoint_every=1,games=3)
    eid=train(cfg,store)
    checkpoints=[c for c in store.checkpoints() if c['experiment']==eid]
    first=next(c for c in checkpoints if c['games']==1)
    final=next(c for c in checkpoints if c['games']==3)
    resumed=train(cfg,store,resume=first['id'])
    latest=next(c for c in store.checkpoints() if c['experiment']==resumed)
    a,_=load_checkpoint(final['path'])
    b,_=load_checkpoint(latest['path'])
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
