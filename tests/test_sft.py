from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from gomoku.api import create_app
from gomoku.demonstrations import validate_demonstration
from gomoku.game import Game, augment
from gomoku.manager import TrainingManager
from gomoku.models import ModelConfig, build_model
from gomoku.sft import SFTConfig, MoveDataset, prepare_sft, split_games, train_sft
from gomoku.storage import Store
from gomoku.training import TrainConfig, train, load_checkpoint, seed_all

BLACK = [0,5,1,6,2,7,3,8,4]
WHITE = [20,0,21,1,10,2,11,3,15,4]


def collected(store):
    return [store.save_demonstration(validate_demonstration(5, moves, name))
            for name, moves in (("black", BLACK), ("white", WHITE))]


def tiny(ids, arch="cnn", **kw):
    args = dict(model=asdict(ModelConfig(architecture=arch,size=5,channels=8,blocks=1,dim=8,layers=1,heads=2)),
                dataset_ids=ids, epochs=2, batch_size=4, checkpoint_every=1, device="cpu", threads=1, seed=17)
    args.update(kw)
    return SFTConfig(**args)


def test_collection_requires_explicit_terminal_approval_and_validates(tmp_path):
    store=Store(tmp_path)
    with TestClient(create_app(store)) as client:
        preview=client.post('/api/demonstrations/position',json=dict(size=5,moves=BLACK)).json()
        assert preview['winner']==1 and preview['terminal']
        assert client.get('/api/demonstrations').json()==[]  # preview does not save
        assert client.post('/api/demonstrations',json=dict(size=5,moves=BLACK[:-1])).status_code==400
        for moves in ([0,0], [-1], [25], BLACK+[9]):
            assert client.post('/api/demonstrations',json=dict(size=5,moves=moves)).status_code==400
        assert client.post('/api/demonstrations',json=dict(size=5,moves=[True])).status_code==422
        assert client.post('/api/demonstrations',json=dict(size=5,moves=BLACK,rules='other')).status_code==400
        assert client.post('/api/demonstrations',json=dict(size=5,moves=BLACK,winner=-1)).status_code==422
        a=client.post('/api/demonstrations',json=dict(size=5,moves=BLACK,name='good game')).json()
        b=client.post('/api/demonstrations',json=dict(size=5,moves=BLACK)).json()
        assert a['id']==b['id'] and len(store.demonstrations())==1
        assert Store(tmp_path).demonstration(a['id'])['moves']==BLACK
        exported=client.get('/api/demonstrations/export').json()
        assert exported['schema']==1 and exported['demonstrations'][0]['winner']==1
        assert store.games()==[]  # collection is independent of AI play history


@pytest.mark.parametrize("moves,winner",[(BLACK,1),(WHITE,-1)])
def test_every_black_and_white_move_with_correct_wdl_and_d4(moves,winner):
    g=dict(size=5,moves=moves,winner=winner)
    dataset=MoveDataset([g],[0])
    assert len(dataset)==len(moves)
    for ply,action in enumerate(moves):
        x,a,w=dataset.sample(ply)
        assert a==action and x[:2].sum()==ply and x[:2,action//5,action%5].sum()==0
        player=1 if ply%2==0 else -1
        assert w.tolist()==[int(winner==player),0,int(winner==-player)]
        for symmetry in range(8):
            xx,aa,ww=dataset.sample(ply,symmetry)
            p=np.zeros(25);p[action]=1
            expected,pp=augment(x,p,symmetry)
            assert np.array_equal(xx,expected) and aa==pp.argmax() and np.array_equal(ww,w)
    # Losing side decisions are retained and remain positive imitation labels.
    assert sum(dataset.sample(i)[2][2] for i in range(len(dataset)))>=4


def test_draw_wdl():
    rng=np.random.default_rng(42)
    for _ in range(100):
        g=Game(5)
        for a in rng.permutation(25):
            if not g.terminal:g.play(int(a))
        if g.winner==0:break
    dataset=MoveDataset([dict(size=5,moves=g.moves,winner=0)],[0])
    assert len(dataset)==25
    assert all(dataset.sample(i)[2].tolist()==[0,1,0] for i in range(25))


def test_split_is_by_whole_game_not_augmented_position():
    games=[dict(size=5,moves=BLACK,winner=1) for _ in range(10)]
    train_ids,val_ids=split_games(games,.2,np.random.default_rng(13))
    assert len(train_ids)==8 and len(val_ids)==2 and not set(train_ids)&set(val_ids)
    assert sorted(train_ids+val_ids)==list(range(10))
    assert split_games(games[:1],.2,np.random.default_rng(1))==([0],[])
    assert split_games(games,0,np.random.default_rng(1))[1]==[]


@pytest.mark.parametrize('arch',['cnn','vit'])
def test_actual_sft_updates_both_heads_and_reloads_for_analysis_and_rl(tmp_path,arch):
    store=Store(tmp_path)
    cfg=tiny([g['id'] for g in collected(store)],arch)
    seed_all(cfg.seed,1)
    initial=build_model(ModelConfig(**cfg.model)).state_dict()
    eid=train_sft(cfg,store)
    assert store.experiment(eid)['status']=='completed'
    ck=next(c for c in store.checkpoints() if c['experiment']==eid)
    net,saved=load_checkpoint(ck['path'])
    assert saved['training_kind']=='sft' and saved['sft']['completed_epochs']==2
    assert saved['sft']['train_samples']+saved['sft']['validation_samples']==19
    assert saved['replay']==[] and saved['updates']>0
    assert all(any(not torch.equal(v,initial[k]) for k,v in net.state_dict().items() if k.startswith(head))
               for head in ('policy','value'))
    assert set(saved['sft_state']['train_indices']).isdisjoint(saved['sft_state']['val_indices'])
    metrics=store.metrics(eid)
    assert metrics[-1]['val_loss']>0 and 0<=metrics[-1]['val_accuracy']<=1
    from gomoku.search import Evaluator,analyze
    a=analyze(Game(5,moves=[0]),Evaluator(net),2)
    assert len(a['top5'])==5 and all(c['action']!=0 for c in a['top5'])
    # A supervised artifact is a normal network initializer for self-play RL.
    rl=TrainConfig(initial_checkpoint=ck['id'],games=1,simulations=1,parallel_games=1,
                   batch_size=4,updates_per_game=1,eval_every=0,device='cpu',threads=1)
    resumed=train(rl,store)
    r=next(c for c in store.checkpoints() if c['experiment']==resumed)
    assert r['model_config']==cfg.model and r['updates']==1 and r['training_kind']=='rl'


@pytest.mark.parametrize('arch',['cnn','vit'])
def test_resume_exact_with_immutable_dataset_even_without_originals(tmp_path,arch):
    store=Store(tmp_path)
    cfg=tiny([g['id'] for g in collected(store)],arch)
    eid=train_sft(cfg,store)
    checkpoints=[c for c in store.checkpoints() if c['experiment']==eid]
    first=next(c for c in checkpoints if c['sft']['completed_epochs']==1)
    final=next(c for c in checkpoints if c['sft']['completed_epochs']==2)
    expected,_=load_checkpoint(final['path'])
    with store.db() as db:db.execute('DELETE FROM demonstrations')
    resumed=train_sft(cfg,store,resume=first['id'])
    actual,saved=load_checkpoint(next(c for c in store.checkpoints() if c['experiment']==resumed)['path'])
    assert saved['sft']['completed_epochs']==2
    assert all(torch.equal(v,actual.state_dict()[k]) for k,v in expected.state_dict().items())


def test_pause_mid_epoch_resume_and_weights_only_initialization(tmp_path,monkeypatch):
    import gomoku.sft as sft
    store=Store(tmp_path)
    cfg=tiny([g['id'] for g in collected(store)],validation_fraction=0)
    eid=train_sft(cfg,store)
    final=next(c for c in store.checkpoints() if c['experiment']==eid)
    expected,_=load_checkpoint(final['path'])
    original=sft.supervised_loss
    paused_once=False
    def request_pause(*args):
        nonlocal paused_once
        result=original(*args)
        if not paused_once:
            paused_once=True
            current=store.experiments()[0]['id']
            store.status(current,'pausing')
        return result
    monkeypatch.setattr(sft,'supervised_loss',request_pause)
    paused=train_sft(cfg,store)
    assert store.experiment(paused)['status']=='paused'
    ck=next(c for c in store.checkpoints() if c['experiment']==paused)
    _,snapshot=load_checkpoint(ck['path'])
    assert snapshot['sft_state']['offset']==4 and snapshot['sft']['completed_epochs']==0
    continued=train_sft(cfg,store,resume=ck['id'])
    net,_=load_checkpoint(next(c for c in store.checkpoints() if c['experiment']==continued)['path'])
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in expected.state_dict().items())
    # New SFT from existing ck starts Adam and counters anew, preserving source bytes.
    before=hashlib.sha256(Path(final['path']).read_bytes()).digest()
    seen=[]
    def inspect(model,batch,device,weight):
        if not seen:
            assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in expected.state_dict().items())
        seen.append(1)
        return original(model,batch,device,weight)
    monkeypatch.setattr(sft,'supervised_loss',inspect)
    warm=train_sft(replace(cfg,initial_checkpoint=final['id'],epochs=1),store)
    _,saved=load_checkpoint(next(c for c in store.checkpoints() if c['experiment']==warm)['path'])
    assert saved['sft']['completed_epochs']==1 and saved['updates']==5
    assert {float(state['step']) for state in saved['optimizer']['state'].values()}=={5.}
    assert saved['initialization']['A']['checkpoint']==final['id']
    assert hashlib.sha256(Path(final['path']).read_bytes()).digest()==before


def test_sft_worker_api_and_shared_training_lock(tmp_path):
    store=Store(tmp_path)
    cfg=tiny([g['id'] for g in collected(store)])
    with TestClient(create_app(store)) as client:
        request=client.post('/api/sft/experiments',json=dict(config=asdict(cfg)))
        assert request.status_code==200,request.text
        eid=request.json()['id']
        # Existing manager serializes both types, rather than starting a second GPU task.
        assert client.post('/api/sft/experiments',json=dict(config=asdict(cfg))).status_code==400
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            result=next(e for e in client.get('/api/experiments').json() if e['id']==eid)
            if result['status'] not in ('created','running'):break
            time.sleep(.1)
        assert result['status']=='completed',result
        ck=next(c for c in store.checkpoints() if c['experiment']==eid)
        assert client.post('/api/experiments',json=dict(config=asdict(TrainConfig()),resume=ck['id'])).status_code==400
        assert client.post('/api/analyze',json=dict(checkpoint=ck['id'],size=5,moves=[],simulations=0)).status_code==200


@pytest.mark.parametrize('change',[dict(dataset_ids=[]),dict(epochs=True),dict(batch_size=0),
                                    dict(validation_fraction=float('nan')),dict(value_weight=-1),dict(augment=1)])
def test_sft_configuration_validation(change):
    with pytest.raises(ValueError):tiny(['id'],**change)


def test_sft_preflight_missing_and_board_mismatch(tmp_path):
    store=Store(tmp_path)
    ids=[g['id'] for g in collected(store)]
    with pytest.raises(KeyError):prepare_sft(tiny(['missing']),store)
    with pytest.raises(ValueError):prepare_sft(replace(tiny(ids),model=asdict(ModelConfig())),store)
    assert store.experiments()==[]


def test_sft_initializes_from_paired_rl_B_own_architecture(tmp_path,monkeypatch):
    import gomoku.sft as sft
    store=Store(tmp_path)
    ids=[g['id'] for g in collected(store)]
    cfg=tiny(ids)
    b=asdict(ModelConfig(architecture='vit',size=5,channels=8,blocks=1,dim=12,layers=1,heads=2))
    eid=train(TrainConfig(model=cfg.model,opponent_model=b,mode='paired',games=1,simulations=1,
                          parallel_games=1,batch_size=4,updates_per_game=1,eval_every=0,
                          device='cpu',threads=1),store)
    source=next(c for c in store.checkpoints() if c['experiment']==eid and c['agent']=='B')
    expected,_=load_checkpoint(source['path'])
    original=sft.supervised_loss
    seen=[]
    def inspect(model,*args):
        if not seen:
            assert model.config==b
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.state_dict().items())
        seen.append(True)
        return original(model,*args)
    monkeypatch.setattr(sft,'supervised_loss',inspect)
    warm=train_sft(replace(cfg,initial_checkpoint=source['id'],epochs=1),store)
    ck=next(c for c in store.checkpoints() if c['experiment']==warm)
    assert ck['model_config']==b and ck['initialization']['A']['source_agent']=='B'


def test_sft_stop_and_failure_are_persisted(tmp_path,monkeypatch):
    import gomoku.sft as sft
    store=Store(tmp_path)
    ids=[g['id'] for g in collected(store)]
    cfg=tiny(ids)
    original=sft.supervised_loss
    def stop_after_update(*args):
        result=original(*args)
        store.status(store.experiments()[0]['id'],'stopping')
        return result
    monkeypatch.setattr(sft,'supervised_loss',stop_after_update)
    eid=train_sft(cfg,store)
    assert store.experiment(eid)['status']=='stopped'
    assert store.metrics(eid)[-1]['updates']==1
    assert next(c for c in store.checkpoints() if c['experiment']==eid)['updates']==1
    def fail(*args):raise RuntimeError('test failure')
    monkeypatch.setattr(sft,'supervised_loss',fail)
    with pytest.raises(RuntimeError,match='test failure'):train_sft(cfg,store)
    failed=store.experiments()[0]
    assert failed['status']=='failed' and 'test failure' in store.metrics(failed['id'])[-1]['error']


def test_sft_disk_budget_includes_just_written_checkpoint(tmp_path,monkeypatch):
    import gomoku.sft as sft
    store=Store(tmp_path)
    cfg=tiny([g['id'] for g in collected(store)],epochs=3,disk_budget_gb=.01)
    original=sft.save_checkpoint
    def larger_artifact(*args,**kwargs):
        cid=original(*args,**kwargs)
        # Simulate a large artifact without allocating a large network in the test.
        with (store.root/'checkpoints'/f'{cid}.pt').open('ab') as f:f.truncate(12*1024**2)
        return cid
    monkeypatch.setattr(sft,'save_checkpoint',larger_artifact)
    eid=train_sft(cfg,store)
    assert store.experiment(eid)['status']=='paused'
    artifacts=[c for c in store.checkpoints() if c['experiment']==eid]
    assert len(artifacts)==1 and artifacts[0]['sft']['completed_epochs']==1
    assert any('预算' in m.get('warning','') for m in store.metrics(eid))


def test_sft_time_budget_saves_recoverable_progress(tmp_path,monkeypatch):
    import gomoku.sft as sft
    store=Store(tmp_path)
    cfg=tiny([g['id'] for g in collected(store)],max_hours=.000001)
    original=sft.supervised_loss
    def slow_batch(*args):
        time.sleep(.01)
        return original(*args)
    monkeypatch.setattr(sft,'supervised_loss',slow_batch)
    eid=train_sft(cfg,store)
    assert store.experiment(eid)['status']=='paused'
    ck=next(c for c in store.checkpoints() if c['experiment']==eid)
    _,saved=load_checkpoint(ck['path'])
    assert saved['updates']<=1 and saved['sft']['completed_epochs']==0


@pytest.mark.parametrize('size',[9,13])
@pytest.mark.parametrize('architecture',['cnn','vit'])
def test_size_categories_load_and_train_matching_games_only(tmp_path,size,architecture):
    store=Store(tmp_path)
    with TestClient(create_app(store)) as client:
        by_size={}
        for n in (9,13):
            moves=[0,n,1,n+1,2,n+2,3,n+3,4]
            response=client.post('/api/demonstrations',json=dict(size=n,moves=moves,name=f'{n}x{n}'))
            assert response.status_code==200,response.text
            by_size[n]=response.json()['id']
        rows=client.get('/api/demonstrations',params=dict(size=size)).json()
        assert [row['id'] for row in rows]==[by_size[size]]
        assert all(row['size']==size for row in rows)
        assert len(client.get('/api/demonstrations').json())==2
        assert len(Store(tmp_path).demonstrations(size))==1
        assert client.get('/api/demonstrations',params=dict(size=20)).status_code==422
        cfg=tiny([by_size[size]],architecture,model=asdict(ModelConfig(
            architecture=architecture,size=size,channels=8,blocks=1,dim=8,layers=1,heads=2)),epochs=1)
        eid=train_sft(cfg,store)
        ck=next(c for c in store.checkpoints() if c['experiment']==eid)
        net,saved=load_checkpoint(ck['path'])
        assert saved['sft']['train_samples']==9 and saved['sft']['validation_samples']==0
        assert {g['size'] for g in saved['sft_state']['games']}=={size}
        assert [g['id'] for g in saved['sft_state']['games']]==[by_size[size]]
        policy,wdl=net(torch.zeros(1,3,size,size))
        assert policy.shape==(1,size**2) and wdl.shape==(1,3) and saved['updates']>0
        before=len(store.experiments())
        mixed=replace(cfg,dataset_ids=list(by_size.values()))
        assert client.post('/api/sft/experiments',json=dict(config=asdict(mixed))).status_code==400
        assert len(store.experiments())==before
        # A source weight fixes the true size even if the request carries a placeholder.
        resolved=prepare_sft(replace(cfg,model=asdict(ModelConfig()),initial_checkpoint=ck['id']),store)
        assert resolved.model['size']==size
