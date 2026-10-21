from dataclasses import asdict
import numpy as np
import pytest
from fastapi.testclient import TestClient
from gomoku.api import create_app
from gomoku.models import ModelConfig
from gomoku.sft import SFTConfig, selected_games, split_selected_games, MoveDataset, train_sft
from gomoku.storage import Store
from gomoku_data.lab import selected_for_sft
from tests_data.test_pipeline import synthetic_corpus


def test_external_selection_and_server_size_permission_gates(tmp_path,monkeypatch):
    synthetic_corpus(tmp_path)
    monkeypatch.setenv("GOMOKU_EXTERNAL_DATA",str(tmp_path/"normalized"))
    cfg=SFTConfig(model=asdict(ModelConfig(size=15,channels=8,blocks=1)),external_sources=["karesis"])
    store=Store(tmp_path/"lab")
    games=selected_games(cfg,store)
    assert games and all(g["external_split"]!='test' for g in games)
    train,val=split_selected_games(games,.0,np.random.default_rng(1))
    assert all(games[i]['external_split']=='train' for i in train)
    assert all(games[i]['external_split']=='validation' for i in val)
    assert len(MoveDataset(games,train))>0
    with pytest.raises(ValueError):selected_games(SFTConfig(external_sources=["karesis"]),store)
    for source in ("renjunet","gomocup","unknown"):
        with pytest.raises(ValueError):SFTConfig(external_sources=[source])
    with TestClient(create_app(store)) as client:
        catalog=client.get('/api/sft/sources').json()
        assert next(r for r in catalog if r['source']=='karesis')['status']=='ready'
        assert next(r for r in catalog if r['source']=='renjunet')['status']=='blocked'


def test_actual_external_sft_preserves_split_and_provenance(tmp_path,monkeypatch):
    synthetic_corpus(tmp_path)
    monkeypatch.setenv("GOMOKU_EXTERNAL_DATA",str(tmp_path/"normalized"))
    store=Store(tmp_path/"lab")
    config=SFTConfig(model=asdict(ModelConfig(size=15,channels=8,blocks=1)),external_sources=["karesis"],
                     epochs=1,batch_size=16,threads=1,device='cpu',checkpoint_every=1)
    eid=train_sft(config,store)
    assert store.experiment(eid)['status']=='completed'
    from gomoku.training import load_checkpoint
    ck=store.checkpoints()[0];_,saved=load_checkpoint(ck['path'])
    state=saved['sft_state']
    assert all(g.get('provenance') and g['external_split']!='test' for g in state['games'])
    assert all(state['games'][i]['external_split']=='train' for i in state['train_indices'])
    assert all(state['games'][i]['external_split']=='validation' for i in state['val_indices'])


def test_collected_duplicate_of_external_test_is_excluded(tmp_path,monkeypatch):
    synthetic_corpus(tmp_path);monkeypatch.setenv('GOMOKU_EXTERNAL_DATA',str(tmp_path/'normalized'))
    from gomoku_data.pipeline import load_games
    from gomoku.demonstrations import validate_demonstration
    store=Store(tmp_path/'lab')
    heldout=next(load_games(tmp_path/'normalized',['karesis'],'test'))
    row=store.save_demonstration(validate_demonstration(15,[m['action'] for m in heldout['moves']]))
    cfg=SFTConfig(model=asdict(ModelConfig(size=15)),dataset_ids=[row['id']],external_sources=['karesis'])
    games=selected_games(cfg,store)
    assert row['id'] not in {g['id'] for g in games}
