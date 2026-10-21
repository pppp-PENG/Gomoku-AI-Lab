"""Reproduce real-corpus selection, rebuild and one-epoch CPU SFT validation.

Run from repository root: .venv/Scripts/python scripts/external_data_smoke.py
Writes only to the --work directory; never to the main training database.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gomoku.sft import SFTConfig, selected_games, split_selected_games, train_sft
from gomoku.models import ModelConfig, build_model
from gomoku.storage import Store
from gomoku.training import seed_all, load_checkpoint
from gomoku_data.lab import catalog, corpus_path
from gomoku_data.pipeline import load_games, read_json, build, write_json, acquire


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,default=Path('work/external-data-smoke'))
    parser.add_argument('--report',type=Path,default=Path('datasets/gomoku/validation.json'))
    args=parser.parse_args()
    if args.work.exists() and any(args.work.iterdir()):
        raise ValueError('choose a new/empty --work directory')
    args.work.mkdir(parents=True,exist_ok=True)
    report={'catalog':catalog(),'combinations':{}}
    lock=read_json('datasets/gomoku/sources.lock.json');cfg=read_json('datasets/gomoku/config.json')
    acquire('data/external_gomoku',lock,['karesis','gomoku_party'],offline=True)
    rebuilt=build('data/external_gomoku',lock,cfg,args.work/'rebuilt')
    report['identical_rebuild']=rebuilt==read_json(corpus_path()/'manifest.json')
    assert report['identical_rebuild']
    for sources in (['karesis'],['gomoku_party'],['karesis','gomoku_party']):
        games=list(load_games(corpus_path(),sources));sets={s:{g['game_hash'] for g in games if g['split']==s} for s in ('train','validation','test')}
        assert not (sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['test']&sets['validation'])
        report['combinations']['+'.join(sources)]={'games':len(games),'splits':{s:len(ids) for s,ids in sets.items()}}
    config=SFTConfig(model=asdict(ModelConfig(size=15,channels=8,blocks=1)),external_sources=['karesis','gomoku_party'],epochs=1,batch_size=256,threads=2,device='cpu',checkpoint_every=1,max_hours=.1,augment=False)
    store=Store(args.work/'training')
    games=selected_games(config,store);ti,vi=split_selected_games(games,config.validation_fraction,np.random.default_rng(config.seed))
    report['training_selection']={'games':len(games),'train_games':len(ti),'validation_games':len(vi),'excluded_test':report['combinations']['karesis+gomoku_party']['splits']['test']}
    seed_all(config.seed,config.threads);initial=build_model(ModelConfig(**config.model)).state_dict()
    eid=train_sft(config,store);assert store.experiment(eid)['status']=='completed'
    ck=next(c for c in store.checkpoints() if c['experiment']==eid);net,saved=load_checkpoint(ck['path'])
    assert all(g['external_split']!='test' and g['provenance'] for g in saved['sft_state']['games'])
    changed={head:any(not torch.equal(v,initial[k]) for k,v in net.state_dict().items() if k.startswith(head)) for head in ('policy','value')}
    assert all(changed.values())
    report['training']={'experiment':eid,'status':'completed','sft':saved['sft'],'updates':saved['updates'],'changed_heads':changed,'checkpoint_rules':saved['rules'],'checkpoint_size':saved['model_config']['size']}
    report['metrics']=store.metrics(eid)
    write_json(args.report,report)
    print(json.dumps({k:v for k,v in report.items() if k!='metrics'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
