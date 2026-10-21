"""Bounded real SFT -> checkpoint -> inference -> RL validation, isolated from user data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid

import torch

from gomoku.demonstrations import validate_demonstration
from gomoku.game import Game
from gomoku.models import ModelConfig, build_model
from gomoku.search import Evaluator, analyze
from gomoku.sft import SFTConfig, train_sft
from gomoku.storage import Store
from gomoku.training import TrainConfig, load_checkpoint, seed_all, train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    parser.add_argument('--data')
    parser.add_argument('--output',default='docs/sft-validation.json')
    args=parser.parse_args()
    root=Path(args.data or f'.test-runs/sft-smoke-{uuid.uuid4().hex}').resolve()
    store=Store(root)
    games=[store.save_demonstration(validate_demonstration(9,moves,name)) for name,moves in
           [('验收黑胜',[0,9,1,10,2,11,3,12,4]),('验收白胜',[72,0,73,1,54,2,55,3,63,4])]]
    report=dict(data=str(root),torch=torch.__version__,cuda=torch.cuda.is_available(),models=[])
    for arch in ('cnn','vit'):
        config=SFTConfig(model=asdict(ModelConfig(architecture=arch,channels=16,blocks=1,dim=16,layers=1,heads=2)),
                         dataset_ids=[g['id'] for g in games],epochs=12,batch_size=8,validation_fraction=0,
                         learning_rate=.003,checkpoint_every=6,device=args.device,threads=2,max_hours=.03)
        seed_all(config.seed,config.threads)
        initial=build_model(ModelConfig(**config.model)).state_dict()
        eid=train_sft(config,store)
        ck=next(c for c in store.checkpoints() if c['experiment']==eid)
        model,saved=load_checkpoint(ck['path'])
        metrics=[m for m in store.metrics(eid) if 'loss' in m]
        changed=sum(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
        assert store.experiment(eid)['status']=='completed' and changed>0
        result=analyze(Game(moves=[40]),Evaluator(model),8)
        assert len(result['top5'])==5 and all(c['action']!=40 for c in result['top5'])
        first=next(c for c in store.checkpoints() if c['experiment']==eid and c['sft']['completed_epochs']==6)
        resumed=train_sft(config,store,resume=first['id'])
        net,_=load_checkpoint(next(c for c in store.checkpoints() if c['experiment']==resumed)['path'])
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in model.state_dict().items())
        source_hash=hashlib.sha256(Path(ck['path']).read_bytes()).hexdigest()
        warm=SFTConfig(**{**asdict(config),'initial_checkpoint':ck['id'],'epochs':2})
        warm_id=train_sft(warm,store)
        assert store.experiment(warm_id)['status']=='completed'
        assert hashlib.sha256(Path(ck['path']).read_bytes()).hexdigest()==source_hash
        rl=TrainConfig(initial_checkpoint=ck['id'],games=1,simulations=2,parallel_games=1,batch_size=8,
                       updates_per_game=1,eval_every=0,device=args.device,threads=2,max_hours=.03)
        rl_id=train(rl,store)
        assert store.experiment(rl_id)['status']=='completed'
        report['models'].append(dict(architecture=arch,experiment=eid,checkpoint=ck['id'],updates=saved['updates'],
                                     samples=saved['sft']['samples_seen'],changed_tensors=changed,
                                     first_epoch=metrics[0],last_epoch=metrics[-1],top5=result['top5'],
                                     resume_exact=True,warm_sft_experiment=warm_id,source_unchanged=True,
                                     rl_experiment=rl_id))
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=args.output,data=str(root),models=[dict(architecture=m['architecture'],
          updates=m['updates'],first_loss=m['first_epoch']['loss'],last_loss=m['last_epoch']['loss'])
          for m in report['models']]),ensure_ascii=False))


if __name__=='__main__':main()
