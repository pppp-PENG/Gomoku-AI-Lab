"""Sequential, bounded, independent multi-seed CNN/ViT experiment runner."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from gomoku.models import ModelConfig
from gomoku.storage import Store
from gomoku.training import TrainConfig,train,load_checkpoint
from gomoku.search import Evaluator
from gomoku.evaluation import evaluate,tactics,benchmark


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--games',type=int,default=200)
    p.add_argument('--simulations',type=int,default=32)
    p.add_argument('--pairs',type=int,default=4)
    p.add_argument('--hours',type=float,default=2.)
    p.add_argument('--output',default='data/research-summary.json')
    args=p.parse_args()
    store=Store()
    rows=[]
    for seed in args.seeds:
        for architecture in ('cnn','vit'):
            config=TrainConfig(model=asdict(ModelConfig(architecture=architecture)),seed=seed,
                               games=args.games,simulations=args.simulations,max_hours=args.hours,mode='independent')
            eid=train(config,store)
            ck=next(c for c in store.checkpoints() if c['experiment']==eid)
            model,_=load_checkpoint(ck['path'])
            evaluator=Evaluator(model)
            for opponent in ('random','heuristic'):
                # Same evaluation openings across architectures and training seeds.
                r=evaluate(evaluator,opponent,9,args.pairs,args.simulations,2026)
                r.update(experiment=eid,checkpoint=ck['id'],architecture=architecture,training_seed=seed,
                         games_trained=ck['games'],training_status=store.experiment(eid)['status'],
                         training_seconds=store.metrics(eid)[-1].get('seconds'),parameters=ck['parameters'],
                         tactics=tactics(evaluator,9,args.simulations),benchmark=benchmark(model,9))
                store.save_evaluation(r)
                rows.append(r)
            # Save incremental results so an interrupted suite retains completed runs.
            output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
            aggregate=[]
            for arch in ('cnn','vit'):
                for opponent in ('random','heuristic'):
                    group=[r for r in rows if r['architecture']==arch and r['opponent']==opponent]
                    if group:
                        aggregate.append(dict(architecture=arch,opponent=opponent,n=len(group),
                                              mean=float(np.mean([r['score'] for r in group])),
                                              std=float(np.std([r['score'] for r in group],ddof=1)) if len(group)>1 else None))
            output.write_text(json.dumps(dict(rows=rows,aggregate=aggregate),indent=2),encoding='utf-8')
    print(args.output)


if __name__=='__main__':
    main()
