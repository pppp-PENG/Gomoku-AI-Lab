"""Bounded real checkpoint continuation, inference and performance evidence."""
from dataclasses import asdict
import cProfile
import io
import json
from pathlib import Path
import pstats
import platform
import time
import torch
from gomoku.game import Game
from gomoku.models import ModelConfig,build_model,estimated_flops
from gomoku.search import Evaluator,analyze,MCTS
from gomoku.storage import Store
from gomoku.training import TrainConfig,load_checkpoint,train
from gomoku.evaluation import benchmark


def main():
    torch.set_num_threads(2)
    store=Store()
    report=dict(python=platform.python_version(),torch=torch.__version__,
                cuda=torch.cuda.is_available(),device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                verification='Smoke evidence only; no architecture-strength conclusion',checkpoints=[],benchmarks=[])
    for arch in ('cnn','vit'):
        checkpoints=[c for c in store.checkpoints() if c['model_config']['architecture']==arch]
        if not checkpoints:
            raise RuntimeError(f'Run smoke-{arch}.json first')
        source=next((c for c in checkpoints if c['games']==4),checkpoints[0])
        before,_=load_checkpoint(source['path'])
        config=TrainConfig(**source['config'])
        config.games=source['games']+2
        eid=train(config,store,resume=source['id'])
        checkpoint=next(c for c in store.checkpoints() if c['experiment']==eid)
        after,ck=load_checkpoint(checkpoint['path'])
        changed=sum(not torch.equal(a,b) for a,b in zip(before.state_dict().values(),after.state_dict().values()))
        assert changed>0 and checkpoint['updates']>source['updates']
        game=Game(moves=[40,0,39,8])
        start=time.perf_counter()
        result=analyze(game,Evaluator(after),16)
        seconds=time.perf_counter()-start
        assert len(result['top5'])==5
        assert all(game.legal_mask()[c['action']] for c in result['candidates'])
        assert all(abs(c['win']+c['draw']+c['loss']-1)<1e-5 for c in result['candidates'])
        report['checkpoints'].append(dict(architecture=arch,source=source['id'],checkpoint=checkpoint['id'],
                                          experiment=eid,games=checkpoint['games'],updates=checkpoint['updates'],
                                          changed_tensors=changed,replay=len(ck['replay']),analysis_ms=seconds*1000,
                                          top5=result['top5'],metrics=store.metrics(eid)))
        profile=cProfile.Profile()
        profile.enable()
        MCTS(Evaluator(after),32).run(game)
        profile.disable()
        stream=io.StringIO()
        pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(15)
        report['checkpoints'][-1]['profile']=stream.getvalue()
    for arch in ('cnn','vit'):
        config=ModelConfig(architecture=arch)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            model=build_model(config).to(device)
            result=benchmark(model,config.size,device)
            result.update(architecture=arch,model_config=asdict(config),
                          parameters=sum(p.numel() for p in model.parameters()),estimated_flops=estimated_flops(config))
            report['benchmarks'].append(result)
    report['evaluations']=store.evaluations()
    directory=Path('docs');directory.mkdir(exist_ok=True)
    (directory/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checkpoints','evaluations')},indent=2))


if __name__=='__main__':
    main()
