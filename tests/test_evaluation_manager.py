from dataclasses import asdict
import time
import pytest
from gomoku.game import Game
from gomoku.evaluation import heuristic,immediate,tactical_positions,evaluate,tactics
from gomoku.manager import TrainingManager
from gomoku.models import ModelConfig,build_model
from gomoku.search import Evaluator
from gomoku.storage import Store
from gomoku.training import TrainConfig


def test_verified_tactics_and_heuristic():
    tasks=tactical_positions()
    assert {name for name,_,_ in tasks}=={'win','defend'}
    for name,game,answers in tasks:
        assert heuristic(game) in answers
        assert answers==immediate(game,game.to_play if name=='win' else -game.to_play)


def test_matched_evaluation():
    c=ModelConfig(size=5,channels=8,blocks=1,dim=8,layers=1,heads=2)
    report=evaluate(Evaluator(build_model(c)),'random',5,2,0,42)
    assert report['games']==4
    assert report['wins']+report['draws']+report['losses']==4
    assert 0<=report['score']<=1 and 0<=report['brier']<=2
    assert sum(b['count'] for b in report['calibration_bins'])>0


def test_manager_immediate_pause_and_resume(tmp_path):
    store=Store(tmp_path)
    manager=TrainingManager(store)
    config=TrainConfig(model=asdict(ModelConfig(channels=8,blocks=1)),games=1000,
                       simulations=2,parallel_games=1,device='cpu',threads=1,eval_every=0)
    eid=manager.start(config)
    with pytest.raises(ValueError):
        manager.start(config)
    manager.control(eid,'pause')
    deadline=time.monotonic()+30
    while manager.process.poll() is None and time.monotonic()<deadline:
        time.sleep(.1)
    if manager.process.poll() is None:
        manager.process.terminate()
        pytest.fail('pause did not finish in 30 seconds')
    assert store.experiment(eid)['status']=='paused'
    checkpoint=next(c for c in store.checkpoints() if c['experiment']==eid)
    assert checkpoint['games']==0
    config.games=1
    resumed=manager.start(config,checkpoint['id'])
    deadline=time.monotonic()+30
    while manager.process.poll() is None and time.monotonic()<deadline:
        time.sleep(.1)
    if manager.process.poll() is None:
        manager.process.terminate()
        pytest.fail('resume did not finish in 30 seconds')
    assert store.experiment(resumed)['status']=='completed'
    assert any(c['games']==1 and c['experiment']==resumed for c in store.checkpoints())
