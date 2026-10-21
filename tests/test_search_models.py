import numpy as np
import pytest
import torch
from gomoku.game import Game
from gomoku.models import ModelConfig, build_model, masked_policy
from gomoku.search import Node, MCTS, Evaluator, backup, select, analyze, search_many


class Uniform:
    def __call__(self,g):
        p=g.legal_mask().astype(float)
        return p/p.sum(), np.array([.2,.3,.5])
    def batch(self,games):
        return [self(g) for g in games]


def test_backup_and_puct():
    parent, child = Node(), Node()
    backup([parent,child], [.8,.1,.1])
    assert child.q == pytest.approx(-.7)
    assert parent.q == pytest.approx(.7)
    assert child.total.tolist() == pytest.approx([.1,.1,.8])
    root=Node(children={0:Node(prior=.1),1:Node(prior=.9)})
    assert select(root,1.5)==1


def test_immediate_win_and_batched_search():
    game=Game(moves=[0,80,1,78,2,76,3,74])
    p,root,_=MCTS(Uniform(),128).run(game)
    assert np.argmax(p)==4 and root.children[4].q==1.
    p2=search_many([game,game],[Uniform(),Uniform()],128,np.random.default_rng(1),False)
    assert all(np.argmax(p)==4 for p in p2)
    assert sum(c.visits for c in root.children.values())==128


def test_wdl_candidate_conversion_and_top5():
    game=Game(moves=[0,80,1,78,2,76,3,74])
    result=analyze(game,Uniform(),8)
    assert result['top5'][0]['action']==4
    assert result['top5'][0]['win']==1
    # Fake evaluator predicts opponent .2 win / .5 loss => current .5 win.
    ordinary=next(c for c in result['candidates'] if c['action']==5)
    assert ordinary['win']==.5 and ordinary['draw']==.3 and ordinary['loss']==.2
    assert all(game.legal_mask()[c['action']] for c in result['top5'])
    assert len(result['candidates'])==73
    assert all(a['win']>=b['win'] for a,b in zip(result['top5'],result['top5'][1:]))


def test_terminal_analysis_has_no_recommendations():
    result=analyze(Game(moves=[0,80,1,78,2,76,3,74,4]),Uniform(),16)
    assert result['top5']==[] and result['candidates']==[]
    assert result['root_wdl']==[0,0,1] and result['simulations']==16


@pytest.mark.parametrize('architecture',['cnn','vit'])
@pytest.mark.parametrize('size',[9,15])
def test_forward_backward_mask(architecture,size):
    c=ModelConfig(architecture=architecture,size=size,channels=8,blocks=1,dim=8,layers=1,heads=2)
    model=build_model(c)
    game=Game(size,moves=[0,8,4])
    x=torch.from_numpy(np.stack([game.encode()]*2))
    p,w=model(x)
    assert p.shape==(2,size**2) and w.shape==(2,3)
    legal=torch.from_numpy(np.stack([game.legal_mask()]*2))
    probs=masked_policy(p,legal)
    assert torch.all(probs[:,[0,8,4]]==0)
    assert torch.allclose(probs.sum(1),torch.ones(2))
    loss=-probs[:,1].log().mean()-w.log_softmax(-1)[:,0].mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
    result=analyze(game,Evaluator(model),2)
    assert len(result['top5'])==5
    with pytest.raises(ValueError):
        masked_policy(p,torch.zeros_like(legal))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('architecture',['cnn','vit'])
def test_cuda_forward_backward(architecture):
    model=build_model(ModelConfig(architecture=architecture,channels=8,blocks=1,dim=8,layers=1,heads=2)).cuda()
    p,w=model(torch.randn(2,3,9,9,device='cuda'))
    (p.square().mean()+w.square().mean()).backward()
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
