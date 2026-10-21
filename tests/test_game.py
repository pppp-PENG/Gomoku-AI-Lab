import numpy as np
import pytest
from gomoku.game import Game, augment, flip_wdl


@pytest.mark.parametrize("stones", [[0,1,2,3,4], [0,9,18,27,36], [0,10,20,30,40], [4,12,20,28,36]])
def test_five_directions(stones):
    game = Game()
    filler = [80,78,76,74]
    for i, action in enumerate(stones):
        game.play(action)
        if i < 4:
            game.play(filler[i])
    assert game.winner == 1 and game.to_play == -1
    assert game.terminal_wdl().tolist() == [0,0,1]
    assert not game.legal_mask().any()
    with pytest.raises(ValueError):
        game.play(60)


def test_illegal_undo_replay_clone():
    game = Game(moves=[0, 80, 1])
    for bad in (0, -1, 81, 1.2, True):
        with pytest.raises(ValueError):
            game.play(bad)
    copied = game.clone()
    copied.play(70)
    assert len(game.moves) == 3
    game.undo(2)
    assert game.moves == [0] and game.to_play == -1
    assert np.array_equal(Game(moves=copied.moves).board, copied.board)
    game.undo(99)
    assert game.moves == [] and game.winner is None


def test_undo_terminal():
    game=Game(moves=[0,80,1,78,2,76,3,74,4])
    game.undo()
    assert not game.terminal and game.to_play == 1


def test_draw_and_encoding():
    rng = np.random.default_rng(42)
    found = False
    for _ in range(100):
        game = Game(5)
        for action in rng.permutation(25):
            if not game.terminal:
                game.play(int(action))
        if game.winner == 0:
            assert game.terminal_wdl().tolist() == [0,1,0]
            found = True
            break
    assert found
    game = Game(moves=[0])
    x=game.encode()
    assert x[0].sum() == 0 and x[1,0,0] == 1 and x[2].sum() == 81
    assert flip_wdl([.1,.2,.7]).tolist() == [.7,.2,.1]


def test_d4_alignment_and_group():
    state = Game(moves=[0,12,4]).encode()
    policy = np.arange(81, dtype=np.float32)
    outputs = []
    for symmetry in range(8):
        x,p = augment(state, policy, symmetry)
        for i in (0,1):
            assert np.array_equal(x[i].ravel(), state[i].ravel()[p.astype(int)])
        assert sorted(p.tolist()) == policy.tolist()
        outputs.append(p.tobytes())
    assert len(set(outputs)) == 8
    x,p=state,policy
    for _ in range(4):
        x,p=augment(x,p,1)
    assert np.array_equal(x,state) and np.array_equal(p,policy)


def test_overline_and_no_wrap():
    game=Game()
    game.board[0,5:9]=1
    game.board[1,0:3]=1
    game.play(9+3)
    assert not game.terminal
    game=Game()
    game.board[0,[0,1,2,4,5]]=1
    game.play(3)
    assert game.winner == 1
