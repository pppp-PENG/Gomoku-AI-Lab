import math
import time
import numpy as np
import torch
from .game import Game, flip_wdl
from .search import MCTS, Evaluator


def immediate(game, player):
    actions = []
    for a in np.flatnonzero(game.legal_mask()):
        test = game.clone()
        test.to_play = player
        test.play(int(a))
        if test.winner == player:
            actions.append(int(a))
    return actions


def heuristic(game):
    wins = immediate(game, game.to_play)
    blocks = immediate(game, -game.to_play)
    if wins or blocks:
        return (wins or blocks)[0]
    scores = {}
    # Count unblocked line windows, emphasizing fours and open threes.
    n, k = game.size, game.connect
    for a in np.flatnonzero(game.legal_mask()):
        r, c = divmod(int(a), n)
        score = -.01*((r-(n-1)/2)**2 + (c-(n-1)/2)**2)
        for player, weight in ((game.to_play, 1.), (-game.to_play, .9)):
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                for offset in range(k):
                    cells = [(r+(j-offset)*dr, c+(j-offset)*dc) for j in range(k)]
                    if not all(0 <= rr < n and 0 <= cc < n for rr, cc in cells):
                        continue
                    values = [game.board[rr, cc] for rr, cc in cells]
                    if -player not in values:
                        count = values.count(player)
                        score += weight*(6**count)
        scores[int(a)] = score
    return max(scores, key=scores.get)


def action_for(game, agent, simulations, rng):
    if isinstance(agent, str):
        return int(rng.choice(np.flatnonzero(game.legal_mask()))) if agent == "random" else heuristic(game)
    if simulations:
        p, _, _ = MCTS(agent, simulations).run(game)
    else:
        p, _ = agent(game)
    return int(np.argmax(p))


def evaluate(agent, opponent="heuristic", size=9, pairs=4, simulations=16, seed=42):
    rng = np.random.default_rng(seed)
    outcomes, predictions, targets, lengths, latencies = [], [], [], [], []
    start = time.monotonic()
    for _ in range(pairs):
        opening = int(rng.integers(size**2))
        for learner in (1, -1):
            game = Game(size, moves=[opening])
            probe = []
            while not game.terminal:
                current = agent if game.to_play == learner else opponent
                if game.to_play == learner:
                    _, w = agent(game)
                    probe.append(w)
                begin = time.monotonic()
                game.play(action_for(game, current, simulations, rng))
                if current is agent:
                    latencies.append(time.monotonic()-begin)
            outcome = 1 if game.winner == learner else .5 if game.winner == 0 else 0
            outcomes.append(outcome)
            lengths.append(len(game.moves))
            for pred in probe:
                predictions.append(pred)
                targets.append([outcome == 1, outcome == .5, outcome == 0])
    score = float(np.mean(outcomes))
    clipped = float(np.clip(score, .01, .99))
    # Interval on per-pair means preserves paired opening/color grouping.
    pair_means = np.asarray(outcomes).reshape(-1, 2).mean(1)
    se = float(pair_means.std(ddof=1)/math.sqrt(pairs)) if pairs > 1 else None
    pred = np.asarray(predictions)
    truth = np.asarray(targets)
    bins = []
    for i in range(10):
        mask = (pred[:, 0] >= i/10) & (pred[:, 0] < (i+1)/10 + (1e-8 if i == 9 else 0))
        if mask.any():
            bins.append(dict(count=int(mask.sum()), predicted=float(pred[mask, 0].mean()),
                             observed=float(truth[mask, 0].mean())))
    return dict(opponent=opponent if isinstance(opponent, str) else "checkpoint", seed=seed,
                pairs=pairs, games=2*pairs, simulations=simulations, score=score,
                wins=outcomes.count(1), draws=outcomes.count(.5), losses=outcomes.count(0),
                score_se=se, elo_difference=400*math.log10(clipped/(1-clipped)),
                elo_censored=score in (0., 1.), average_length=float(np.mean(lengths)),
                seconds=time.monotonic()-start, latency_ms=float(np.mean(latencies)*1000),
                brier=float(np.square(pred-truth).sum(1).mean()), calibration_bins=bins,
                ece=sum(b["count"]*abs(b["predicted"]-b["observed"]) for b in bins)/len(predictions),
                calibration_note="Correlated positions; calibration is descriptive, not independent samples.")


def tactical_positions(size=9):
    """Generated and engine-verified one-move win / mandatory defense tasks."""
    tasks = []
    for row in range(min(size-1, 4)):
        moves = []
        for col in range(4):
            moves.extend([row*size+col, (size-1)*size+col])
        game = Game(size, moves=moves)
        winning = immediate(game, game.to_play)
        if winning:
            tasks.append(("win", game, winning))
        defense = Game(size, moves=moves[:-1])
        threats = immediate(defense, -defense.to_play)
        if threats and not immediate(defense, defense.to_play):
            tasks.append(("defend", defense, threats))
    return tasks


def tactics(evaluator, size=9, simulations=16):
    rows = []
    for name, game, answers in tactical_positions(size):
        a = action_for(game, evaluator, simulations, np.random.default_rng(0))
        rows.append(dict(kind=name, moves=game.moves, answers=answers, action=a, correct=a in answers))
    return dict(accuracy=float(np.mean([r["correct"] for r in rows])), count=len(rows), items=rows)


def benchmark(model, size, device="cpu", repeats=30):
    model.eval()
    x = torch.zeros(1, 3, size, size, device=device)
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
    return dict(device=device, batch=1, repeats=repeats, median_ms=float(np.median(samples)),
                p95_ms=float(np.percentile(samples, 95)), positions_per_second=1000/float(np.mean(samples)),
                gpu_peak_mb=torch.cuda.max_memory_allocated()/1024**2 if device == "cuda" else None)
