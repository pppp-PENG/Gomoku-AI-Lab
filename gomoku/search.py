from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import torch
from .game import Game, flip_wdl
from .models import masked_policy


class Evaluator:
    def __init__(self, model, device="cpu"):
        self.model = model.eval()
        self.device = device

    @torch.inference_mode()
    def batch(self, games):
        if not games:
            return []
        x = torch.from_numpy(np.stack([g.encode() for g in games])).to(self.device)
        legal = torch.from_numpy(np.stack([g.legal_mask() for g in games])).to(self.device)
        logits, values = self.model(x)
        p = masked_policy(logits, legal).cpu().numpy()
        w = values.softmax(-1).cpu().numpy()
        return list(zip(p, w))

    def __call__(self, game):
        return self.batch([game])[0]


@dataclass
class Node:
    prior: float = 1.
    visits: int = 0
    # WDL totals belong to the parent player for this edge.
    total: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def q(self):
        return float((self.total[0] - self.total[2]) / self.visits) if self.visits else 0.


def select(node, c_puct):
    scale = math.sqrt(max(1, sum(ch.visits for ch in node.children.values())))
    return max(node.children, key=lambda a: node.children[a].q +
               c_puct * node.children[a].prior * scale / (1 + node.children[a].visits))


def backup(path, leaf_wdl):
    wdl = np.asarray(leaf_wdl)
    for node in reversed(path):
        wdl = flip_wdl(wdl)
        node.total += wdl
        node.visits += 1


class MCTS:
    def __init__(self, evaluator, simulations=32, c_puct=1.5, rng=None):
        if simulations < 1:
            raise ValueError("simulations must be positive")
        self.evaluator, self.simulations, self.c_puct = evaluator, simulations, c_puct
        self.rng = rng or np.random.default_rng(0)

    def expand(self, node, game):
        policy, wdl = self.evaluator(game)
        node.children = {int(a): Node(float(policy[a])) for a in np.flatnonzero(game.legal_mask())}
        return wdl

    def run(self, game, noise=False):
        if game.terminal:
            raise ValueError("cannot search terminal game")
        root = Node()
        root_wdl = self.expand(root, game)
        if noise:
            eta = self.rng.dirichlet(np.full(len(root.children), .3))
            for ch, e in zip(root.children.values(), eta):
                ch.prior = .75*ch.prior + .25*e
        for _ in range(self.simulations):
            state, node, path = game.clone(), root, []
            while node.children and not state.terminal:
                a = select(node, self.c_puct)
                node = node.children[a]
                state.play(a)
                path.append(node)
            value = state.terminal_wdl() if state.terminal else self.expand(node, state)
            backup(path, value)
        visits = np.zeros(game.size**2, dtype=np.float32)
        for a, ch in root.children.items():
            visits[a] = ch.visits
        return visits / visits.sum(), root, root_wdl


def choose(policy, temperature, rng):
    if temperature <= 0:
        return int(np.argmax(policy))
    p = np.power(policy.astype(np.float64), 1/temperature)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def analyze(game, evaluator, simulations=0):
    """Rank all actions by child WDL win probability, never by policy prior."""
    if game.terminal:
        return dict(candidates=[], top5=[], root_wdl=game.terminal_wdl().tolist(),
                    simulations=simulations, ranking="terminal",
                    semantics="Terminal WDL from the next player's perspective; no legal candidates")
    prior, root_wdl = evaluator(game)
    visits, root = np.zeros(game.size**2), None
    if simulations:
        visits, root, _ = MCTS(evaluator, simulations).run(game)
    actions = np.flatnonzero(game.legal_mask())
    children, unresolved, values = [], [], {}
    for a in actions:
        child = game.clone().play(int(a))
        if child.terminal:
            values[int(a)] = flip_wdl(child.terminal_wdl())
        else:
            children.append(child)
            unresolved.append(int(a))
    # Bound the batch for larger boards / laptop memory.
    for start in range(0, len(children), 64):
        for a, (_, wdl) in zip(unresolved[start:start+64], evaluator.batch(children[start:start+64])):
            values[a] = flip_wdl(wdl)
    candidates = []
    for action in actions:
        a = int(action)
        w = values[a]
        ch = root.children[a] if root else None
        candidates.append(dict(action=a, win=float(w[0]), draw=float(w[1]), loss=float(w[2]),
                               value=float(w[0]-w[2]), policy=float(prior[a]),
                               visit_fraction=float(visits[a]), visits=ch.visits if ch else 0,
                               search_q=ch.q if ch and ch.visits else None,
                               search_wdl=(ch.total/ch.visits).tolist() if ch and ch.visits else None))
    candidates.sort(key=lambda c: (-c["win"], -c["value"], c["action"]))
    return dict(candidates=candidates, top5=candidates[:5], root_wdl=root_wdl.tolist(),
                ranking="child_value_win_probability", simulations=simulations,
                semantics="WDL is current-player prediction; search_q is W-L, not win probability")


def search_many(games, evaluators, simulations, rng, noise=True):
    """One leaf per active tree per round; batch leaves sharing a network."""
    roots = [Node() for _ in games]

    def evaluate_requests(requests):
        groups = {}
        for i, node, state, path in requests:
            groups.setdefault(evaluators[i], []).append((i, node, state, path))
        for evaluator, items in groups.items():
            outputs = evaluator.batch([item[2] for item in items])
            for (_, node, state, path), (prior, wdl) in zip(items, outputs):
                node.children = {int(a): Node(float(prior[a])) for a in np.flatnonzero(state.legal_mask())}
                if path:
                    backup(path, wdl)

    evaluate_requests([(i, roots[i], g, []) for i, g in enumerate(games)])
    if noise:
        for root in roots:
            eta = rng.dirichlet(np.full(len(root.children), .3))
            for ch, e in zip(root.children.values(), eta):
                ch.prior = .75*ch.prior + .25*e
    for _ in range(simulations):
        requests = []
        for i, game in enumerate(games):
            state, node, path = game.clone(), roots[i], []
            while node.children and not state.terminal:
                a = select(node, 1.5)
                node = node.children[a]
                state.play(a)
                path.append(node)
            if state.terminal:
                backup(path, state.terminal_wdl())
            else:
                requests.append((i, node, state, path))
        evaluate_requests(requests)
    policies = []
    for root, game in zip(roots, games):
        p = np.zeros(game.size**2, dtype=np.float32)
        for a, ch in root.children.items():
            p[a] = ch.visits
        policies.append(p/p.sum())
    return policies
