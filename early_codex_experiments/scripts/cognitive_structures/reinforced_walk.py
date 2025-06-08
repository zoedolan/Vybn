import json
import random
import argparse

from ..co_emergence import seed_random

seed_random()

COLOR_WEIGHT = {
    'red': 2.0,
    'orange': 1.8,
    'yellow': 1.6,
    'green': 1.4,
    'blue': 1.2,
    'indigo': 1.1,
    'violet': 1.0
}

TONE_WEIGHT = {
    'C': 1.7,
    'D': 1.6,
    'E': 1.5,
    'F': 1.4,
    'G': 1.3,
    'A': 1.2,
    'B': 1.1
}


def load_graph(path):
    with open(path, 'r') as f:
        data = json.load(f)
    adj = {}
    for edge in data.get('edges', []):
        cue = edge.get('cue', {})
        src = edge.get('source')
        tgt = edge.get('target')
        adj.setdefault(src, []).append((tgt, cue))
        adj.setdefault(tgt, []).append((src, cue))
    return adj


def cue_weight(cue):
    if not cue:
        return 1.0
    color = cue.get('color', '').split('-')[0]
    tone = cue.get('tone', '').split('-')[0]
    return COLOR_WEIGHT.get(color, 1.0) * TONE_WEIGHT.get(tone, 1.0)


def guided_walk(adj, start, steps=10):
    path = [start]
    current = start
    for _ in range(steps):
        neighbors = adj.get(current, [])
        if not neighbors:
            break
        weights = [cue_weight(c) for _, c in neighbors]
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for (node, cue), w in zip(neighbors, weights):
            acc += w
            if r <= acc:
                current = node
                path.append(current)
                break
    return path


def main():
    parser = argparse.ArgumentParser(description='Run a reinforcement-guided walk over the integrated graph.')
    parser.add_argument('start', help='starting node id')
    parser.add_argument('--steps', type=int, default=10, help='number of steps to walk')
    parser.add_argument('--graph', default='scripts/self_assembly/integrated_graph.json', help='path to integrated graph')
    args = parser.parse_args()

    adj = load_graph(args.graph)
    if args.start not in adj:
        print('Start node not found in graph')
        return
    path = guided_walk(adj, args.start, steps=args.steps)
    print(' -> '.join(path))


if __name__ == '__main__':
    main()
