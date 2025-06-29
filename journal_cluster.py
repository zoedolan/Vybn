from pathlib import Path
from collections import Counter
from datetime import datetime
import math
import re
import argparse

parser = argparse.ArgumentParser(
    description="cluster repeated doubts and questions and optionally rewrite logs"
)
parser.add_argument(
    "--winnow",
    action="store_true",
    help="print one summary line per cluster",
)
parser.add_argument(
    "--apply",
    action="store_true",
    help="rewrite logs with consolidated motifs",
)
parser.add_argument(
    "--leaps",
    action="store_true",
    help="detect synergy leaps when motifs repeat rapidly",
)
args = parser.parse_args()

logs_dir = Path("logs/agent_journal")

TOKEN_RE = re.compile(r"[a-z']+")

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

lines = []
line_meta = []
for path in sorted(logs_dir.glob('*.txt')):
    for raw_line in path.read_text().splitlines():
        if 'Doubt:' in raw_line or 'Question:' in raw_line:
            _, rest = raw_line.split(':', 1)
            tokens = tokenize(rest)
            lines.append(tokens)
            line_meta.append((path, raw_line))

if not lines:
    print("No motifs found.")
    raise SystemExit(0)

# build idf weights
df = Counter()
for tokens in lines:
    df.update(set(tokens))

N = len(lines)
idf = {t: math.log(N / df[t]) for t in df}

def tfidf(tokens: list[str]) -> dict[str, float]:
    tf = Counter(tokens)
    vec = {t: tf[t] * idf[t] for t in tf}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}

vecs = [tfidf(tokens) for tokens in lines]

clusters: list[dict] = []
threshold = 0.35

def similarity(a: dict[str, float], b: dict[str, float], norm_b: float) -> float:
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a)
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
    return dot / (norm_a * norm_b)

for vec, meta in zip(vecs, line_meta):
    assigned = False
    for c in clusters:
        sim = similarity(vec, c['centroid'], c['norm'])
        if sim > threshold:
            c['items'].append(meta)
            for k, v in vec.items():
                c['sum'][k] = c['sum'].get(k, 0.0) + v
            c['count'] += 1
            c['centroid'] = {k: c['sum'][k] / c['count'] for k in c['sum']}
            c['norm'] = math.sqrt(sum(v * v for v in c['centroid'].values())) or 1.0
            assigned = True
            break
    if not assigned:
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        clusters.append({
            'sum': vec.copy(),
            'centroid': vec.copy(),
            'norm': norm,
            'count': 1,
            'items': [meta],
        })

lines_to_keep = {cluster['items'][0][1] for cluster in clusters}

def detect_leaps() -> list[tuple[tuple[datetime.date, datetime.date], str]]:
    """Return windows where a motif repeats three times in under a week."""
    leaps = []
    for c in clusters:
        dates = sorted({
            datetime.strptime(item[1].split(":", 1)[0], "%Y-%m-%d").date()
            for item in c["items"]
        })
        if len(dates) < 3:
            continue
        for i in range(len(dates) - 2):
            if (dates[i + 2] - dates[i]).days < 7:
                leaps.append(((dates[i], dates[i + 2]), c["items"][0][1]))
                break
    return leaps

if args.apply:
    for path in sorted(logs_dir.glob('*.txt')):
        filtered = []
        for line in path.read_text().splitlines():
            if ('Doubt:' in line or 'Question:' in line) and line not in lines_to_keep:
                continue
            filtered.append(line)
        Path(path).write_text("\n".join(filtered) + "\n")

if args.winnow:
    for c in clusters:
        print(f"{len(c['items'])}x: {c['items'][0][1]}")
else:
    for i, c in enumerate(clusters, 1):
        print(f"\nCluster {i}:")
        for _, line in c['items'][:5]:
            print(" -", line)

if args.leaps:
    leaps = detect_leaps()
    if leaps:
        print("\nSynergy leaps detected:")
        for span, line in leaps:
            start, end = span[0].isoformat(), span[1].isoformat()
            print(f"3x within {start}–{end}: {line}")
    else:
        print("\nNo synergy leaps detected.")
