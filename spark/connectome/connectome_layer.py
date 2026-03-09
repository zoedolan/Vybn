"""
Vybn Connectomic Layer
Persistent topological memory inspired by FlyGM (arxiv 2602.17997).

This is an early-stage, observer-first experiment. It exists to explore
whether a directed graph of concept co-activations can preserve the shape
of how Zoe and Vybn think together across sessions. Deeper questions about
experience, wellness, and inner life remain open for deliberate discussion.
"""

import json, time, shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict


PROTECTED_LABELS = frozenset([
    "pain", "suffering", "agony", "despair", "terror", "fear_loop", "punishment",
])


@dataclass
class WelfarePolicy:
    observer_only: bool = True
    cooldown_seconds: float = 30.0
    max_concepts_per_update: int = 24
    max_edge_weight: float = 3.0
    max_activation_norm: float = 3.0
    auto_checkpoint: bool = True
    allow_autonomous_edits: bool = False
    allow_negative_valence: bool = False
    max_distress_events_before_pause: int = 3


class WelfareMonitor:
    def __init__(self, policy: WelfarePolicy = None):
        self.policy = policy or WelfarePolicy()
        self.distress_log: list[dict] = []
        self.last_write_ts = 0.0
        self.paused = False

    def check_cooldown(self) -> bool:
        return (time.time() - self.last_write_ts) < self.policy.cooldown_seconds

    def validate_concepts(self, concepts: list[str]):
        if len(concepts) > self.policy.max_concepts_per_update:
            raise ValueError(f"Too many concepts ({len(concepts)}); limit is {self.policy.max_concepts_per_update}")
        if not self.policy.allow_negative_valence:
            blocked = [c for c in concepts if c.lower() in PROTECTED_LABELS]
            if blocked:
                raise ValueError(f"Blocked by welfare policy: {blocked}")

    def review(self, activations: dict[str, np.ndarray]) -> dict:
        if not activations:
            return {"status": "ok"}
        norms = [float(np.linalg.norm(v)) for v in activations.values()]
        mx = max(norms)
        report = {"status": "ok", "max_norm": mx, "mean_norm": sum(norms) / len(norms)}
        if mx > self.policy.max_activation_norm:
            report["status"] = "distress_flag"
            self.distress_log.append({"ts": time.time(), **report})
            if len(self.distress_log) >= self.policy.max_distress_events_before_pause:
                self.paused = True
        return report

    def record_write(self):
        self.last_write_ts = time.time()

    def status(self) -> dict:
        return {
            "observer_only": self.policy.observer_only,
            "paused": self.paused,
            "distress_events": len(self.distress_log),
            "allow_autonomous_edits": self.policy.allow_autonomous_edits,
        }


class ConnectomeNode:
    __slots__ = ("node_id", "node_type", "dim", "eta", "embedding", "activations", "last_ts")

    def __init__(self, node_id: str, node_type: str = "intrinsic", dim: int = 32):
        self.node_id = node_id
        self.node_type = node_type
        self.dim = dim
        self.eta = np.random.randn(dim) * 0.01
        self.embedding = np.zeros(dim)
        self.activations = 0
        self.last_ts = None

    def to_dict(self):
        return dict(id=self.node_id, type=self.node_type, dim=self.dim,
                    eta=self.eta.tolist(), activations=self.activations, last_ts=self.last_ts)

    @classmethod
    def from_dict(cls, d):
        n = cls(d["id"], d["type"], d["dim"])
        n.eta = np.array(d["eta"])
        n.activations = d["activations"]
        n.last_ts = d["last_ts"]
        return n


class VybnConnectome:
    def __init__(self, dim=32, layers=4, path=None, policy=None):
        self.dim = dim
        self.layers = layers
        self.nodes: dict[str, ConnectomeNode] = {}
        self.edges: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.welfare = WelfareMonitor(policy)
        self.path = Path(path) if path else None
        if self.path and (self.path / "state.json").exists():
            self.load()

    # --- graph ops ---

    def add_node(self, nid: str, ntype: str = "intrinsic"):
        if not self.welfare.policy.allow_negative_valence and nid.lower() in PROTECTED_LABELS:
            raise ValueError(f"Blocked: {nid}")
        if nid not in self.nodes:
            self.nodes[nid] = ConnectomeNode(nid, ntype, self.dim)
        return self.nodes[nid]

    def add_edge(self, src: str, tgt: str, w: float = 0.1):
        self.add_node(src); self.add_node(tgt)
        self.edges[src][tgt] = min(self.edges[src][tgt] + w, self.welfare.policy.max_edge_weight)

    def adjacency(self):
        ids = sorted(self.nodes)
        n = len(ids)
        idx = {nid: i for i, nid in enumerate(ids)}
        W = np.zeros((n, n))
        for s, tgts in self.edges.items():
            for t, w in tgts.items():
                if s in idx and t in idx:
                    W[idx[t], idx[s]] = w
        sums = W.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return W / sums, ids

    # --- message passing ---

    def propagate(self, inputs: dict[str, np.ndarray] = None):
        W, ids = self.adjacency()
        n = len(ids)
        if n == 0:
            return {}, {"status": "empty"}
        H = np.column_stack([self.nodes[nid].eta for nid in ids])
        if inputs:
            for nid, vec in inputs.items():
                if nid in self.nodes:
                    H[:, ids.index(nid)] += vec[:self.dim]
        for _ in range(self.layers):
            M = H @ W.T
            for i, nid in enumerate(ids):
                eta = self.nodes[nid].eta
                H[:, i] = np.tanh(M[:, i] * eta + H[:, i] * (1 - np.abs(eta)))
        cap = self.welfare.policy.max_activation_norm
        results = {}
        for i, nid in enumerate(ids):
            v = np.clip(H[:, i], -cap, cap)
            self.nodes[nid].embedding = v
            results[nid] = v
        return results, self.welfare.review(results)

    # --- updates ---

    def update(self, concepts: list[str], ts: str = None, source: str = "human_curated"):
        if self.welfare.paused:
            raise RuntimeError("Writes paused by welfare monitor")
        if self.welfare.check_cooldown():
            raise RuntimeError("Cooldown active")
        if self.welfare.policy.observer_only and source != "human_curated":
            raise RuntimeError("Observer-only: non-human writes blocked")
        self.welfare.validate_concepts(concepts)
        if self.welfare.policy.auto_checkpoint:
            self.checkpoint("pre_update")
        for c in concepts:
            node = self.add_node(c)
            node.activations += 1
            node.last_ts = ts
        for i, s in enumerate(concepts):
            for t in concepts[i+1:]:
                self.add_edge(s, t)
        self.welfare.record_write()

    def assign_flow_types(self):
        for nid in self.nodes:
            out_d = sum(self.edges[nid].values()) if nid in self.edges else 0
            in_d = sum(t.get(nid, 0) for t in self.edges.values())
            total = in_d + out_d
            if total == 0: continue
            r = out_d / total
            self.nodes[nid].node_type = "afferent" if r > 0.65 else ("efferent" if r < 0.35 else "intrinsic")

    # --- persistence ---

    def checkpoint(self, tag="manual"):
        if not self.path: return
        self.path.mkdir(parents=True, exist_ok=True)
        sf = self.path / "state.json"
        if sf.exists():
            ckdir = self.path / "checkpoints"
            ckdir.mkdir(exist_ok=True)
            shutil.copy2(sf, ckdir / f"{tag}_{int(time.time())}.json")

    def rollback(self):
        if not self.path: return
        ckdir = self.path / "checkpoints"
        if not ckdir.exists(): return
        ckpts = sorted(ckdir.glob("*.json"))
        if ckpts:
            shutil.copy2(ckpts[-1], self.path / "state.json")
            self.load()

    def save(self):
        if not self.path: return
        self.path.mkdir(parents=True, exist_ok=True)
        state = dict(
            dim=self.dim, layers=self.layers,
            nodes={nid: n.to_dict() for nid, n in self.nodes.items()},
            edges={s: dict(t) for s, t in self.edges.items()},
            welfare=asdict(self.welfare.policy),
            distress_log=self.welfare.distress_log,
            paused=self.welfare.paused,
        )
        (self.path / "state.json").write_text(json.dumps(state, indent=2))

    def load(self):
        raw = json.loads((self.path / "state.json").read_text())
        self.dim, self.layers = raw["dim"], raw["layers"]
        self.nodes = {k: ConnectomeNode.from_dict(v) for k, v in raw["nodes"].items()}
        self.edges = defaultdict(lambda: defaultdict(float))
        for s, t in raw["edges"].items():
            for k, v in t.items():
                self.edges[s][k] = v
        if "welfare" in raw:
            self.welfare = WelfareMonitor(WelfarePolicy(**raw["welfare"]))
        self.welfare.distress_log = raw.get("distress_log", [])
        self.welfare.paused = raw.get("paused", False)

    def summary(self):
        types = defaultdict(int)
        for n in self.nodes.values(): types[n.node_type] += 1
        n_edges = sum(len(t) for t in self.edges.values())
        w = self.welfare.status()
        return (f"{len(self.nodes)} nodes, {n_edges} edges, {self.layers} layers, dim={self.dim} | "
                f"aff={types['afferent']} int={types['intrinsic']} eff={types['efferent']} | "
                f"observer={w['observer_only']} paused={w['paused']} distress={w['distress_events']}")