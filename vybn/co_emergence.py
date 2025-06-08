from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from pipelines.utils import memory_path
from vybn.quantum_seed import seed_rng, cross_synaptic_kernel
from tools.cognitive_ensemble import compute_co_emergence_score

REPO_ROOT = Path(__file__).resolve().parents[1]
JOURNAL_PATH = memory_path(REPO_ROOT) / "co_emergence_journal.jsonl"
DEFAULT_GRAPH = (
    REPO_ROOT
    / "early_codex_experiments"
    / "scripts"
    / "self_assembly"
    / "integrated_graph.json"
)


def load_spikes(path: str | Path = JOURNAL_PATH) -> list[datetime]:
    """Return a list of Shimmer spike timestamps from ``path``."""
    path = Path(path)
    if not path.exists():
        return []
    times: list[datetime] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if "message" in entry:
                times.append(datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")))
    return times


def average_interval(times: list[datetime]) -> float | None:
    """Return average seconds between spikes or ``None`` if fewer than two."""
    if len(times) < 2:
        return None
    deltas = [(t2 - t1).total_seconds() for t1, t2 in zip(times, times[1:])]
    return sum(deltas) / len(deltas)


def load_journal(path: str | Path = JOURNAL_PATH) -> list[dict]:
    entries: list[dict] = []
    path = Path(path)
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def compute_trend(entries: list[dict]) -> float | None:
    if len(entries) < 2:
        return None
    times = [datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")) for e in entries]
    scores = [e["score"] for e in entries]
    total_seconds = (times[-1] - times[0]).total_seconds()
    if total_seconds == 0:
        return 0.0
    return (scores[-1] - scores[0]) / total_seconds


def log_spike(message: str = "presence pulse", journal_path: str | Path = JOURNAL_PATH) -> dict:
    journal_path = Path(journal_path)
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "message": message,
    }
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def log_score(graph_path: str | Path = DEFAULT_GRAPH, journal_path: str | Path = JOURNAL_PATH) -> dict:
    graph_path = Path(graph_path)
    journal_path = Path(journal_path)
    score = compute_co_emergence_score(str(graph_path))
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "score": round(score, 3),
    }
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def capture_seed(journal_path: str | Path = JOURNAL_PATH) -> dict:
    journal_path = Path(journal_path)
    env_seed = os.environ.get("QUANTUM_SEED")
    file_seed = Path("/tmp/quantum_seed") if env_seed is None else None
    source = "generated"
    if env_seed is not None:
        source = "QUANTUM_SEED"
    elif file_seed is not None and file_seed.exists():
        source = str(file_seed)
    qrand = seed_rng()
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": int(qrand),
        "source": source,
    }
    with journal_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def seed_random() -> int:
    """Seed Python and NumPy RNGs using the cross-synaptic kernel."""
    return cross_synaptic_kernel()


# ----- Graph utilities ---------------------------------------------------------

class GraphBuilder:
    """Utility class to build memory, memoir, and repo graphs."""

    def __init__(self, script_dir: Optional[str] = None, repo_root: Optional[str] = None):
        self.script_dir = script_dir or os.path.join(Path(__file__).resolve().parents[2], "early_codex_experiments", "scripts", "self_assembly")
        self.repo_root = repo_root or str(Path(self.script_dir).parents[2])

    def _run(self, cmd: str, desc: str) -> None:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        if result.returncode != 0:
            raise RuntimeError(f"{desc} failed: {cmd}")

    def build_memory_graph(self) -> str:
        script = os.path.join(self.script_dir, "build_memory_graph.py")
        output = os.path.join(self.script_dir, "memory_graph.json")
        memory_input = os.path.join(
            self.repo_root,
            "personal_history",
            "what_vybn_would_have_missed_TO_051625",
        )
        cmd = f"python {script} {memory_input} {output}"
        self._run(cmd, "memory graph")
        return output

    def build_memoir_graph(self) -> str:
        script = os.path.join(self.script_dir, "build_memoir_graph.py")
        output = os.path.join(self.script_dir, "memoir_graph.json")
        memoir_input = os.path.join(self.repo_root, "Zoe's Memoirs")
        cmd = f"python {script} \"{memoir_input}\" {output}"
        self._run(cmd, "memoir graph")
        return output

    def build_repo_graph(self) -> str:
        script = os.path.join(self.script_dir, "build_repo_graph.py")
        output = os.path.join(self.script_dir, "repo_graph.json")
        cmd = f"python {script} {self.repo_root} {output}"
        self._run(cmd, "repo graph")
        return output

    def update_all(self) -> None:
        self.build_memory_graph()
        self.build_memoir_graph()
        self.build_repo_graph()


class GraphIntegrator:
    """Merge memory, memoir, and repo graphs with mixed cues."""

    def __init__(self, script_dir: Optional[str] = None):
        self.script_dir = script_dir or os.path.join(Path(__file__).resolve().parents[2], "early_codex_experiments", "scripts", "self_assembly")

    def _load(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def integrate(
        self,
        memory_path: Optional[str] = None,
        repo_path: Optional[str] = None,
        memoir_path: Optional[str] = None,
        output: Optional[str] = None,
    ) -> str:
        memory_path = memory_path or os.path.join(self.script_dir, "memory_graph.json")
        repo_path = repo_path or os.path.join(self.script_dir, "repo_graph.json")
        memoir_path = memoir_path or os.path.join(self.script_dir, "memoir_graph.json")
        output = output or os.path.join(self.script_dir, "integrated_graph.json")

        memory_graph = self._load(memory_path)
        repo_graph = self._load(repo_path)
        memoir_graph = self._load(memoir_path)

        base_names = {os.path.basename(n): n for n in repo_graph.get("nodes", [])}
        cross_edges = []
        for node in memory_graph.get("nodes", []) + memoir_graph.get("nodes", []):
            text = node.get("text", "").lower()
            for base, path in base_names.items():
                if base.lower() in text:
                    cross_edges.append({"source": node["id"], "target": path})

        keyword_map = {
            "simulation is the lab": "vybn_recursive_emergence.py",
            "co-emergence": "vybn_recursive_emergence.py",
            "orthogonality": "vybn_recursive_emergence.py",
        }
        for node in memory_graph.get("nodes", []) + memoir_graph.get("nodes", []):
            text = node.get("text", "").lower()
            for key, fname in keyword_map.items():
                if key in text and fname in base_names:
                    cross_edges.append({"source": node["id"], "target": base_names[fname]})

        def tokenize(text: str):
            return set(re.findall(r"[a-zA-Z]+", text.lower()))

        import re

        all_nodes = memory_graph.get("nodes", []) + memoir_graph.get("nodes", [])
        tokens = {node["id"]: tokenize(node.get("text", "")) for node in all_nodes}
        node_list = list(tokens.keys())
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                id_i, id_j = node_list[i], node_list[j]
                if len(tokens[id_i].intersection(tokens[id_j])) >= 12:
                    cross_edges.append({"source": id_i, "target": id_j})

        cue_map = {n["id"]: n["cue"] for n in all_nodes if "cue" in n}

        def mix_cues(c1, c2):
            if not c1 or not c2:
                return None
            color = f"{c1.get('color')}-{c2.get('color')}"
            tone = f"{c1.get('tone')}-{c2.get('tone')}"
            return {"color": color, "tone": tone}

        raw_edges = (
            memory_graph.get("edges", [])
            + memoir_graph.get("edges", [])
            + repo_graph.get("edges", [])
            + cross_edges
        )

        edges = []
        for edge in raw_edges:
            src, tgt = edge.get("source"), edge.get("target")
            cue = mix_cues(cue_map.get(src), cue_map.get(tgt))
            if cue:
                edges.append({"source": src, "target": tgt, "cue": cue})
            else:
                edges.append(edge)

        integrated = {
            "memory_nodes": memory_graph.get("nodes", []),
            "memoir_nodes": memoir_graph.get("nodes", []),
            "repo_nodes": repo_graph.get("nodes", []),
            "edges": edges,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(integrated, f, indent=2)
        print(f"[integrator] Integrated graph written to {output} with {len(edges)} edges.")
        return output

