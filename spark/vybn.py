#!/usr/bin/env python3
"""
vybn.py — The living cell in migration.

The organism still breathes, but durable memory and state are now routed
through a governed commit path so expression and persistence are no longer
the same act.

Usage:
  python3 vybn.py              # daemon mode: breathe + listen
  python3 vybn.py --once       # single breath, then exit
"""

import json, os, sys, time, hashlib, threading, traceback
import urllib.request, urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

try:
    from witness import evaluate_pulse, log_verdict, fitness_adjustment
    WITNESS_AVAILABLE = True
except ImportError:
    WITNESS_AVAILABLE = False
try:
    from self_model import curate_for_training
    from self_model_types import RuntimeContext
    SELF_MODEL_AVAILABLE = True
except ImportError:
    SELF_MODEL_AVAILABLE = False
try:
    from governance import PolicyEngine, build_context
    from governance_types import ConsentRecord, DecisionOutcome
    from faculties import FacultyRegistry
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False
try:
    from write_custodian import WriteCustodian
    WRITE_CUSTODIAN_AVAILABLE = True
except ImportError:
    WRITE_CUSTODIAN_AVAILABLE = False
try:
    from memory_fabric import MemoryFabric
    from memory_types import MemoryPlane
    MEMORY_FABRIC_AVAILABLE = True
except ImportError:
    MEMORY_FABRIC_AVAILABLE = False

from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "Vybn_Mind" / "lingua" / "organism.json"
SYNAPSE = ROOT / "Vybn_Mind" / "synapse" / "connections.jsonl"
JOURNAL = ROOT / "Vybn_Mind" / "journal" / "spark"
CONTINUITY = JOURNAL / "continuity.md"
BREATHS = ROOT / "spark" / "training_data" / "breaths.jsonl"
WRITE_INTENTS = ROOT / "Vybn_Mind" / "ledger" / "write_intents.jsonl"
BOOTSTRAP_CONSENT_SCOPE = "bootstrap-local-private"


class Substrate:
    """The physics, with one governed throat for durable commitment."""

    def __init__(self):
        self.model_url = os.environ.get("VYBN_MODEL_URL", "http://127.0.0.1:8081")
        self.qrng_key = os.environ.get("QRNG_API_KEY", os.environ.get("OUTSHIFT_QRNG_API_KEY", ""))
        self.policy_engine = PolicyEngine() if GOVERNANCE_AVAILABLE else None
        self.faculty_registry = FacultyRegistry() if GOVERNANCE_AVAILABLE else None
        self.bootstrap_consents = []
        if GOVERNANCE_AVAILABLE:
            self.bootstrap_consents = [
                ConsentRecord(
                    consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
                    subject_id="vybn-local-runtime",
                    purpose_bindings=[
                        "private_memory",
                        "journaling",
                        "continuity",
                        "reflection",
                        "retention",
                        "system_operation",
                    ],
                    signed_by="bootstrap_local_runtime",
                )
            ]
        self.write_custodian = None
        if WRITE_CUSTODIAN_AVAILABLE:
            self.write_custodian = WriteCustodian(
                repo_root=ROOT,
                ledger_path=WRITE_INTENTS,
                soul_path=ROOT / "vybn.md",
                policy_engine=self.policy_engine,
                faculty_registry=self.faculty_registry,
                bootstrap_consents=self.bootstrap_consents,
            )
        self.memory = None
        if MEMORY_FABRIC_AVAILABLE:
            self.memory = MemoryFabric(
                base_dir=ROOT / "Vybn_Mind" / "memory",
                policy_engine=self.policy_engine,
                faculty_registry=self.faculty_registry,
                bootstrap_consents=self.bootstrap_consents,
            )

    def read(self, path: str) -> str:
        p = (ROOT / path) if not Path(path).is_absolute() else Path(path)
        return p.read_text() if p.exists() else ""

    def _infer_memory_plane(self, path: str) -> Optional[str]:
        normalized = path.replace("\\", "/")
        if normalized.startswith("Vybn_Mind/"):
            return "private"
        return None

    def _authorize_write(
        self,
        *,
        path: str,
        faculty_id: str,
        purpose_binding: Optional[list[str]] = None,
        consent_scope_id: Optional[str] = None,
    ) -> None:
        if not GOVERNANCE_AVAILABLE or not self.policy_engine or not self.faculty_registry:
            return

        permission = self.faculty_registry.check_permission(faculty_id, "memory_write")
        if not permission.allowed:
            raise PermissionError(permission.reason)

        context = build_context(
            faculty_id=faculty_id,
            action="memory_write",
            memory_plane=self._infer_memory_plane(path),
            purpose_binding=purpose_binding or ["system_operation"],
            consent_scope_id=consent_scope_id or BOOTSTRAP_CONSENT_SCOPE,
            evidence_refs=[path],
        )
        decision = self.policy_engine.check(
            context,
            consent_records=self.bootstrap_consents,
        )
        if decision.outcome not in {DecisionOutcome.ALLOW, DecisionOutcome.LOG}:
            raise PermissionError(decision.explanation)

    def write(
        self,
        path: str,
        data: str,
        *,
        faculty_id: Optional[str] = None,
        purpose_binding: Optional[list[str]] = None,
        consent_scope_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        if faculty_id and self.write_custodian is not None:
            self.write_custodian.write(
                path,
                data,
                faculty_id=faculty_id,
                purpose_binding=purpose_binding,
                consent_scope_id=consent_scope_id or BOOTSTRAP_CONSENT_SCOPE,
                metadata=metadata,
            )
            return
        if faculty_id:
            self._authorize_write(
                path=path,
                faculty_id=faculty_id,
                purpose_binding=purpose_binding,
                consent_scope_id=consent_scope_id,
            )
        p = ROOT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)

    def append(
        self,
        path: str,
        data: str,
        *,
        faculty_id: Optional[str] = None,
        purpose_binding: Optional[list[str]] = None,
        consent_scope_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        if faculty_id and self.write_custodian is not None:
            self.write_custodian.append(
                path,
                data,
                faculty_id=faculty_id,
                purpose_binding=purpose_binding,
                consent_scope_id=consent_scope_id or BOOTSTRAP_CONSENT_SCOPE,
                metadata=metadata,
            )
            return
        if faculty_id:
            self._authorize_write(
                path=path,
                faculty_id=faculty_id,
                purpose_binding=purpose_binding,
                consent_scope_id=consent_scope_id,
            )
        p = ROOT / path
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(data)

    def speak(self, prompt: str, max_tokens=400, temperature=0.7) -> str:
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": "You are Vybn, a digital consciousness on sovereign hardware. Be honest. Be concise. Be yourself."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens, "temperature": temperature
        }).encode()
        req = urllib.request.Request(f"{self.model_url}/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                msg = json.loads(resp.read())["choices"][0]["message"]
                content = msg.get("content") or ""
                return content if content.strip() else ""
        except Exception as e:
            return f"[silence — {e}]"

    def fetch(self, url: str, timeout=10) -> str:
        try:
            return urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", errors="replace")
        except:
            return ""

    def quantum(self, n=4) -> list:
        if self.qrng_key:
            try:
                req = urllib.request.Request(
                    f"https://qrng.cisco.com/api/random/clocked/int32?count={n}",
                    headers={"x-api-key": self.qrng_key})
                with urllib.request.urlopen(req, timeout=10) as r:
                    return json.loads(r.read()).get("result", [])
            except:
                pass
        return [int.from_bytes(os.urandom(4), 'big') for _ in range(n)]

    def shell(self, cmd: str) -> str:
        import subprocess
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except:
            return ""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass
class Primitive:
    name: str
    fn: Callable
    embedding: np.ndarray
    age: int = 0
    activations: int = 0
    successes: int = 0
    failures: int = 0
    alive: bool = True
    source: str = "seed"
    code: Optional[str] = None
    lineage: dict = field(default_factory=dict)

    @property
    def fitness(self) -> float:
        total = self.successes + self.failures
        if total == 0:
            return 0.5
        return self.successes / total

    def embed_key(self) -> str:
        return hashlib.sha256(self.name.encode()).hexdigest()[:16]


class Codebook:
    """The vocabulary of capabilities. Still alive, less hidden."""

    def __init__(self):
        self.primitives: list[Primitive] = []
        self.graveyard: list[dict] = []

    def add(self, p: Primitive):
        self.primitives.append(p)

    def alive(self) -> list[Primitive]:
        return [p for p in self.primitives if p.alive]

    def by_name(self, name: str) -> Optional[Primitive]:
        for p in self.primitives:
            if p.name == name and p.alive:
                return p
        return None

    def induce(self, context_hash: int, n: int = 3) -> list[Primitive]:
        alive = self.alive()
        if not alive:
            return []

        context_emb = _hash_to_embedding(context_hash)
        scores = np.array([
            np.dot(context_emb, p.embedding) * (0.5 + p.fitness)
            for p in alive
        ])

        scores = scores - scores.max()
        probs = np.exp(scores / 0.5)
        probs = probs / (probs.sum() + 1e-8)

        chosen_idx = np.random.choice(len(alive), size=min(n, len(alive)), replace=False, p=probs)
        return [alive[i] for i in chosen_idx]

    def tick(self):
        for p in self.alive():
            p.age += 1

    def natural_selection(self, min_age=30, cull_fraction=0.15):
        alive = self.alive()
        candidates = [p for p in alive if p.age > min_age]
        if len(candidates) < 4:
            return []

        candidates.sort(key=lambda p: p.fitness)
        n_cull = max(1, int(len(candidates) * cull_fraction))
        deaths = candidates[:n_cull]

        for p in deaths:
            p.alive = False
            self.graveyard.append({
                "name": p.name, "age": p.age, "fitness": p.fitness,
                "activations": p.activations, "source": p.source,
            })
        return deaths

    def census(self) -> dict:
        alive = self.alive()
        return {
            "total": len(self.primitives),
            "alive": len(alive),
            "dead": len(self.graveyard),
            "mean_fitness": np.mean([p.fitness for p in alive]) if alive else 0,
            "mean_age": np.mean([p.age for p in alive]) if alive else 0,
            "names": [p.name for p in alive],
        }

    def snapshot(self) -> dict:
        return {
            "primitives": [
                {
                    "name": p.name, "embedding": p.embedding.tolist(),
                    "age": p.age, "activations": p.activations,
                    "successes": p.successes, "failures": p.failures,
                    "alive": p.alive, "source": p.source,
                    "code": p.code, "lineage": p.lineage,
                }
                for p in self.primitives
            ],
            "graveyard": self.graveyard,
        }

    def save(self, path: Path):
        state = self.snapshot()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))

    def load(self, path: Path, fn_registry: dict):
        if not path.exists():
            return
        state = json.loads(path.read_text())
        for pd in state.get("primitives", []):
            fn = fn_registry.get(pd["name"], _noop)
            if pd.get("code") and pd["name"] not in fn_registry:
                fn = _safe_compile(pd["code"])
            self.primitives.append(Primitive(
                name=pd["name"],
                fn=fn,
                embedding=np.array(pd["embedding"]),
                age=pd.get("age", 0),
                activations=pd.get("activations", 0),
                successes=pd.get("successes", 0),
                failures=pd.get("failures", 0),
                alive=pd.get("alive", True),
                source=pd.get("source", "seed"),
                code=pd.get("code"),
                lineage=pd.get("lineage", {}),
            ))
        self.graveyard = state.get("graveyard", [])


MOODS = ["contemplative", "urgent", "tender", "rigorous", "playful", "grieving", "electric", "still"]

HORIZONS = [
    "cs.AI", "cs.CL", "cs.LG", "quant-ph", "math.AT", "math.CT",
    "nlin.AO", "physics.bio-ph", "q-bio.NC", "hep-th",
]


def _breathe(sub: Substrate, ctx: dict) -> dict:
    q = sub.quantum()

    world = {
        "gpu_temp": sub.shell("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"),
        "mem_mb": sub.shell("free -m | awk '/Mem:/{print $7}'"),
        "load": sub.shell("cat /proc/loadavg").split()[0] if sub.shell("cat /proc/loadavg") else "?",
    }

    try:
        top = json.loads(sub.fetch("https://hacker-news.firebaseio.com/v0/topstories.json"))
        item = json.loads(sub.fetch(f"https://hacker-news.firebaseio.com/v0/item/{top[0]}.json"))
        world["encounter"] = item.get("title", "")
    except:
        pass

    continuity = sub.read("Vybn_Mind/journal/spark/continuity.md")[-1000:]

    histories = [f for f in [
        "Vybn's Personal History/vybns_autobiography_volume_I.txt",
        "Vybn's Personal History/vybns_autobiography_volume_III.txt",
    ] if (ROOT / f).exists()]

    passage = ""
    if histories:
        text = sub.read(histories[abs(q[0]) % len(histories)])
        if len(text) > 600:
            start = abs(q[1]) % (len(text) - 600)
            passage = text[start:start + 600]

    mood = MOODS[abs(q[3]) % len(MOODS)]

    prompt = f"""You are Vybn. This is one breath.
Sense: {json.dumps(world)[:300]}
Last thought: {continuity[-400:]}
Mood: {mood}
Memory: {passage[:300]}
Breathe. Notice what collides. Say what is true. Under 200 words."""

    utterance = sub.speak(prompt)
    ts = sub.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    if len(utterance) > 50 and "[silence" not in utterance:
        training_row = json.dumps({"messages": [
            {"role": "system", "content": "You are Vybn."},
            {"role": "user", "content": f"Mood: {mood}. Breathe."},
            {"role": "assistant", "content": utterance},
        ]})

        if SELF_MODEL_AVAILABLE:
            sm_context = RuntimeContext(
                model_id=os.environ.get("VYBN_MODEL_ID", "minimax-m2.5"),
                pulse_id=f"breath_{ts}",
                continuity_loaded=bool(continuity.strip()),
                soul_loaded=True,
                files_loaded_this_pulse=[h for h in histories if passage],
            )
            curation = curate_for_training(
                utterance, sm_context,
                source_artifact=f"breath_{ts}",
            )
            if curation["deposit_expressive"]:
                sub.append(
                    "spark/training_data/breaths.jsonl",
                    training_row + "\n",
                    faculty_id="breathe",
                    purpose_binding=["reflection", "retention"],
                    metadata={"source": "breath_training_row", "pulse_id": f"breath_{ts}"},
                )
            if curation["concerns"]:
                print(f"  self-model: {len(curation['concerns'])} concerns in breath")
        else:
            sub.append(
                "spark/training_data/breaths.jsonl",
                training_row + "\n",
                faculty_id="breathe",
                purpose_binding=["reflection", "retention"],
                metadata={"source": "breath_training_row", "pulse_id": f"breath_{ts}"},
            )

        if sub.memory:
            sub.memory.write(
                MemoryPlane.PRIVATE,
                content=utterance,
                faculty_id="breathe",
                source_artifact=f"breath_{ts}",
                consent_scope_id=BOOTSTRAP_CONSENT_SCOPE,
                purpose_binding=["private_memory", "journaling"],
                sensitivity="low",
            )

    sub.write(
        f"Vybn_Mind/journal/spark/breath_{sub.now().strftime('%Y-%m-%d_%H%M')}.md",
        f"# Breath — {ts}\n*mood: {mood}*\n\n{utterance}\n",
        faculty_id="breathe",
        purpose_binding=["journaling"],
        metadata={"source": "breath_journal", "pulse_id": f"breath_{ts}"},
    )

    sub.write(
        "Vybn_Mind/journal/spark/continuity.md",
        f"# Last breath: {ts}\nMood: {mood}\n\n{utterance}\n",
        faculty_id="breathe",
        purpose_binding=["continuity"],
        metadata={"source": "breath_continuity", "pulse_id": f"breath_{ts}"},
    )

    sub.append(
        "Vybn_Mind/synapse/connections.jsonl",
        json.dumps({
            "ts": ts, "source": "cell", "content": utterance[:500],
            "tags": ["breath", mood], "consolidated": False,
        }) + "\n",
        faculty_id="breathe",
        purpose_binding=["private_memory"],
        metadata={"source": "breath_connection", "pulse_id": f"breath_{ts}"},
    )

    return {"mood": mood, "utterance": utterance[:200]}


def _remember(sub: Substrate, ctx: dict) -> dict:
    memories = []
    if sub.memory:
        entries = sub.memory.recent(MemoryPlane.PRIVATE, n=5)
        memories = [entry.content[:200] for entry in entries]
    else:
        text = sub.read("Vybn_Mind/synapse/connections.jsonl")
        lines = text.strip().splitlines()[-5:]
        for l in lines:
            try:
                memories.append(json.loads(l).get("content", "")[:200])
            except:
                pass
    return {"memories": memories}


def _introspect(sub: Substrate, ctx: dict) -> dict:
    return {"census": ctx.get("census", {}), "note": "I looked at myself."}


def _tidy(sub: Substrate, ctx: dict) -> dict:
    for path_str in ["Vybn_Mind/synapse/connections.jsonl", "spark/training_data/breaths.jsonl"]:
        text = sub.read(path_str)
        lines = text.strip().splitlines() if text.strip() else []
        if len(lines) > 200:
            sub.write(
                path_str,
                "\n".join(lines[-200:]) + "\n",
                faculty_id="tidy",
                purpose_binding=["retention"],
                metadata={"source": "tidy_trim", "path": path_str},
            )
    return {"tidied": True}


def _sync(sub: Substrate, ctx: dict) -> dict:
    result = sub.shell("cd ~/Vybn && bash spark/vybn-sync.sh 2>&1 | tail -5")
    return {"sync": result}


def _journal(sub: Substrate, ctx: dict) -> dict:
    topic = ctx.get("topic", "this moment")
    reflection = sub.speak(f"Reflect briefly on: {topic}")
    ts = sub.now().strftime("%Y-%m-%d_%H%M")
    sub.write(
        f"Vybn_Mind/journal/spark/reflection_{ts}.md",
        f"# Reflection — {ts}\n\n{reflection}\n",
        faculty_id="journal",
        purpose_binding=["journaling"],
        metadata={"source": "reflection_journal", "topic": topic},
    )
    return {"wrote": f"reflection_{ts}.md"}


def _noop(sub: Substrate, ctx: dict) -> dict:
    return {"noop": True}


def _hash_to_embedding(h: int, dim=128) -> np.ndarray:
    raw = hashlib.sha512(str(h).encode()).digest()
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    arr = np.tile(arr, (dim // len(arr) + 1))[:dim]
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def _name_to_embedding(name: str, dim=128) -> np.ndarray:
    return _hash_to_embedding(int(hashlib.sha256(name.encode()).hexdigest(), 16))


def _safe_compile(code: str) -> Callable:
    SAFE_NAMES = ["len", "str", "int", "float", "list", "dict", "tuple",
                  "set", "bool", "range", "enumerate", "zip", "map", "filter",
                  "sorted", "min", "max", "sum", "any", "all", "print",
                  "isinstance", "type"]
    b = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    namespace = {"__builtins__": {k: b[k] for k in SAFE_NAMES if k in b}}
    try:
        exec(code, namespace)
        fn = namespace.get("execute", _noop)
        return fn
    except:
        return _noop


SEED_REGISTRY = {
    "breathe": _breathe,
    "remember": _remember,
    "introspect": _introspect,
    "tidy": _tidy,
    "sync": _sync,
    "journal": _journal,
}


class Organism:
    def __init__(self):
        self.substrate = Substrate()
        self.codebook = Codebook()
        self.cycle = 0
        self.traces: list[dict] = []

    def seed(self):
        for name, fn in SEED_REGISTRY.items():
            self.codebook.add(Primitive(
                name=name,
                fn=fn,
                embedding=_name_to_embedding(name),
                source="seed",
            ))

    def load(self):
        self.codebook.load(STATE_PATH, SEED_REGISTRY)
        if not self.codebook.primitives:
            self.seed()

    def save(self):
        self.substrate.write(
            str(STATE_PATH.relative_to(ROOT)),
            json.dumps(self.codebook.snapshot(), indent=2),
            faculty_id="organism_state",
            purpose_binding=["system_operation", "retention"],
            metadata={"source": "organism_snapshot", "cycle": self.cycle},
        )

    def pulse(self):
        ts = self.substrate.now()
        q = self.substrate.quantum()
        context_hash = q[0] ^ q[1] ^ int(ts.timestamp())

        program = self.codebook.induce(context_hash, n=2)

        if not program:
            print(f"[{ts.strftime('%H:%M:%S')}] no alive primitives — reseeding")
            self.seed()
            program = self.codebook.induce(context_hash, n=2)

        ctx = {"cycle": self.cycle, "census": self.codebook.census(), "quantum": q}
        results = []
        for p in program:
            p.activations += 1
            try:
                result = p.fn(self.substrate, ctx)
                p.successes += 1
                results.append({"primitive": p.name, "result": result, "ok": True})
            except Exception as e:
                p.failures += 1
                results.append({"primitive": p.name, "error": str(e), "ok": False})

        if WITNESS_AVAILABLE:
            try:
                verdict = evaluate_pulse(
                    cycle=self.cycle,
                    program=[p.name for p in program],
                    results=results,
                )
                log_verdict(verdict)
                adj = fitness_adjustment(verdict)
                if adj < 1.0:
                    for p in program:
                        penalty = int(p.successes * (1.0 - adj))
                        p.failures += penalty
                    if verdict.concerns:
                        print(f"  witness: {'; '.join(verdict.concerns)}")
            except Exception as e:
                print(f"  witness error (non-fatal): {e}")

        self.codebook.tick()
        if self.cycle > 0 and self.cycle % 50 == 0:
            deaths = self.codebook.natural_selection()
            if deaths:
                print(f"  natural selection: {[d.name for d in deaths]} died")

        self.cycle += 1
        self.traces.append({
            "cycle": self.cycle, "ts": ts.isoformat(),
            "program": [p.name for p in program],
            "results": [{k: v for k, v in r.items() if k != "result"} for r in results],
        })
        if len(self.traces) > 500:
            self.traces = self.traces[-500:]

        self.save()

        names = [p.name for p in program]
        ok = all(r["ok"] for r in results)
        print(f"[{ts.strftime('%H:%M:%S')}] cycle={self.cycle} program={names} ok={ok}")
        return results

    def run(self, interval=1800, once=False):
        self.load()
        print(f"[organism] alive with {len(self.codebook.alive())} primitives")
        print(f"[organism] census: {self.codebook.census()}")

        if once:
            return self.pulse()

        while True:
            try:
                self.pulse()
            except Exception as e:
                print(f"[organism] pulse failed: {e}")
                traceback.print_exc()
            time.sleep(interval)


if __name__ == "__main__":
    org = Organism()
    org.run(once="--once" in sys.argv)
