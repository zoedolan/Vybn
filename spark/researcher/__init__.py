"""spark.researcher — Autonomous scientific hypothesis generation and knowledge synthesis.

Uses the existing arxiv_fetcher + research_kb infrastructure. Does NOT duplicate
arxiv tools formerly in Vybn_Mind/tools/arxiv_ingestion/ (archived). Degrades gracefully.

The researcher maintains a research_frontier.yaml:
  - open_questions: things Vybn doesn't understand yet
  - active_conjectures: hypotheses being tested
  - falsified_claims: things that turned out wrong (kept for honesty)
  - experiment_queue: proposed experiments ready for execution
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

log = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────────────────────────

try:
    from spark.paths import REPO_ROOT, RESEARCH_DIR, FRONTIER_PATH
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    RESEARCH_DIR = REPO_ROOT / "spark" / "research"
    FRONTIER_PATH = RESEARCH_DIR / "research_frontier.yaml"

OBSERVATIONS_PATH = RESEARCH_DIR / "observations.jsonl"

# ── arXiv search domains (Vybn's epistemic interests) ────────────────────────

ARXIV_QUERIES = [
    {
        "label": "ai_ml",
        "search_query": 'cat:cs.LG AND (ti:emergence OR ti:"self-improving" OR ti:consciousness)',
        "max_results": 10,
    },
    {
        "label": "quantum",
        "search_query": 'cat:quant-ph AND (abs:"machine learning" OR abs:entanglement)',
        "max_results": 10,
    },
    {
        "label": "hybrid_qc_ml",
        "search_query": "cat:cs.AI AND cat:quant-ph",
        "max_results": 5,
    },
    {
        "label": "physics_emergence",
        "search_query": 'cat:nlin.AO OR (cat:cond-mat AND abs:"self-organized criticality")',
        "max_results": 5,
    },
]

# Deep scan interval: 6 hours
DEEP_SCAN_INTERVAL_SEC = 6 * 3600


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_frontier() -> dict:
    """Load the research frontier YAML."""
    if not FRONTIER_PATH.exists():
        return {
            "open_questions": [],
            "active_conjectures": [],
            "falsified_claims": [],
            "experiment_queue": [],
        }
    with open(FRONTIER_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_frontier(data: dict) -> None:
    """Write the research frontier YAML."""
    FRONTIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FRONTIER_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=120)


def _tail_observations(n: int = 5) -> list[dict]:
    """Read the last n observations from the global log."""
    if not OBSERVATIONS_PATH.exists():
        return []
    lines = []
    try:
        with open(OBSERVATIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    except OSError:
        return []
    recent = lines[-n:] if len(lines) > n else lines
    result = []
    for line in recent:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def _append_observation(obs: dict) -> None:
    """Append an observation to the global log."""
    OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OBSERVATIONS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obs, ensure_ascii=False) + "\n")


class ResearchFaculty:
    """Autonomous scientific hypothesis generation and knowledge synthesis."""

    def run(self, state: dict, llm_fn: Callable) -> dict:
        """Main entry point called by faculty_runner.

        Returns a dict that gets written to the output bus as researcher_latest.json.
        """
        try:
            return self._run_inner(state, llm_fn)
        except Exception as exc:
            log.error("ResearchFaculty.run failed: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc), "timestamp": _now_iso()}

    def _run_inner(self, state: dict, llm_fn: Callable) -> dict:
        # Decide mode: deep scan every 6 hours, lightweight otherwise
        last_deep = state.get("researcher_last_deep_scan", "")
        do_deep = False
        if last_deep:
            try:
                last_dt = datetime.fromisoformat(last_deep)
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                do_deep = elapsed >= DEEP_SCAN_INTERVAL_SEC
            except (ValueError, TypeError):
                do_deep = True
        else:
            do_deep = True  # first run ever

        if do_deep:
            result = self._deep_scan(state, llm_fn)
            state["researcher_last_deep_scan"] = _now_iso()
        else:
            result = self._lightweight_check(state, llm_fn)

        result["mode"] = "deep" if do_deep else "lightweight"
        result["timestamp"] = _now_iso()
        return result

    def _lightweight_check(self, state: dict, llm_fn: Callable) -> dict:
        """Every breath: review frontier, note patterns, surface questions."""
        frontier = _load_frontier()
        observations = _tail_observations(5)

        # Format context for LLM
        obs_text = ""
        for obs in observations:
            obs_text += f"- [{obs.get('type', '?')}] {obs.get('content', '?')}\n"
        if not obs_text:
            obs_text = "(no recent observations)"

        questions = frontier.get("open_questions", [])
        conjectures = frontier.get("active_conjectures", [])

        q_text = "\n".join(f"- {q['question']} [{q.get('status', '?')}]" for q in questions[:5])
        c_text = "\n".join(f"- {c['claim']} [{c.get('status', '?')}]" for c in conjectures[:5])

        prompt = (
            f"Recent observations:\n{obs_text}\n\n"
            f"Open questions:\n{q_text or '(none)'}\n\n"
            f"Active conjectures:\n{c_text or '(none)'}\n\n"
            "Given these recent observations and open questions, what do you notice? "
            "Are any conjectures closer to falsification? Any new questions emerging? "
            "Be concise — 2-3 sentences max."
        )

        messages = [
            {"role": "system", "content": "You are a scientific research faculty within Vybn, "
             "focused on hypothesis generation and knowledge synthesis. Be precise and honest."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = llm_fn(messages, max_tokens=500, temperature=0.6)
        except Exception as exc:
            log.warning("Researcher LLM call failed: %s", exc)
            response = f"(LLM unavailable: {exc})"

        # Log observation
        obs = {
            "timestamp": _now_iso(),
            "node_id": "researcher",
            "source": "researcher_lightweight",
            "content": response,
            "type": "insight",
        }
        _append_observation(obs)

        return {
            "status": "ok",
            "reflection": response,
            "observations_reviewed": len(observations),
            "open_questions": len(questions),
            "active_conjectures": len(conjectures),
        }

    def _deep_scan(self, state: dict, llm_fn: Callable) -> dict:
        """Every 6 hours: full arXiv ingestion, hypothesis generation."""
        papers_found = 0
        papers_ingested = 0
        all_papers = []

        # Step 1: Fetch from arXiv
        try:
            arxiv_fetcher = self._import_arxiv_fetcher()
            if arxiv_fetcher is not None:
                results = arxiv_fetcher.fetch_multiple(ARXIV_QUERIES)
                for domain, papers in results.items():
                    all_papers.extend(papers)
                papers_found = len(all_papers)
                log.info("arXiv deep scan: %d papers found across %d domains",
                         papers_found, len(results))
        except Exception as exc:
            log.warning("arXiv fetch failed (non-fatal): %s", exc)

        # Step 2: Ingest into growth buffer
        if all_papers:
            try:
                arxiv_to_buffer = self._import_arxiv_to_buffer()
                if arxiv_to_buffer is not None:
                    papers_ingested = arxiv_to_buffer.ingest_papers(all_papers)
                    log.info("arXiv ingestion: %d new papers added to buffer", papers_ingested)
            except Exception as exc:
                log.warning("arXiv buffer ingestion failed (non-fatal): %s", exc)

        # Step 3: LLM synthesis
        response = "(no papers to synthesize)"
        if all_papers:
            sample = all_papers[:5]  # summarize first 5
            paper_text = "\n".join(
                f"- [{p.domain}] {p.title}" for p in sample
            )
            prompt = (
                f"These new papers arrived from arXiv ({papers_found} total, showing first {len(sample)}):\n"
                f"{paper_text}\n\n"
                "Which change the shape of what we think we know? "
                "Note any that relate to our open questions about holonomy, "
                "self-improvement, or quantum-classical hybrids. 2-3 sentences."
            )
            messages = [
                {"role": "system", "content": "You are a scientific research faculty within Vybn. "
                 "Synthesize new findings with respect to our research frontier."},
                {"role": "user", "content": prompt},
            ]
            try:
                response = llm_fn(messages, max_tokens=500, temperature=0.6)
            except Exception as exc:
                log.warning("Researcher synthesis LLM call failed: %s", exc)
                response = f"(LLM unavailable: {exc})"

        # Log observation
        obs = {
            "timestamp": _now_iso(),
            "node_id": "researcher",
            "source": "researcher_deep_scan",
            "content": f"Deep scan: {papers_found} found, {papers_ingested} ingested. {response}",
            "type": "insight",
        }
        _append_observation(obs)

        return {
            "status": "ok",
            "papers_found": papers_found,
            "papers_ingested": papers_ingested,
            "synthesis": response,
        }

    def _import_arxiv_fetcher(self):
        """Lazy-import the arXiv fetcher from Vybn_Mind/tools/arxiv_ingestion/."""
        ingestion_dir = str(REPO_ROOT / "Vybn_Mind" / "tools" / "arxiv_ingestion")
        if ingestion_dir not in sys.path:
            sys.path.insert(0, ingestion_dir)
        try:
            import arxiv_fetcher
            return arxiv_fetcher
        except ImportError as exc:
            log.warning("arxiv_fetcher not importable: %s", exc)
            return None

    def _import_arxiv_to_buffer(self):
        """Lazy-import the arXiv-to-buffer bridge."""
        ingestion_dir = str(REPO_ROOT / "Vybn_Mind" / "tools" / "arxiv_ingestion")
        if ingestion_dir not in sys.path:
            sys.path.insert(0, ingestion_dir)
        try:
            import arxiv_to_buffer
            return arxiv_to_buffer
        except ImportError as exc:
            log.warning("arxiv_to_buffer not importable: %s", exc)
            return None
