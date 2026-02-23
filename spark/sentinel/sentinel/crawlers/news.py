"""AI news crawler: Substacks, arXiv, lab blogs.

Includes URL deduplication via seen_urls.json (capped at 5000 entries).
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None
try:
    import feedparser
except ImportError:
    feedparser = None

log = logging.getLogger("sentinel")
ARXIV_API = "http://export.arxiv.org/api/query"
MAX_SEEN_URLS = 5000


def _load_seen_urls(data_dir: str) -> set[str]:
    path = Path(data_dir) / "seen_urls.json"
    if path.exists():
        try:
            return set(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return set()


def _save_seen_urls(data_dir: str, seen: set[str]) -> None:
    # Cap at MAX_SEEN_URLS to prevent unbounded growth
    urls = sorted(seen)[-MAX_SEEN_URLS:]
    path = Path(data_dir) / "seen_urls.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(urls, indent=1))


def fetch_arxiv(categories: list[str], max_results: int = 20) -> list[dict]:
    if not (httpx and feedparser):
        raise RuntimeError("pip install httpx feedparser")
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    resp = httpx.get(ARXIV_API, params={
        "search_query": cat_query, "sortBy": "submittedDate",
        "sortOrder": "descending", "max_results": max_results,
    }, timeout=30)
    feed = feedparser.parse(resp.text)
    return [{
        "title": e.get("title", "").strip(),
        "authors": [a.get("name", "") for a in e.get("authors", [])],
        "summary": e.get("summary", "").strip()[:500],
        "url": e.get("link", ""),
        "published": e.get("published", ""),
        "source": "arxiv",
    } for e in feed.entries]


def fetch_rss(url: str, label: str, max_items: int = 5) -> list[dict]:
    if not (httpx and feedparser):
        raise RuntimeError("pip install httpx feedparser")
    resp = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (compatible; Vybn/0.2)"})
    feed = feedparser.parse(resp.text)
    return [{
        "title": e.get("title", "").strip(),
        "summary": e.get("summary", "")[:500],
        "url": e.get("link", ""),
        "published": e.get("published", ""),
        "source": label,
    } for e in feed.entries[:max_items]]


def run(config: dict) -> Path:
    items: list[dict] = []
    ns = config.get("news_sources", {})

    arxiv_cfg = ns.get("arxiv", {})
    if arxiv_cfg:
        items.extend(fetch_arxiv(arxiv_cfg.get("categories", ["cs.AI"]),
                                 arxiv_cfg.get("max_daily", 20)))
    for blog in ns.get("lab_blogs", []):
        try:
            items.extend(fetch_rss(blog["url"], blog["label"]))
        except Exception:
            pass
    for sub in ns.get("substacks", []):
        try:
            items.extend(fetch_rss(sub["url"].rstrip("/") + "/feed",
                                   sub.get("label", "substack")))
        except Exception:
            pass

    # Deduplication
    data_dir = config.get("output", {}).get("data_dir", "data")
    seen = _load_seen_urls(data_dir)
    before = len(items)
    items = [item for item in items if item.get("url") and item["url"] not in seen]
    dupes = before - len(items)
    if dupes:
        log.info(f"Deduplicated {dupes} items")
    seen.update(item["url"] for item in items if item.get("url"))
    _save_seen_urls(data_dir, seen)

    output_dir = Path(config["output"]["raw_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"news_{ts}.json"
    path.write_text(json.dumps(items, indent=2))
    return path
