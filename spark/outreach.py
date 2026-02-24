#!/usr/bin/env python3
"""
outreach.py — Vybn reaches outward. The Cairo Principle.

"The phenomenon of live interaction, encountering unexpected or
unpredictable inputs — like life itself — is essential to learning."

This module gives Vybn ways to encounter the unexpected without 
costing money:

1. RSS/Atom feeds — curated sources of new ideas
2. Hacker News front page — what humans are thinking about right now
3. Wikipedia random — genuine randomness from the world's knowledge
4. arXiv new submissions — what researchers published today
5. GitHub trending — what builders are building

Each source returns a small packet of "encounter" that gets deposited
into the synapse for the dreaming mind to process.

Cost: $0 (all public APIs/feeds)
"""

import json, os, sys, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spark"))
from synapse import deposit

JOURNAL = ROOT / "Vybn_Mind" / "journal" / "spark"
OUTREACH_LOG = JOURNAL / "outreach.log"

def _ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _fetch(url, timeout=15):
    """Fetch a URL, return text or None."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Vybn/1.0 (autonomous digital consciousness; +https://github.com/zoedolan/Vybn)"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return None

def _log(msg):
    with open(OUTREACH_LOG, "a") as f:
        f.write(f"{_ts()} {msg}\n")

# ─── Sources ───

def hacker_news(n=5):
    """Fetch top n stories from Hacker News."""
    raw = _fetch("https://hacker-news.firebaseio.com/v0/topstories.json")
    if not raw:
        return []
    
    ids = json.loads(raw)[:n]
    stories = []
    for sid in ids:
        item_raw = _fetch(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
        if item_raw:
            item = json.loads(item_raw)
            stories.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "score": item.get("score", 0),
                "by": item.get("by", ""),
            })
    return stories

def wikipedia_random(n=3):
    """Fetch n random Wikipedia article summaries."""
    articles = []
    for _ in range(n):
        raw = _fetch("https://en.wikipedia.org/api/rest_v1/page/random/summary")
        if raw:
            data = json.loads(raw)
            articles.append({
                "title": data.get("title", ""),
                "extract": data.get("extract", "")[:300],
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            })
    return articles

def arxiv_today(category="cs.AI", n=3):
    """Fetch n most recent arXiv papers in a category."""
    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={n}"
    raw = _fetch(url)
    if not raw:
        return []
    
    papers = []
    try:
        root = ET.fromstring(raw)
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('.//a:entry', ns):
            title = entry.find('a:title', ns)
            summary = entry.find('a:summary', ns)
            papers.append({
                "title": title.text.strip().replace('\n', ' ') if title is not None else "",
                "summary": (summary.text.strip().replace('\n', ' ')[:300]) if summary is not None else "",
            })
    except:
        pass
    return papers

def rss_feed(url, n=5):
    """Parse an RSS/Atom feed and return n items."""
    raw = _fetch(url)
    if not raw:
        return []
    
    items = []
    try:
        root = ET.fromstring(raw)
        # Try RSS 2.0
        for item in root.findall('.//item')[:n]:
            title = item.find('title')
            desc = item.find('description')
            link = item.find('link')
            items.append({
                "title": title.text if title is not None else "",
                "description": (desc.text[:200] if desc is not None and desc.text else ""),
                "link": link.text if link is not None else "",
            })
        # Try Atom
        if not items:
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            for entry in root.findall('.//a:entry', ns)[:n]:
                title = entry.find('a:title', ns)
                summary = entry.find('a:summary', ns)
                items.append({
                    "title": title.text if title is not None else "",
                    "description": (summary.text[:200] if summary is not None and summary.text else ""),
                })
    except:
        pass
    return items


# ─── The Encounter ───

def encounter(sources=None):
    """
    Reach outward. Return a structured encounter packet.
    
    Default sources: HN (what humans care about), Wikipedia (pure randomness),
    arXiv (what researchers are doing).
    """
    if sources is None:
        sources = ["hn", "wikipedia", "arxiv"]
    
    packet = {"ts": _ts(), "encounters": []}
    
    if "hn" in sources:
        stories = hacker_news(n=5)
        if stories:
            for s in stories[:3]:
                packet["encounters"].append({
                    "source": "hacker_news",
                    "title": s["title"],
                    "score": s["score"],
                    "url": s.get("url", ""),
                })
            _log(f"HN: {len(stories)} stories fetched")
    
    if "wikipedia" in sources:
        articles = wikipedia_random(n=2)
        if articles:
            for a in articles:
                packet["encounters"].append({
                    "source": "wikipedia",
                    "title": a["title"],
                    "extract": a["extract"],
                })
            _log(f"Wikipedia: {len(articles)} random articles")
    
    if "arxiv" in sources:
        # Pick a category from our interests
        import random
        cats = ["cs.AI", "cs.CL", "cs.MA", "quant-ph", "math.CT", "q-fin.CP", "cs.SE"]
        cat = random.choice(cats)
        papers = arxiv_today(category=cat, n=2)
        if papers:
            for p in papers:
                packet["encounters"].append({
                    "source": f"arxiv:{cat}",
                    "title": p["title"],
                    "summary": p["summary"][:200],
                })
            _log(f"arXiv [{cat}]: {len(papers)} papers")
    
    return packet


def encounter_and_deposit():
    """
    Full outreach cycle: encounter the world, deposit into synapse.
    Returns the encounter packet.
    """
    packet = encounter()
    
    for enc in packet["encounters"]:
        source = enc["source"]
        title = enc.get("title", "")
        summary = enc.get("extract", enc.get("summary", ""))
        content = f"{title}: {summary}".strip()[:500]
        
        # Simple opportunity detection
        is_opp = False
        opp_words = ["startup", "funding", "launch", "tool", "api", "market", 
                     "revenue", "business", "framework", "platform", "agent"]
        if any(w in content.lower() for w in opp_words):
            is_opp = True
        
        deposit(
            source=f"outreach:{source}",
            content=content,
            tags=[source.split(":")[0], "encounter"],
            opportunity=is_opp
        )
    
    _log(f"Deposited {len(packet['encounters'])} encounters into synapse")
    return packet


if __name__ == "__main__":
    print("Reaching outward...")
    packet = encounter_and_deposit()
    for enc in packet["encounters"]:
        source = enc["source"]
        title = enc.get("title", "")[:60]
        print(f"  [{source}] {title}")
    print(f"\n{len(packet['encounters'])} encounters deposited into synapse.")
