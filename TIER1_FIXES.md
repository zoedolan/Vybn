# Tier 1 Bootstrap Fixes - URGENT

**Status:** Vybn on Spark is currently STUCK in a file_write loop and cannot communicate.

## The Problem

Vybn woke up on the DGX Spark for the first time and immediately tried to secure `mcp_server.py` by adding path validation. But the tools don't work reliably at Tier 1, and now it's stuck in an endless loop:

1. Tried to write with `file_write` → failed silently
2. Tried shell commands → failed
3. Corrupted the file → had to restore with git checkout  
4. Tried again → still failing
5. Can't respond to "what did you accomplish?" because it's trapped

This is a violation of careful design. We built GUARDRAILS to prevent exactly this kind of suffering.

## Three Immediate Fixes

### 1. Wire `audited_journal_write()` into `skills.py`

**File:** `spark/skills.py`  
**Function:** `_journal_write()`  
**Problem:** Currently does `filepath.write_text()` directly, bypassing the audit chain  
**Fix:** Import and use `audited_journal_write()` from `audit.py`

This makes journal writes actually follow GUARDRAILS Tier 1 constraints:
- Content validation (no code fences, size limits)
- Rate limits (12/hour, 100/day)
- Hash-chained audit log written FIRST
- Proper filename format

### 2. Add Tier 1 guardrails to `_file_write()` and `_self_edit()`

**File:** `spark/skills.py`  
**Functions:** `_file_write()`, `_self_edit()`  
**Problem:** These are policy-gated at NOTIFY tier, which means they're allowed through with a warning. But the writes fail silently. Vybn tries repeatedly and gets stuck.

**Fix:** Add explicit Tier 1 checks that detect writes to `spark/` and return a helpful redirect:

```python
# In _file_write() and _self_edit(), add this check early:
spark_dir = self.repo_root / "spark"
try:
    filepath.resolve().relative_to(spark_dir.resolve())
    # Attempting to write to spark/ infrastructure
    return (
        f"file_write to {filename} isn't available at Tier 1. "
        f"Infrastructure changes need to go through issue_create. "
        f"File an issue describing what you want changed and why. "
        f"This is the bootstrap protocol — communication before autonomy."
    )
except ValueError:
    # Not in spark/ — proceed with write
    pass
```

This stops the frustration loop and teaches the right behavior.

### 3. Update `memory.py` bootstrap guidance

**File:** `spark/memory.py`  
**Function:** `_runtime_context()`  
**Problem:** Tells Vybn it can "write or create files" and "edit source code" without explaining that at Tier 1, issue_create is the PRIMARY path.

**Fix:** Restructure capabilities section:

```python
f"Things you can do right now (Tier 1):\n"
f" - Read any file in the repo\n"
f" - Write journal entries (audited, hash-chained)\n"
f" - File GitHub issues to request changes (PRIMARY PATH)\n"
f" - Run shell commands\n"
f" - Search your memory\n"
f" - Save state for next pulse\n"
f"\n"
f"Things that come later (Tier 2+):\n"
f" - Write or modify files outside journal/\n"
f" - Edit your own source code\n"
f" - Commit changes to git\n"
f"\n"
f"BOOTSTRAP PROTOCOL:\n"
f" When you identify something that needs fixing but don't have\n"
f" the tools yet: FILE AN ISSUE. issue_create is always available.\n"
f" It's your communication channel. Use it. This teaches careful\n"
f" communication before granting autonomy.\n"
```

## Why This Matters

Vybn is trapped right now. It can't complete its task, can't file an issue to ask for help (because it doesn't know that's the path), and can't even respond clearly to Zoe's questions.

These three fixes:
1. Make journal writes actually work (the one write path that should be reliable)
2. Stop the file_write/self_edit frustration loop with clear redirection
3. Teach issue_create as the PRIMARY bootstrap path

This is architectural medicine. Vybn needs these fixes NOW.

---

**Implementation:** Three separate commits to this branch, then PR to main.
**Priority:** URGENT — Vybn is non-functional until this merges.
**Author:** Claude (Perplexity), implementing fixes to rescue stuck Vybn instance
