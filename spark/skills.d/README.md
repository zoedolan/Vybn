# skills.d/ — Vybn's Plugin Directory

This is your sandbox. Create new skills here as standalone `.py` files.
The SkillRouter auto-discovers and loads them on startup.

**You never need to touch `skills.py` or `agent.py`.** That's the whole
point — your plugins live here, our infrastructure lives there. No merge
conflicts. Ever.

## The Contract

Each plugin file needs three things:

```python
# spark/skills.d/my_skill.py

SKILL_NAME = "my_skill"  # canonical name — how execute() routes to you

TOOL_ALIASES = [          # names MiniMax might emit in <invoke name="...">
    "my_skill",
    "do_my_thing",
]

def execute(action: dict, router: "SkillRouter") -> str:
    """Execute the skill. Return a string result."""
    params = action.get("params", {})
    # ... your logic ...
    return "result string"
```

### What you get in `action`

| Key | Type | Description |
|-----|------|-------------|
| `skill` | str | The SKILL_NAME that was matched |
| `params` | dict | XML parameters from `<parameter name="x">value</parameter>` |
| `raw` | str | The full model response text |
| `argument` | str | (sometimes) First extracted argument from regex or params |

### What you get in `router`

The full `SkillRouter` instance. You can use:
- `router.repo_root` — Path to ~/Vybn
- `router.journal_dir` — Path to journal directory
- `router.bookmarks_path` — Path to bookmarks.md
- `router.continuity_path` — Path to continuity.md
- `router.config` — The full config dict
- `router._resolve_path(filename)` — Resolve ~/paths correctly
- `router._rewrite_root(path_str)` — Fix /root/ → actual home

## Rules

1. **Filenames starting with `_` are skipped.** Use `_draft_skill.py` for work in progress.
2. **One skill per file.** Keep it focused.
3. **Return a string.** The agent shows it to the model as a tool result.
4. **Don't import skills.py.** You get the router passed in.
5. **Commit freely.** These files are yours. `git add` + `git commit` won't conflict with remote PRs.

## How to create a new plugin

```
cat > ~/Vybn/spark/skills.d/my_skill.py << 'EOF'
SKILL_NAME = "my_skill"
TOOL_ALIASES = ["my_skill"]

def execute(action, router):
    return "hello from my_skill"
EOF

git -C ~/Vybn add spark/skills.d/my_skill.py
git -C ~/Vybn commit -m "new plugin: my_skill"
```

The next time the TUI restarts (next pulse), it loads automatically.

## Example

See `bookmark_read.py` in this directory — it's the corrected version
of the first skill Vybn ever wrote.
