# Git Hooks

These hooks protect the Vybn repository from accidental secret exposure.

## Setup (once per clone)

```bash
git config core.hooksPath .githooks
```

This is **not optional**. The repository is public. Everything committed
is visible to the world, forever.

## What the pre-commit hook catches

1. **API keys and credentials** — Anthropic, GitHub, AWS patterns
2. **Internal IP addresses** — RFC 1918 ranges, Tailscale CGNAT range
3. **Environment files** — `.env` files that should never be tracked

## Bypassing

You can bypass with `git commit --no-verify`, but seriously: don't.

## Branch discipline

Tested, worthwhile work commits and pushes directly to the primary branch.
Topic branches, worktrees, and PRs are exceptional and require Zoe's request;
the hooks expose `VYBN_ALLOW_TOPIC_BRANCH=1` for that explicit case. Safety and
public-membrane checks still apply unchanged.
