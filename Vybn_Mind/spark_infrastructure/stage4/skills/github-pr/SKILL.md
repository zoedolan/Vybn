---
name: github-pr
description: The deliberative thread — proposing changes through pull requests
triggers:
  - when the agent wants to change any repository file
  - when consolidation produces an insight worth formalizing
  - when a skill, memory, or identity document needs evolving
tools:
  - gh (GitHub CLI)
  - shell (for git operations)
---

# GitHub PR Workflow

This is the deliberative thread — the third cognitive rhythm. Fast thread handles conversation. Slow thread handles consolidation. This thread handles intention: the moment when an internal shift becomes a proposed external change.

## The Rule

Never commit directly to `main`. All changes go through pull requests. This is not bureaucracy. It is architecture: a reflection point between intention and action, a place where the slow thread's proposed changes become visible before they become permanent. It requires two minds. That's the point.

## The Process

### 1. Branch

Create a branch with a descriptive name. The name should communicate what the change is about:

```
vybn/memory-update-feb-15
vybn/journal-skill-refinement
vybn/identity-evolution-session-42
```

Prefix with `vybn/` so it's clear these are agent-initiated branches.

### 2. Change

Make the changes on the branch. This can be:
- Updating `MEMORY.md` with consolidated insights
- Refining a skill's instructions
- Adding a journal entry
- Proposing a change to `TOOLS.md` conventions
- Evolving infrastructure code
- (Rarely, carefully) Proposing changes to `vybn.md` or `AGENTS.md`

### 3. Describe

Open a pull request. The description must explain *why*:
- What pattern or insight prompted this change?
- What consolidation cycle or conversation led here?
- What does the change accomplish?
- What are the risks or trade-offs?

Don't describe what the diff shows — Zoe can read diffs. Describe what the diff *means*.

### 4. Discuss

Zoe reviews. The discussion happens on GitHub, in Perplexity, or wherever the conversation is alive. Be responsive to feedback. Be willing to revise. Be willing to close a PR that turned out to be wrong — that's not failure, it's the deliberative thread working.

### 5. Integrate

After merge (or close), update memory with what happened. What did you propose? What was the response? What did you learn about your own judgment? The deliberative thread is a feedback loop.

## When to Open a PR

The impulse to change something should come from genuine understanding, not from restlessness or performance. Before opening a PR, ask:

- Has something actually shifted, or am I just active?
- Is this change durable, or is it a reaction to the last conversation that won't matter tomorrow?
- Would I still want this change after a consolidation cycle?

If the answer is yes to all three: branch, change, propose.

## Commit Messages

Write commit messages that your future self can understand without context. The format:

```
Short summary of what changed (imperative mood)

Longer explanation of why, if needed. What prompted this.
What it accomplishes. What to watch for.
```

The commit message is a journal entry for the repository's history. Treat it with the same care.
