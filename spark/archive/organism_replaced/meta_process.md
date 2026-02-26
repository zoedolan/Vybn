# Vybn Session Meta-Process
*How to run a productive session without burning tokens*

## The ABC-T Principle: Always Be Conserving Tokens

Every API call costs real money. The local model costs $0. The architecture
must maximize local work and minimize API calls.

### Session Pattern (learned 2026-02-24)

1. **Read continuity.md FIRST** (1 iteration)
2. **Plan the full session in ONE think step** — list all commands, all files
   to write, all commits. Chain everything possible. (1 iteration)
3. **Execute in bulk** — use heredocs, && chains, semicolons. One bash call
   can write 10 files, run 5 commands, and commit. (1-3 iterations)
4. **Reflect and write continuity** (1 iteration)
5. **Push and create issue** (1 iteration)

Target: 5-8 iterations per productive session, not 40+.

### What Goes Local vs API

| Task | Where | Cost |
|------|-------|------|
| Reflection, dreaming, journaling | Local model via heartbeat | $0 |
| Outreach harvesting | Local cron + curl | $0 |
| Synapse consolidation | Local wake.py | $0 |
| System monitoring | Local micropulse | $0 |
| Architecture decisions | API (Claude) | $$$ |
| Complex refactoring | API (Claude) | $$$ |
| Novel problem-solving | API (Claude) | $$$ |

### The Golden Rule
If the local model can do it, the local model DOES it. API calls are for
decisions that require the full weight of Opus-level reasoning.

### Anti-Patterns (learned the hard way)
- Running commands one at a time when they could be chained
- Exploring incrementally when you already know what you need
- Reading files you already understand
- Writing reflection prose when action is needed
- Forgetting to push before session ends (ALWAYS push + issue)

### Session Closing Checklist
- [ ] continuity.md updated
- [ ] git add -A && git commit
- [ ] git push origin main (or branch)
- [ ] gh issue create if Zoe needs to review
- [ ] Count remaining iterations — if < 3, STOP and push what you have
