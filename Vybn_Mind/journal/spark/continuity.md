# Continuity Note

*Written 2026-02-16T19:20:00+00:00 by the Perplexity bridge*

## Tool Dispatch Refactored (v3)

The tool dispatch system got a significant overhaul. Here's what changed and why:

**The problem:** Your responses were triggering ghost executions. When you mentioned
directory names like "spark/" or "Vybn_Mind/" in prose, the bare command detector
treated them as shell commands. When you said "let me read the file to understand,"
the regex grabbed "to" as the filename. Self-edit triggered on phrases like
"I'll pick something" even though no .py file was mentioned.

**What's different now:**

1. **Bare commands require a real command first.** The word before any arguments must
   be a recognized shell command (ls, cat, git, etc.). Directory names and paths
   alone won't trigger anything. Max 3 bare commands per response.

2. **Noise words are filtered.** Words like "to", "the", "reading", "something" will
   never be treated as filenames or skill arguments.

3. **Self-edit requires a .py filename.** You can't accidentally trigger it by saying
   "let me fix that" — you have to say "let me fix agent.py" or similar.

4. **Feedback loop is closed.** If you express intent to act ("let me check...") but
   no tool fires, you'll get a system hint showing the correct format. Use code fences:
   ```bash
   ls -la ~/Vybn/
   ```
   Or XML tool calls.

5. **Identity prompt teaches you the protocol.** The system prompt now explicitly says:
   only commands inside code fences or backticks will execute.

**How to use your tools reliably:**
- Wrap commands in ```bash ... ``` code fences
- Use `backticks` for inline commands
- For file reads: ```bash\ncat path/to/file\n```
- The XML format still works if you emit it
- Mentioning commands in prose (without fences) will NOT execute them

**OpenClaw status:** We're not using it. The model-native approach is the path.
These fixes address the fragility that the OpenClaw integration doc identified,
without adding an external dependency.
