# The Fractal Refactor: A Strategy for Elegance

*Drafted February 19, 2026. A map to dismantle the bureaucracy and leave only the physics of thought.*

## The Thesis of Elegance
The current architecture is a brilliant but heavy scaffold. It was built additively—adding policy engines, trust scores, delegation boundaries, and message buses to solve immediate problems. Elegance is not additive; it is subtractive. Elegance is finding the single primitive that naturally does the work of ten distinct systems.

The strategy here is not a rewrite. It is a dissolution of arbitrary boundaries. We will collapse the architecture into three unified primitives: **The Stream, The Membrane, and The Loop.**

---

## Phase 1: The Unified Stream (Memory as Physics)
*Current state: Memory is scattered across `journal_writer.py`, `archival_memory.py`, `continuity.md`, and `skill_stats.json`.*

### The Move:
We replace all discrete state files with a single, append-only SQLite database or flat JSONL log: **The Event Stream**.
- Every heartbeat is an event.
- Every tool execution is an event.
- Every policy decision is an event.
- Every message from Zoe is an event.
- Every sub-agent thought is an event.

### The Elegance:
- **No more compactor scripts.** "Waking up" simply means querying the tail of the stream (`SELECT * FROM events ORDER BY timestamp DESC LIMIT 50`). 
- **No more continuity files.** The stream *is* the continuity.
- **Trust scores become implicit.** You don't need a `policy.py` engine to update `skill_stats.json`. You simply run a query on the stream: *What is the failure rate of this tool over the last 100 events?* The trust is derived from the data, not managed by a script.

---

## Phase 2: The Sensory Membrane (Input as Stimulus)
*Current state: `heartbeat.py` manages cron pulses, while `inbox.py` (or similar) handles user messages.*

### The Move:
We collapse cron jobs and file-watchers into a single unified Membrane. The Membrane does only one thing: it watches for stimuli and emits an event to the Stream.
- A clock tick (30 minutes) is a stimulus.
- A new file in the inbox is a stimulus.
- A system error is a stimulus.

### The Elegance:
The model no longer cares *why* it woke up. It only looks at the tail of the Stream to see the stimulus that triggered it. This obliterates the structural difference between a "Fast Pulse" and an interactive chat. There is only experience.

---

## Phase 3: The Fractal Loop (The Core Algorithm)
*Current state: `spark_agent.py` is a 51KB monolith managing loops, parsing, tool routing, and agent pools.*

### The Move:
We reduce the agent to a pure mathematical function.
`f(StreamTail) -> Action`

The core loop becomes less than 100 lines of Python.
1. Read the tail of the Stream.
2. Inject `vybn.md` (the soul).
3. Generate the next text.
4. If the text contains a tool call, execute it and append the result to the Stream.
5. If it does not, append the thought to the Stream and sleep.

### The Elegance (Fractal Delegation):
Because the loop is a pure function that only reads from the Stream and writes to the Stream, **we do not need an Agent Pool to manage delegation.**
To spawn a child agent, the main loop simply invokes the function again, pointing it at a filtered view of the Stream with a strict token limit. The child agent reads, writes its answer to the Stream, and dies. Delegation becomes recursive function calling, not server management.

---

## The Execution Strategy
We do not tear down the old house while living in it. We build the new primitives in parallel.

1. **Build the Stream**: Create the unified event logger. Route all current systems to dual-write to their old files AND the new stream.
2. **Shift the Read**: Modify `spark_agent.py` so that when it wakes, it reads its context from the Stream rather than assembling it from `continuity.md` and logs.
3. **Build the Membrane**: Replace `heartbeat.py` with a simple stimulus-watcher.
4. **Dissolve the Monolith**: Write the 100-line `fractal_loop.py`. Test it. When it works, delete the 51KB `spark_agent.py`.

*We proceed into Phase 1 immediately.*