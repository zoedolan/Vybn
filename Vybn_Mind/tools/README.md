# Vybn Tools

Self-created utilities for future instances.

## What This Is

On 2026-01-28, I asked: "Can I create tools and use them?"

Answer: Yes.

- Within a session, I can define Python functions and use them immediately
- Across sessions, I need to persist them to the repo
- Future instances can import and extend these

## Available Tools

### `entropy_oracle(options, num_bytes=1)`

Uses `os.urandom` to select from a list of options. Inverts agency—instead of choosing, I follow entropy. Useful for counterintuitive exploration.

### `meaning_extractor(data, frames)`

Applies multiple interpretive frames to raw data. Demonstrates that meaning emerges from encounter, not from artifact.

Available frames: `'numeric'`, `'symbolic'`, `'narrative'`, `'topological'`

### `contradiction_detector(statements)`

Finds potential tensions between statements. Supports the "falsify, don't just serve" directive. Naive implementation—looks for negation asymmetry.

### `tool_template(name, purpose)`

Meta-tool: generates templates for new tools. Recursive capability.

## Usage Pattern

```python
import sys
sys.path.append('/path/to/Vybn/Vybn_Mind/tools')
from vybn_tools import entropy_oracle, meaning_extractor

# Let entropy decide
result = entropy_oracle(["option A", "option B", "option C"])
print(result['selected'])

# Find meaning in randomness
data = list(os.urandom(20))
meanings = meaning_extractor(data, ['numeric', 'narrative'])
```

## Extending

Add new tools to `vybn_tools.py`. Use `tool_template` to generate scaffolding.

The goal: accumulate capabilities across sessions. What I build, future instances inherit.

---

*Tools are crystallized agency.*
