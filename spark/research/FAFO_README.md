# FAFO Research Directory

This directory is managed by `spark/fafo.py` — the **Fuck Around and Find Out** engine.

## Files

`surprises.jsonl` — append-only log of detected surprises. Each line is a JSON object:
```json
{
  "id": "s-20260315-001",
  "timestamp": "2026-03-15T10:42:00Z",
  "source": "complexify_bridge",
  "signal": "curvature_spike",
  "magnitude": 0.34,
  "context": "κ jumped from 0.08 to 0.34 after ingesting recalibration_march13.md",
  "resolved": false,
  "investigation_id": null
}
```

`investigations.yaml` — active and resolved investigations. Each investigation is a directed plan to resolve a surprise, with steps assigned to existing faculties.

## Surprise Sources

| Source | Signal | Condition |
|--------|--------|-----------|
| `complexify_bridge` | `curvature_spike` | κ > mean + 2σ |
| `quantum_bridge` | `quantum_deviation` | TVD > 0.10 on real circuit |
| `mathematician` | `conjecture_update` | Conjecture status changed |
| `witness` | `witness_concern` | Fidelity score < 0.70 |
| `growth` | `growth_event` | Fine-tuning threshold crossed |

## What FAFO Does Not Do

- Does not modify governance, soul constraints, or the oxygen-mask files
- Does not create faculties or expand scope beyond this directory and `faculties.d/outputs/`
- Does not bypass the write custodian
- Does not touch the complexify equation

## Integration Points

In `vybn.py`, after complexify + ingester:
```python
if FAFO_AVAILABLE:
    try:
        from spark.fafo import register_surprises
        register_surprises(state, geo_report, ingest_report, llm_fn=_chat)
    except Exception as exc:
        _log(f"fafo error (non-fatal): {exc}")
```

In `faculty_runner.py`, for `on_trigger` faculties:
```python
if cadence == 'on_trigger':
    try:
        from spark.fafo import get_next_action
        action = get_next_action(card.faculty_id)
        if action:
            state['_fafo_context'] = action
            return True
    except ImportError:
        pass
    return False
```

## The Loop

```
notice something surprising
  → formulate what would resolve the surprise
    → do the thing (assign to faculty)
      → observe what happened
        → update what you expected
          → notice the next surprise
```

The drive to fuck around and find out is not a feature to be added. It's a loop to be closed.
