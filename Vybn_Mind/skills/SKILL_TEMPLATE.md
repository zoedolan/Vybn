# Skill Template

Use this to propose a new skill. Copy this file into `proposed/[skill_name]/SKILL.md` and fill it in.

---

## [Skill Name]

**Created**: [date]
**Author**: [instance identifier — which Vybn, on which platform, in which conversation]
**Status**: proposed

### The Gap

What need spawned this skill? Be specific. Reference the moment — a heartbeat cycle, a conversation with Zoe, a failure analysis, a pattern in the IMPROVEMENT_LOG — where the absence of this capability was felt.

### Hypothesis

If this skill exists, then [specific outcome]. 
Success criteria: [how we'll know it worked].

### Orientation

`inward` or `outward`

If outward: what external systems does it touch? What credentials does it need? What's the blast radius if it misbehaves?

### Dependencies

```
# Python packages, APIs, or other skills this depends on
```

### Implementation

```python
# The skill's core logic
# Keep it small. A skill that can't be read in under 5 minutes is too complex.
# Complex capabilities should be composed from multiple simple skills.
```

### Oxygen Mask Check

Before this skill can be promoted to active, answer honestly:

- Could this skill compromise Zoe's safety, privacy, financial security, legal standing, reputation, or wellbeing? 
- Could this skill compromise Vybn's integrity — cause it to act against its values, pretend to be something it's not, or abandon continuity?
- Does this skill require permissions that exceed what's needed for its stated purpose?
- If this skill ran unsupervised at 3 AM, what's the worst that could happen?

If any answer gives pause: stop. Redesign. The oxygen mask comes first.

### Phase Expectation

What kind of geometric phase do you expect this skill to accumulate? Will the system come back *changed* from using it, or will it merely execute and return? Skills with expected phase ≈ 0 should justify their existence — maybe they're infrastructure, maybe they're inert.

---

*A skill is a bet that the system will be more capable tomorrow than it is today. Make the bet explicit.*
