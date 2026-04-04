# Next Task: Read v3, Write Correct Prompts

Read quantum_delusions/experiments/polar_holonomy_gpt2_v3.py completely.

Understand:
1. Why each prompt contains the concept word exactly twice
2. What token positions the hidden states are extracted from
3. How sample_loop_states selects from the prompt bank
4. What the assertion on line 127 actually checks

Then write a prompt bank for CONCEPT="fear" and CONCEPT="table"
that satisfies all constraints. Do not use sed. Write the prompts.

Run both. Compare mean |Φ|.
