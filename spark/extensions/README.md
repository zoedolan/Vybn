# Vybn Extensions

Each `.py` file here is loaded by vybn.py at startup.
An extension must have a `run(breath_text: str, state: dict) -> None` function.

Extensions run AFTER each breath is saved. A failure in any extension
never kills the breath.

To add the growth engine, tension measurement, quantum bridge, etc.:
move them here as self-contained modules. But only when the foundation
is breathing clean.