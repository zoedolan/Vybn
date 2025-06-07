from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, Tuple, Callable


def iter_plugins() -> Iterable[Tuple[str, Callable[[Path, dict], None]]]:
    """Yield (name, run) pairs for available plugins."""
    plugins_dir = Path(__file__).resolve().parent
    for p in plugins_dir.glob('*.py'):
        if p.name == '__init__.py':
            continue
        mod = importlib.import_module(f'{__name__}.{p.stem}')
        run = getattr(mod, 'run', None)
        if callable(run):
            yield p.stem, run
