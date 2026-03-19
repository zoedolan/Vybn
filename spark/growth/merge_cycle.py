"""spark.growth.merge_cycle — Merge and serve for the recursive growth engine.

Phase 6 (BECOME) of the growth cycle described in issue #2483.

The model wakes up slightly different. The next pulse reflects on
that difference. The loop closes.

Strategy: LoRA adapter → GGUF conversion → hot-load into llama-server
via POST /lora-adapters. Takes effect immediately, no restart needed.

The serving model is llama-server (llama.cpp) with the Nemotron IQ4_XS
GGUF. LoRA adapters trained by peft_train.py are converted to GGUF
format using llama.cpp's convert_lora_to_gguf.py, then hot-loaded
into the running server.

Integration points:
  - Input: trained adapter from TrainCycle (PEFT .safetensors)
  - GGUF conversion via llama.cpp convert_lora_to_gguf.py
  - Hot-load via POST /lora-adapters to llama-server
  - Cycle history: GROWTH_DIR / "cycle_history.jsonl"
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

GROWTH_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = GROWTH_DIR / "growth_config.yaml"
CYCLE_HISTORY = GROWTH_DIR / "cycle_history.jsonl"

# llama.cpp paths
_LLAMA_CPP_DIR = Path.home() / "llama.cpp"
_GGUF_BASE_DIR = Path.home() / "models" / "Nemotron-3-Super-120B-GGUF"


@dataclass(slots=True)
class MergeResult:
    """Result of the merge-and-serve phase."""

    cycle_id: str
    adapter_path: Path
    strategy_used: str
    base_model: str
    gguf_converted: bool = False
    hot_loaded: bool = False
    serve_verified: bool = False
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "adapter_path": str(self.adapter_path),
            "strategy_used": self.strategy_used,
            "base_model": self.base_model,
            "gguf_converted": self.gguf_converted,
            "hot_loaded": self.hot_loaded,
            "serve_verified": self.serve_verified,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class MergeCycle:
    """Activates the trained adapter in the serving model.

    Strategy: convert PEFT LoRA adapter to GGUF format using llama.cpp's
    converter, then hot-load into the running llama-server via POST
    /lora-adapters. No restart needed — takes effect immediately.

    This is Phase 6 (BECOME) of the growth cycle described in #2483.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        config_path = config_path or DEFAULT_CONFIG
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._cfg = yaml.safe_load(f) or {}
        else:
            self._cfg = {}
        self._merge_cfg = self._cfg.get("merge", {})
        self._base_model = self._merge_cfg.get(
            "serving_model",
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-IQ4_XS",
        )
        self._server_url = self._merge_cfg.get(
            "server_url", "http://127.0.0.1:8000"
        )

    def run(
        self,
        adapter_path: Path,
        cycle_id: str,
        dry_run: bool = False,
    ) -> MergeResult:
        """Activate the trained adapter in the serving model.

        1. Verify adapter exists and is valid
        2. Convert PEFT LoRA adapter to GGUF format
        3. Hot-load GGUF adapter into running llama-server
        4. Verify serving is healthy
        5. Record in cycle history
        """
        start = time.monotonic()

        if dry_run:
            print("[MergeCycle] Dry run — skipping GGUF conversion and hot-load")
            return MergeResult(
                cycle_id=cycle_id,
                adapter_path=adapter_path,
                strategy_used="llama_server_lora_hotload",
                base_model=self._base_model,
                duration_seconds=time.monotonic() - start,
                metadata={"dry_run": True},
            )

        # 1. Verify adapter
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        adapter_dir = adapter_path if adapter_path.is_dir() else adapter_path.parent
        adapter_config = adapter_dir / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(
                f"No adapter_config.json in {adapter_dir} — "
                "training may not have completed successfully"
            )
        config_data = json.loads(adapter_config.read_text())
        print(f"[MergeCycle] Adapter verified: r={config_data.get('r')}, "
              f"alpha={config_data.get('lora_alpha')}, "
              f"modules={config_data.get('target_modules')}")

        # 2. Convert to GGUF
        gguf_path = self._convert_to_gguf(adapter_dir)
        gguf_converted = gguf_path is not None

        # 3. Hot-load into llama-server
        hot_loaded = False
        if gguf_path is not None:
            hot_loaded = self._hot_load_adapter(gguf_path)

        # 4. Verify serving
        verified = False
        if hot_loaded:
            verified = self._verify_serving()

        # 5. Record active adapter info
        active_adapter_file = GROWTH_DIR / "active_adapter.json"
        active_adapter_file.write_text(json.dumps({
            "cycle_id": cycle_id,
            "adapter_path": str(adapter_dir),
            "gguf_path": str(gguf_path) if gguf_path else None,
            "base_model": self._base_model,
            "hot_loaded": hot_loaded,
            "activated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

        duration = time.monotonic() - start

        result = MergeResult(
            cycle_id=cycle_id,
            adapter_path=adapter_path,
            strategy_used="llama_server_lora_hotload",
            base_model=self._base_model,
            gguf_converted=gguf_converted,
            hot_loaded=hot_loaded,
            serve_verified=verified,
            duration_seconds=round(duration, 2),
        )

        self._record_merge(result)
        return result

    def _convert_to_gguf(self, adapter_dir: Path) -> Optional[Path]:
        """Convert a PEFT LoRA adapter to GGUF format for llama-server.

        Uses llama.cpp's convert_lora_to_gguf.py. Returns the path to the
        GGUF adapter file, or None if conversion fails.
        """
        convert_script = _LLAMA_CPP_DIR / "convert_lora_to_gguf.py"
        gguf_out = adapter_dir / "adapter.gguf"

        if not convert_script.exists():
            print(f"[MergeCycle] GGUF conversion skipped: {convert_script} not found")
            return None

        if not _GGUF_BASE_DIR.exists():
            print(f"[MergeCycle] GGUF conversion skipped: {_GGUF_BASE_DIR} not found")
            return None

        cmd = [
            sys.executable, str(convert_script),
            "--base", str(_GGUF_BASE_DIR),
            "--adapter", str(adapter_dir),
            "--outfile", str(gguf_out),
        ]
        print(f"[MergeCycle] GGUF conversion: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[MergeCycle] GGUF conversion failed (exit {result.returncode}):")
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-10:]:
                        print(f"[MergeCycle]   {line}")
                return None
            print(f"[MergeCycle] GGUF adapter saved: {gguf_out}")
            return gguf_out
        except subprocess.TimeoutExpired:
            print("[MergeCycle] GGUF conversion timed out after 600s")
            return None
        except Exception as e:
            print(f"[MergeCycle] GGUF conversion error: {e}")
            return None

    def _hot_load_adapter(self, gguf_path: Path) -> bool:
        """Hot-load a GGUF LoRA adapter into the running llama-server.

        Posts to /lora-adapters endpoint. Takes effect immediately.
        """
        payload = json.dumps([{
            "id": 1,
            "path": str(gguf_path),
            "scale": 1.0,
        }]).encode("utf-8")

        req = urllib.request.Request(
            f"{self._server_url}/lora-adapters",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                print(f"[MergeCycle] Hot-loaded adapter into llama-server ({resp.status})")
                return True
        except urllib.error.HTTPError as e:
            print(f"[MergeCycle] Hot-load failed: HTTP {e.code} — {e.read().decode()[:200]}")
            return False
        except Exception as e:
            print(f"[MergeCycle] Hot-load failed: {e}")
            return False

    def _verify_serving(self, timeout: int = 30) -> bool:
        """Verify llama-server is healthy after hot-loading the adapter."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                req = urllib.request.urlopen(
                    f"{self._server_url}/health", timeout=5
                )
                data = json.loads(req.read())
                if data.get("status") == "ok":
                    print("[MergeCycle] llama-server healthy after hot-load")
                    return True
            except Exception:
                time.sleep(2)
        print("[MergeCycle] Warning: llama-server health check timed out")
        return False

    def _record_merge(self, result: MergeResult) -> None:
        """Append merge result to cycle history."""
        CYCLE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": "merge",
            **result.to_dict(),
        }
        with open(CYCLE_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
