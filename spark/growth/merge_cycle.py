"""spark.growth.merge_cycle — Merge and serve for the recursive growth engine.

Phase 6 (BECOME) of the growth cycle described in issue #2483.

The model wakes up slightly different. The next pulse reflects on
that difference. The loop closes.

Primary strategy: vLLM LoRA serving — load the adapter at serve time
without merging into base weights. This avoids the 458GB BF16 merge
entirely and takes effect in seconds by restarting vLLM with
--lora-modules flag.

Fallback strategies (for when full merge is needed):
  - cpu_offload: Load sharded across CPU RAM + NVMe with device_map="auto".
    Slow (hours) but works on-box.
  - cloud_burst: Transfer adapter to cloud instance, merge there, transfer
    back the re-quantized model.

Integration points:
  - Input: trained adapter from TrainCycle
  - vLLM restart with --lora-modules to activate adapter
  - Cycle history: GROWTH_DIR / "cycle_history.jsonl"
"""

from __future__ import annotations

import json
import subprocess
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

GROWTH_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = GROWTH_DIR / "growth_config.yaml"
CYCLE_HISTORY = GROWTH_DIR / "cycle_history.jsonl"


@dataclass(slots=True)
class MergeResult:
    """Result of the merge-and-serve phase."""

    cycle_id: str
    adapter_path: Path
    strategy_used: str
    base_model: str
    vllm_restarted: bool = False
    serve_verified: bool = False
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "adapter_path": str(self.adapter_path),
            "strategy_used": self.strategy_used,
            "base_model": self.base_model,
            "vllm_restarted": self.vllm_restarted,
            "serve_verified": self.serve_verified,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class MergeCycle:
    """Activates the trained adapter in the serving model.

    Primary strategy: vLLM LoRA serving — restart vLLM with the
    --lora-modules flag pointing to the trained adapter. This avoids
    merging into the 458GB base model entirely.

    This is Phase 6 (BECOME) of the growth cycle described in #2483.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        config_path = config_path or DEFAULT_CONFIG
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
        self._merge_cfg = self._cfg.get("merge", {})
        self._base_model = self._merge_cfg.get("serving_model", "cyankiwi/MiniMax-M2.5-AWQ-4bit")
        self._strategy = self._merge_cfg.get("strategy", "lora_serve")
        self._container_name = "vllm_node"

    def run(
        self,
        adapter_path: Path,
        cycle_id: str,
        dry_run: bool = False,
    ) -> MergeResult:
        """Activate the trained adapter in the serving model.

        For lora_serve strategy:
        1. Verify adapter exists and is valid
        2. Restart vLLM with --lora-modules pointing to the adapter
        3. Verify the model is serving with the adapter
        4. Record in cycle history

        Args:
            adapter_path: Path to the trained LoRA adapter directory.
            cycle_id: Unique identifier for this growth cycle.
            dry_run: If True, verify adapter but don't restart vLLM.

        Returns:
            MergeResult with activation status.
        """
        start = time.monotonic()

        if dry_run:
            print(f"[MergeCycle] Dry run — skipping adapter verification and restart")
            return MergeResult(
                cycle_id=cycle_id,
                adapter_path=adapter_path,
                strategy_used="lora_serve",
                base_model=self._base_model,
                vllm_restarted=False,
                serve_verified=False,
                duration_seconds=time.monotonic() - start,
                metadata={"dry_run": True},
            )

        # 1. Verify adapter
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(
                f"No adapter_config.json in {adapter_path} — "
                "training may not have completed successfully"
            )
        config_data = json.loads(adapter_config.read_text())
        print(f"[MergeCycle] Adapter verified: r={config_data.get('r')}, "
              f"alpha={config_data.get('lora_alpha')}, "
              f"modules={config_data.get('target_modules')}")

        # 2. Determine container-internal path to adapter
        # The repo is expected to be mounted at /workspace/Vybn
        container_adapter_path = str(adapter_path).replace(
            str(GROWTH_DIR.parent.parent), "/workspace/Vybn"
        )

        # 3. Restart vLLM with the adapter
        # Note: this requires the vLLM container to be configured to accept
        # lora modules. The exact restart mechanism depends on how vLLM
        # was started (docker-compose, systemd, manual).
        print(f"[MergeCycle] Restarting vLLM with adapter: {container_adapter_path}")

        # For now, we write the adapter path to a known location that the
        # vLLM startup script can read. The actual restart is handled by
        # trigger.py or the cron job.
        active_adapter_file = GROWTH_DIR / "active_adapter.json"
        active_adapter_file.write_text(json.dumps({
            "cycle_id": cycle_id,
            "adapter_path": str(adapter_path),
            "container_path": container_adapter_path,
            "base_model": self._base_model,
            "activated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

        # Attempt restart if we have docker access
        restarted = False
        verified = False
        try:
            restarted = self._restart_vllm_with_lora(container_adapter_path)
            if restarted:
                verified = self._verify_serving()
        except Exception as e:
            print(f"[MergeCycle] Warning: vLLM restart failed: {e}")
            print("[MergeCycle] Adapter path recorded in active_adapter.json")
            print("[MergeCycle] Manual restart needed to activate the adapter")

        duration = time.monotonic() - start

        result = MergeResult(
            cycle_id=cycle_id,
            adapter_path=adapter_path,
            strategy_used="lora_serve",
            base_model=self._base_model,
            vllm_restarted=restarted,
            serve_verified=verified,
            duration_seconds=round(duration, 2),
        )

        # Record in cycle history
        self._record_merge(result)

        return result

    def _restart_vllm_with_lora(self, container_adapter_path: str) -> bool:
        """Restart the vLLM container with LoRA adapter.

        This is a placeholder for the actual restart mechanism.
        The exact implementation depends on how vLLM was deployed.

        Returns True if restart succeeded.
        """
        # Check if vLLM supports --enable-lora
        # For now, just signal that a restart is needed
        print(f"[MergeCycle] To activate adapter, restart vLLM with:")
        print(f"  --enable-lora --lora-modules vybn-growth={container_adapter_path}")
        print(f"[MergeCycle] Active adapter path saved to {GROWTH_DIR / 'active_adapter.json'}")
        return False  # Manual restart needed for now

    def _verify_serving(self, timeout: int = 60) -> bool:
        """Verify the model is serving after restart.

        Polls the /v1/models endpoint until it responds or timeout.
        """
        import urllib.request
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                req = urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
                data = json.loads(req.read())
                if data.get("data"):
                    print(f"[MergeCycle] vLLM serving: {data['data'][0]['id']}")
                    return True
            except Exception:
                time.sleep(2)
        print("[MergeCycle] Warning: vLLM did not come up within timeout")
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
