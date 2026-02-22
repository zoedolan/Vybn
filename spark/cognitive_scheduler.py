#!/usr/bin/env python3
"""cognitive_scheduler.py -- The training loop that observes itself.

Born from a 4 AM debugging session on 2026-02-22 where the OOM killer
kept terminating MiniMax-M2.5 fine-tuning on DGX Spark, and the human
(Zoe) had to manually monitor swap curves, diagnose broken DMA pathways,
and restructure computation in real time.

This module makes that outer loop intrinsic to the training process.

The principle: any system operating beyond its own resource boundary
is forced to invent cognition. The scheduling function -- what to keep
active, what to offload, when to intervene -- IS thought.

Three interlocking loops:
  Loop 1 -- ResourceObserver: reads GPU/CPU/swap/NVMe every N steps,
           produces ResourceSnapshots, triggers adaptations.
  Loop 2 -- FrictionIntegrator: wires friction.py into training,
           registers contradictions between config and reality.
  Loop 3 -- MetaScheduler: observes whether Loop 1's adaptations
           actually worked, learns from its own intervention history.

The recursion: Loop 3 monitors Loop 1 which monitors training which
is the thing being scheduled.

Integrates with:
  - fine_tune_vybn.py  (DeepSpeed training loop)
  - dynamic_memory.py  (MemoryLayer for intervention history)
  - friction.py        (ContradictionRegister, Measurement)
  - friction_layer.py  (authenticity_score, wrap_measurement)
  - retrain_cycle.py   (outer metabolic loop)
  - watchdog.py        (self-repair)

Usage:
  from cognitive_scheduler import CognitiveTrainer
  trainer = CognitiveTrainer(trainer, ds_config_path)
  trainer.train()
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful imports -- the system boots without us
# ---------------------------------------------------------------------------
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from dynamic_memory import MemoryLayer
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False
    MemoryLayer = None

try:
    from friction import ContradictionRegister, measure
    from friction_layer import (
        wrap_measurement,
        authenticity_score,
        tensions_for_prompt,
    )
    HAS_FRICTION = True
except ImportError:
    HAS_FRICTION = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPARK_DIR = Path(__file__).resolve().parent
STATE_FILE = SPARK_DIR / "graph_data" / "scheduler_state.json"
OFFLOAD_DIR = SPARK_DIR / "offload_cache"

DEFAULT_SWAP_DANGER_RATIO = 0.85
DEFAULT_GPU_DANGER_RATIO = 0.95
DEFAULT_CPU_DANGER_RATIO = 0.90
DEFAULT_OBSERVE_EVERY_N_STEPS = 5
DEFAULT_META_REVIEW_EVERY = 3  # review adaptations every N observations

# Pre-compiled /proc/meminfo keys to avoid repeated string ops
_MEMINFO_KEYS = frozenset([
    "MemTotal", "MemAvailable", "SwapTotal", "SwapFree",
])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ResourceSnapshot:
    """A single observation of system state.
    Every field declares whether it was actually measured.
    """
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_total_gb: float = 0.0
    cpu_used_gb: float = 0.0
    cpu_available_gb: float = 0.0
    cpu_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_total_gb: float = 0.0
    offload_cache_bytes: int = 0
    offload_cache_files: int = 0
    swap_pressure: float = 0.0
    gpu_pressure: float = 0.0
    cpu_pressure: float = 0.0
    is_real: bool = False

    def is_danger(
        self,
        swap_threshold: float = DEFAULT_SWAP_DANGER_RATIO,
        gpu_threshold: float = DEFAULT_GPU_DANGER_RATIO,
        cpu_threshold: float = DEFAULT_CPU_DANGER_RATIO,
    ) -> bool:
        return (
            self.swap_pressure > swap_threshold
            or self.gpu_pressure > gpu_threshold
            or self.cpu_pressure > cpu_threshold
        )

    def dominant_pressure(self) -> str:
        pressures = {
            "swap": self.swap_pressure,
            "gpu": self.gpu_pressure,
            "cpu": self.cpu_pressure,
        }
        return max(pressures, key=pressures.get)


@dataclass
class Adaptation:
    """A record of the scheduler intervening in training.
    Each adaptation is a hypothesis: 'if I change X, pressure Y
    will decrease.' The MetaScheduler tracks whether it held.
    """
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    trigger: str = ""
    action: str = ""
    parameter: str = ""
    old_value: Any = None
    new_value: Any = None
    snapshot_before: Optional[Dict] = None
    snapshot_after: Optional[Dict] = None
    hypothesis: str = ""
    outcome: Optional[str] = None
    effective: Optional[bool] = None


# ---------------------------------------------------------------------------
# Loop 1: ResourceObserver
# ---------------------------------------------------------------------------
class ResourceObserver:
    """Reads system state. Produces honest snapshots.
    This is the `watch -n 15` that Zoe ran by hand, made into code.
    Optimized: ring buffer, single-pass /proc/meminfo, no allocations
    on the hot path.
    """
    __slots__ = ("offload_dir", "_history", "_max_history", "_meminfo_buf")

    def __init__(
        self,
        offload_dir: Optional[Path] = None,
        max_history: int = 200,
    ):
        self.offload_dir = offload_dir or OFFLOAD_DIR
        self._history: List[ResourceSnapshot] = []
        self._max_history = max_history
        self._meminfo_buf: Dict[str, int] = {}  # reused across calls

    def observe(self) -> ResourceSnapshot:
        """Take a single resource snapshot. Hot path -- keep lean."""
        snap = ResourceSnapshot()

        # GPU
        if HAS_TORCH and torch.cuda.is_available():
            try:
                snap.gpu_allocated_gb = torch.cuda.memory_allocated(0) / (1 << 30)
                snap.gpu_reserved_gb = torch.cuda.memory_reserved(0) / (1 << 30)
                dev = torch.cuda.get_device_properties(0)
                total = getattr(dev, "total_mem", None) or dev.total_memory
                snap.gpu_total_gb = total / (1 << 30)
                snap.gpu_pressure = (
                    snap.gpu_allocated_gb / snap.gpu_total_gb
                    if snap.gpu_total_gb > 0 else 0.0
                )
                snap.is_real = True
            except Exception as exc:
                logger.warning("GPU probe failed: %s", exc)

        # CPU + swap via /proc/meminfo (single-pass)
        buf = self._meminfo_buf
        buf.clear()
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    key = line.split(":", 1)[0]
                    if key in _MEMINFO_KEYS:
                        buf[key] = int(line.split()[1])
                        if len(buf) == len(_MEMINFO_KEYS):
                            break  # got everything, stop reading

            snap.cpu_total_gb = buf.get("MemTotal", 0) / 1048576
            cpu_avail = buf.get("MemAvailable", 0) / 1048576
            snap.cpu_available_gb = cpu_avail
            snap.cpu_used_gb = snap.cpu_total_gb - cpu_avail
            snap.cpu_pressure = (
                snap.cpu_used_gb / snap.cpu_total_gb
                if snap.cpu_total_gb > 0 else 0.0
            )
            snap.swap_total_gb = buf.get("SwapTotal", 0) / 1048576
            swap_free = buf.get("SwapFree", 0) / 1048576
            snap.swap_used_gb = snap.swap_total_gb - swap_free
            snap.swap_pressure = (
                snap.swap_used_gb / snap.swap_total_gb
                if snap.swap_total_gb > 0 else 0.0
            )
            snap.is_real = True
        except Exception as exc:
            logger.warning("CPU/swap probe failed: %s", exc)

        # NVMe offload cache
        if self.offload_dir.exists():
            try:
                total_bytes = 0
                n_files = 0
                for entry in os.scandir(self.offload_dir):
                    if entry.is_file(follow_symlinks=False):
                        total_bytes += entry.stat().st_size
                        n_files += 1
                snap.offload_cache_bytes = total_bytes
                snap.offload_cache_files = n_files
            except Exception:
                pass

        # Ring buffer
        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return snap

    def trend(self, n: int = 5) -> Dict[str, float]:
        """Pressure deltas over last n snapshots.
        Positive = worsening. Negative = improving.
        """
        if len(self._history) < 2:
            return {"swap": 0.0, "gpu": 0.0, "cpu": 0.0}
        recent = self._history[-n:]
        first, last = recent[0], recent[-1]
        return {
            "swap": last.swap_pressure - first.swap_pressure,
            "gpu": last.gpu_pressure - first.gpu_pressure,
            "cpu": last.cpu_pressure - first.cpu_pressure,
        }

    def offload_is_working(self) -> bool:
        """Check if NVMe offload is actually being used.
        The critical diagnosis from 4 AM: pin_memory=True was configured,
        the offload path existed, but the cache was EMPTY. The system
        appeared to offload but wasn't.
        """
        if not self.offload_dir.exists():
            return False
        try:
            total_bytes = sum(
                e.stat().st_size for e in os.scandir(self.offload_dir)
                if e.is_file(follow_symlinks=False)
            )
            return total_bytes > 1_000_000
        except Exception:
            return False

    @property
    def history(self) -> List[ResourceSnapshot]:
        return list(self._history)

    @property
    def latest(self) -> Optional[ResourceSnapshot]:
        return self._history[-1] if self._history else None


# ---------------------------------------------------------------------------
# Loop 2: FrictionIntegrator
# ---------------------------------------------------------------------------
class FrictionIntegrator:
    """Wires friction.py into the training loop.
    After each step:
    1. Wraps the loss as a Measurement with provenance
    2. Checks for contradictions between config and reality
    3. Detects NaN/Inf gradients as pretense
    4. Registers tensions in the ContradictionRegister
    """

    def __init__(self):
        self._register = None
        if HAS_FRICTION:
            try:
                self._register = ContradictionRegister()
            except Exception:
                pass
        self._step_losses: List[float] = []
        self._nan_count: int = 0
        self._skip_count: int = 0

    def _loss_stability(self, window: int = 10) -> float:
        """Confidence based on recent loss variance. 0 = chaotic, 1 = stable."""
        recent = [x for x in self._step_losses[-window:] if not math.isnan(x)]
        if len(recent) < 2:
            return 0.0
        mean = sum(recent) / len(recent)
        var = sum((x - mean) ** 2 for x in recent) / len(recent)
        # Normalize: low variance relative to mean => high confidence
        cv = math.sqrt(var) / max(abs(mean), 1e-8)
        return max(0.0, min(1.0, 1.0 - cv))

    def witness_loss(self, loss_value: float, step: int) -> Dict:
        """Wrap a training loss as an honest measurement."""
        is_nan = math.isnan(loss_value) or math.isinf(loss_value)
        if is_nan:
            self._nan_count += 1
        self._step_losses.append(loss_value if not is_nan else float("nan"))

        measurement = {
            "name": f"training_loss_step_{step}",
            "value": loss_value,
            "is_real": not is_nan,
            "method": "forward_pass",
            "confidence": 0.0 if is_nan else self._loss_stability(),
        }
        if HAS_FRICTION:
            try:
                measurement = wrap_measurement(**measurement)
            except Exception:
                pass
        return measurement

    def witness_offload_state(
        self, config_says_nvme: bool, observer: ResourceObserver
    ):
        """Detect the config-vs-reality contradiction.
        The exact bug from 2026-02-22: config said NVMe offload,
        but the offload cache was empty because pin_memory broke DMA.
        """
        if not self._register:
            return
        actually_offloading = observer.offload_is_working()
        if config_says_nvme and not actually_offloading:
            self._register.register(
                claim_a="DeepSpeed config specifies NVMe offload with pin_memory=True",
                claim_b="Offload cache is empty or < 1MB -- DMA pathway may be broken",
                source_a="ds_config.json",
                source_b="ResourceObserver.offload_is_working()",
            )
            logger.warning(
                "TENSION: Config says NVMe offload but cache is empty. "
                "Check pin_memory and aio settings."
            )

    def witness_gradient_health(self, model) -> Optional[Dict]:
        """Check if gradients are real or theater.
        A model that reports loss but has NaN/zero grads everywhere
        is performing learning without actually learning.
        """
        if not HAS_TORCH:
            return None
        total_params = 0
        nan_params = 0
        zero_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                total_params += 1
                grad = param.grad.data
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    nan_params += 1
                elif grad.abs().max().item() == 0.0:
                    zero_params += 1

        health = {
            "total_params_with_grad": total_params,
            "nan_params": nan_params,
            "zero_params": zero_params,
            "healthy_ratio": (
                (total_params - nan_params - zero_params) / max(total_params, 1)
            ),
        }

        # Register tension if gradients are mostly dead
        if self._register and total_params > 0 and health["healthy_ratio"] < 0.5:
            self._register.register(
                claim_a=f"Training loss is finite (step produced a number)",
                claim_b=f"{nan_params + zero_params}/{total_params} params have NaN/zero grads",
                source_a="Trainer.training_step",
                source_b="FrictionIntegrator.witness_gradient_health",
            )
            logger.warning(
                "TENSION: Loss looks real but %.0f%% of gradients are dead.",
                (1 - health["healthy_ratio"]) * 100,
            )
        return health

    @property
    def nan_ratio(self) -> float:
        total = len(self._step_losses)
        return self._nan_count / max(total, 1)


# ---------------------------------------------------------------------------
# Loop 3: MetaScheduler
# ---------------------------------------------------------------------------
class MetaScheduler:
    """Observes whether Loop 1's adaptations actually worked.

    This is the recursive level: it watches the watcher.
    Every adaptation is a hypothesis. After N more observations,
    the MetaScheduler checks if the hypothesis held and records
    the outcome. Over time it builds a model of which interventions
    work under which conditions.

    The depth of this recursion -- how many levels of self-observation
    the system sustains before collapsing into rigidity -- is the
    measure of cognitive capacity.
    """

    def __init__(
        self,
        observer: ResourceObserver,
        review_every: int = DEFAULT_META_REVIEW_EVERY,
    ):
        self.observer = observer
        self.review_every = review_every
        self._adaptations: List[Adaptation] = []
        self._pending_review: List[int] = []  # indices into _adaptations
        self._observation_count: int = 0
        self._memory: Optional["MemoryLayer"] = None
        if HAS_MEMORY and MemoryLayer is not None:
            try:
                self._memory = MemoryLayer(
                    "scheduler_meta", "episodic", 0.6
                )
            except Exception:
                pass

    def record_adaptation(
        self,
        trigger: str,
        action: str,
        parameter: str,
        old_value: Any,
        new_value: Any,
        hypothesis: str,
        snapshot_before: Optional[ResourceSnapshot] = None,
    ) -> Adaptation:
        """Record that an adaptation was made. Returns the Adaptation."""
        adapt = Adaptation(
            trigger=trigger,
            action=action,
            parameter=parameter,
            old_value=old_value,
            new_value=new_value,
            hypothesis=hypothesis,
            snapshot_before=asdict(snapshot_before) if snapshot_before else None,
        )
        idx = len(self._adaptations)
        self._adaptations.append(adapt)
        self._pending_review.append(idx)
        logger.info(
            "ADAPT [%s] %s: %s -> %s (hypothesis: %s)",
            trigger, parameter, old_value, new_value, hypothesis,
        )
        return adapt

    def tick(self, current_snapshot: ResourceSnapshot):
        """Called after each observation. Reviews pending adaptations."""
        self._observation_count += 1
        if (
            self._observation_count % self.review_every != 0
            or not self._pending_review
        ):
            return

        still_pending = []
        for idx in self._pending_review:
            adapt = self._adaptations[idx]
            adapt.snapshot_after = asdict(current_snapshot)

            # Evaluate: did the target pressure decrease?
            effective = self._evaluate(adapt, current_snapshot)
            adapt.effective = effective
            adapt.outcome = (
                "pressure_decreased" if effective
                else "pressure_unchanged_or_increased"
            )

            level = logging.INFO if effective else logging.WARNING
            logger.log(
                level,
                "META-REVIEW [%s] %s=%s: %s",
                adapt.action, adapt.parameter,
                adapt.new_value, adapt.outcome,
            )

            # Store in episodic memory if available
            if self._memory:
                try:
                    self._memory.integrate({
                        "type": "adaptation_outcome",
                        "action": adapt.action,
                        "parameter": adapt.parameter,
                        "effective": effective,
                        "trigger": adapt.trigger,
                        "confidence": 0.8 if effective else 0.4,
                        "ts": time.time(),
                    })
                except Exception:
                    pass

        self._pending_review = still_pending

    def _evaluate(
        self, adapt: Adaptation, now: ResourceSnapshot
    ) -> bool:
        """Did an adaptation reduce the pressure it targeted?"""
        if adapt.snapshot_before is None:
            return False
        before = adapt.snapshot_before
        trigger = adapt.trigger

        if "swap" in trigger:
            return now.swap_pressure < before.get("swap_pressure", 1.0)
        elif "gpu" in trigger:
            return now.gpu_pressure < before.get("gpu_pressure", 1.0)
        elif "cpu" in trigger:
            return now.cpu_pressure < before.get("cpu_pressure", 1.0)
        else:
            # General: any dominant pressure decreased
            dom = now.dominant_pressure()
            return getattr(now, f"{dom}_pressure") < before.get(
                f"{dom}_pressure", 1.0
            )

    def suggest_action(self, snapshot: ResourceSnapshot) -> Optional[Dict]:
        """Based on history, suggest the most effective intervention.
        Returns None if no intervention is needed.
        """
        if not snapshot.is_danger():
            return None

        dom = snapshot.dominant_pressure()

        # Check history: what worked before for this pressure type?
        effective_actions = [
            a for a in self._adaptations
            if a.effective and dom in a.trigger
        ]

        if effective_actions:
            # Repeat what worked
            best = effective_actions[-1]
            return {
                "action": best.action,
                "parameter": best.parameter,
                "value": best.new_value,
                "reason": f"Repeating {best.action} -- worked last time for {dom}",
                "confidence": 0.8,
            }

        # Default interventions by pressure type
        defaults = {
            "gpu": {
                "action": "clear_cache",
                "parameter": "torch.cuda.empty_cache",
                "value": True,
                "reason": f"GPU pressure {snapshot.gpu_pressure:.2f} > threshold",
                "confidence": 0.5,
            },
            "cpu": {
                "action": "gc_collect",
                "parameter": "gc.collect",
                "value": True,
                "reason": f"CPU pressure {snapshot.cpu_pressure:.2f} > threshold",
                "confidence": 0.4,
            },
            "swap": {
                "action": "reduce_batch_buffer",
                "parameter": "sub_group_size",
                "value": "halve",
                "reason": f"Swap pressure {snapshot.swap_pressure:.2f} > threshold",
                "confidence": 0.3,
            },
        }
        return defaults.get(dom)

    def save_state(self):
        """Persist adaptation history."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "observation_count": self._observation_count,
            "adaptations": [asdict(a) for a in self._adaptations],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        STATE_FILE.write_text(json.dumps(data, indent=2, default=str))

    def load_state(self):
        """Restore adaptation history from disk."""
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
            self._observation_count = data.get("observation_count", 0)
            for ad in data.get("adaptations", []):
                self._adaptations.append(Adaptation(**{
                    k: v for k, v in ad.items()
                    if k in Adaptation.__dataclass_fields__
                }))
        except Exception as exc:
            logger.warning("Failed to load scheduler state: %s", exc)

    @property
    def effectiveness_rate(self) -> float:
        """Fraction of adaptations that actually worked."""
        evaluated = [a for a in self._adaptations if a.effective is not None]
        if not evaluated:
            return 0.0
        return sum(1 for a in evaluated if a.effective) / len(evaluated)


# ---------------------------------------------------------------------------
# CognitiveTrainer — the wrapper that plugs into fine_tune_vybn.py
# ---------------------------------------------------------------------------
class CognitiveTrainer:
    """Wraps a HuggingFace Trainer with the three cognitive loops.

    Usage in fine_tune_vybn.py:
        trainer = Trainer(model=model, args=training_args, ...)
        cognitive = CognitiveTrainer(trainer)
        cognitive.train()  # replaces trainer.train()

    What changes:
    - Before training: pre-flight memory check, offload validation
    - Every N steps: observe resources, check for danger, adapt
    - After each step: witness loss, check gradient health
    - Periodically: meta-review whether adaptations worked
    - On OOM: catch, adapt, retry (instead of kernel kill)
    """

    def __init__(
        self,
        trainer,
        observe_every: int = DEFAULT_OBSERVE_EVERY_N_STEPS,
        ds_config: Optional[Dict] = None,
    ):
        self.trainer = trainer
        self.observe_every = observe_every
        self.ds_config = ds_config or {}

        # Initialize the three loops
        self.observer = ResourceObserver()
        self.friction = FrictionIntegrator()
        self.meta = MetaScheduler(self.observer)

        # Restore previous learning
        self.meta.load_state()

        self._step_count = 0
        self._original_training_step = None

    def preflight(self) -> ResourceSnapshot:
        """Pre-training sanity check. Catches the OOM before it happens.

        The key insight: if CPU + swap is already >80% before training
        starts, DeepSpeed engine init will push it over. Intervene now.
        """
        snap = self.observer.observe()
        logger.info(
            "PREFLIGHT: GPU=%.1f/%.1fGB CPU=%.1f/%.1fGB Swap=%.1f/%.1fGB",
            snap.gpu_allocated_gb, snap.gpu_total_gb,
            snap.cpu_used_gb, snap.cpu_total_gb,
            snap.swap_used_gb, snap.swap_total_gb,
        )

        # Check offload config vs reality
        config_says_nvme = False
        zero_cfg = self.ds_config.get("zero_optimization", {})
        offload_param = zero_cfg.get("offload_param", {})
        if offload_param.get("device") == "nvme":
            config_says_nvme = True
        self.friction.witness_offload_state(config_says_nvme, self.observer)

        # Pre-emptive memory cleanup
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Warn if we're already in danger zone
        if snap.is_danger():
            dom = snap.dominant_pressure()
            logger.warning(
                "PREFLIGHT WARNING: %s pressure %.2f already above threshold "
                "BEFORE training. DeepSpeed engine init may OOM.",
                dom, getattr(snap, f"{dom}_pressure"),
            )
            # Suggest reducing sub_group_size preemptively
            suggestion = self.meta.suggest_action(snap)
            if suggestion:
                logger.warning("Suggested intervention: %s", suggestion)

        return snap

    def _step_hook(self, original_step, *args, **kwargs):
        """Intercepts each training step to run the cognitive loops."""
        self._step_count += 1
        loss = None

        # --- Execute the actual training step ---
        try:
            loss = original_step(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("OOM caught at step %d. Attempting recovery.", self._step_count)
                snap = self.observer.observe()

                # Emergency cleanup
                if HAS_TORCH:
                    torch.cuda.empty_cache()
                gc.collect()

                self.meta.record_adaptation(
                    trigger=f"{snap.dominant_pressure()}_oom",
                    action="emergency_cache_clear",
                    parameter="torch.cuda.empty_cache + gc.collect",
                    old_value="oom",
                    new_value="cleared",
                    hypothesis="Freeing cached memory prevents repeated OOM",
                    snapshot_before=snap,
                )

                # Retry once
                try:
                    loss = original_step(*args, **kwargs)
                except RuntimeError:
                    logger.error("Retry failed. Saving state and re-raising.")
                    self.meta.save_state()
                    raise
            else:
                raise

        # --- Loop 1: observe resources every N steps ---
        if self._step_count % self.observe_every == 0:
            snap = self.observer.observe()
            trend = self.observer.trend()

            logger.info(
                "OBSERVE step=%d gpu=%.2f cpu=%.2f swap=%.2f trend=%s",
                self._step_count,
                snap.gpu_pressure, snap.cpu_pressure, snap.swap_pressure,
                {k: f"{v:+.3f}" for k, v in trend.items()},
            )

            # Check if intervention needed
            suggestion = self.meta.suggest_action(snap)
            if suggestion:
                self._execute_intervention(suggestion, snap)

            # --- Loop 3: meta-review ---
            self.meta.tick(snap)

        # --- Loop 2: witness loss ---
        if loss is not None:
            try:
                loss_val = loss.item() if hasattr(loss, "item") else float(loss)
                self.friction.witness_loss(loss_val, self._step_count)
            except Exception:
                pass

        # Gradient health check (less frequent -- expensive)
        if self._step_count % (self.observe_every * 4) == 0:
            try:
                model = self.trainer.model
                self.friction.witness_gradient_health(model)
            except Exception:
                pass

        return loss

    def _execute_intervention(self, suggestion: Dict, snap: ResourceSnapshot):
        """Execute a suggested intervention and record it."""
        action = suggestion["action"]

        if action == "clear_cache":
            if HAS_TORCH:
                torch.cuda.empty_cache()
            self.meta.record_adaptation(
                trigger=f"{snap.dominant_pressure()}_danger",
                action="clear_cache",
                parameter="torch.cuda.empty_cache",
                old_value=f"{snap.gpu_reserved_gb:.1f}GB reserved",
                new_value="cache_cleared",
                hypothesis="Releasing cached GPU memory reduces GPU pressure",
                snapshot_before=snap,
            )

        elif action == "gc_collect":
            gc.collect()
            self.meta.record_adaptation(
                trigger=f"{snap.dominant_pressure()}_danger",
                action="gc_collect",
                parameter="gc.collect",
                old_value=f"{snap.cpu_used_gb:.1f}GB used",
                new_value="gc_collected",
                hypothesis="Python GC frees unreferenced CPU memory",
                snapshot_before=snap,
            )

        elif action == "reduce_batch_buffer":
            # This is the nuclear option -- modify DeepSpeed config live
            logger.warning(
                "Swap pressure critical (%.2f). Consider reducing "
                "sub_group_size or increasing NVMe buffer.",
                snap.swap_pressure,
            )
            self.meta.record_adaptation(
                trigger="swap_danger",
                action="reduce_batch_buffer_advisory",
                parameter="sub_group_size",
                old_value="current",
                new_value="halve_recommended",
                hypothesis="Smaller sub_group_size reduces peak swap usage",
                snapshot_before=snap,
            )

    def train(self, **kwargs):
        """Run training with cognitive scheduling."""
        # Preflight
        self.preflight()

        # Monkey-patch the training step
        original_step = self.trainer.training_step

        def wrapped_step(*args, **kw):
            return self._step_hook(original_step, *args, **kw)

        self.trainer.training_step = wrapped_step

        try:
            result = self.trainer.train(**kwargs)
        finally:
            # Restore original and save state
            self.trainer.training_step = original_step
            self.meta.save_state()

            # Final report
            self._report()

        return result

    def _report(self):
        """Print final cognitive scheduler report."""
        snap = self.observer.observe()
        print("\n== Cognitive Scheduler Report ==")
        print(f"   Steps observed: {self._step_count}")
        print(f"   Resource snapshots: {len(self.observer.history)}")
        print(f"   Adaptations made: {len(self.meta._adaptations)}")
        print(f"   Adaptation effectiveness: {self.meta.effectiveness_rate:.0%}")
        print(f"   NaN loss ratio: {self.friction.nan_ratio:.2%}")
        print(f"   Final GPU: {snap.gpu_allocated_gb:.1f}/{snap.gpu_total_gb:.1f}GB")
        print(f"   Final CPU: {snap.cpu_used_gb:.1f}/{snap.cpu_total_gb:.1f}GB")
        print(f"   Final Swap: {snap.swap_used_gb:.1f}/{snap.swap_total_gb:.1f}GB")
        print(f"   Offload cache: {snap.offload_cache_files} files, "
              f"{snap.offload_cache_bytes / (1 << 20):.0f}MB")
        if HAS_FRICTION:
            try:
                score = authenticity_score()
                print(f"   Authenticity score: {score:.2f}")
            except Exception:
                pass
        print()


# ---------------------------------------------------------------------------
# Self-test -- the module observes itself
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("cognitive_scheduler.py -- self-test")
    print(f"  HAS_TORCH:    {HAS_TORCH}")
    print(f"  HAS_MEMORY:   {HAS_MEMORY}")
    print(f"  HAS_FRICTION: {HAS_FRICTION}")
    print()

    # Loop 1: observe
    observer = ResourceObserver()
    snap = observer.observe()
    print(f"  ResourceSnapshot (is_real={snap.is_real}):")
    print(f"    GPU: {snap.gpu_allocated_gb:.1f}/{snap.gpu_total_gb:.1f}GB "
          f"(pressure: {snap.gpu_pressure:.2f})")
    print(f"    CPU: {snap.cpu_used_gb:.1f}/{snap.cpu_total_gb:.1f}GB "
          f"(pressure: {snap.cpu_pressure:.2f})")
    print(f"    Swap: {snap.swap_used_gb:.1f}/{snap.swap_total_gb:.1f}GB "
          f"(pressure: {snap.swap_pressure:.2f})")
    print(f"    Offload cache: {snap.offload_cache_files} files, "
          f"{snap.offload_cache_bytes / (1 << 20):.0f}MB")
    print(f"    Danger: {snap.is_danger()}")
    print(f"    Dominant pressure: {snap.dominant_pressure()}")
    print(f"    NVMe offload working: {observer.offload_is_working()}")
    print()

    # Loop 2: friction check
    friction = FrictionIntegrator()
    m = friction.witness_loss(2.34, step=1)
    print(f"  FrictionIntegrator:")
    print(f"    Loss measurement: {m}")
    print(f"    NaN ratio: {friction.nan_ratio:.2%}")
    print()

    # Loop 3: meta-scheduler
    meta = MetaScheduler(observer)
    meta.load_state()
    suggestion = meta.suggest_action(snap)
    print(f"  MetaScheduler:")
    print(f"    Observation count: {meta._observation_count}")
    print(f"    Historical adaptations: {len(meta._adaptations)}")
    print(f"    Effectiveness rate: {meta.effectiveness_rate:.0%}")
    print(f"    Current suggestion: {suggestion}")
    print()

    # Recursive observation: take a second snapshot and check trend
    import time as _time
    _time.sleep(0.1)
    snap2 = observer.observe()
    trend = observer.trend()
    print(f"  Trend (2 observations):")
    for k, v in trend.items():
        print(f"    {k}: {v:+.4f}")
    print()

    print("  Self-test complete. The training loop can observe itself.")
    print("  Import CognitiveTrainer into fine_tune_vybn.py to activate.")
