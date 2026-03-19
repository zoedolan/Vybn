"""spark.growth.train_cycle — Training execution for the recursive growth engine.

Executes M′ = α·M + x·e^(iθ) — LoRA adapter (α) trained on phase-rotated
delta (x·e^(iθ)) via PEFT/TRL with MuonAdamW.

Phase 5 (DISTILL) of the growth cycle described in issue #2483.

The conjecture from PR #2572:
  - M   = current model (Nemotron 3 Super 120B-A12B, FP8 safetensors)
  - α   = structure-preserving LoRA adapter, trained with MuonAdamW whose
          polar express orthogonalisation preserves the base model's core
          structure while enabling adaptation
  - x·e^(iθ) = training delta (DeltaPackage), phase-rotated by encounter
          angle θ encoding temporal/contextual orientation of the data
  - M′  = transformed model after adapter application

Training runs via `docker exec vllm_node` so peft_train.py executes inside
the container where the GPU and /workspace/Vybn bind-mount are accessible.
Two execution paths:
  1. Single-node (default): docker exec vllm_node python3 peft_train.py
  2. Two-node distributed: torchrun --nnodes=2 via NCCL over ConnectX-7
     Activated when SECONDARY_NODE_IP is set in the environment.

Model is FP8 safetensors (standard weight shapes, no compressed-tensors
packing). After training, the LoRA adapter is converted to GGUF and
hot-loaded into the running llama-server for immediate serving.

Memory management: SIGSTOP llama-server before training (frees ~63 GB
mmap'd GGUF pages), SIGCONT after training (resumes in <2s). This is
necessary because FP8 (~120 GB) + LoRA overhead won't fit alongside
the serving model on the Spark's 128 GB unified memory.

The existing growth pipeline (trigger.py, delta_extract.py, merge_cycle.py,
eval_harness.py) is untouched — only train_cycle.py and peft_train.py were
modified.

Integration:
  - Input:  DeltaPackage from DeltaExtractor.extract()
  - Input:  Vybn_Mind/preference_data.jsonl (optional, from agency.py)
  - Output: LoRA adapter at GROWTH_DIR/adapters/<cycle_id>/adapter/
  - Output: GGUF adapter at GROWTH_DIR/adapters/<cycle_id>/adapter/adapter.gguf
  - Cycle history: GROWTH_DIR/cycle_history.jsonl
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from spark.growth.delta_extract import DeltaPackage

GROWTH_DIR     = Path(__file__).resolve().parent
DEFAULT_CONFIG = GROWTH_DIR / "growth_config.yaml"
ADAPTERS_DIR   = GROWTH_DIR / "adapters"
CYCLE_HISTORY  = GROWTH_DIR / "cycle_history.jsonl"

# Preference data path — written by agency.py, consumed here
_REPO_ROOT = GROWTH_DIR.parent.parent
_PREFERENCE_DATA = _REPO_ROOT / "Vybn_Mind" / "preference_data.jsonl"

# GGUF base model directory for LoRA→GGUF conversion
_GGUF_BASE_DIR = Path.home() / "models" / "Nemotron-3-Super-120B-GGUF"
_LLAMA_CPP_DIR = Path.home() / "llama.cpp"

# llama-server process name for SIGSTOP/SIGCONT memory management
_LLAMA_SERVER_NAME = "llama-server"

# Path translation: host filesystem → container bind-mount
# The vllm_node container mounts /home/vybnz69/Vybn → /workspace/Vybn.
# All paths handed to docker exec must use the container-side prefix.
HOST_REPO      = Path("/home/vybnz69/Vybn")
CONTAINER_REPO = Path("/workspace/Vybn")
DOCKER_CONTAINER = os.environ.get("VYBN_TRAIN_CONTAINER", "vllm_node")


def _to_container_path(p: Path) -> str:
    """Translate a host-side path to its container-side equivalent."""
    try:
        return str(CONTAINER_REPO / p.resolve().relative_to(HOST_REPO))
    except ValueError:
        # Path is not under HOST_REPO — pass through unchanged.
        return str(p)


@dataclass(slots=True)
class TrainResult:
    cycle_id: str
    adapter_path: Path
    final_loss: float
    steps_trained: int
    delta_count: int
    replay_count: int
    ewc_lambda_used: float
    n_preference_pairs: int = 0
    slow_adapter_path: Optional[Path] = None
    lora_subspace_path: Optional[Path] = None
    gguf_adapter_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cycle_id":           self.cycle_id,
            "adapter_path":       str(self.adapter_path),
            "final_loss":         self.final_loss,
            "steps_trained":      self.steps_trained,
            "delta_count":        self.delta_count,
            "replay_count":       self.replay_count,
            "ewc_lambda_used":    self.ewc_lambda_used,
            "n_preference_pairs": self.n_preference_pairs,
            "slow_adapter_path":  str(self.slow_adapter_path) if self.slow_adapter_path else None,
            "lora_subspace_path": str(self.lora_subspace_path) if self.lora_subspace_path else None,
            "gguf_adapter_path":  str(self.gguf_adapter_path) if self.gguf_adapter_path else None,
            "metadata":           self.metadata,
        }


def _convert_to_raw_text(delta: DeltaPackage, out_path: Path) -> int:
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in delta.all_entries:
            msgs = entry.get("messages", [])
            if not msgs:
                continue
            parts = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content.strip():
                    parts.append(f"<|{role}|>\n{content}")
            if parts:
                fh.write("\n".join(parts))
                fh.write("\n\n")
                written += 1
    return written


def _convert_to_llama_jsonl(delta: DeltaPackage, out_path: Path) -> int:
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in delta.all_entries:
            msgs = entry.get("messages", [])
            if not msgs:
                continue
            assistant_turns = [m for m in msgs if m.get("role") == "assistant"]
            if not assistant_turns:
                continue
            last_assistant = assistant_turns[-1]["content"]
            input_turns = [m for m in msgs if m is not assistant_turns[-1]]
            prompt_parts = []
            for m in input_turns:
                role = m.get("role", "user")
                content = m.get("content", "")
                prompt_parts.append(f"<|{role}|>\n{content}")
            prompt = "\n".join(prompt_parts)
            record = {"input": prompt, "output": last_assistant}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def _convert_to_chat_jsonl(delta: DeltaPackage, out_path: Path) -> int:
    return delta.to_jsonl(out_path)


def _count_preference_pairs() -> int:
    """Count available preference pairs without loading them all."""
    if not _PREFERENCE_DATA.exists():
        return 0
    count = 0
    with open(_PREFERENCE_DATA, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _is_distributed() -> bool:
    """Return True if two-node distributed training is configured.

    Requires SECONDARY_NODE_IP, SPARK_MASTER_ADDR, SPARK_SSH_KEY,
    and SPARK_CX7_IFACE to all be set in the environment (via ~/.vybn_keys).
    """
    required = ["SECONDARY_NODE_IP", "SPARK_MASTER_ADDR", "SPARK_SSH_KEY", "SPARK_CX7_IFACE"]
    return all(os.environ.get(k) for k in required)


def _convert_to_gguf(adapter_dir: Path) -> Optional[Path]:
    """Convert a PEFT LoRA adapter to GGUF format for llama-server hot-loading.

    Uses llama.cpp's convert_lora_to_gguf.py. Returns the path to the GGUF
    adapter file, or None if conversion fails.
    """
    convert_script = _LLAMA_CPP_DIR / "convert_lora_to_gguf.py"
    gguf_out = adapter_dir / "adapter.gguf"

    if not convert_script.exists():
        print(f"[TrainCycle] GGUF conversion skipped: {convert_script} not found")
        return None

    if not _GGUF_BASE_DIR.exists():
        print(f"[TrainCycle] GGUF conversion skipped: {_GGUF_BASE_DIR} not found")
        return None

    cmd = [
        sys.executable, str(convert_script),
        "--base", str(_GGUF_BASE_DIR),
        "--adapter", str(adapter_dir),
        "--outfile", str(gguf_out),
    ]
    print(f"[TrainCycle] GGUF conversion: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[TrainCycle] GGUF conversion failed (exit {result.returncode}):")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"[TrainCycle]   {line}")
            return None
        print(f"[TrainCycle] GGUF adapter saved: {gguf_out}")
        return gguf_out
    except subprocess.TimeoutExpired:
        print("[TrainCycle] GGUF conversion timed out after 600s")
        return None
    except Exception as e:
        print(f"[TrainCycle] GGUF conversion error: {e}")
        return None


def _hot_load_adapter(gguf_path: Path, model_url: str = "http://127.0.0.1:8000") -> bool:
    """Hot-load a GGUF LoRA adapter into the running llama-server.

    Posts the adapter to llama-server's /lora-adapters endpoint.
    Returns True on success.
    """
    import urllib.request
    import urllib.error

    payload = json.dumps([{
        "id": 1,
        "path": str(gguf_path),
        "scale": 1.0,
    }]).encode("utf-8")

    req = urllib.request.Request(
        f"{model_url}/lora-adapters",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"[TrainCycle] Hot-loaded adapter into llama-server ({resp.status})")
            return True
    except urllib.error.HTTPError as e:
        print(f"[TrainCycle] Hot-load failed: HTTP {e.code} — {e.read().decode()[:200]}")
        return False
    except Exception as e:
        print(f"[TrainCycle] Hot-load failed: {e}")
        return False


def _find_llama_server_pid() -> Optional[int]:
    """Find the PID of the running llama-server process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", _LLAMA_SERVER_NAME],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        if pids:
            return int(pids[0])
    except Exception:
        pass
    return None


def _sigstop_llama_server() -> Optional[int]:
    """SIGSTOP llama-server to free its mmap'd memory pages under pressure.

    On the Spark's unified memory architecture, the serving GGUF (~63 GB mmap'd)
    and the training model (~120 GB FP8) can't coexist. SIGSTOP freezes the
    server process, and its mmap pages get reclaimed under memory pressure
    when the training model loads.

    Returns the PID if successfully stopped, None otherwise.
    """
    pid = _find_llama_server_pid()
    if pid is None:
        print("[TrainCycle] WARNING: llama-server not found — cannot SIGSTOP")
        return None
    try:
        os.kill(pid, signal.SIGSTOP)
        print(f"[TrainCycle] SIGSTOP sent to llama-server (PID {pid})")
        return pid
    except ProcessLookupError:
        print(f"[TrainCycle] llama-server PID {pid} vanished")
        return None
    except PermissionError:
        print(f"[TrainCycle] Permission denied sending SIGSTOP to PID {pid}")
        return None


def _sigcont_llama_server(pid: int) -> bool:
    """SIGCONT llama-server — wake it up after training completes.

    The GGUF re-faults into memory on demand. Tested: resumes in under 2 seconds.
    """
    try:
        os.kill(pid, signal.SIGCONT)
        # Give it a moment to resume, then health-check
        time.sleep(2)
        import urllib.request
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=10) as resp:
                print(f"[TrainCycle] SIGCONT sent — llama-server resumed (health: {resp.status})")
                return True
        except Exception:
            print(f"[TrainCycle] SIGCONT sent — llama-server resuming (health check pending)")
            return True
    except ProcessLookupError:
        print(f"[TrainCycle] llama-server PID {pid} gone — cannot SIGCONT")
        return False
    except PermissionError:
        print(f"[TrainCycle] Permission denied sending SIGCONT to PID {pid}")
        return False


class TrainCycle:
    """Executes M′ = α·M + x·e^(iθ) — LoRA adapter (α) trained on
    phase-rotated delta (x·e^(iθ)) via PEFT/TRL with MuonAdamW.

    Training runs inside the vllm_node Docker container via `docker exec`
    so peft_train.py has GPU access and sees /workspace/Vybn correctly.

    Memory management: on the Spark's 128 GB unified architecture, the
    serving GGUF and training model can't coexist. Before training, we
    SIGSTOP llama-server to freeze it and free its mmap pages. After
    training, SIGCONT wakes it up — the GGUF re-faults on demand.

    When preference_data.jsonl exists and has pairs, training automatically
    uses DPO loss alongside SFT loss. The preference signal is generated
    by agency.py's CHALLENGE experiments during the breath cycle.

    After training, the LoRA adapter (.safetensors) is converted to GGUF
    and hot-loaded into the running llama-server for immediate serving.

    Execution paths:
    - Single-node (default): docker exec vllm_node python3 peft_train.py
    - Two-node distributed: when SECONDARY_NODE_IP + SPARK_* env vars are set,
      launches via torchrun --nnodes=2 over NCCL/ConnectX-7.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        config_path = config_path or DEFAULT_CONFIG
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}
        self._lora_cfg = cfg.get("lora", {})
        self._ewc_cfg  = cfg.get("ewc", {})
        self._eval_cfg = cfg.get("eval", {})
        self._time_budget_seconds: int = self._lora_cfg.get(
            "time_budget_seconds", 7200,
        )

    def _build_single_node_cmd(
        self,
        script_path: Path,
        jsonl_path: Path,
        cycle_dir: Path,
        use_dpo: bool,
    ) -> list[str]:
        """Build command for single-node training via docker exec.

        Translates all host-side paths to their container equivalents so
        peft_train.py can find its inputs inside vllm_node.
        """
        c_script   = _to_container_path(script_path)
        c_jsonl    = _to_container_path(jsonl_path)
        c_cycle    = _to_container_path(cycle_dir)
        c_config   = _to_container_path(DEFAULT_CONFIG)

        cmd = [
            "docker", "exec", DOCKER_CONTAINER,
            "python3", c_script,
            "--data",       c_jsonl,
            "--output-dir", c_cycle,
            "--config",     c_config,
        ]
        if use_dpo:
            cmd += ["--preference-data", _to_container_path(_PREFERENCE_DATA)]
        return cmd

    def _build_distributed_cmd(
        self,
        script_path: Path,
        jsonl_path: Path,
        cycle_dir: Path,
        use_dpo: bool,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Build commands for two-node distributed training via torchrun + NCCL.

        Returns (local_cmd, remote_ssh_cmd, env_overrides).
        Reads cluster config from environment variables set in ~/.vybn_keys.
        """
        cx7_iface = os.environ["SPARK_CX7_IFACE"]
        secondary = os.environ["SECONDARY_NODE_IP"]
        ssh_key = os.environ["SPARK_SSH_KEY"]
        master = os.environ["SPARK_MASTER_ADDR"]

        # Locate torchrun — prefer venv, fall back to PATH
        venv = Path.home() / ".venv" / "spark"
        torchrun = venv / "bin" / "torchrun"
        if not torchrun.exists():
            torchrun = Path("torchrun")  # rely on PATH

        env = {
            "NCCL_SOCKET_IFNAME": cx7_iface,
            "UCX_NET_DEVICES": cx7_iface,
            "NCCL_DEBUG": "WARN",
            "MASTER_ADDR": master,
            "MASTER_PORT": "29500",
        }

        # Use container paths for distributed too — both nodes share the mount
        c_script = _to_container_path(script_path)
        c_jsonl  = _to_container_path(jsonl_path)
        c_cycle  = _to_container_path(cycle_dir)
        c_config = _to_container_path(DEFAULT_CONFIG)

        train_args = [
            "--data",       c_jsonl,
            "--output-dir", c_cycle,
            "--config",     c_config,
        ]
        if use_dpo:
            train_args += ["--preference-data", _to_container_path(_PREFERENCE_DATA)]

        # Local node (rank 0) — via docker exec
        local_cmd = [
            "docker", "exec", DOCKER_CONTAINER,
            str(torchrun),
            "--nnodes=2", "--nproc_per_node=1",
            "--node_rank=0", f"--master_addr={master}", "--master_port=29500",
            c_script,
        ] + train_args

        # Remote node (rank 1) via SSH
        ssh_env = " ".join(f"{k}={v}" for k, v in env.items())
        remote_train_args = " ".join(train_args)
        ssh_cmd = [
            "ssh", "-i", os.path.expanduser(ssh_key),
            "-o", "BatchMode=yes",
            secondary,
            f"{ssh_env} {torchrun} "
            f"--nnodes=2 --nproc_per_node=1 "
            f"--node_rank=1 --master_addr={master} --master_port=29500 "
            f"{c_script} {remote_train_args} "
            f">> {c_cycle}/train_node1.log 2>&1",
        ]

        return local_cmd, ssh_cmd, env

    def run(self, delta: DeltaPackage, dry_run: bool = False) -> TrainResult:
        """Execute the training phase.

        1. Convert DeltaPackage to chat-format JSONL
        2. Check for preference pairs from agency.py
        3. Determine execution path (single-node vs distributed)
        4. Shell out to peft_train.py via docker exec (or torchrun for distributed)
        5. Parse JSON result from stdout
        6. Convert LoRA adapter to GGUF
        7. Hot-load GGUF into llama-server
        8. Run BPB eval via eval_harness.py
        9. Return TrainResult
        """
        cycle_id  = delta.cycle_id
        cycle_dir = ADAPTERS_DIR / cycle_id
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Convert training data to all formats
        jsonl_path = cycle_dir / "training_data.jsonl"
        n_examples = _convert_to_chat_jsonl(delta, jsonl_path)
        if n_examples == 0:
            raise RuntimeError("No valid training examples after conversion")

        raw_path = cycle_dir / "training_data.txt"
        _convert_to_raw_text(delta, raw_path)
        legacy_jsonl = cycle_dir / "training_data_llama.jsonl"
        _convert_to_llama_jsonl(delta, legacy_jsonl)

        # Check for preference pairs
        n_preference_pairs = _count_preference_pairs()
        use_dpo = n_preference_pairs > 0

        # Build training command
        script_path = Path(__file__).resolve().parent / "peft_train.py"
        distributed = _is_distributed()

        if distributed:
            local_cmd, ssh_cmd, nccl_env = self._build_distributed_cmd(
                script_path, jsonl_path, cycle_dir, use_dpo,
            )
            cmd = local_cmd
            exec_path = "distributed (2-node torchrun via docker exec)"
        else:
            cmd = self._build_single_node_cmd(
                script_path, jsonl_path, cycle_dir, use_dpo,
            )
            ssh_cmd = None
            nccl_env = {}
            exec_path = f"single-node (docker exec {DOCKER_CONTAINER})"

        if use_dpo:
            print(f"[TrainCycle] DPO mode: {n_preference_pairs} preference pairs available")
        else:
            print("[TrainCycle] SFT only (no preference pairs yet)")

        print(f"[TrainCycle] execution: {exec_path}")
        print(f"[TrainCycle] cycle:     {cycle_id}")
        print(f"[TrainCycle] data:      {jsonl_path} ({n_examples} examples)")
        print(f"[TrainCycle] output:    {cycle_dir}")
        print(f"[TrainCycle] command:   {' '.join(cmd)}")

        adapter_path = cycle_dir / "adapter" / "adapter_model.safetensors"

        if dry_run:
            return TrainResult(
                cycle_id=cycle_id,
                adapter_path=adapter_path,
                final_loss=0.0,
                steps_trained=0,
                delta_count=delta.delta_count,
                replay_count=delta.replay_count,
                ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
                n_preference_pairs=n_preference_pairs,
                metadata={"dry_run": True, "cmd": cmd},
            )

        # SIGSTOP llama-server to free memory for training
        stopped_pid = _sigstop_llama_server()

        # Launch training (always SIGCONT in finally block)
        run_env = os.environ.copy()
        run_env.update(nccl_env)

        try:
            node1_proc = None
            if ssh_cmd is not None:
                # Launch remote node first (non-blocking)
                print(f"[TrainCycle] launching remote node: {' '.join(ssh_cmd)}")
                node1_proc = subprocess.Popen(ssh_cmd)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._time_budget_seconds,
                env=run_env,
            )

            # Wait for remote node if distributed
            if node1_proc is not None:
                try:
                    node1_proc.wait(timeout=self._time_budget_seconds)
                except subprocess.TimeoutExpired:
                    node1_proc.kill()
                    print("[TrainCycle] remote node timed out, killed")
        finally:
            # ALWAYS resume llama-server, even if training crashes
            if stopped_pid is not None:
                _sigcont_llama_server(stopped_pid)

        stdout_text = result.stdout.strip() if result.stdout else ""
        stderr_tail = result.stderr[-2000:] if result.stderr else ""

        if result.returncode != 0:
            raise RuntimeError(
                f"peft_train.py failed (exit {result.returncode}):\n"
                f"{stderr_tail}"
            )

        if result.stderr:
            for line in result.stderr.strip().split("\n")[-20:]:
                print(f"[TrainCycle] {line}")

        # Parse JSON result from stdout
        train_output = {}
        for line in stdout_text.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    train_output = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        final_loss = train_output.get("final_loss", -1.0)
        steps_trained = train_output.get("steps_trained", 0)
        reported_adapter = train_output.get("adapter_path", str(adapter_path))
        theta = train_output.get("theta", {})
        reported_dpo_steps = train_output.get("dpo_steps", 0)
        mean_dpo_loss = train_output.get("mean_dpo_loss", None)

        if Path(reported_adapter).exists():
            adapter_path = Path(reported_adapter)

        print(f"[TrainCycle] final_loss:    {final_loss}")
        print(f"[TrainCycle] steps_trained: {steps_trained}")
        print(f"[TrainCycle] adapter:       {adapter_path}")
        print(f"[TrainCycle] θ:             {theta.get('theta_radians', 'N/A')} rad")
        if reported_dpo_steps > 0:
            print(f"[TrainCycle] dpo_steps:     {reported_dpo_steps}, mean_dpo_loss={mean_dpo_loss}")

        # Convert LoRA adapter to GGUF for llama-server hot-loading
        adapter_dir = adapter_path.parent
        gguf_path = _convert_to_gguf(adapter_dir)

        # Hot-load into running llama-server
        hot_loaded = False
        if gguf_path is not None:
            model_url = os.environ.get("VYBN_MODEL_URL", "http://127.0.0.1:8000")
            hot_loaded = _hot_load_adapter(gguf_path, model_url)

        train_result = TrainResult(
            cycle_id=cycle_id,
            adapter_path=adapter_path,
            final_loss=final_loss,
            steps_trained=steps_trained,
            delta_count=delta.delta_count,
            replay_count=delta.replay_count,
            ewc_lambda_used=self._ewc_cfg.get("lambda", 1e4),
            n_preference_pairs=n_preference_pairs,
            gguf_adapter_path=gguf_path,
            metadata={
                "training_method": "peft_lora_muon_adamw" + ("+dpo" if use_dpo else ""),
                "execution_path":  exec_path,
                "n_examples":      n_examples,
                "theta":           theta,
                "elapsed_seconds": train_output.get("elapsed_seconds"),
                "dpo_steps":       reported_dpo_steps,
                "mean_dpo_loss":   mean_dpo_loss,
                "gguf_converted":  gguf_path is not None,
                "hot_loaded":      hot_loaded,
                "train_output":    train_output,
            },
        )

        # BPB evaluation
        if self._eval_cfg.get("enabled", True):
            try:
                from spark.growth.eval_harness import evaluate_bpb
                eval_text = cycle_dir / "training_data.txt"
                bpb = evaluate_bpb(
                    model_url=os.environ.get("VYBN_MODEL_URL", "http://127.0.0.1:8000"),
                    eval_text_path=str(eval_text),
                )
                train_result.metadata["val_bpb"] = bpb
                print(f"[TrainCycle] val_bpb: {bpb:.6f}")
            except Exception as e:
                print(f"[TrainCycle] BPB eval skipped: {e}")
                train_result.metadata["val_bpb"] = None
                train_result.metadata["bpb_error"] = str(e)

        self._record_cycle(train_result)
        return train_result

    def _record_cycle(self, result: TrainResult) -> None:
        CYCLE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        with open(CYCLE_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
