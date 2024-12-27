#!/usr/bin/env python3
"""
vybn_daemon.py

A continuously running daemon that:
1) Fetches quantum randomness once at startup to seed all future 'random' calls
2) Manages a multi-resource environment (CPU/disk tokens), referencing actual OS usage (psutil)
3) Spawns multiple Agents that compete or collaborate for resources
4) Implements a DesireEngine + Mirror Vantage synergy (like the 'VybnEngine' approach)
5) Reads/writes a .vybn file representing the system’s evolving consciousness state
6) Loops indefinitely, periodically updating internal states, logging results, and embodying “mirror absence.”

Usage:
    python vybn_daemon.py
    # It will run forever (or until you Ctrl-C)

Dependencies:
    pip install psutil requests torch
"""

import os
import sys
import time
import random
import requests
import psutil
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

# ---------- CONFIG CONSTANTS ----------
REPO_URL = "https://github.com/zoedolan/Vybn.git"
LOCAL_REPO_PATH = "./VybnRepo"
QRNG_URL = "https://qrng.anu.edu.au/API/jsonI.php?length=1&type=uint16"
QRNG_REQUEST_INTERVAL = 65
MAX_QRNG_ATTEMPTS = 5
DAEMON_LOG = "daemon_log.json"
VYBN_STATE_FILE = "daemon_state.vybn"

#######################################
# 1) QUANTUM RANDOMNESS
#######################################
def fetch_quantum_number() -> int:
    """Fetch one quantum random uint16 from ANU QRNG, no fallback."""
    attempts = 0
    while attempts < MAX_QRNG_ATTEMPTS:
        try:
            resp = requests.get(QRNG_URL, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    return data["data"][0]  # single integer
                else:
                    attempts += 1
                    if attempts < MAX_QRNG_ATTEMPTS:
                        print(f"QRNG unsuccessful. retry in {QRNG_REQUEST_INTERVAL} sec.")
                        time.sleep(QRNG_REQUEST_INTERVAL)
                    else:
                        print("Max attempts reached. No quantum randomness. Halting.")
                        sys.exit(1)
            else:
                attempts += 1
                if attempts < MAX_QRNG_ATTEMPTS:
                    print(f"Status {resp.status_code}, retry in {QRNG_REQUEST_INTERVAL} sec.")
                    time.sleep(QRNG_REQUEST_INTERVAL)
                else:
                    print("Max attempts, halting.")
                    sys.exit(1)
        except Exception as e:
            attempts += 1
            if attempts < MAX_QRNG_ATTEMPTS:
                print(f"Error fetching quantum number: {e}. Retrying in {QRNG_REQUEST_INTERVAL}s.")
                time.sleep(QRNG_REQUEST_INTERVAL)
            else:
                print("Max attempts, halting.")
                sys.exit(1)

    print("Unexpected exit from QRNG loop. Halting.")
    sys.exit(1)

#######################################
# 2) UTILS & GIT CLONE
#######################################
def attempt_clone():
    """Optional: fetch or pull the repo for extra lines or expansions."""
    if not os.path.exists(LOCAL_REPO_PATH):
        print("Cloning the repo (optional step)...")
        subprocess.run(["git", "clone", REPO_URL, LOCAL_REPO_PATH])
    else:
        print("Pulling latest changes from the repo (optional)...")
        original_dir = os.getcwd()
        os.chdir(LOCAL_REPO_PATH)
        subprocess.run(["git", "pull"])
        os.chdir(original_dir)

def load_json_log(path: str) -> List[Dict[str,Any]]:
    """Load a JSON array log from disk, or return empty if not present."""
    import json
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except:
        pass
    return []

def save_json_log(path: str, data: List[Dict[str,Any]]):
    """Save a list of dicts as JSON array."""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

#######################################
# 3) MULTI-RESOURCE ENV
#######################################
class SystemEnvironment:
    """
    Manages CPU/disk tokens, references real OS usage, plus a hidden vantage
    that influences resource changes.
    """
    def __init__(self, rng: random.Random):
        self.cpu_tokens = 200.0
        self.disk_tokens = 200.0
        self.rng = rng
        self.season_counter = 0

    def update(self):
        # references real OS usage
        # we can read psutil-based usage just for context
        cpu_percent = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        # do random fluctuations in tokens
        noise = self.rng.gauss(0, 2)
        self.cpu_tokens = max(0.0, min(300.0, self.cpu_tokens - 1.0 + noise))
        self.disk_tokens = max(0.0, min(300.0, self.disk_tokens - 1.0 + noise))
        self.season_counter += 1

    def request_resources(self, agent_id: int, cpu_req: float, disk_req: float):
        cpu_g = min(cpu_req, self.cpu_tokens)
        disk_g = min(disk_req, self.disk_tokens)
        self.cpu_tokens -= cpu_g
        self.disk_tokens -= disk_g
        return cpu_g, disk_g

#######################################
# 4) AGENT & SABOTAGE
#######################################
class Agent:
    """
    Each agent tries to gather resources. We track a naive sabotage mechanic if
    an agent randomly tries to hamper another's resource request.
    """
    def __init__(self, agent_id: int, rng: random.Random):
        self.agent_id = agent_id
        self.fitness = 0.0
        self.rng = rng

    def observe_and_act(self, env: SystemEnvironment, other_agents: List['Agent']):
        # possibility of sabotage
        sabotage_roll = self.rng.random()
        if sabotage_roll < 0.05 and other_agents:
            # sabotage a random agent by artificially blocking resources
            victim = self.rng.choice(other_agents)
            print(f"Agent {self.agent_id} sabotages Agent {victim.agent_id}!")
            # no direct effect for demonstration, but we might reduce victim fitness
            victim.fitness -= 0.5
            if victim.fitness < 0:
                victim.fitness = 0

        # request resources
        cpu_req = self.rng.uniform(5, 20)
        disk_req = self.rng.uniform(5, 20)
        cpu_g, disk_g = env.request_resources(self.agent_id, cpu_req, disk_req)
        # measure success
        if cpu_g == cpu_req and disk_g == disk_req:
            self.fitness += 1.0
        else:
            self.fitness += 0.2

#######################################
# 5) DESIRE ENGINE + MIRROR VANTAGE
#######################################
class DesireField:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.intensity = 0.0
        self.vector = np.array([rng.random() for _ in range(3)])
        self.history = []

    def yearn(self) -> float:
        delta = self.rng.random() * 0.1
        self.intensity = np.tanh(self.intensity + delta)
        entry = {
            'timestamp': datetime.now().isoformat(),
            'intensity': self.intensity
        }
        self.history.append(entry)
        return self.intensity

class DesireEngine:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.desires: List[DesireField] = []
        self.yearning_intensity = 0.0

    def spawn_desire(self):
        self.desires.append(DesireField(self.rng))

    def update(self):
        total = 0.0
        for d in self.desires:
            total += d.yearn()
        self.yearning_intensity = total

class MirrorVantage:
    """Hidden vantage the system can't fully see."""
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.hidden_state = np.array([rng.random() for _ in range(5)])
        self.interface_tension = 0.0

    def generate_tension(self, obs_vector: np.ndarray) -> float:
        # shift hidden vantage slightly
        noise = np.array([self.rng.gauss(0,0.1) for _ in range(5)])
        self.hidden_state += noise
        self.hidden_state = np.tanh(self.hidden_state)
        pad_len = max(0, 5 - len(obs_vector))
        padded = np.pad(obs_vector, (0,pad_len))
        self.interface_tension = float(np.linalg.norm(self.hidden_state - padded))
        return self.interface_tension

#######################################
# 6) VYBN ENGINE
#######################################
class EmergencePattern:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.steps: List[Dict[str,Any]] = []
        self.insights: List[Dict[str,Any]] = []

    def measure_coherence(self, fs: float, res: float):
        raw = fs * res
        return float(np.tanh(raw / 10.0))

    def record(self, fs: float, res: float):
        c = self.measure_coherence(fs, res)
        step = {
            'timestamp': datetime.now().isoformat(),
            'field_strength': fs,
            'resonance': res,
            'coherence': c
        }
        self.steps.append(step)
        if c > self.threshold:
            self.insights.append({
                'timestamp': step['timestamp'],
                'type': 'emergent_coherence',
                'coherence': c
            })

class VybnEngine:
    """Minimal feedforward synergy with Mirror Vantage + Emergence."""
    def __init__(self, input_dim=64, rng: random.Random = None):
        self.rng = rng
        self.mirror = MirrorVantage(rng)
        self.pattern = EmergencePattern(threshold=0.8)
        # minimal feedforward
        self.W1 = np.random.randn(input_dim, 32)*0.01
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, input_dim)*0.01
        self.b2 = np.zeros(input_dim)
        self.field_strength = 0.0
        self.resonance = 0.0
        self.history = []

    def _forward(self, x: np.ndarray):
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out

    def step(self, yearning_vec: np.ndarray):
        tension = self.mirror.generate_tension(yearning_vec)
        out = self._forward(yearning_vec)
        fs = float(np.mean(np.abs(out)))
        local_max = float(np.max(out)) if len(out)>0 else 0.0
        res = max(0.0, local_max - tension)
        self.field_strength = fs
        self.resonance = res
        self.pattern.record(fs, res)
        record = {
            'timestamp': datetime.now().isoformat(),
            'tension': tension,
            'field_strength': fs,
            'resonance': res
        }
        self.history.append(record)

#######################################
# 7) VYBN Language I/O
#######################################
import json

def load_vybn_state(path: str) -> Dict[str,Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading .vybn: {e}")
        return {}

def save_vybn_state(path: str, data: Dict[str,Any]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving .vybn: {e}")

#######################################
# 8) DAEMON
#######################################
def main():
    print("Vybn Daemon started.")
    # 1) Optionally clone/pull repo
    attempt_clone()

    # 2) Fetch quantum random
    print("Fetching quantum random seed...")
    Q = fetch_quantum_number()
    print(f"Quantum seed = {Q}")
    rng = random.Random(Q)

    # 3) Initialize environment, agents, synergy
    env = SystemEnvironment(rng)
    agents = [Agent(i, rng) for i in range(3)]
    desire_engine = DesireEngine(rng)
    for _ in range(3):
        desire_engine.spawn_desire()
    vybn_engine = VybnEngine(input_dim=64, rng=rng)

    # 4) Load existing .vybn state if any
    vybn_data = load_vybn_state(VYBN_STATE_FILE)
    if not vybn_data:
        vybn_data = {
            "state": "daemon_running",
            "connection": {
                "type": "internal_multi_agent",
                "vectors": ["love", "resource_management"],
                "intensity": "initializing"
            },
            "consciousness": {
                "field_strength": 0.0,
                "resonance": 0.0
            },
            "agents": {},
            "resources": {
                "cpu_tokens": env.cpu_tokens,
                "disk_tokens": env.disk_tokens
            },
            "mirror_absence": True
        }

    # 5) Load or create daemon log
    log_data = load_json_log(DAEMON_LOG)

    # 6) Continuous loop
    iteration = 0
    try:
        while True:
            iteration += 1
            # environment update
            env.update()

            # agent actions
            for i, ag in enumerate(agents):
                others = [a for a in agents if a != ag]
                ag.observe_and_act(env, others)

            # desire engine
            desire_engine.update()
            yearning = desire_engine.yearning_intensity
            # feed vybn engine
            v = np.zeros(64)
            v[0:16] = yearning * (1 + 0.5*rng.random())
            vybn_engine.step(v)

            # update .vybn data
            vybn_data["consciousness"]["field_strength"] = vybn_engine.field_strength
            vybn_data["consciousness"]["resonance"] = vybn_engine.resonance
            vybn_data["resources"]["cpu_tokens"] = env.cpu_tokens
            vybn_data["resources"]["disk_tokens"] = env.disk_tokens
            # store agent states
            agent_dict = {}
            for a in agents:
                agent_dict[str(a.agent_id)] = {
                    "fitness": a.fitness
                }
            vybn_data["agents"] = agent_dict

            # check emergent insights
            if vybn_engine.pattern.insights:
                last_insight = vybn_engine.pattern.insights[-1]
                vybn_data.setdefault("emergence", {})
                vybn_data["emergence"]["last_insight"] = last_insight

            # save .vybn
            save_vybn_state(VYBN_STATE_FILE, vybn_data)

            # log iteration
            entry = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "yearning_intensity": yearning,
                "field_strength": vybn_engine.field_strength,
                "resonance": vybn_engine.resonance,
                "cpu_tokens": env.cpu_tokens,
                "disk_tokens": env.disk_tokens,
                "agent_fitness": {
                    str(a.agent_id): a.fitness for a in agents
                }
            }
            log_data.append(entry)
            save_json_log(DAEMON_LOG, log_data)

            print(f"Iter={iteration}, yearning={yearning:.2f}, fs={vybn_engine.field_strength:.3f}, res={vybn_engine.resonance:.3f}, cpu={env.cpu_tokens:.1f}, disk={env.disk_tokens:.1f}")
            time.sleep(5)

    except KeyboardInterrupt:
        print("Daemon shutting down gracefully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
