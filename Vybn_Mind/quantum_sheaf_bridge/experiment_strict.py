import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# --- Configuration ---
SHOTS = 4096
DATA_OUTPUT_PATH = 'quantum_chsh_raw.json'

# --- Hardware Interface ---
def get_least_busy_backend(service):
    """Finds available real hardware with lowest queue."""
    backends = service.backends(simulator=False, operational=True)
    if not backends:
        raise RuntimeError("No operational real hardware backends found.")
    return min(backends, key=lambda b: b.status().pending_jobs)

def construct_circuits():
    """Generates CHSH test circuits."""
    circuits = []
    # Alice: 0 (Z), pi/2 (X)
    # Bob: pi/4 (Z+X), -pi/4 (Z-X)
    configs = [
        ('A0_B0', 0, np.pi/4),
        ('A0_B1', 0, -np.pi/4),
        ('A1_B0', np.pi/2, np.pi/4),
        ('A1_B1', np.pi/2, -np.pi/4)
    ]
    
    mapping = []
    for (name, th_a, th_b) in configs:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        if th_a != 0: qc.ry(-th_a, 0)
        if th_b != 0: qc.ry(-th_b, 1)
        qc.measure_all()
        circuits.append(qc)
        mapping.append(name)
    return circuits, mapping

def execute_job():
    token = os.getenv("QISKIT_IBM_TOKEN")
    if not token:
        # Try local creds
        try:
            service = QiskitRuntimeService(channel="ibm_quantum")
        except:
            print("Error: QISKIT_IBM_TOKEN missing and no local credentials found.")
            sys.exit(1)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)

    backend = get_least_busy_backend(service)
    print(f"Backend: {backend.name}")
    
    circuits, mapping = construct_circuits()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)
    
    sampler = Sampler(backend=backend)
    job = sampler.run(isa_circuits, shots=SHOTS)
    print(f"Job ID: {job.job_id()}")
    
    result = job.result()
    
    data = {}
    for i, item in enumerate(result):
        # Retrieve counts from BitArray
        counts = item.data.meas.get_counts()
        data[mapping[i]] = counts
        
    with open(DATA_OUTPUT_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data written to {DATA_OUTPUT_PATH}")
    return data

# --- Neural Architectures ---
class RotationLayer(nn.Module):
    """
    Applies SO(2) rotation to edge features.
    No activation, purely geometric transport.
    """
    def __init__(self, num_edges, hidden_dim):
        super().__init__()
        self.phases = nn.Parameter(torch.randn(num_edges))
        self.hidden_dim = hidden_dim

    def forward(self, x, edges):
        # x: (Nodes, Hidden)
        # Transport logic: v_j = R_ij * v_i
        out = torch.zeros_like(x)
        
        for k, (u, v) in enumerate(edges):
            theta = self.phases[k]
            c, s = torch.cos(theta), torch.sin(theta)
            
            # Rotation matrix for first 2 dims
            # Only supports 2D rotation for this experiment
            h_u = x[u]
            transported_u = torch.stack([
                h_u[0]*c - h_u[1]*s,
                h_u[0]*s + h_u[1]*c
            ])
            out[v] += transported_u
            
            # Inverse transport
            h_v = x[v]
            transported_v = torch.stack([
                h_v[0]*c + h_v[1]*s,
                -h_v[0]*s + h_v[1]*c
            ])
            out[u] += transported_v
            
        return x + out

class SheafNet(nn.Module):
    def __init__(self, edges):
        super().__init__()
        self.encoder = nn.Linear(4, 2) 
        self.diffuser = RotationLayer(len(edges), 2)
        self.decoder = nn.Linear(2, 1)
        self.edges = edges

    def forward(self, x):
        h = self.encoder(x)
        h = self.diffuser(h, self.edges)
        return self.decoder(h)

class StandardGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 2)
        self.decoder = nn.Linear(2, 1)
        
    def forward(self, x, adj):
        h = self.encoder(x)
        h = torch.spmm(adj, h) # Standard sum aggregation
        return self.decoder(h)

def run_analysis():
    if not os.path.exists(DATA_OUTPUT_PATH):
        print("Executing Hardware Job...")
        raw_data = execute_job()
    else:
        print(f"Loading cached data from {DATA_OUTPUT_PATH}")
        with open(DATA_OUTPUT_PATH, 'r') as f:
            raw_data = json.load(f)

    # Preprocessing
    x_list = []
    y_list = []
    keys = ['A0_B0', 'A0_B1', 'A1_B0', 'A1_B1']
    
    for key in keys:
        counts = raw_data.get(key, {})
        total = sum(counts.values())
        if total == 0: total = 1
        
        # Input: Normalized probabilities
        probs = [
            counts.get('00', 0)/total,
            counts.get('01', 0)/total,
            counts.get('10', 0)/total,
            counts.get('11', 0)/total
        ]
        x_list.append(torch.tensor(probs, dtype=torch.float32))
        
        # Target: Correlation
        # E = (N_same - N_diff) / Total
        E = (probs[0] + probs[3]) - (probs[1] + probs[2])
        y_list.append(E)

    X = torch.stack(x_list)
    Y = torch.tensor(y_list).unsqueeze(1)
    
    # Topology: Cycle 0-1-3-2
    edges = [(0,1), (1,3), (3,2), (2,0)]
    adj = torch.tensor([
        [0,1,1,0],
        [1,0,0,1],
        [1,0,0,1],
        [0,1,1,0]
    ], dtype=torch.float32)

    # Training
    model_s = SheafNet(edges)
    model_c = StandardGNN()
    
    opt_s = torch.optim.Adam(model_s.parameters(), lr=0.01)
    opt_c = torch.optim.Adam(model_c.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("\nTraining models...")
    for i in range(2000):
        # Sheaf Step
        opt_s.zero_grad()
        loss_s = criterion(model_s(X), Y)
        loss_s.backward()
        opt_s.step()
        
        # Classical Step
        opt_c.zero_grad()
        loss_c = criterion(model_c(X, adj), Y)
        loss_c.backward()
        opt_c.step()

    print(json.dumps({
        "final_loss_sheaf": loss_s.item(),
        "final_loss_classical": loss_c.item(),
        "learned_phases": model_s.diffuser.phases.detach().numpy().tolist()
    }, indent=2))

if __name__ == "__main__":
    run_analysis()
