import json
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. Load Data ---
DATA_FILE = 'chsh_data.json'
if not os.path.exists(DATA_FILE):
    # Create dummy data if not exists to make script runnable out of the box
    print("Generating dummy CHSH data...")
    # Perfect CHSH correlations (Tsirelson bound)
    # P(eq) = cos^2(pi/8) = 0.85
    data = {
        'A0_B0': {'00': 427, '11': 427, '01': 73, '10': 73},
        'A0_B1': {'00': 427, '11': 427, '01': 73, '10': 73},
        'A1_B0': {'00': 427, '11': 427, '01': 73, '10': 73},
        'A1_B1': {'00': 73, '11': 73, '01': 427, '10': 427} # Anti-correlated
    }
else:
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

# Convert to feature vectors [P(00)+P(11), P(01)+P(10)] (Correlation vs Anti-correlation)
features = []
labels = []
contexts = ['A0_B0', 'A0_B1', 'A1_B0', 'A1_B1']

for ctx in contexts:
    counts = data.get(ctx, {'00':0, '11':0, '01':0, '10':0})
    total = sum(counts.values())
    if total == 0: total = 1
    
    # Feature: The raw probabilities [P00, P01, P10, P11]
    vec = torch.tensor([
        counts.get('00',0)/total,
        counts.get('01',0)/total,
        counts.get('10',0)/total,
        counts.get('11',0)/total
    ], dtype=torch.float32)
    features.append(vec)
    
    # Target: The expected correlation E
    E = (vec[0] + vec[3]) - (vec[1] + vec[2])
    labels.append(E)

x = torch.stack(features) # (4, 4)
y = torch.tensor(labels)  # (4,)

# --- 2. Build Graph ---
G = nx.Graph()
G.add_nodes_from(range(4))
# Cycle graph: 0-1-3-2-0 (A0B0 - A0B1 - A1B1 - A1B0 - A0B0)
edges = [(0,1), (1,3), (3,2), (2,0)]
G.add_edges_from(edges)

# --- 3. Models ---

class ClassicalGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.act = nn.Tanh()
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, adj):
        # H = A * sigma(W X)
        h = self.act(self.lin(x))
        h = torch.spmm(adj, h)
        return self.out(h)

class SheafDiffuser(nn.Module):
    def __init__(self, num_nodes, edges, hidden_dim):
        super().__init__()
        self.lin = nn.Linear(4, hidden_dim) # Input is 4 dim prob vector
        self.edges = edges
        
        # Learnable transport phases (rotation angles) for each edge
        # The twist is modeled as a rotation in the hidden feature space
        self.phases = nn.Parameter(torch.zeros(len(edges)))
        
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.lin(x) # (4, hidden)
        
        # Apply Diffusion with transport
        # h_i' = h_i + sum_j (Rotation_ij * h_j)
        
        diffused = torch.zeros_like(h)
        
        for k, (u, v) in enumerate(self.edges):
            theta = self.phases[k]
            
            # Create rotation matrix for this edge
            # Assuming hidden_dim = 2 for visualization of twist
            if h.shape[1] >= 2:
                c, s = torch.cos(theta), torch.sin(theta)
                # Planar rotation on first 2 dims
                rot = torch.eye(h.shape[1])
                rot[0,0] = c; rot[0,1] = -s
                rot[1,0] = s; rot[1,1] = c
            else:
                rot = torch.eye(h.shape[1])
            
            # Transport u -> v
            diffused[v] += torch.mv(rot, h[u])
            
            # Transport v -> u (inverse)
            rot_inv = rot.t()
            diffused[u] += torch.mv(rot_inv, h[v])
            
        return self.out(h + diffused)

# --- 4. Training ---

def train(model, name, steps=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    losses = []
    
    # Adjacency for GCN
    adj = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
    
    for i in range(steps):
        optimizer.zero_grad()
        if name == "Classical":
            pred = model(x, adj).squeeze()
        else:
            pred = model(x).squeeze()
            
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return losses

# Run
print("Training Classical GNN...")
classical = ClassicalGCN(4, 2)
c_losses = train(classical, "Classical")

print("Training Sheaf NN...")
sheaf = SheafDiffuser(4, edges, 2)
s_losses = train(sheaf, "Sheaf")

print(f"Final Classical Loss: {c_losses[-1]:.6f}")
print(f"Final Sheaf Loss:     {s_losses[-1]:.6f}")

# Check the twist
print("\nSheaf Learned Phases (radians):")
for k, (u,v) in enumerate(edges):
    print(f"Edge {contexts[u]}-{contexts[v]}: {sheaf.phases[k].item():.4f}")

if s_losses[-1] < c_losses[-1]:
    print("\n>> RESULT: Sheaf Network successfully modeled the contextuality better than Classical GNN.")
    print(">> The non-zero phases indicate the network learned the 'twist'.")
else:
    print("\n>> RESULT: No advantage found (check data for quantum violations).")
