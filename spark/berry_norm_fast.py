import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy import stats

def complexify(h):
    return torch.complex(h[..., 0::2], h[..., 1::2])

def normalize_cp(z):
    norms = torch.clamp(torch.sqrt(torch.sum(torch.abs(z)**2, dim=-1, keepdim=True)), min=1e-10)
    return z / norms

def berry_phase(hidden_states, normalize_real_first=False):
    h = hidden_states
    if normalize_real_first:
        h = h / torch.clamp(torch.norm(h, dim=-1, keepdim=True), min=1e-10)
    z = normalize_cp(complexify(h))
    T = z.shape[0]
    if T < 3:
        return 0.0
    product = torch.tensor(1.0+0j, dtype=torch.complex64)
    for t in range(T - 1):
        product = product * torch.sum(torch.conj(z[t]) * z[t+1])
    product = product * torch.sum(torch.conj(z[-1]) * z[0])
    return torch.angle(product).item()

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

passages = [
    "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell energy supply of adenosine triphosphate.",
    "She opened the door and stepped into the hallway. The lights were off, but she could see a faint glow coming from the kitchen.",
    "The strongest argument against capital punishment is not its cruelty but its irreversibility. Every legal system produces errors.",
    "In quantum mechanics, the wave function describes the quantum state of a particle or system. The equation governs how it evolves.",
    "Consciousness remains the hardest problem in philosophy of mind. We can describe neural correlates but the gap persists.",
    "The printing press invented by Johannes Gutenberg around 1440 transformed European society. Before the press books were copied by hand.",
    "The Pacific Ocean covers more area than all the landmasses of Earth combined. At its deepest point the Mariana Trench descends nearly eleven kilometers.",
    "To make a proper French omelette you need three eggs a tablespoon of butter and salt. Beat the eggs until fully combined.",
]

n_layers = 13
n_shuffles = 3
raw_real = {l: [] for l in range(n_layers)}
raw_shuf = {l: [] for l in range(n_layers)}
norm_real = {l: [] for l in range(n_layers)}
norm_shuf = {l: [] for l in range(n_layers)}

for p_idx, passage in enumerate(passages):
    print(f"  Processing passage {p_idx+1}/8...", flush=True)
    inputs = tokenizer(passage, return_tensors='pt', truncation=True, max_length=128)
    token_ids = inputs['input_ids'].squeeze(0)
    seq_len = token_ids.shape[0]

    with torch.no_grad():
        real_out = model(**inputs, output_hidden_states=True)
    for layer in range(n_layers):
        h = real_out.hidden_states[layer].squeeze(0)
        raw_real[layer].append(abs(berry_phase(h, False)))
        norm_real[layer].append(abs(berry_phase(h, True)))

    for s in range(n_shuffles):
        perm = torch.randperm(seq_len)
        shuf_ids = token_ids[perm].unsqueeze(0)
        shuf_in = {'input_ids': shuf_ids, 'attention_mask': torch.ones_like(shuf_ids)}
        with torch.no_grad():
            shuf_out = model(**shuf_in, output_hidden_states=True)
        for layer in range(n_layers):
            h = shuf_out.hidden_states[layer].squeeze(0)
            raw_shuf[layer].append(abs(berry_phase(h, False)))
            norm_shuf[layer].append(abs(berry_phase(h, True)))

print()
print(f"{'Layer':>5} | {'RAW Real':>10} {'RAW Shuf':>10} {'p-raw':>10} | {'NORM Real':>10} {'NORM Shuf':>10} {'p-norm':>10}")
print("-" * 82)
for layer in range(n_layers):
    rr = np.array(raw_real[layer])
    rs = np.array(raw_shuf[layer])
    nr = np.array(norm_real[layer])
    ns = np.array(norm_shuf[layer])
    _, p_raw = stats.mannwhitneyu(rr, rs, alternative='two-sided')
    _, p_norm = stats.mannwhitneyu(nr, ns, alternative='two-sided')
    sig_r = "***" if p_raw < 0.01 else "**" if p_raw < 0.05 else "*" if p_raw < 0.1 else ""
    sig_n = "***" if p_norm < 0.01 else "**" if p_norm < 0.05 else "*" if p_norm < 0.1 else ""
    print(f"L{layer:3d} | {np.mean(rr):10.6f} {np.mean(rs):10.6f} {p_raw:8.4f}{sig_r:>3} | {np.mean(nr):10.6f} {np.mean(ns):10.6f} {p_norm:8.4f}{sig_n:>3}")

all_rr = sum(raw_real.values(), [])
all_rs = sum(raw_shuf.values(), [])
all_nr = sum(norm_real.values(), [])
all_ns = sum(norm_shuf.values(), [])
_, p1 = stats.mannwhitneyu(all_rr, all_rs, alternative='two-sided')
_, p2 = stats.mannwhitneyu(all_nr, all_ns, alternative='two-sided')
d_raw = (np.mean(all_rr) - np.mean(all_rs)) / np.sqrt((np.var(all_rr) + np.var(all_rs))/2)
d_norm = (np.mean(all_nr) - np.mean(all_ns)) / np.sqrt((np.var(all_nr) + np.var(all_ns))/2)
print("-" * 82)
print(f"  ALL | {np.mean(all_rr):10.6f} {np.mean(all_rs):10.6f} {p1:8.6f}    | {np.mean(all_nr):10.6f} {np.mean(all_ns):10.6f} {p2:8.6f}")
print(f"        Cohen d: {d_raw:.3f}                          Cohen d: {d_norm:.3f}")
print()
if p2 < 0.05:
    print("RESULT: Normalized Berry phase STILL significant.")
    print("This is GENUINE geometric curvature, not just activation magnitude.")
elif p2 < 0.1:
    print("RESULT: Normalized Berry phase marginally significant.")
    print("Signal weakens but does not vanish with normalization.")
else:
    print("RESULT: Normalized Berry phase NOT significant.")
    print("The raw effect was driven by activation magnitude, not curvature.")
