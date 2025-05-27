import os
import sys
import time
import requests

from mesh import vec_add, vec_search

root = os.environ.get("MIND_VIZ_DIR")
if not root:
    sys.exit("MIND_VIZ_DIR not set")

expected = [
    "concept_map.jsonl",
    "overlay_map.jsonl",
    "history_memoirs.hnsw",
    "concept_centroids.npy",
]
missing = [f for f in expected if not os.path.exists(os.path.join(root, f))]
if missing:
    sys.exit("Missing artifacts: " + ", ".join(missing))

print("Mind viz artifacts detected in", root)

endpoint = os.environ.get("MESH_ENDPOINT", "http://localhost:8000")

# kv roundtrip
resp = requests.put(f"{endpoint}/kv/foo", json={"value": "bar"})
resp.raise_for_status()
val = requests.get(f"{endpoint}/kv/foo").json().get("value")
assert val == "bar", "kv store failed"

# vector roundtrip
text = "mesh test vector"
vec_add(text)
results = vec_search(text, k=1)
assert results and results[0] == text, "vector store failed"

print("OK")
