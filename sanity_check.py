import os
import sys

root = os.environ.get('MIND_VIZ_DIR')
if not root:
    sys.exit('MIND_VIZ_DIR not set')

expected = ['concept_map.jsonl', 'overlay_map.jsonl', 'history_memoirs.hnsw', 'concept_centroids.npy']
missing = [f for f in expected if not os.path.exists(os.path.join(root, f))]
if missing:
    sys.exit('Missing artifacts: ' + ', '.join(missing))

print('Mind viz artifacts detected in', root)
