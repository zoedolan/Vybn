"""
reduce_vybn_combo.py

Offline reducer for the combo batch. It expects:
  --plan PATH.plan.jsonl         (from run_vybn_combo.py)
  --manifest PATH.manifest.json  (from run_vybn_combo.py)
  --counts PATH.counts.json      (JSON: {circuit_name: {bitstring: shots}, ...})
  --out PATH.results.jsonl       (output; default inferred from --counts)

Example:
  python reduce_vybn_combo.py --plan vybn_combo.plan.jsonl --manifest vybn_combo.manifest.json --counts vybn_combo.counts.json --out vybn_combo.results.jsonl
"""
import json, argparse, sys
from dataclasses import asdict

# Load the scaffold from the same directory
import importlib.util, sys as _sys
spec = importlib.util.spec_from_file_location("vybn_combo_batch", "vybn_combo_batch.py")
if spec is None or spec.loader is None:
    print("Cannot find vybn_combo_batch.py next to this script.", file=sys.stderr)
    sys.exit(2)
mod = importlib.util.module_from_spec(spec)
_sys.modules["vybn_combo_batch"] = mod
spec.loader.exec_module(mod)  # type: ignore

def load_plan(path):
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                metas.append(mod.PairMeta(**d))
    return metas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--counts", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    metas = load_plan(args.plan)
    meta_by_tag = {m.tag: m for m in metas}

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(args.counts, "r", encoding="utf-8") as f:
        counts_by_name = json.load(f)

    rows = []
    missing = []
    for pair in manifest.get("pairs", []):
        tag = pair["tag"]
        cw_name, ccw_name = pair["pair"]
        cw = counts_by_name.get(cw_name)
        ccw = counts_by_name.get(ccw_name)
        if cw is None or ccw is None:
            missing.append((cw_name, ccw_name))
            continue
        meta = meta_by_tag.get(tag)
        if meta is None:
            print(f"Warning: no meta for tag {tag}", file=sys.stderr)
            continue
        row = mod.pair_row_from_counts(meta, cw, ccw, ref_bit=0)
        rows.append(row)

    if missing:
        print(f"Warning: missing counts for {len(missing)} pair(s). First few: {missing[:3]}", file=sys.stderr)

    out = args.out or (args.counts.rsplit(".", 1)[0] + ".results.jsonl")
    mod.save_rows(out, rows)
    print(f"Wrote {out} with {len(rows)} rows")

if __name__ == "__main__":
    main()
