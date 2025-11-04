
"""
post_reducer_qca.py — Walk a Vybn counts.json and emit per-qubit Δ, slopes, MI, and ZZ for QCA pairs.

Usage:
  python post_reducer_qca.py --counts vybn_combo.counts.json --out qca_post_reduce.csv

Notes:
- Detects QCA pairs by keys starting with "qca_" and pairing *_cw with *_ccw.
- For staticlinks (3 bits per string) we treat all measured bits as matter.
- For linkreg (5 bits per string) we treat the first 3 bits as matter and ignore the last 2 link bits for MI/ZZ.
- Bit indices are taken "leftmost=0" for reproducibility; if your endianness differs, flip by passing --right-endian.
"""
import json, math, re, argparse
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd

def parse_area_from_key(key: str) -> float:
    m = re.search(r"_a([0-9]+\.[0-9]+)", key)
    if m: return float(m.group(1))
    m = re.search(r"_a([0-9]+)", key)
    return float(m.group(1)) if m else float("nan")

def group_pairs(counts_dict):
    bases = defaultdict(dict)
    for name in counts_dict.keys():
        if name.endswith("_cw"):
            bases[name[:-3]]['cw'] = name
        elif name.endswith("_ccw"):
            bases[name[:-4]]['ccw'] = name
    return {b:(v.get('cw'),v.get('ccw')) for b,v in bases.items() if 'cw' in v and 'ccw' in v}

def bit_prob(counts, bit_index, target='1', endian='left'):
    tot = sum(counts.values())
    if tot == 0: return float('nan')
    acc = 0
    for bstr, c in counts.items():
        idx = bit_index if endian=='left' else (len(bstr)-1-bit_index)
        if 0 <= idx < len(bstr) and bstr[idx]==target:
            acc += c
    return acc / tot

def mi_two_bits(counts, i, j, endian='left'):
    tot = sum(counts.values())
    if tot == 0: return float('nan')
    joint = Counter()
    for bstr, c in counts.items():
        bi = bstr[i] if endian=='left' else bstr[len(bstr)-1-i]
        bj = bstr[j] if endian=='left' else bstr[len(bstr)-1-j]
        joint[(bi,bj)] += c
    ps = {k: v/tot for k,v in joint.items()}
    pi = {'0': ps.get(('0','0'),0)+ps.get(('0','1'),0),
          '1': ps.get(('1','0'),0)+ps.get(('1','1'),0)}
    pj = {'0': ps.get(('0','0'),0)+ps.get(('1','0'),0),
          '1': ps.get(('0','1'),0)+ps.get(('1','1'),0)}
    mi = 0.0
    for a in ('0','1'):
        for b in ('0','1'):
            p = ps.get((a,b),0)
            if p>0 and pi[a]>0 and pj[b]>0:
                mi += p * math.log(p/(pi[a]*pj[b]), 2)
    return mi

def zz_corr(counts, i, j, endian='left'):
    tot = sum(counts.values())
    if tot == 0: return float('nan')
    val = 0.0
    for bstr, c in counts.items():
        bi = bstr[i] if endian=='left' else bstr[len(bstr)-1-i]
        bj = bstr[j] if endian=='left' else bstr[len(bstr)-1-j]
        zi = 1.0 if bi=='0' else -1.0
        zj = 1.0 if bj=='0' else -1.0
        val += zi*zj*c
    return val/tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--out", default="qca_post_reduce.csv")
    ap.add_argument("--json-out", default="qca_post_reduce.json")
    ap.add_argument("--right-endian", action="store_true")
    args = ap.parse_args()
    endian = 'right' if args.right_endian else 'left'

    counts_data = json.loads(Path(args.counts).read_text(encoding="utf-8"))
    pairs = group_pairs(counts_data)

    rows = []
    for base, (cw_key, ccw_key) in pairs.items():
        if not base.startswith("qca_") and not base.startswith("comm_"):
            continue
        family = "qca" if base.startswith("qca_") else "comm"
        area = parse_area_from_key(base)
        cw = counts_data[cw_key]; ccw = counts_data[ccw_key]
        sample_bstr = next(iter(cw.keys())) if cw else next(iter(ccw.keys()))
        width = len(sample_bstr)

        # matter indices
        if family=="qca" and width>=5:
            matter_indices = [0,1,2]   # first 3 bits = matter
        elif family=="qca" and width==3:
            matter_indices = [0,1,2]
        else:
            matter_indices = list(range(width))

        # per-bit Δ and slope
        deltas = []
        slopes = []
        for i in matter_indices:
            p1_cw = bit_prob(cw, i, '1', endian=endian)
            p1_ccw = bit_prob(ccw, i, '1', endian=endian)
            delta = (p1_cw - p1_ccw) if (math.isfinite(p1_cw) and math.isfinite(p1_ccw)) else float('nan')
            slope = (delta/area) if (area and math.isfinite(area) and area!=0) else float('nan')
            deltas.append(delta); slopes.append(slope)

        # MI/ZZ on nearest neighbors
        mi01_cw=mi01_ccw=mi12_cw=mi12_ccw=float('nan')
        zz01_cw=zz01_ccw=zz12_cw=zz12_ccw=float('nan')
        if len(matter_indices)>=2:
            i0,i1 = matter_indices[0], matter_indices[1]
            mi01_cw, mi01_ccw = mi_two_bits(cw,i0,i1,endian), mi_two_bits(ccw,i0,i1,endian)
            zz01_cw, zz01_ccw = zz_corr(cw,i0,i1,endian), zz_corr(ccw,i0,i1,endian)
        if len(matter_indices)>=3:
            i1,i2 = matter_indices[1], matter_indices[2]
            mi12_cw, mi12_ccw = mi_two_bits(cw,i1,i2,endian), mi_two_bits(ccw,i1,i2,endian)
            zz12_cw, zz12_ccw = zz_corr(cw,i1,i2,endian), zz_corr(ccw,i1,i2,endian)

        row = {
            "tag": base, "family": family, "area": area, "width": width, "n_matter": len(matter_indices),
            "delta_bits": deltas, "slope_per_area_bits": slopes,
            "mi01_cw": mi01_cw, "mi01_ccw": mi01_ccw, "mi12_cw": mi12_cw, "mi12_ccw": mi12_ccw,
            "zz01_cw": zz01_cw, "zz01_ccw": zz01_ccw, "zz12_cw": zz12_cw, "zz12_ccw": zz12_ccw,
            "cw_key": cw_key, "ccw_key": ccw_key
        }
        rows.append(row)

    # Expand to a flat table
    recs = []
    for r in rows:
        rec = {k:v for k,v in r.items() if k not in ("delta_bits","slope_per_area_bits")}
        for i in range(max(3, len(r["delta_bits"]))):
            rec[f"delta_bit{i}"] = r["delta_bits"][i] if i < len(r["delta_bits"]) else float('nan')
            rec[f"slope_per_area_bit{i}"] = r["slope_per_area_bits"][i] if i < len(r["slope_per_area_bits"]) else float('nan')
        recs.append(rec)

    df = pd.DataFrame(recs).sort_values(by=["family","area"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    Path(args.json_out).write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} and {args.json_out}")

if __name__ == "__main__":
    main()
