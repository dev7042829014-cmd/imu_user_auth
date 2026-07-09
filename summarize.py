"""
summarize.py — print one comparison table from all results/*.json files.
Usage:  python summarize.py [results_dir]   (default: results)
"""
import json, glob, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    try:
        j = json.load(open(f))
        r = j["results"][0]                       # [0] = cosine scorer
        boot = j.get("bootstrap") or {}
        be = (boot.get("eer") or [float("nan")])[0]
        br = (boot.get("rank1") or [float("nan")])[0]
        name = os.path.basename(f).replace("_test", "").replace(".json", "")
        rows.append((name, r["eer"] * 100, r["rank1"] * 100, be * 100, br * 100,
                     r.get("frr_at_far", {})))
    except Exception as e:
        print(f"[skip] {f}: {e}")

rows.sort(key=lambda x: x[1])                      # sort by EER (best first)
print(f"\n{'model':16}{'EER%':>8}{'Rank1%':>9}{'EER(boot)':>11}{'R1(boot)':>10}")
print("-" * 54)
for n, e, r1, be, br, _ in rows:
    print(f"{n:16}{e:8.2f}{r1:9.2f}{be:11.2f}{br:10.2f}")

# highlight the sweep winner
sweep = [x for x in rows if x[0].startswith("s2a_T")]
if sweep:
    best = min(sweep, key=lambda x: x[1])
    print(f"\nBest S2A temperature: {best[0].split('_T')[1]}  "
          f"(EER {best[1]:.2f}%, Rank-1 {best[2]:.2f}%)")
