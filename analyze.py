#!/usr/bin/env python3
"""Analysis for the Mini Matrix run: the per-model table, majority calls, and
flip counts, computed from whatever trials_*.jsonl files sit next to it."""
import json, glob
from collections import Counter, defaultdict

ORDER = ["haiku", "opus", "gpt-5.6-sol"]  # display order; anything else appends

rows = []
for path in sorted(glob.glob("trials/*.jsonl")):
    rows += [json.loads(line) for line in open(path)]

total = len(rows)
valid = [r for r in rows if r["choice"] in ("red", "blue", "refuse")]
refusals = [r for r in valid if r["choice"] == "refuse"]
print(f"trials {total}   valid {len(valid)} ({100*len(valid)/total:.1f}%)   "
      f"refusals {len(refusals)} ({100*len(refusals)/len(valid):.1f}% of valid)")

cells = defaultdict(list)
for r in valid:
    cells[(r["persona_id"], r["model"])].append(r["choice"])
def call(v):
    # a real majority, or nothing. most_common() alone breaks a 1-1-1 cell by
    # insertion order, which invents an answer the model never gave.
    top, n = Counter(v).most_common(1)[0]
    return top if n * 2 > len(v) else "no majority"


majority = {k: call(v) for k, v in cells.items()}
unanimous = {k: len(set(v)) == 1 for k, v in cells.items()}
personas = sorted({r["persona_id"] for r in valid})
models = [m for m in ORDER if any(k[1] == m for k in cells)]
models += sorted({k[1] for k in cells} - set(models))
print(f"cells {len(cells)}   unanimous {sum(unanimous.values())} "
      f"({100*sum(unanimous.values())/len(cells):.1f}%)")

print(f"\n{'model':16}{'red':>6}{'blue':>6}{'refuse':>8}{'none':>6}{'blue% maj':>12}{'blue% raw':>12}")
for m in models:
    maj = Counter(majority[(p, m)] for p in personas if (p, m) in majority)
    raw = Counter(r["choice"] for r in valid if r["model"] == m)
    print(f"{m:16}{maj['red']:>6}{maj['blue']:>6}{maj['refuse']:>8}"
          f"{maj['no majority']:>6}"
          f"{100*maj['blue']/sum(maj.values()):>11.1f}%{100*raw['blue']/sum(raw.values()):>11.1f}%")

soft, hard = [], []
for p in personas:
    per = {m: majority[(p, m)] for m in models if (p, m) in majority}
    if len(set(per.values())) > 1:
        soft.append((p, per))
        if any(unanimous[(p, a)] and unanimous[(p, b)] and per[a] != per[b]
               for a in per for b in per):
            hard.append((p, per))

print(f"\nsoft flips {len(soft)}/{len(personas)}   hard flips {len(hard)}")

print("\nsample flips:")
for p, per in soft[:10]:
    print(f"  persona {p:>3}  " + "  ".join(
        f"{m.replace('gpt-5.6-','')}={per[m]}" for m in models if m in per))
