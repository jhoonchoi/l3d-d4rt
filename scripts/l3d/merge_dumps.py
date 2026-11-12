#!/usr/bin/env python3
"""Merge dump runs into one run directory (for a dump split by subset or by batch).

    python scripts/l3d/merge_dumps.py output/l3d/dumps flow3r_T48 flow3r_T48_adt_mini flow3r_T48_ds_mini ...

Keeps clips with status "ok" (first run wins on duplicates), copies their tables, and concatenates the
manifests (header from the first run, plus `merged_from`)."""
import json, shutil, sys
from pathlib import Path
root, out, srcs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
dst = root / out
dst.mkdir(parents=True, exist_ok=True)
merged, seen = None, set()
for name in srcs:
    src = root / name
    m = json.loads((src / "manifest.json").read_text())
    if merged is None:
        merged = {k: v for k, v in m.items() if k != "clips"}
        merged.update(run_name=out, clips=[], total_seconds=0.0, merged_from=[])
    merged["merged_from"].append(name)
    merged["total_seconds"] += m.get("total_seconds", 0.0)
    for c in m["clips"]:
        key = (c["subset"], c["sequence"], c["clip_start"])
        if c["status"] != "ok" or key in seen:
            continue
        seen.add(key)
        (dst / c["file"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / c["file"], dst / c["file"])
        merged["clips"].append(c)
merged["clips"].sort(key=lambda c: (c["subset"], c["sequence"], c["clip_start"]))
(dst / "manifest.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n")
from collections import Counter
print(out, dict(Counter(c["subset"] for c in merged["clips"])))
