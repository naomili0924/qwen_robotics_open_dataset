#!/usr/bin/env python
"""Derive an HSSD scene dataset without door objects (and other given categories).

HSSD places closed door leaves as objects, which cut a house into rooms on the navmesh.  This
writes <out>/ with symlinks to the original assets and filtered scene_instance files, plus
scene_dataset_config.json, usable with generate_habitat_pointnav.py --dataset-config.

    python scripts/prepare_hssd.py --hssd /dev/shm/hnod/habitat_data/hssd-hab --out /dev/shm/hnod/habitat_data/hssd-hab-nodoors
"""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hssd", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", nargs="*", default=["door"], help="main_category values to remove")
    args = ap.parse_args()
    src, out = Path(args.hssd).resolve(), Path(args.out)
    cats = {}
    with open(src / "metadata/fpmodels-with-decomposed.csv") as f:
        for r in csv.DictReader(f):
            cats[r["id"]] = r["main_category"]
    out.mkdir(parents=True, exist_ok=True)
    for d in ("stages", "objects", "semantics", "metadata"):
        if not (out / d).exists():
            os.symlink(src / d, out / d)
    cfg = json.load(open(src / "hssd-hab.scene_dataset_config.json"))
    json.dump(cfg, open(out / "hssd-hab.scene_dataset_config.json", "w"), indent=1)
    (out / "scenes").mkdir(exist_ok=True)
    dropped = total = 0
    for f in sorted((src / "scenes").glob("*.scene_instance.json")):
        sc = json.load(open(f))
        keep = [o for o in sc["object_instances"] if cats.get(o["template_name"].split("/")[-1], "") not in args.drop]
        dropped += len(sc["object_instances"]) - len(keep)
        total += len(sc["object_instances"])
        sc["object_instances"] = keep
        json.dump(sc, open(out / "scenes" / f.name, "w"))
    print(f"{out}: dropped {dropped} of {total} object instances ({', '.join(args.drop)})")


if __name__ == "__main__":
    main()
