"""Turn a multi-geometry GLB into a sample object directory (input.glb + parts/ + names.json).

    python finetune/import_glb.py --glb data_toolkit/assets/example.glb --out <root>/example \
        --names wall chimney door planter window soil mushroom window roof pot
Names default to the GLB geometry names; pass --names to override (one per kept geometry).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--out", required=True, help="Object directory to create")
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--min_faces", type=int, default=1)
    args = parser.parse_args()
    obj = common.prepare_object_from_glb(args.glb, args.out, names=args.names, min_faces=args.min_faces)
    names = obj.names()
    print(f"{obj.path}: {len(obj.part_files())} parts")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")


if __name__ == "__main__":
    main()
