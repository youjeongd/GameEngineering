# build_lvis_annotations.py
import os
import json
import argparse
from pathlib import Path

def main(args):
    image_root = Path(args.image_root)
    assert image_root.is_dir(), f"{image_root} is not a directory"

    uids = []
    for d in sorted(image_root.iterdir()):
        if not d.is_dir():
            continue
        pngs = list(d.glob("*.png"))
        if len(pngs) >= 7:
            uids.append(d.name)

    print(f"Found {len(uids)} objects with >=7 views")

    meta = {"all": uids}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote meta to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True,
                        help="data/objaverse/rendering_zero123plus")
    parser.add_argument("--out_json", type=str, required=True,
                        help="data/objaverse/lvis-annotations.json")
    args = parser.parse_args()
    main(args)
