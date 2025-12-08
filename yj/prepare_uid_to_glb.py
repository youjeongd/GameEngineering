# prepare_uid_to_glb.py
import json
import objaverse
from pathlib import Path
from tqdm import tqdm
import argparse

def main(args):
    with open(args.uids_json, "r") as f:
        uids = json.load(f)

    print(f"Total uids in list = {len(uids)}")

    # 이미 다운로드된 경우, load_objects는 단순히 로컬 경로만 리턴함
    uid_to_path = objaverse.load_objects(
        uids=uids,
        download_processes=args.download_processes,
    )

    # JSON으로 저장
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(uid_to_path, f, indent=2)

    print(f"Saved mapping for {len(uid_to_path)} objects to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uids_json", type=str, required=True,
                        help="10k UID 리스트가 들어있는 JSON")
    parser.add_argument("--out_json", type=str, default="uid_to_glb.json")
    parser.add_argument("--download_dir", type=str, default="~/.objaverse")
    parser.add_argument("--download_processes", type=int, default=16)
    args = parser.parse_args()
    main(args)
