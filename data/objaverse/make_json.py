import json
from pathlib import Path

# 1) 여기만 네 실제 루트 경로로 바꿔주면 됨
ROOT_DIR = Path("C:/Users/yjdoh/OneDrive/Desktop/GameEngineering/data/objaverse/rendering_zero123plus")   # 예: "/data/objaverse/rendering_zero123plus"

# 2) 바로 아래에 있는 "폴더"들만 UID로 수집
uids = sorted([
    p.name for p in ROOT_DIR.iterdir()
    if p.is_dir()
])

print(f"총 {len(uids)}개 UID 발견")

# 4) JSON 파일로 저장
OUT_PATH = "C:/Users/yjdoh/OneDrive/Desktop/GameEngineering/data/objaverse/rendering_zero123plus/uid_list.json"   # 원하는 이름/경로로 변경 가능

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(uids, f, ensure_ascii=False, indent=2)

print(f"JSON 저장 완료: {OUT_PATH}")
