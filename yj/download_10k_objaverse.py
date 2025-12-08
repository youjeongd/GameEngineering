import os
import json
from pathlib import Path
import objaverse
from tqdm import tqdm

# -------------------------------------------------------
# 이미 다운로드된 UID 탐색
# -------------------------------------------------------
def get_downloaded_uids(base_path=r"C:/Users/yjdoh/.objaverse/hf-objaverse-v1/glbs"):
    base_path = os.path.expanduser(base_path)
    downloaded = set()

    if not os.path.exists(base_path):
        return downloaded

    for shard in os.listdir(base_path):
        shard_path = os.path.join(base_path, shard)
        if not os.path.isdir(shard_path):
            continue

        for fname in os.listdir(shard_path):
            if fname.endswith(".glb"):
                uid = Path(fname).stem
                downloaded.add(uid)
    return downloaded

# 10k UID 리스트 만들기
def create_uid_subset_json(
    output_json="uids_10k.json",
    glb_base=r"C:/Users/yjdoh/.objaverse/hf-objaverse-v1/glbs",
):

    downloaded_uids = get_downloaded_uids(glb_base)

    with open(output_json, "w") as f:
        json.dump(list(downloaded_uids), f, indent=2)

    print(f"Saved {len(downloaded_uids)} downloaded UIDs → {output_json}")
    print("Saving JSON to:", os.path.abspath(output_json))



# -------------------------------------------------------
# UID 리스트 기반 다운로드 함수 (속도 최적화)
# -------------------------------------------------------
def download_from_uid_list(
    json_path="uids_to_download.json",
    failed_path="failed_uids.json",
    done_path="done_uids.json",
    download_processes=12,   # 네트워크 빠르면 더 높여도 OK
):
    # UID 로드
    with open(json_path, "r") as f:
        uid_list = json.load(f)

    # 이미 다운로드된 UID 체크
    downloaded = get_downloaded_uids()
    print(f"✔ 이미 다운로드된 UID 개수: {len(downloaded)}")

    # 기존 완료 기록 불러오기
    completed = set()
    if os.path.exists(done_path):
        completed = set(json.load(open(done_path)))

    # 다운로드해야 하는 UID 최종 필터링
    remaining_uids = [uid for uid in uid_list if uid not in downloaded and uid not in completed]
    print(f"📌 남은 다운로드 대상 UID: {len(remaining_uids)}")

    if len(remaining_uids) == 0:
        print("🎉 다운로드할 UID가 없습니다!")
        return

    # 실패 UID 기록 초기화
    failed = []

    print("🚀 다운로드 시작... (최적화 병렬 다운로드)")

    try:
        # Objaverse는 내부에서 shard 단위 + 프로세스 단위 병렬 다운로드 지원
        uid_to_path = objaverse.load_objects(
            uids=remaining_uids,
            download_processes=download_processes,
        )
    except Exception as e:
        print("❗ 다운로드 중 오류 발생:", e)
        failed.extend(remaining_uids)

    # 다운로드 결과 확인
    uid_to_path = {k: v for k, v in uid_to_path.items() if v is not None}

    print(f"✅ 성공적으로 다운로드된 개수: {len(uid_to_path)}")

    # 성공한 UID 기록 업데이트
    new_done = list(completed.union(uid_to_path.keys()))
    with open(done_path, "w") as f:
        json.dump(new_done, f, indent=2)

    # 실패한 UID 기록
    failed = [uid for uid in remaining_uids if uid not in uid_to_path]
    print(f"❌ 실패한 UID 개수: {len(failed)}")

    if failed:
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"❗ 실패 UID 목록이 {failed_path} 에 저장되었습니다")

    print("🎯 전체 작업 완료!")


# -------------------------------------------------------
# 실행
# -------------------------------------------------------
if __name__ == "__main__":
    create_uid_subset_json()

