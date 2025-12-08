# blender_render_zero123plus.py
import bpy
import math
import mathutils
import os
import sys
import json
import argparse
import random
from pathlib import Path

RENDER_RES = 320  # 논문에서는 입력 이미지를 320x320으로 resize해서 쓰지만,
                  # 여기서는 여유 있게 512로 렌더 후 dataloader에서 resize 해도 됨.

def parse_args():
    # Blender에서는 -- 이후의 인자만 파싱해야 함
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--uid_to_glb", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True,
                        help="data/objaverse/rendering_zero123plus 같은 루트")
    parser.add_argument("--start", type=int, default=0,
                        help="uid 리스트 인덱스 시작 (분산 실행용)")
    parser.add_argument("--end", type=int, default=-1,
                        help="uid 리스트 인덱스 끝 (exclusive), -1이면 끝까지")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)

# ---------- 수학 / 카메라 유틸 ----------

def spherical_to_cart(radius, az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = radius * math.cos(el) * math.cos(az)
    y = radius * math.cos(el) * math.sin(az)
    z = radius * math.sin(el)
    return mathutils.Vector((x, y, z))

def look_at(obj, target=mathutils.Vector((0.0, 0.0, 0.0))):
    direction = target - obj.location
    # Blender: -Z가 바라보는 방향, Y가 위
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def sample_query_pose(rng):
    az = rng.uniform(0.0, 360.0)
    el = rng.uniform(-10.0, 30.0)
    dist = rng.uniform(2.2, 3.0)
    return az, el, dist

def get_all_poses(rng):
    # query (idx 0) + 6 targets (idx 1~6)
    q_az, q_el, q_dist = sample_query_pose(rng)

    poses = []
    # 0: query
    poses.append((q_az, q_el, q_dist))

    # 1~6: target, 논문에 나온 분포
    relative_azimuths = [30, 90, 150, 210, 270, 330]
    elevations =        [20, -10, 20, -10, 20, -10]
    for rel_az, el in zip(relative_azimuths, elevations):
        az = q_az + rel_az
        poses.append((az, el, q_dist))

    return poses

# ---------- 씬 세팅 및 메쉬 정규화 ----------

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    # 기본 카메라/라이트 하나씩 만들어두기
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    light_data = bpy.data.lights.new(name="Light", type='AREA')
    light_obj = bpy.data.objects.new("Light", light_data)
    bpy.context.collection.objects.link(light_obj)

    return cam_obj, light_obj


# ============================================================
# GPU 디바이스 자동 설정 (RTX 40 시리즈 포함)
# ============================================================
def setup_cycles_devices():
    import bpy

    scene = bpy.context.scene
    prefs = bpy.context.preferences

    # Cycles addon이 로딩되지 않은 경우 보호
    if "cycles" not in prefs.addons:
        print("⚠ Cycles addon not found. Falling back to CPU rendering.")
        scene.cycles.device = "CPU"
        return

    cycles_prefs = prefs.addons["cycles"].preferences

    # 디바이스 목록 새로 로드
    try:
        cycles_prefs.refresh_devices()
    except Exception as e:
        print(f"⚠ Failed to refresh GPU devices: {e}")
        scene.cycles.device = "CPU"
        return

    # 가능한 backend 선택 (OPTIX > CUDA > CPU)
    backend_candidates = ["OPTIX", "CUDA"]
    backend = None
    for b in backend_candidates:
        try:
            cycles_prefs.compute_device_type = b
            backend = b
            break
        except:
            continue

    if backend is None:
        print("⚠ No GPU backend available. Using CPU.")
        scene.cycles.device = "CPU"
        return

    print(f"Using Cycles backend: {backend}")

    gpu_found = False
    for dev in cycles_prefs.devices:
        # GPU 감지
        if dev.type in {"CUDA", "OPTIX"} and "NVIDIA" in dev.name:
            dev.use = True
            gpu_found = True
            print(f"👍 GPU Enabled: {dev.name} ({dev.type})")
        else:
            dev.use = False  # CPU, ONEAPI 등은 비활성화
            print(f"⏹ Disabled: {dev.name} ({dev.type})")

    if not gpu_found:
        print("⚠ No NVIDIA GPU available. Falling back to CPU.")
        scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "GPU"

def setup_render_engine():

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    #scene.cycles.device = 'GPU'  # GPU 없으면 'CPU'로 둬도 됨

    setup_cycles_devices()

    print(f"Render device: {scene.cycles.device}")

    scene.render.resolution_x = RENDER_RES
    scene.render.resolution_y = RENDER_RES
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True  # 알파 채널

    cycles = scene.cycles
    cycles.samples = 32         # 24~32 권장이라고 했던 그 값
    cycles.use_adaptive_sampling = True
    cycles.use_denoising = False
    cycles.use_denoising_pass = False

    cycles.use_progressive_refine = False
    cycles.use_persistent_data = True

    cycles.max_bounces = 3
    cycles.diffuse_bounces = 1
    cycles.glossy_bounces = 1
    cycles.transmission_bounces = 1
    cycles.transparent_max_bounces = 1
    cycles.use_caustics_reflective = False
    cycles.use_caustics_refractive = False


def normalize_object_to_unit_sphere(obj):
    # 모든 방향으로 1 정도 크기에 들어오도록 스케일링
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    dims = obj.dimensions
    max_dim = max(dims.x, dims.y, dims.z)
    if max_dim > 0:
        scale = 1.0 / max_dim
        obj.scale = (scale, scale, scale)
    obj.location = (0.0, 0.0, 0.0)

def import_glb(glb_path):
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    # 방금 들어온 메쉬들만 골라서 하나로 합치기
    imported = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if len(imported) == 0:
        return None

    if len(imported) > 1:
        bpy.context.view_layer.objects.active = imported[0]
        for obj in imported[1:]:
            obj.select_set(True)
        bpy.ops.object.join()
        main_obj = imported[0]
    else:
        main_obj = imported[0]

    return main_obj

# ---------- uid 하나 렌더링 ----------

def render_one_uid(uid, glb_path, out_dir, rng):
    print(f"[{uid}] rendering from {glb_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 씬 리셋
    clear_scene()
    setup_render_engine()
    scene = bpy.context.scene

    # GLB Import
    main_obj = import_glb(glb_path)
    if main_obj is None:
        print(f"  >> Failed to import {glb_path}")
        return

    normalize_object_to_unit_sphere(main_obj)

    # 카메라 / 라이트
    cam = [obj for obj in bpy.data.objects if obj.type == 'CAMERA'][0]
    light = [obj for obj in bpy.data.objects if obj.type == 'LIGHT'][0]

    # 카메라 FOV 50°
    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(50.0)

    # 라이트: 카메라 근처에서 오브젝트 쪽 비춤
    light.data.energy = 3000
    light.data.size = 5

    poses = get_all_poses(rng)

    for idx, (az, el, dist) in enumerate(poses):
        cam.location = spherical_to_cart(dist, az, el)
        look_at(cam, mathutils.Vector((0.0, 0.0, 0.0)))

        # 라이트도 카메라 근처에서 비슷하게 배치
        light.location = cam.location
        look_at(light, mathutils.Vector((0.0, 0.0, 0.0)))

        scene.camera = cam
        scene.render.filepath = str(out_dir / f"{idx:03d}.png")
        bpy.ops.render.render(write_still=True)

# ---------- 메인 ----------

def main():
    args = parse_args()

    with open(args.uid_to_glb, "r") as f:
        uid_to_glb = json.load(f)

    uids = sorted(uid_to_glb.keys())
    if args.end < 0 or args.end > len(uids):
        end = len(uids)
    else:
        end = args.end

    uids = uids[args.start:end]
    print(f"Total UIDs to render in this run: {len(uids)}")

    rng = random.Random(args.seed)

    for i, uid in enumerate(uids):
        glb_path = uid_to_glb[uid]
        out_dir = os.path.join(args.output_root, uid)

        # uid마다 seed를 다르게 (재현 가능)
        uid_seed = hash(uid) ^ args.seed
        rng.seed(uid_seed)

        try:
            render_one_uid(uid, glb_path, out_dir, rng)
        except Exception as e:
            print(f"Error while rendering {uid}: {e}")

if __name__ == "__main__":
    main()
