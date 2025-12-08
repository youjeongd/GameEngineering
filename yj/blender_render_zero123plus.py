# blender_render_zero123plus.py

import bpy
import os
import json
import math
import random
import sys
from mathutils import Vector

import argparse


# -----------------------
# 기본 세팅 함수들
# -----------------------

def clear_scene():
    # 모든 오브젝트 삭제
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # 메쉬 데이터 정리
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

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




def setup_render(resolution=256, samples=24):
    setup_cycles_devices()
    scene = bpy.context.scene

    # 우선 GPU 세팅을 강제로 적용
    setup_cycles_devices()

    # 해상도
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    # 출력 포맷
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Cycles 최적화 옵션
    cycles = scene.cycles
    cycles.samples = samples               # 속도 중심: 24 ~ 32 권장
    cycles.use_adaptive_sampling = True
    cycles.use_denoising = True
    cycles.use_denoising_pass = False
    cycles.use_progressive_refine = False
    cycles.use_persistent_data = True     # 캐싱으로 큰 속도 향상

    # Bounces 줄여서 속도 향상
    cycles.max_bounces = 3
    cycles.diffuse_bounces = 1
    cycles.glossy_bounces = 1
    cycles.transmission_bounces = 1
    cycles.transparent_max_bounces = 1
    cycles.use_caustics_reflective = False
    cycles.use_caustics_refractive = False

    # 투명 배경
    scene.render.film_transparent = True



def setup_world_and_lights():
    # World 밝은 회색
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    bg = None
    for n in nodes:
        if n.type == 'BACKGROUND':
            bg = n
            break
    if bg is None:
        bg = nodes.new(type='ShaderNodeBackground')
    bg.inputs[0].default_value = (0.9, 0.9, 0.9, 1.0)  # 밝은 회색
    bg.inputs[1].default_value = 1.0

    # 기존 라이트 삭제
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # 3점 조명
    lights = []

    # Key light
    light_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    light_data.energy = 1500
    light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (3.0, -3.0, 3.0)
    lights.append(light_obj)

    # Fill light
    light_data = bpy.data.lights.new(name="FillLight", type='AREA')
    light_data.energy = 800
    light_obj = bpy.data.objects.new(name="FillLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (-3.0, 3.0, 2.0)
    lights.append(light_obj)

    # Rim light
    light_data = bpy.data.lights.new(name="RimLight", type='AREA')
    light_data.energy = 600
    light_obj = bpy.data.objects.new(name="RimLight", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (-2.0, -3.0, 4.0)
    lights.append(light_obj)

    return lights


def create_camera(fov_degree=50.0):
    # 카메라 생성
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # FOV 세팅
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(fov_degree)
    return cam_obj


# -----------------------
# Bounding Box 정규화
# -----------------------

def get_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']


def compute_world_bbox(objs):
    """월드 좌표계에서 여러 mesh의 bbox (min, max) 계산."""
    if not objs:
        return None, None

    min_v = Vector((float('inf'), float('inf'), float('inf')))
    max_v = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in objs:
        # obj.bound_box: 8개 로컬 좌표
        for v in obj.bound_box:
            wv = obj.matrix_world @ Vector(v)
            min_v.x = min(min_v.x, wv.x)
            min_v.y = min(min_v.y, wv.y)
            min_v.z = min(min_v.z, wv.z)

            max_v.x = max(max_v.x, wv.x)
            max_v.y = max(max_v.y, wv.y)
            max_v.z = max(max_v.z, wv.z)

    return min_v, max_v


def normalize_objects_to_unit_box(objs, target_max_dim=1.0):
    """
    - world-space bounding box 계산
    - 중심을 원점으로 이동
    - 최대 길이를 target_max_dim(기본 1.0) 으로 맞춤
    """
    if not objs:
        return None

    min_v, max_v = compute_world_bbox(objs)
    if min_v is None:
        return None

    size = max_v - min_v
    max_dim = max(size.x, size.y, size.z)

    if max_dim == 0:
        return None

    center = (min_v + max_v) * 0.5
    scale_factor = target_max_dim / max_dim

    # 중심을 원점으로, scale 적용
    for obj in objs:
        obj.location = (obj.location - center) * scale_factor
        obj.scale = obj.scale * scale_factor

    return max_dim  # 원래 bbox의 최대 길이 (normalize 전)


# -----------------------
# 카메라 배치 & 렌더
# -----------------------

def look_at(obj, target=Vector((0.0, 0.0, 0.0))):
    direction = target - obj.location
    # Z-up, -Z forward, Y up (Blender 기본 카메라)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()


def compute_camera_distance_from_bbox(original_max_dim, fov_degree=50.0, margin=3.0):
    """
    original_max_dim: 정규화 전에 bbox 최대 길이
    fov와 object 크기에 기반해 카메라 거리 계산.
    - 정규화로 실제 max_dim은 1이지만,
      original_max_dim을 써서 객체가 지나치게 크거나 작은 경우를 평균적으로 커버.
    """
    if original_max_dim is None or original_max_dim <= 0:
        # fallback: 대충 유니트 스케일 기준
        radius = 0.5
    else:
        # 정규화 후에는 1이 되지만, 원래 크기를 반영해서 약간 조정하고 싶다면
        # radius를 0.5 * (original_max_dim / 평균값) 같은 식으로 비례하도록 바꿀 수도 있음.
        radius = 0.5

    fov_rad = math.radians(fov_degree)
    base_dist = radius / math.tan(fov_rad / 2.0)
    return base_dist * margin


def render_views_for_object(cam_obj, out_dir, fov_degree=50.0):
    """
    7-view orbit 렌더링 (Zero123++ 스타일)
    000.png ~ 006.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # Zero123++에서 사용했다고 알려진 6 view + 1 ref 구조 (예시)
    relative_azimuths = [0,  30, 90, 150, 210, 270, 330]  # 첫 번째(0deg)를 cond로 쓸 수 있음
    elevations        = [0,  20, -10, 20,  -10, 20, -10]

    # bbox 기반 거리 계산은 밖에서 해두고, 여기서는 cam_obj.location 의 길이를 사용해도 됨
    # 여기서는 cam_obj.location의 길이를 유지한 채 방향만 바꾼다고 가정
    base_dist = cam_obj.location.length  # 이미 세팅된 거리

    for idx, (az_deg, el_deg) in enumerate(zip(relative_azimuths, elevations)):
        az = math.radians(az_deg)
        el = math.radians(el_deg)

        # 구 좌표 → 데카르트 좌표
        x = base_dist * math.cos(el) * math.cos(az)
        y = base_dist * math.cos(el) * math.sin(az)
        z = base_dist * math.sin(el)

        cam_obj.location = Vector((x, y, z))
        look_at(cam_obj, Vector((0.0, 0.0, 0.0)))

        # 출력 경로
        filepath = os.path.join(out_dir, f"{idx:03d}.png")
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)


# -----------------------
# GLB 로딩
# -----------------------

def import_glb(path):
    before_objs = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    after_objs = set(bpy.context.scene.objects)
    new_objs = list(after_objs - before_objs)
    # mesh만 필터링
    mesh_objs = [obj for obj in new_objs if obj.type == 'MESH']
    return mesh_objs


# -----------------------
# 메인 루프
# -----------------------

def process_uid(uid, glb_path, output_root, fov_degree=50.0):
    out_dir = os.path.join(output_root, uid)

    # 이미 7장 다 있으면 스킵
    if os.path.exists(out_dir):
        pngs = [f for f in os.listdir(out_dir) if f.endswith(".png")]
        if len(pngs) >= 7:
            return

    clear_scene()
    setup_render(resolution=256, samples=24)
    setup_world_and_lights()
    cam = create_camera(fov_degree=fov_degree)

    try:
        mesh_objs = import_glb(glb_path)
        if not mesh_objs:
            return

        # bbox 정규화
        original_max_dim = normalize_objects_to_unit_box(mesh_objs, target_max_dim=1.0)

        # bbox 기반 카메라 거리 계산
        cam_dist = compute_camera_distance_from_bbox(original_max_dim, fov_degree=fov_degree, margin=2.0)
        # 초기 위치: 정면 (y-축 쪽)
        cam.location = Vector((0.0, -cam_dist, 0.0))
        look_at(cam, Vector((0.0, 0.0, 0.0)))

        render_views_for_object(cam, out_dir, fov_degree=fov_degree)

    except Exception as e:
        print(f"[{uid}] rendering failed: {e}")


def main():
    # Blender에서 실행 시 argv 처리
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--uid_to_glb", type=str, required=True,
                        help="uid -> glb path 매핑이 들어있는 JSON")
    parser.add_argument("--output_root", type=str, required=True,
                        help="렌더링된 이미지를 저장할 루트 디렉토리")
    parser.add_argument("--start", type=int, default=0,
                        help="UID 리스트에서 시작 index (포함)")
    parser.add_argument("--end", type=int, default=None,
                        help="UID 리스트에서 끝 index (미포함)")
    parser.add_argument("--seed", type=int, default=0,
                        help="랜덤 시드 (사용 시 확장 가능)")
    args = parser.parse_args(argv)

    random.seed(args.seed)

    with open(args.uid_to_glb, "r") as f:
        uid_to_glb = json.load(f)

    uids = list(uid_to_glb.keys())
    uids.sort()

    n = len(uids)
    start = max(0, args.start)
    end = n if args.end is None else min(args.end, n)

    print(f"Total UIDs = {n}, rendering [{start}, {end})")

    for i in range(start, end):
        uid = uids[i]
        glb_path = uid_to_glb[uid]
        print(f"[{uid}] rendering from {glb_path}")
        process_uid(uid, glb_path, args.output_root, fov_degree=50.0)


if __name__ == "__main__":
    main()

