"""Blender 안에서 실행하는 스크립트: 지정한 본들의 head/tail 월드 좌표 + 실제 회전(quaternion)을
프레임마다 CSV로 저장.

2026-08-29 수정: 기존 head/tail-only 버전에 world-space rotation quaternion 4개 컬럼을 추가했다.
이유: 본이 자기 세로축(=head->tail 방향) 둘레로 회전하면 head/tail 좌표에는 그 회전이 전혀 안
남는다(자기 자신을 축으로 도는 회전이라 두 끝점이 그대로 있음). 이 "숨은 회전"을 잡아내려면
본의 실제 회전값을 직접 받아야 한다. (rotation_value_test.csv 분석에서 spine.001의 몸통 twist가
head/tail로는 안 보이고 quaternion으로만 확인된 사례 — scripts/convert_csv_to_npz.py 주변 메모 참고)

사용법:
1. Blender에서 위쪽 탭 중 "Scripting"으로 이동
2. 왼쪽 텍스트 에디터에서 New 누르고 이 파일 내용을 전부 붙여넣기
3. 아래 ARMATURE_NAME을 실제 아마추어(리그) 오브젝트 이름으로 바꾸기
   (왼쪽 Outliner에서 아마추어 오브젝트를 클릭하면 이름이 보임)
4. 아래 TARGET_BONES 리스트가 실제 본 이름과 일치하는지 확인
   (다르면 콘솔에 "누락된 본" 경고가 출력됨)
5. 위쪽 Run Script(▶) 버튼 클릭
6. OUT_PATH에 지정된 파일이 생성됨
"""
import bpy
import csv

ARMATURE_NAME = "metarig.003"  # <-- 실제 아마추어 오브젝트 이름으로 수정
OUT_PATH = "C:/Users/CKIRUser/Downloads/export_raw2.csv"

# 뽑고 싶은 본만 지정
TARGET_BONES = [
    "shin.R",
    "thigh.R",
    "shin.L",
    "thigh.L",
    "spine.001",
    "upper_arm.R",
    "forearm.R",
    "upper_arm.L",
    "forearm.L",
]


def bone_world_quat(arm_obj, pbone):
    """본의 world-space 회전을 quaternion(w,x,y,z)으로 반환.

    pbone.matrix는 armature 오브젝트 로컬 공간(pose space) 기준이라, head/tail을
    world로 바꿀 때와 똑같이 arm_obj.matrix_world를 곱해 world 기준으로 맞춘다.
    to_quaternion()은 이동/스케일 성분을 무시하고 회전만 뽑아준다.
    """
    world_matrix = arm_obj.matrix_world @ pbone.matrix
    return world_matrix.to_quaternion()


def main():
    arm_obj = bpy.data.objects.get(ARMATURE_NAME)
    if arm_obj is None:
        print(f"[에러] '{ARMATURE_NAME}' 오브젝트를 못 찾음. Outliner에서 정확한 이름을 확인하세요.")
        print("현재 씬의 오브젝트 목록:", [o.name for o in bpy.data.objects])
        return

    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end

    # 실제 존재하는 본 이름만 필터링 + 누락 본 경고
    all_bone_names = {pb.name for pb in arm_obj.pose.bones}
    bone_names = [b for b in TARGET_BONES if b in all_bone_names]
    missing = [b for b in TARGET_BONES if b not in all_bone_names]
    if missing:
        print(f"[경고] 다음 본을 아마추어에서 찾지 못해 건너뜁니다: {missing}")
        print("실제 아마추어의 전체 본 목록:", sorted(all_bone_names))
    if not bone_names:
        print("[에러] TARGET_BONES 중 아마추어에 존재하는 본이 하나도 없습니다. 이름을 확인하세요.")
        return

    rows = []
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        row = {"frame": frame}
        for bone_name in bone_names:
            pbone = arm_obj.pose.bones[bone_name]
            # pose bone의 head/tail을 오브젝트 로컬 -> 월드로 변환
            head_world = arm_obj.matrix_world @ pbone.head
            tail_world = arm_obj.matrix_world @ pbone.tail
            row[f"{bone_name}__head_x"] = head_world.x
            row[f"{bone_name}__head_y"] = head_world.y
            row[f"{bone_name}__head_z"] = head_world.z
            row[f"{bone_name}__tail_x"] = tail_world.x
            row[f"{bone_name}__tail_y"] = tail_world.y
            row[f"{bone_name}__tail_z"] = tail_world.z

            # --- world-space 회전 quaternion (자기축 회전까지 포함) ---
            quat = bone_world_quat(arm_obj, pbone)
            row[f"{bone_name}__quat_w"] = quat.w
            row[f"{bone_name}__quat_x"] = quat.x
            row[f"{bone_name}__quat_y"] = quat.y
            row[f"{bone_name}__quat_z"] = quat.z

        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"저장 완료: {OUT_PATH} ({len(rows)} 프레임, 본 {len(bone_names)}개: {bone_names})")


main()
