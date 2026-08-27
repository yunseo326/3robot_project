"""Blender에서 export한 본 좌표 CSV(head/tail 월드좌표)를 관절각도 npz로 역산 변환.

2robot_project/blender/convert_bvh_to_npz.py의 "좌표만으로 역산" 방식을 이어받되,
우리 hip은 2robot(ant, 1축 yaw)과 달리 yaw+roll+pitch 3축 직렬이라 허벅지 벡터
하나(자유도 2)만으로는 3자유도를 못 푼다 — 같은 허벅지 방향을 만드는 (yaw,roll,pitch)
조합이 무한히 많다(진짜 부정형). 이를 풀기 위해 정강이 벡터를 추가로 쓴다:

핵심 원리
---------
1. 무릎 각도는 hip 회전과 무관하게 바로 나온다.
   무릎 관절은 hip_pitch_link의 로컬 Y축 둘레 1자유도 힌지이고, 허벅지 방향은 그 프레임의
   -Z축과 정의상 같다. Y축은 -Z에 수직이므로, "허벅지→정강이" 회전각(=무릎각) 크기는
   hip이 어떻게 돌아가 있든 상관없이 두 벡터 사이 각도(arccos(dot))와 정확히 같다.
2. hip의 3축(yaw,roll,pitch)은 허벅지 방향(R_hip의 3번째 열) + 무릎굽힘축(R_hip의 2번째
   열, = cross(허벅지방향, 정강이방향)과 같은 방향) 두 벡터로 완전히 결정된다. 회전행렬의
   두 열을 알면 나머지 한 열(1번째)은 외적으로 나오고, 거기서 yaw/roll/pitch를 대수적으로
   추출한다 (character.xml의 조인트 순서 Rz(yaw)·Rx(roll)·Ry(pitch)를 그대로 유도).
3. 무릎이 거의 펴진 프레임(허벅지∥정강이)에서는 cross()가 0에 가까워 무릎축을 못 구한다
   — 이 프레임에서는 시각적으로도 hip 비틀림이 안 보이는 구간이라(펴진 다리를 축 방향으로
   돌려도 겉보기 자세가 그대로) 크게 문제되지 않는다. 이전 프레임의 축을 허벅지 방향
   변화량만큼 평행이동(parallel transport)해서 이어붙인다.

몸통(root)
----------
2robot은 root를 아예 고정값(항등 쿼터니언)으로 박아넣었다 — 제자리 걷기였기 때문.
우리는 실제로 이동/회전하는 애니메이션이라 매 프레임 실제로 계산해야 한다:
  - 위치: spine 본의 head
  - 방향: spine 본의 head->tail("위" 방향)과 오른쪽/왼쪽 허벅지 head 사이 벡터("옆" 방향)
    두 개로 정규직교 프레임을 만들어 회전행렬 -> 쿼터니언으로 변환한다.

전제 조건 (Blender export 스크립트 확인!)
-----------------------------------------
본마다 head와 tail 좌표가 **둘 다** 있어야 한다(방향 벡터 계산에 필수). 사용자가 붙여준
스크립트는 head만 저장하고 있었다 — tail도 저장하도록 고쳐야 이 스크립트가 동작한다.

qpos/qvel 레이아웃은 models/character.xml과 정확히 일치해야 한다:
  qpos[0:3]=root xyz, qpos[3:7]=root quat(w,x,y,z),
  qpos[7:15]=[l_hip_yaw,l_hip_roll,l_hip_pitch,l_knee, r_hip_yaw,r_hip_roll,r_hip_pitch,r_knee]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

CTRL_DT = 0.02  # envs/biped_mimic_gym.py의 CTRL_DT와 반드시 일치해야 함 (50Hz)
KNEE_AXIS_EPS = 1e-3  # 이 값보다 |cross(thigh,shin)|가 작으면 "거의 편 다리"로 보고
                       # 평행이동으로 축을 이어붙인다.

# 본 이름 기본값 — Blender 쪽 TARGET_BONES와 다르면 CLI 인자로 덮어쓸 것.
ROOT_BONE = "spine.002"
LEG_BONES = {
    "l": dict(thigh="thigh.L", shin="shin.L"),
    "r": dict(thigh="thigh.R", shin="shin.R"),
}


def apply_bone_name_args(args):
    global ROOT_BONE
    ROOT_BONE = args.root_bone
    LEG_BONES["l"]["thigh"] = args.thigh_l
    LEG_BONES["l"]["shin"] = args.shin_l
    LEG_BONES["r"]["thigh"] = args.thigh_r
    LEG_BONES["r"]["shin"] = args.shin_r
JOINT_ORDER = ["l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee",
               "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee"]


def bone_vec(df, bone, part, t):
    return df.loc[t, [f"{bone}__{part}_x", f"{bone}__{part}_y", f"{bone}__{part}_z"]].to_numpy(dtype=float)


def unit(v, fallback=None):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return fallback if fallback is not None else v
    return v / n


def nearest_rotation(M):
    """노이즈 섞인 3x3 행렬을 가장 가까운 진짜 회전행렬로 투영 (SVD 극분해)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # 반사(reflection) 방지
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rodrigues_transport(v_from, v_to, axis_from):
    """v_from -> v_to로 가는 최소 회전을 axis_from에 적용해 옮긴다 (평행이동).
    무릎이 편 프레임에서 이전 프레임의 무릎굽힘축을 이어붙일 때 쓴다."""
    a, b = unit(v_from), unit(v_to)
    cos_t = np.clip(np.dot(a, b), -1.0, 1.0)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return axis_from
    axis = axis / s
    sin_t = s
    # 로드리게스 회전 공식
    return (axis_from * cos_t
            + np.cross(axis, axis_from) * sin_t
            + axis * np.dot(axis, axis_from) * (1 - cos_t))


def build_root_frames(df, n):
    """프레임별 root 위치(n,3)와 회전행렬(n,3,3)을 만든다.
    로컬 축 정의: x=forward, y=left, z=up (character.xml 관절 오프셋 부호와 일치:
    l_* 본이 +y, r_*가 -y에 붙어있으므로 +y=왼쪽)."""
    positions = np.zeros((n, 3))
    rotations = np.zeros((n, 3, 3))
    for t in range(n):
        root_head = bone_vec(df, ROOT_BONE, "head", t)
        root_tail = bone_vec(df, ROOT_BONE, "tail", t)
        up_raw = root_tail - root_head

        l_hip_head = bone_vec(df, LEG_BONES["l"]["thigh"], "head", t)
        r_hip_head = bone_vec(df, LEG_BONES["r"]["thigh"], "head", t)
        left_raw = l_hip_head - r_hip_head

        z_axis = unit(up_raw)
        left_axis = unit(left_raw - np.dot(left_raw, z_axis) * z_axis)
        x_axis = np.cross(left_axis, z_axis)  # forward = left x up (오른손 좌표계)

        R = nearest_rotation(np.column_stack([x_axis, left_axis, z_axis]))
        positions[t] = root_head
        rotations[t] = R
    return positions, rotations


def hip_euler_from_matrix(R):
    """R = Rz(yaw)·Rx(roll)·Ry(pitch) 분해 (character.xml 조인트 순서와 일치).
    유도 과정은 이 파일 docstring 참고. roll이 ±90도 안(우리 조인트 한계 ±30도라 안전)
    이라는 전제 하에 성립하는 닫힌형 공식."""
    roll = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
    yaw = np.arctan2(-R[0, 1], R[1, 1])
    pitch = np.arctan2(-R[2, 0], R[2, 2])
    return yaw, roll, pitch


def compute_leg_joint_angles(df, n, root_pos, root_rot, side):
    """한쪽 다리의 (yaw,roll,pitch,knee) 시계열을 반환. shape (n,4)."""
    return compute_pair_joint_angles(df, n, root_pos, root_rot,
                                      LEG_BONES[side]["thigh"], LEG_BONES[side]["shin"])


def compute_pair_joint_angles(df, n, root_pos, root_rot, thigh_bone, shin_bone):
    """근위(hip/shoulder 역할) + 원위(knee/elbow 역할) 본 쌍 하나의 (yaw,roll,pitch,bend)
    시계열을 반환. shape (n,4). 다리(thigh/shin)든 팔(upper_arm/forearm)이든 뼈대
    구조(3축 근위 관절 + 1축 원위 힌지)가 같으면 그대로 재사용 가능."""
    angles = np.zeros((n, 4))
    prev_thigh_local = None
    prev_axis_local = np.array([0.0, 1.0, 0.0])  # rest pose 기본값

    for t in range(n):
        thigh_head = bone_vec(df, thigh_bone, "head", t)
        thigh_tail = bone_vec(df, thigh_bone, "tail", t)
        shin_head = bone_vec(df, shin_bone, "head", t)
        shin_tail = bone_vec(df, shin_bone, "tail", t)

        thigh_dir_world = unit(thigh_tail - thigh_head)
        shin_dir_world = unit(shin_tail - shin_head)

        Rw2l = root_rot[t].T  # world -> root-local
        thigh_local = Rw2l @ thigh_dir_world
        shin_local = Rw2l @ shin_dir_world

        # 무릎각: hip 회전과 무관하게 두 벡터 사이 각도로 바로 나옴
        knee_mag = np.arccos(np.clip(np.dot(thigh_local, shin_local), -1.0, 1.0))

        # 무릎굽힘축(=hip의 로컬 Y축) 추정
        raw_axis = np.cross(thigh_local, shin_local)
        axis_norm = np.linalg.norm(raw_axis)
        if axis_norm > KNEE_AXIS_EPS:
            knee_axis = raw_axis / axis_norm
        elif prev_thigh_local is not None:
            # 다리가 거의 펴짐 -> 이전 프레임 축을 허벅지 방향 변화량만큼 평행이동
            knee_axis = unit(rodrigues_transport(prev_thigh_local, thigh_local, prev_axis_local))
        else:
            knee_axis = prev_axis_local

        # 부호 보정: 무릎각은 양수(0~2.09rad, character.xml 관절범위)여야 함.
        # cross(thigh,shin) 방향이 "양의 축"과 반대일 수 있어 axis 부호를 knee_mag 부호계로 흡수.
        sign_check = np.dot(np.cross(thigh_local, knee_axis), shin_local)
        knee = knee_mag if sign_check <= 0 else -knee_mag
        # (부호 규약은 character.xml 실제 재생 후 반대로 보이면 위 부호를 뒤집을 것 — 주석 참고)

        # R_hip 조립: 3번째 열=-thigh_local, 2번째 열=knee_axis, 1번째 열=2x3 외적
        col3 = -thigh_local
        col2 = knee_axis
        col1 = np.cross(col2, col3)
        R_hip = nearest_rotation(np.column_stack([col1, col2, col3]))

        yaw, roll, pitch = hip_euler_from_matrix(R_hip)
        angles[t] = [yaw, roll, pitch, knee]

        prev_thigh_local = thigh_local
        prev_axis_local = knee_axis

    return angles


def resample(t_src, arr, t_dst):
    """스칼라/벡터 채널 선형보간 (쿼터니언 제외)."""
    out = np.zeros((len(t_dst),) + arr.shape[1:])
    for i in range(arr.shape[1]):
        out[:, i] = np.interp(t_dst, t_src, arr[:, i])
    return out


def resample_quat(t_src, quats_xyzw, t_dst):
    """쿼터니언 전용 slerp 리샘플. quats_xyzw: (n,4) scipy 순서(x,y,z,w)."""
    slerp = Slerp(t_src, Rotation.from_quat(quats_xyzw))
    return slerp(t_dst).as_quat()


def foot_contact_from_bones(df, n, thresh):
    """발 본이 따로 없어(발목 없는 설계) shin tail(=발 위치 근사)의 z높이로 접촉 판정."""
    contact = np.zeros((n, 2), dtype=np.int8)
    for t in range(n):
        l_z = bone_vec(df, LEG_BONES["l"]["shin"], "tail", t)[2]
        r_z = bone_vec(df, LEG_BONES["r"]["shin"], "tail", t)[2]
        contact[t, 0] = 1 if l_z <= thresh else 0
        contact[t, 1] = 1 if r_z <= thresh else 0
    return contact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="export_raw.csv")
    parser.add_argument("--out", type=str, default="references/blender_walk.npz")
    parser.add_argument("--model", type=str, default="models/character.xml")
    parser.add_argument("--fps", type=float, default=30.0, help="Blender 씬 프레임레이트")
    parser.add_argument("--foot-height-thresh", type=float, default=0.005,
                         help="이 높이(m) 이하면 발이 바닥에 닿은 것으로 판정")
    parser.add_argument("--root-bone", type=str, default=ROOT_BONE)
    parser.add_argument("--thigh-l", type=str, default=LEG_BONES["l"]["thigh"])
    parser.add_argument("--shin-l", type=str, default=LEG_BONES["l"]["shin"])
    parser.add_argument("--thigh-r", type=str, default=LEG_BONES["r"]["thigh"])
    parser.add_argument("--shin-r", type=str, default=LEG_BONES["r"]["shin"])
    args = parser.parse_args()
    apply_bone_name_args(args)

    root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(root, args.csv)
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"입력 프레임 수: {n} (Blender {args.fps} fps)")

    root_pos, root_rot = build_root_frames(df, n)
    l_angles = compute_leg_joint_angles(df, n, root_pos, root_rot, "l")
    r_angles = compute_leg_joint_angles(df, n, root_pos, root_rot, "r")

    print("\n=== 관절각 요약 (도) ===")
    for name, arr, col in [("l_hip_yaw", l_angles, 0), ("l_hip_roll", l_angles, 1),
                            ("l_hip_pitch", l_angles, 2), ("l_knee", l_angles, 3),
                            ("r_hip_yaw", r_angles, 0), ("r_hip_roll", r_angles, 1),
                            ("r_hip_pitch", r_angles, 2), ("r_knee", r_angles, 3)]:
        deg = np.degrees(arr[:, col])
        print(f"  {name:12s} min={deg.min():7.2f}  max={deg.max():7.2f}")

    quats_xyzw = Rotation.from_matrix(root_rot).as_quat()  # (n,4) x,y,z,w

    # ---- 원본 Blender 프레임 시간축 -> CTRL_DT(0.02s) 균일 그리드로 리샘플 ----
    t_src = np.arange(n) / args.fps
    duration = t_src[-1]
    t_dst = np.arange(0.0, duration, CTRL_DT)
    m = len(t_dst)
    print(f"\n리샘플: {args.fps}fps {n}프레임 -> {1/CTRL_DT:.0f}Hz {m}프레임 (길이 {duration:.2f}s)")

    root_pos_rs = resample(t_src, root_pos, t_dst)
    l_angles_rs = resample(t_src, l_angles, t_dst)
    r_angles_rs = resample(t_src, r_angles, t_dst)
    quat_rs_xyzw = resample_quat(t_src, quats_xyzw, t_dst)
    quat_rs = np.column_stack([quat_rs_xyzw[:, 3], quat_rs_xyzw[:, 0],
                                quat_rs_xyzw[:, 1], quat_rs_xyzw[:, 2]])  # xyzw -> wxyz

    joint_angles = np.zeros((m, 8))
    joint_angles[:, 0:3] = l_angles_rs[:, 0:3]
    joint_angles[:, 3] = l_angles_rs[:, 3]
    joint_angles[:, 4:7] = r_angles_rs[:, 0:3]
    joint_angles[:, 7] = r_angles_rs[:, 3]

    qpos = np.zeros((m, 15))
    qpos[:, 0:3] = root_pos_rs
    qpos[:, 3:7] = quat_rs
    qpos[:, 7:15] = joint_angles

    foot_contact_src = foot_contact_from_bones(df, n, args.foot_height_thresh)
    foot_contact = np.zeros((m, 2), dtype=np.int8)
    for i in range(2):
        foot_contact[:, i] = np.round(np.interp(t_dst, t_src, foot_contact_src[:, i])).astype(np.int8)

    model_path = os.path.join(root, args.model)
    model = mujoco.MjModel.from_xml_path(model_path)
    qvel = np.zeros((m, model.nv))
    for i in range(1, m):
        mujoco.mj_differentiatePos(model, qvel[i], CTRL_DT, qpos[i - 1], qpos[i])
    qvel[0] = qvel[1]

    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, qpos=qpos, qvel=qvel, foot_contact=foot_contact)
    print(f"\n저장 완료: {out_path} (qpos {qpos.shape}, qvel {qvel.shape}, foot_contact {foot_contact.shape})")
    print("주의: scripts/check_feasibility.py 사전검사 ①~⑤를 통과하는지 확인 후 RL을 돌릴 것.")


if __name__ == "__main__":
    main()
