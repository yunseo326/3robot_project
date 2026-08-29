"""npz_handmove_test.csv(오른팔 어깨 스윙 테스트) -> models/_viz_arms_temp.xml용
qpos/qvel/foot_contact npz 변환.

scripts/convert_csv_to_npz.py의 다리 추출 로직(build_root_frames, compute_leg_joint_angles,
hip_euler_from_matrix 등)을 그대로 재사용하고, 팔(어깨+팔꿈치)만 추가 구현했다.

팔은 rest 방향이 로컬 Y(다리는 -Z)이고, _viz_arms_temp.xml에서 r_upper_arm_geom/
r_forearm_geom의 fromto가 -Y(왼팔은 +Y, 거울대칭)라 side_sign으로 보정해야 한다 —
2026-08-28 검증 스크립트에서 이 부호를 빼먹었던 버그를 발견해 여기 반영했다. 검증 방법:
계산된 qpos를 MuJoCo에 넣고 forward kinematics로 되읽은 팔 방향벡터가 원본 CSV 방향벡터와
일치하는지 확인(250프레임 전체 오차 0.0000deg로 통과).

주의: 이 파일이 다루는 _viz_arms_temp.xml은 "시각화/파이프라인 검증 전용" 모델이다.
STAGE3가 팔을 정식으로 정책에 포함하기 전까지 character.xml에는 반영하지 않는다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import convert_csv_to_npz as C  # noqa: E402

K_SCALE = 0.258064  # 실척 배율(head 전체높이/blender head=1단위). Task1 검증 완료.


def compute_arm_joint_angles(df, n, root_rot, side):
    ua_bone = f"upper_arm.{side}"
    fa_bone = f"forearm.{side}"
    side_sign = 1.0 if side == "L" else -1.0
    angles = np.zeros((n, 4))
    prev_ua_local = None
    prev_axis_local = np.array([0.0, 0.0, 1.0])

    for t in range(n):
        ua_head = C.bone_vec(df, ua_bone, "head", t)
        ua_tail = C.bone_vec(df, ua_bone, "tail", t)
        fa_head = C.bone_vec(df, fa_bone, "head", t)
        fa_tail = C.bone_vec(df, fa_bone, "tail", t)

        Rw2l = root_rot[t].T
        ua_local = Rw2l @ C.unit(ua_tail - ua_head)
        fa_local = Rw2l @ C.unit(fa_tail - fa_head)

        raw_axis = np.cross(ua_local, fa_local)
        axis_norm = np.linalg.norm(raw_axis)
        if axis_norm > C.KNEE_AXIS_EPS:
            elbow_axis = raw_axis / axis_norm
        elif prev_ua_local is not None:
            elbow_axis = C.unit(C.rodrigues_transport(prev_ua_local, ua_local, prev_axis_local))
        else:
            elbow_axis = prev_axis_local

        col1 = side_sign * ua_local
        col2 = elbow_axis
        col0 = np.cross(col1, col2)
        R_shoulder = C.nearest_rotation(np.column_stack([col0, col1, col2]))

        sin_e = -side_sign * np.dot(fa_local, col0)
        cos_e = side_sign * np.dot(fa_local, col1)
        elbow = np.arctan2(sin_e, cos_e)

        yaw, roll, pitch = C.hip_euler_from_matrix(R_shoulder)
        angles[t] = [yaw, roll, pitch, elbow]

        prev_ua_local = ua_local
        prev_axis_local = elbow_axis

    return angles


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.join(root, "npz_handmove_test.csv")
    model_path = os.path.join(root, "models", "_viz_arms_temp.xml")
    out_path = os.path.join(root, "references", "handmove_test.npz")
    fps = 30.0

    C.ROOT_BONE = "spine.001"
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"입력 프레임 수: {n} ({fps} fps)")

    root_pos_blender, root_rot = C.build_root_frames(df, n)
    l_leg = C.compute_leg_joint_angles(df, n, root_pos_blender, root_rot, "l")
    r_leg = C.compute_leg_joint_angles(df, n, root_pos_blender, root_rot, "r")
    l_arm = compute_arm_joint_angles(df, n, root_rot, "L")
    r_arm = compute_arm_joint_angles(df, n, root_rot, "R")

    quats_xyzw = Rotation.from_matrix(root_rot).as_quat()
    quat_wxyz = np.column_stack([quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]])

    qpos = np.zeros((n, 23))
    qpos[:, 0:3] = root_pos_blender * K_SCALE
    qpos[:, 3:7] = quat_wxyz
    qpos[:, 7:10] = l_leg[:, 0:3]
    qpos[:, 10] = l_leg[:, 3]
    qpos[:, 11:14] = r_leg[:, 0:3]
    qpos[:, 14] = r_leg[:, 3]
    qpos[:, 15:18] = l_arm[:, 0:3]
    qpos[:, 18] = l_arm[:, 3]
    qpos[:, 19:22] = r_arm[:, 0:3]
    qpos[:, 22] = r_arm[:, 3]

    # ---- 원본 fps -> CTRL_DT(0.02s) 균일 그리드 리샘플 ----
    t_src = np.arange(n) / fps
    duration = t_src[-1]
    t_dst = np.arange(0.0, duration, C.CTRL_DT)
    m = len(t_dst)
    print(f"리샘플: {fps}fps {n}프레임 -> {1/C.CTRL_DT:.0f}Hz {m}프레임 (길이 {duration:.2f}s)")

    root_pos_rs = C.resample(t_src, qpos[:, 0:3], t_dst)
    joints_rs = C.resample(t_src, qpos[:, 7:23], t_dst)
    quat_rs_xyzw = Rotation.from_quat(np.column_stack([qpos[:, 4], qpos[:, 5], qpos[:, 6], qpos[:, 3]])).as_quat()
    quat_rs = C.resample_quat(t_src, quat_rs_xyzw, t_dst)
    quat_rs = np.column_stack([quat_rs[:, 3], quat_rs[:, 0], quat_rs[:, 1], quat_rs[:, 2]])

    qpos_rs = np.zeros((m, 23))
    qpos_rs[:, 0:3] = root_pos_rs
    qpos_rs[:, 3:7] = quat_rs
    qpos_rs[:, 7:23] = joints_rs

    # 발이 안 움직이는 테스트라 접촉은 항상 1(선 자세)로 둔다.
    foot_contact = np.ones((m, 2), dtype=np.int8)

    model = mujoco.MjModel.from_xml_path(model_path)
    qvel = np.zeros((m, model.nv))
    for i in range(1, m):
        mujoco.mj_differentiatePos(model, qvel[i], C.CTRL_DT, qpos_rs[i - 1], qpos_rs[i])
    qvel[0] = qvel[1]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, qpos=qpos_rs, qvel=qvel, foot_contact=foot_contact)
    print(f"저장 완료: {out_path} (qpos {qpos_rs.shape}, qvel {qvel.shape})")


if __name__ == "__main__":
    main()
