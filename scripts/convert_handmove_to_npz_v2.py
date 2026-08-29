"""npz_handmove_test_ver2_with_r.csv(쿼터니언 포함) -> models/_viz_arms_temp.xml용
qpos/qvel/foot_contact npz 변환. convert_handmove_to_npz.py(v1)의 후속.

v1과의 차이 (원인: v1의 진동 버그)
----------------------------------
v1은 팔꿈치 굽힘 평면을 cross(upper_arm_dir, forearm_dir)로 추정했다. 이번 동작은
팔을 편 채(팔꿈치≈0) 어깨만 회전시키므로 두 벡터가 거의 평행 -> 외적이 구조적으로
불안정 -> r_elbow가 진짜 값(0)이 아니라 미세한 노이즈(프레임간 부호가 413번 중
104번 뒤집힘)를 담고 있었다. 이게 학습된 정책이 "떠는" 것처럼 보인 근본 원인으로 추정된다.

이 파일은 본마다 world-space rotation quaternion이 있다는 전제(ver2 CSV, 다리
회전축 검증 실험과 동일한 export_bones_with_rotation.py 산출물)를 이용해 외적 없이
직접 관절각을 구한다:

1. 어깨(3축): R_bone_world(t) = quat(t)를 행렬로. root-local 좌표계로 옮긴 뒤
   frame 0(=t-pose.csv와 일치 확인된 rest pose)을 기준으로
     R_joint(t) = R_bone_rootlocal(0)^T @ R_bone_rootlocal(t)
   로 "rest 대비 순수 관절회전"만 분리한다. R_joint = Rz(yaw)Rx(roll)Ry(pitch) 형태이므로
   기존 hip_euler_from_matrix를 그대로 재사용해 분해한다(다리와 팔 다 이 형태의 직렬
   힌지체인이라 재사용 가능 - convert_csv_to_npz.py 참고).
2. 팔꿈치(1축): rel(t) = ua_quat(t)^-1 * fa_quat(t) (upper_arm 로컬 프레임에서 본 forearm
   방향), rel_rest = rel(0)로 보정한 rel_joint(t) = rel_rest^-1 * rel(t)의 회전각을
   2*arccos(|w|)로 직접 구한다. 축 추정이 아예 필요 없어 팔이 펴진 구간에서도 안정적이다.

검증: 기존 방식과 동일하게 qpos를 MuJoCo forward kinematics에 넣어 되읽은 방향벡터를
원본 CSV와 대조(round-trip). 아래 main()에서 실행 시 콘솔에 오차를 출력한다.
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


def bone_quat_xyzw(df, bone, t):
    w = df.loc[t, f"{bone}__quat_w"]
    x = df.loc[t, f"{bone}__quat_x"]
    y = df.loc[t, f"{bone}__quat_y"]
    z = df.loc[t, f"{bone}__quat_z"]
    return np.array([x, y, z, w], dtype=float)


def compute_shoulder_angles(df, n, root_rot, side):
    """side: 'L' or 'R'. R_joint(t) = R_rootlocal(0)^T @ R_rootlocal(t)를
    hip_euler_from_matrix로 분해해 (yaw,roll,pitch) 시계열 반환. shape (n,3)."""
    ua_bone = f"upper_arm.{side}"
    angles = np.zeros((n, 3))

    ua_world_mats = Rotation.from_quat(
        np.array([bone_quat_xyzw(df, ua_bone, t) for t in range(n)])
    ).as_matrix()

    rootlocal_mats = np.einsum("tij,tjk->tik", root_rot.transpose(0, 2, 1), ua_world_mats)
    rest_inv = rootlocal_mats[0].T

    for t in range(n):
        R_joint = rest_inv @ rootlocal_mats[t]
        R_joint = C.nearest_rotation(R_joint)
        yaw, roll, pitch = C.hip_euler_from_matrix(R_joint)
        angles[t] = [yaw, roll, pitch]

    return angles


def compute_elbow_angle(df, n, side):
    """side: 'L' or 'R'. 팔꿈치 상대회전각(스칼라, rad) 시계열. shape (n,)."""
    ua_bone = f"upper_arm.{side}"
    fa_bone = f"forearm.{side}"

    ua_q = Rotation.from_quat(np.array([bone_quat_xyzw(df, ua_bone, t) for t in range(n)]))
    fa_q = Rotation.from_quat(np.array([bone_quat_xyzw(df, fa_bone, t) for t in range(n)]))
    rel = ua_q.inv() * fa_q  # forearm orientation in upper_arm local frame

    rel_rest_inv = rel[0].inv()
    elbow = np.zeros(n)
    for t in range(n):
        joint = rel_rest_inv * rel[t]
        w = np.clip(abs(joint.as_quat()[3]), -1.0, 1.0)
        elbow[t] = 2.0 * np.arccos(w)
    return elbow


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.join(root, "npz_handmove_test_ver2_with_r.csv")
    model_path = os.path.join(root, "models", "_viz_arms_temp.xml")
    out_path = os.path.join(root, "references", "handmove_test_v2.npz")
    fps = 30.0

    C.ROOT_BONE = "spine.001"
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"입력 프레임 수: {n} ({fps} fps)")

    root_pos_blender, root_rot = C.build_root_frames(df, n)
    l_leg = C.compute_leg_joint_angles(df, n, root_pos_blender, root_rot, "l")
    r_leg = C.compute_leg_joint_angles(df, n, root_pos_blender, root_rot, "r")

    l_shoulder = compute_shoulder_angles(df, n, root_rot, "L")
    r_shoulder = compute_shoulder_angles(df, n, root_rot, "R")
    l_elbow = compute_elbow_angle(df, n, "L")
    r_elbow = compute_elbow_angle(df, n, "R")

    print("\n=== 팔 관절각 요약 (도) ===")
    for name, arr in [("l_shoulder_yaw", l_shoulder[:, 0]), ("l_shoulder_roll", l_shoulder[:, 1]),
                       ("l_shoulder_pitch", l_shoulder[:, 2]), ("l_elbow", l_elbow),
                       ("r_shoulder_yaw", r_shoulder[:, 0]), ("r_shoulder_roll", r_shoulder[:, 1]),
                       ("r_shoulder_pitch", r_shoulder[:, 2]), ("r_elbow", r_elbow)]:
        deg = np.degrees(arr)
        d1 = np.diff(deg)
        sc = np.sum(np.diff(np.sign(d1)) != 0) if len(d1) > 1 else 0
        print(f"  {name:18s} min={deg.min():8.3f} max={deg.max():8.3f} max|diff|={np.max(np.abs(d1)):.5f} sign_changes={sc}/{max(len(d1)-1,0)}")

    quats_xyzw = Rotation.from_matrix(root_rot).as_quat()
    quat_wxyz = np.column_stack([quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]])

    qpos = np.zeros((n, 23))
    qpos[:, 0:3] = root_pos_blender * K_SCALE
    qpos[:, 3:7] = quat_wxyz
    qpos[:, 7:10] = l_leg[:, 0:3]
    qpos[:, 10] = l_leg[:, 3]
    qpos[:, 11:14] = r_leg[:, 0:3]
    qpos[:, 14] = r_leg[:, 3]
    qpos[:, 15:18] = l_shoulder
    qpos[:, 18] = l_elbow
    qpos[:, 19:22] = r_shoulder
    qpos[:, 22] = r_elbow

    # ---- round-trip 검증: qpos -> FK -> 팔 방향벡터를 원본 CSV와 대조 ----
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    max_err_deg = 0.0
    for t in range(n):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for side in ["R", "L"]:
            for part, geom in [("upper_arm", f"{side.lower()}_upper_arm_geom"),
                                ("forearm", f"{side.lower()}_forearm_geom")]:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                bodyid = model.geom_bodyid[gid]
                xmat = data.xmat[bodyid].reshape(3, 3)
                sim_dir = xmat @ np.array([0.0, 1.0 if side == "L" else -1.0, 0.0])

                head = C.bone_vec(df, f"{part}.{side}", "head", t)
                tail = C.bone_vec(df, f"{part}.{side}", "tail", t)
                csv_dir_local = root_rot[t].T @ C.unit(tail - head)
                # sim_dir는 root-local(직접 계산했으므로 root 회전 미적용) 대신
                # xmat은 world 기준이라 qpos의 root quat 회전이 이미 반영돼 있음.
                # root_rot[t]와 qpos root quat이 같은 값이므로 world 좌표계에서 비교.
                csv_dir_world = C.unit(tail - head)
                cos_a = np.clip(np.dot(sim_dir, csv_dir_world), -1.0, 1.0)
                err_deg = np.degrees(np.arccos(cos_a))
                max_err_deg = max(max_err_deg, err_deg)

    print(f"\nround-trip 최대 방향벡터 오차: {max_err_deg:.6f} deg")

    # ---- 리샘플: 원본 fps -> CTRL_DT(0.02s) ----
    t_src = np.arange(n) / fps
    duration = t_src[-1]
    t_dst = np.arange(0.0, duration, C.CTRL_DT)
    m = len(t_dst)
    print(f"리샘플: {fps}fps {n}프레임 -> {1/C.CTRL_DT:.0f}Hz {m}프레임 (길이 {duration:.2f}s)")

    root_pos_rs = C.resample(t_src, qpos[:, 0:3], t_dst)
    joints_rs = C.resample(t_src, qpos[:, 7:23], t_dst)
    quat_rs_xyzw = Rotation.from_quat(
        np.column_stack([qpos[:, 4], qpos[:, 5], qpos[:, 6], qpos[:, 3]])
    ).as_quat()
    quat_rs = C.resample_quat(t_src, quat_rs_xyzw, t_dst)
    quat_rs = np.column_stack([quat_rs[:, 3], quat_rs[:, 0], quat_rs[:, 1], quat_rs[:, 2]])

    qpos_rs = np.zeros((m, 23))
    qpos_rs[:, 0:3] = root_pos_rs
    qpos_rs[:, 3:7] = quat_rs
    qpos_rs[:, 7:23] = joints_rs

    foot_contact = np.ones((m, 2), dtype=np.int8)

    qvel = np.zeros((m, model.nv))
    for i in range(1, m):
        mujoco.mj_differentiatePos(model, qvel[i], C.CTRL_DT, qpos_rs[i - 1], qpos_rs[i])
    qvel[0] = qvel[1]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, qpos=qpos_rs, qvel=qvel, foot_contact=foot_contact)
    print(f"저장 완료: {out_path} (qpos {qpos_rs.shape}, qvel {qvel.shape})")


if __name__ == "__main__":
    main()
