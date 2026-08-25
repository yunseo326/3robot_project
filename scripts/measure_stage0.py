"""STAGE 0 측정 ①~⑤ — Robotis OP3 몸으로 측정 방법론을 검증한다.

CLAUDE.md STAGE 0: "하지 않을 것 — 캐릭터 디자인, 커스텀 MJCF 작성"이므로
아직 존재하지 않는 우리 로봇이 아니라 기성 OP3 모델로 측정한다 (근거는
logs/question.md 참고). STAGE 1에서 우리 몸이 정해지면 같은 방법론을 그
몸에 다시 적용한다.

GPU/MJX가 필요 없다 — 정적/운동학적 측정이라 순정 mujoco(CPU)로 충분하다.

주의: 이 OP3 MJX 자산은 발(foot1/foot2) 지오메트리를 제외한 모든 바디의
contype/conaffinity가 0으로 꺼져있다 (MJX GPU 학습 속도를 위한 표준 최적화 —
불필요한 self-collision 쌍을 물리 엔진 단계에서 아예 계산하지 않는다). 그래서
`data.ncon`으로는 허벅지-허벅지, 머리-바닥 접촉을 절대 감지할 수 없다 (실제로
처음 시도에서 "허벅지 접촉"으로 잡힌 각도는 사실 발끼리의 접촉이었다 — 발만
contype/conaffinity=1이라 그것만 걸린 것). 그래서 이 스크립트는 엔진의 충돌
판정을 쓰지 않고, 각 지오메트리를 world-frame axis-aligned bounding box(AABB)로
직접 계산해 겹침 여부를 판정한다.

측정 항목:
  ① 역진자 시간상수 τ = sqrt(h_com / g)
  ② 정책 제어 주파수가 τ 안에 5회 이상 들어가는가 (Op3Joystick 기본 ctrl_dt 기준)
  ③ 고관절 최대 벌림각 — 다리를 움직여 허벅지끼리 처음 닿는 각도
  ④ 명령 속도 상한 — min(2*L*sin(θ)*f_step, sqrt(0.5*g*L))
  ⑤ 머리 접지 각도 — 몸통을 전/후/좌/우로 기울여 머리가 지면에 닿는 각도
"""

import numpy as np
import mujoco
from mujoco_playground import registry

G = 9.81


def joint_qpos_addr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[jid]


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def set_stand_pose(model, data, key_name="stand"):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    data.qpos[:] = model.key_qpos[kid]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# ① 역진자 시간상수
# ---------------------------------------------------------------------------

def measure_tau(model, data):
    set_stand_pose(model, data, "stand")
    root_bid = body_id(model, "body_link")
    h_com = data.subtree_com[root_bid][2]
    tau = np.sqrt(h_com / G)
    return h_com, tau


# ---------------------------------------------------------------------------
# ③ 고관절 최대 벌림각 (허벅지-허벅지 접촉이 처음 생기는 각도)
# ---------------------------------------------------------------------------

LEG_GEOM_PREFIXES = ("l_hip", "l_knee", "l_ank", "r_hip", "r_knee", "r_ank")


def geom_world_aabb(model, data, gid):
    """지오메트리를 로컬 박스(half-extent=geom_size)로 근사해, geom_xmat으로
    8개 꼭짓점을 world frame으로 변환한 뒤 축정렬 바운딩박스를 리턴한다.
    mesh 지오메트리의 geom_size는 mujoco가 자동 계산한 bounding half-extent다."""
    pos = data.geom_xpos[gid]
    mat = data.geom_xmat[gid].reshape(3, 3)
    sx, sy, sz = model.geom_size[gid]
    corners = np.array([
        [sx * a, sy * b, sz * c]
        for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)
    ])
    world_corners = pos + corners @ mat.T
    return world_corners.min(axis=0), world_corners.max(axis=0)


def _aabb_overlap(lo1, hi1, lo2, hi2):
    return np.all(lo1 <= hi2) and np.all(lo2 <= hi1)


# 허벅지 판정에 쓸 지오메트리 — 발(foot1/foot2) 자체는 제외한다 (발끼리 닿는 건
# 별개 현상: §4-C2 발 면적 실험에서 다룬다. 여기서는 순수 허벅지-종아리 부피만 본다).
LEG_BODY_PREFIXES_EXCLUDING_FOOT_GEOMS = True


def _leg_side_geoms(model):
    left, right = [], []
    for g in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "foot" in gname:
            continue  # 발 자체는 별개 현상이라 제외
        b = model.geom_bodyid[g]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if bname.startswith("l_"):
            left.append(g)
        elif bname.startswith("r_"):
            right.append(g)
    return set(left), set(right)


def _cross_leg_contact(model, data, left_geoms, right_geoms):
    for gl in left_geoms:
        lo1, hi1 = geom_world_aabb(model, data, gl)
        for gr in right_geoms:
            lo2, hi2 = geom_world_aabb(model, data, gr)
            if _aabb_overlap(lo1, hi1, lo2, hi2):
                return True
    return False


def hip_roll_sign_for_abduction(model, data):
    """l_hip_roll을 +방향으로 움직였을 때 왼쪽 무릎이 몸 중심선에서 멀어지면(외전) +1,
    가까워지면(내전) -1을 리턴한다 — 부호 관례를 가정하지 않고 실측으로 정한다."""
    l_roll_adr = joint_qpos_addr(model, "l_hip_roll")
    l_knee_bid = body_id(model, "l_knee_link")
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    base_qpos = model.key_qpos[kid].copy()

    data.qpos[:] = base_qpos
    mujoco.mj_forward(model, data)
    y0 = data.xpos[l_knee_bid][1]

    theta = np.deg2rad(10)
    data.qpos[:] = base_qpos
    data.qpos[l_roll_adr] = base_qpos[l_roll_adr] + theta
    mujoco.mj_forward(model, data)
    y1 = data.xpos[l_knee_bid][1]

    # y0 부호 기준으로 "중심선(0)에서 멀어졌는지"를 판정한다.
    moved_outward = abs(y1) > abs(y0)
    return +1 if moved_outward else -1


def measure_hip_abduction(model, data, direction, step_deg=0.5, max_deg=89.0):
    """direction: +1 또는 -1 (l_hip_roll = direction*theta, r_hip_roll = -direction*theta).
    +1 = 외전(벌림, 다리가 중심선에서 멀어짐) 방향이 되도록 direction 부호는 호출부에서
    hip_roll_sign_for_abduction()으로 미리 정한다."""
    left_geoms, right_geoms = _leg_side_geoms(model)
    l_roll_adr = joint_qpos_addr(model, "l_hip_roll")
    r_roll_adr = joint_qpos_addr(model, "r_hip_roll")

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    base_qpos = model.key_qpos[kid].copy()
    l_roll0 = base_qpos[l_roll_adr]
    r_roll0 = base_qpos[r_roll_adr]

    theta = 0.0
    while theta <= np.deg2rad(max_deg):
        data.qpos[:] = base_qpos
        data.qpos[l_roll_adr] = l_roll0 + direction * theta
        data.qpos[r_roll_adr] = r_roll0 - direction * theta
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        if _cross_leg_contact(model, data, left_geoms, right_geoms):
            return np.rad2deg(theta)
        theta += np.deg2rad(step_deg)
    return None  # max_deg까지 안 닿음


# ---------------------------------------------------------------------------
# ⑤ 머리 접지 각도 (전/후/좌/우로 기울여 head 지오미터가 바닥에 닿는 각도)
# ---------------------------------------------------------------------------

HEAD_BODY_NAMES = ("head_pan_link", "head_tilt_link")


def _head_geoms(model):
    out = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if bname in HEAD_BODY_NAMES:
            out.append(g)
    return set(out)


def _floor_contact_with(model, data, target_geoms, floor_geom=0):
    for g in target_geoms:
        lo, _hi = geom_world_aabb(model, data, g)
        if lo[2] <= 0.0:
            return True
    return False


def measure_head_ground_angle(model, data, axis, sign, step_deg=0.5, max_deg=89.0):
    """axis: 'pitch'(전후, y축) 또는 'roll'(좌우, x축). sign: +1/-1.

    stand 자세를 유지한 채, 두 발의 중점을 피벗으로 몸 전체를 강체로 회전시켜
    (관절각은 고정) 머리가 바닥에 닿는 각도를 찾는다.
    """
    head_geoms = _head_geoms(model)

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    base_qpos = model.key_qpos[kid].copy()

    # stand 자세에서 두 발 중점(피벗)을 계산한다.
    data.qpos[:] = base_qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    l_foot_bid = body_id(model, "l_ank_roll_link")
    r_foot_bid = body_id(model, "r_ank_roll_link")
    pivot = (data.xpos[l_foot_bid] + data.xpos[r_foot_bid]) / 2.0
    pivot[2] = 0.0

    root_pos0 = base_qpos[0:3].copy()
    root_quat0 = base_qpos[3:7].copy()  # (w, x, y, z)

    rot_axis = np.array([0.0, 1.0, 0.0]) if axis == "pitch" else np.array([1.0, 0.0, 0.0])

    theta = 0.0
    while theta <= np.deg2rad(max_deg):
        ang = sign * theta
        half = ang / 2.0
        dq = np.array([np.cos(half), *(np.sin(half) * rot_axis)])  # (w,x,y,z)

        # 새 root 자세 = dq * root_quat0, root 위치는 pivot 기준 회전
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, dq, root_quat0)

        rel = root_pos0 - pivot
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, dq)
        rotmat = rotmat.reshape(3, 3)
        new_pos = pivot + rotmat @ rel

        data.qpos[:] = base_qpos
        data.qpos[0:3] = new_pos
        data.qpos[3:7] = new_quat
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        if _floor_contact_with(model, data, head_geoms):
            return np.rad2deg(theta)
        theta += np.deg2rad(step_deg)
    return None


# ---------------------------------------------------------------------------
# ④ 다리 길이 (hip_pitch 관절 ~ 발바닥, 무릎/발목 각도 0인 편 다리 기준)
# ---------------------------------------------------------------------------

def measure_leg_length(model, data):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    base_qpos = model.key_qpos[kid].copy()
    data.qpos[:] = base_qpos
    for jn in ("l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll"):
        data.qpos[joint_qpos_addr(model, jn)] = 0.0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    hip_bid = body_id(model, "l_hip_pitch_link")
    hip_z = data.xpos[hip_bid][2]

    foot_bottom_z = None
    for g in ("l_foot1", "l_foot2"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        z = data.geom_xpos[gid][2] - model.geom_size[gid][2]
        foot_bottom_z = z if foot_bottom_z is None else min(foot_bottom_z, z)

    return hip_z - foot_bottom_z


def main():
    env = registry.load("Op3Joystick")
    model = env.mj_model
    data = mujoco.MjData(model)

    print("=" * 60)
    h_com, tau = measure_tau(model, data)
    print(f"① h_com = {h_com:.4f} m,  tau = sqrt(h_com/g) = {tau:.4f} s")

    ctrl_dt = 0.02  # Op3Joystick 기본 설정
    ratio = tau / ctrl_dt
    print(f"② ctrl_dt = {ctrl_dt} s ({1/ctrl_dt:.0f} Hz),  tau 안에 {ratio:.1f}회  ->  "
          f"{'통과 (>=5)' if ratio >= 5 else '미달 (<5)'}")

    print("-" * 60)
    print("③ 고관절 최대 벌림각:")
    abd_sign = hip_roll_sign_for_abduction(model, data)
    theta_abduction = measure_hip_abduction(model, data, direction=abd_sign)
    theta_adduction = measure_hip_abduction(model, data, direction=-abd_sign)
    print(f"   외전(다리를 벌림, 무릎이 중심선에서 멀어지는 방향): "
          f"{theta_abduction} deg에서 접촉" if theta_abduction is not None
          else "   외전 방향: 89도까지 허벅지 접촉 없음")
    print(f"   내전(다리가 서로 교차하는 방향, 참고용): "
          f"{theta_adduction} deg에서 접촉" if theta_adduction is not None
          else "   내전 방향: 89도까지 접촉 없음")
    print("   -> CLAUDE.md가 말하는 '벌림각'은 외전 값을 쓴다.")

    print("-" * 60)
    leg_length = measure_leg_length(model, data)
    print(f"④ 다리 길이 L (hip_pitch~발바닥, 편 다리): {leg_length:.4f} m")
    v_fall = np.sqrt(0.5 * G * leg_length)
    f_step = 1.0 / (2 * tau)
    if theta_abduction is not None:
        theta_rad = np.deg2rad(theta_abduction)
        v_swing = 2 * leg_length * np.sin(theta_rad) * f_step
        v_cmd = min(v_swing, v_fall)
        print(f"   f_step=1/(2*tau)={f_step:.3f} Hz  ->  v_swing={v_swing:.4f} m/s, "
              f"v_fall_limit={v_fall:.4f} m/s  =>  명령 속도 상한 = {v_cmd:.4f} m/s")
    else:
        print(f"   외전 방향이 89도까지 허벅지에 안 걸려서, OP3는 이 항목에서 허벅지 충돌이 "
              f"체결(binding) 제약이 아니다. v_fall_limit={v_fall:.4f} m/s가 사실상의 상한.")
        print(f"   (참고로 90도 근접값을 넣어도 sin(theta)~1이라 v_swing이 v_fall_limit을 "
              f"넘어서므로 결론은 같다: 명령 속도 상한 ≈ {v_fall:.4f} m/s)")

    print("-" * 60)
    print("⑤ 머리 접지 각도 (전/후/좌/우):")
    results = {}
    for label, axis, sign in [
        ("전방(pitch+)", "pitch", +1),
        ("후방(pitch-)", "pitch", -1),
        ("좌측(roll+)", "roll", +1),
        ("우측(roll-)", "roll", -1),
    ]:
        ang = measure_head_ground_angle(model, data, axis, sign)
        results[label] = ang
        print(f"   {label}: {ang} deg에서 머리-바닥 접촉" if ang is not None
              else f"   {label}: 89도까지 접촉 없음")

    print("=" * 60)


if __name__ == "__main__":
    main()
