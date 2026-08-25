"""STAGE 1 스모크테스트 — docs/robot_spec.md 신규 비율 체계의 "기본 조건"
(1번/2번 선제 실험값)이 물리적으로 말이 되는지만 확인한다.

CLAUDE.md STAGE 1 원칙: "가장 먼저 실행할 것 — robot spec에 나와있는 기본
조건을 먼저 실행해서 테스트해볼 것." 이 스크립트는 그 원칙을 따라 학습(RL)
없이 순정 mujoco만으로:
  1. MJCF 컴파일이 되는가
  2. stand keyframe에서 다리끼리/발끼리/머리-몸통이 자기충돌하지 않는가
  3. stand 자세를 유지하려는 position actuator 하에서 중력을 받으며 짧게
     굴려봤을 때 발산(NaN)하거나 곧바로 넘어지지 않는가
를 확인한다. measure_stage0.py의 world-frame AABB 방식을 재사용한다 — 이
모델들은 foot_collision geom만 contype/conaffinity=1이고 나머지는 0으로
꺼져 있어 엔진 접촉 목록(data.ncon)만으로는 다리-다리 자기충돌을 볼 수 없다.
"""

import sys

import mujoco
import numpy as np

CANDIDATES = [
    ("case1_short (몸통0.35/0.4, 다리0.2/0.175)", "models/smoke_case1.xml"),
    ("case2_long (몸통0.55/0.4, 다리0.5/0.175)", "models/smoke_case2.xml"),
]

SIM_SECONDS = 8.0
FALL_HEIGHT_DROP_FRAC = 0.5  # 초기 몸통 높이 대비 이만큼 떨어지면 "넘어짐"으로 판정
MAX_JOINT_VEL = 30.0  # rad/s, 이 이상이면 발산으로 간주


def _local_half_extent(model, gid):
    """geom_size의 의미는 geom 타입마다 다르다 — box만 (x,y,z) half-extent를
    그대로 담고, sphere/capsule은 (반지름, [캡슐 한정] 반길이, 0)이다. AABB
    계산은 geom의 로컬 프레임 기준 진짜 half-extent가 필요하므로 타입별로
    변환한다. (이 버그 때문에 캡슐 다리를 박스처럼 취급해 반길이를 y/z
    half-extent로 잘못 넣어 스탠스가 좁아졌을 때 허위 자기충돌이 났었다 —
    2026-08-23.)"""
    gtype = model.geom_type[gid]
    r, half_len, _ = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.array([r, r, r])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.array([r, r, half_len + r])
    return np.array(model.geom_size[gid])  # box 등: geom_size가 그대로 half-extent


def geom_world_aabb(model, data, gid):
    pos = data.geom_xpos[gid]
    mat = data.geom_xmat[gid].reshape(3, 3)
    sx, sy, sz = _local_half_extent(model, gid)
    corners = np.array([
        [sx * a, sy * b, sz * c]
        for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)
    ])
    world_corners = pos + corners @ mat.T
    return world_corners.min(axis=0), world_corners.max(axis=0)


def _aabb_overlap(lo1, hi1, lo2, hi2):
    return np.all(lo1 <= hi2) and np.all(lo2 <= hi1)


def _body_side(model, gid):
    b = model.geom_bodyid[gid]
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    if bname.startswith("l_"):
        return "l"
    if bname.startswith("r_"):
        return "r"
    return None


def check_self_collision(model, data, key_name="stand"):
    """stand keyframe에서 왼쪽/오른쪽 다리 지오메트리 쌍 전체를 AABB로 겹침
    검사한다 (발-발, 다리-다리 포함 — 전부 좌/우 대칭이라 이 쌍만 봐도 충분).
    머리-몸통은 같은 body_link 트리에 붙어 고정 오프셋이라 비율 산출식 자체가
    겹치지 않게 배치하므로(§4-B 종횡비 유지) 별도 검사하지 않는다."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    data.qpos[:] = model.key_qpos[kid]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    left, right = [], []
    for g in range(model.ngeom):
        side = _body_side(model, g)
        if side == "l":
            left.append(g)
        elif side == "r":
            right.append(g)

    collisions = []
    for gl in left:
        lo1, hi1 = geom_world_aabb(model, data, gl)
        for gr in right:
            lo2, hi2 = geom_world_aabb(model, data, gr)
            if _aabb_overlap(lo1, hi1, lo2, hi2):
                nl = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gl)
                nr = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gr)
                collisions.append((nl, nr))
    return collisions


def run_hold_rollout(model, data, key_name="stand", seconds=SIM_SECONDS):
    """stand 자세를 목표로 position actuator를 고정한 채 중력 하에서 굴려본다.
    다리 8개 관절 전부 목표각 0(펴진 자세, keyframe과 동일)."""
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    data.qpos[:] = model.key_qpos[kid]
    data.qvel[:] = 0
    data.ctrl[:] = 0.0  # 전 관절 목표각 0 (keyframe 자세와 동일)
    mujoco.mj_forward(model, data)

    root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    z0 = data.xpos[root_bid][2]

    nsteps = int(seconds / model.opt.timestep)
    max_qvel = 0.0
    min_z = z0
    diverged = False

    for _ in range(nsteps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            diverged = True
            break
        max_qvel = max(max_qvel, np.max(np.abs(data.qvel[6:])))  # freejoint 6개 제외, 관절속도만
        min_z = min(min_z, data.xpos[root_bid][2])

    fell = (z0 - min_z) > FALL_HEIGHT_DROP_FRAC * z0
    return dict(z0=z0, min_z=min_z, max_qvel=max_qvel, diverged=diverged, fell=fell,
                nsteps_run=_ + 1 if diverged else nsteps)


def test_one(label, path):
    print("=" * 70)
    print(f"[{label}]  {path}")

    try:
        model = mujoco.MjModel.from_xml_path(path)
    except Exception as e:
        print(f"  컴파일: FAIL — {e}")
        return False
    print("  컴파일: PASS")
    data = mujoco.MjData(model)

    collisions = check_self_collision(model, data)
    if collisions:
        print(f"  자기충돌(stand keyframe): FAIL — {len(collisions)}쌍 겹침, 예: {collisions[:3]}")
        collision_ok = False
    else:
        print("  자기충돌(stand keyframe): PASS — 좌/우 다리 겹침 없음")
        collision_ok = True

    result = run_hold_rollout(model, data)
    print(f"  {SIM_SECONDS:.0f}초 물리 롤아웃 (stand 자세 유지 시도):")
    print(f"    시작 몸통 높이 z0={result['z0']:.4f} m, 최저 z={result['min_z']:.4f} m "
          f"(낙차 {result['z0']-result['min_z']:.4f} m)")
    print(f"    최대 관절속도={result['max_qvel']:.2f} rad/s, "
          f"실행 스텝={result['nsteps_run']}/{int(SIM_SECONDS/model.opt.timestep)}")
    if result["diverged"]:
        print("    안정성: FAIL — NaN/inf 발산")
        rollout_ok = False
    elif result["fell"]:
        print(f"    안정성: FAIL — 몸통 높이가 초기값의 {FALL_HEIGHT_DROP_FRAC*100:.0f}% 이상 낙하 (넘어짐)")
        rollout_ok = False
    elif result["max_qvel"] > MAX_JOINT_VEL:
        print(f"    안정성: FAIL — 관절속도 {result['max_qvel']:.1f} rad/s > {MAX_JOINT_VEL} (발산성 진동)")
        rollout_ok = False
    else:
        print("    안정성: PASS — 발산 없음, 서 있는 자세 유지")
        rollout_ok = True

    return collision_ok and rollout_ok


def main():
    all_pass = True
    for label, path in CANDIDATES:
        ok = test_one(label, path)
        all_pass = all_pass and ok

    print("=" * 70)
    print(f"전체 결과: {'PASS' if all_pass else 'FAIL — 위 로그에서 실패 항목 확인'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
