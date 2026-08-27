"""사전검사 ① 지지다각형만 독립 실행 — CLAUDE.md `check_feasibility.py`의 5개 검사
중 이것 하나만 뗀 버전 (②~⑤는 별도).

닿은 발들의 다각형 안에 무게중심(CoM) xy가 있는가를 프레임마다 확인한다.
물리 스텝 없이 매 프레임 qpos를 그대로 mj_forward(정기구학)만 해서 판정하므로
자기충돌/발산 여부와 무관하게 항상 돌아간다.

접촉 판정은 (이전 즉석검사에서 썼던 Blender 본 높이 휴리스틱과 달리) **실제 로봇
발 geom의 world z높이**로 한다 — 발목이 없는 설계라 shin(발목 지점) 높이는 실제
지면 접촉점보다 5~9mm 위에 있어서 그걸로 판정하면 항상 "접지 없음"으로 잘못 나온다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import numpy as np

FOOT_GEOMS = {
    "l": ["l_foot_heel", "l_foot_mid", "l_foot_toe"],
    "r": ["r_foot_heel", "r_foot_mid", "r_foot_toe"],
}
SUPPORT_MARGIN = 0.02  # 발 두께 등 여유 (m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="references/npz_test.npz")
    parser.add_argument("--xml", default="models/character.xml")
    parser.add_argument("--foot-height-thresh", type=float, default=0.002,
                         help="발 geom의 이 높이(m) 이하면 접지로 판정")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    xml_path = args.xml if os.path.isabs(args.xml) else os.path.join(root, args.xml)
    ref_path = args.reference if os.path.isabs(args.reference) else os.path.join(root, args.reference)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    qpos = np.load(ref_path)["qpos"]
    n = qpos.shape[0]

    def gid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    foot_gids = {side: [gid(g) for g in names] for side, names in FOOT_GEOMS.items()}

    min_foot_z_overall = np.inf
    checked = 0
    violations = []

    for t in range(n):
        data.qpos[:] = qpos[t]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        foot_z = {side: [data.geom_xpos[g][2] for g in gids] for side, gids in foot_gids.items()}
        min_foot_z_overall = min(min_foot_z_overall, min(foot_z["l"] + foot_z["r"]))

        contact_pts = []
        for side, gids in foot_gids.items():
            zs = foot_z[side]
            if min(zs) <= args.foot_height_thresh:
                # heel/toe(양 끝)만 다각형 꼭짓점으로 사용 — mid는 heel-toe 사이라 불필요
                contact_pts.append(data.geom_xpos[gids[0]][:2])  # heel
                contact_pts.append(data.geom_xpos[gids[-1]][:2])  # toe

        if not contact_pts:
            continue
        checked += 1

        pts = np.array(contact_pts)
        com_xy = data.subtree_com[0][:2]
        lo_xy = pts.min(axis=0) - SUPPORT_MARGIN
        hi_xy = pts.max(axis=0) + SUPPORT_MARGIN
        if np.any(com_xy < lo_xy) or np.any(com_xy > hi_xy):
            margin_violation = max(
                float(np.max(lo_xy - com_xy)), float(np.max(com_xy - hi_xy)), 0.0)
            violations.append((t, margin_violation))

    print(f"입력 프레임: {n}, 접지 판정된 프레임: {checked}")
    print(f"발 최저 높이(전체 프레임 중 최소): {min_foot_z_overall:.5f} m "
          f"(임계값 {args.foot_height_thresh} m)")

    if checked == 0:
        print("경고: 접지로 판정된 프레임이 0개입니다 — --foot-height-thresh를 "
              "위 '발 최저 높이' 근처로 올려서 재시도하세요.")
        sys.exit(2)

    if violations:
        print(f"\n[지지다각형] FAIL — {len(violations)}/{checked} 접지프레임에서 CoM 이탈")
        worst = max(violations, key=lambda v: v[1])
        print(f"  가장 심한 프레임: t={worst[0]}, 이탈량={worst[1]*1000:.1f}mm")
        for t, m in violations[:10]:
            print(f"  frame {t}: 이탈 {m*1000:.1f}mm")
        if len(violations) > 10:
            print(f"  ... 외 {len(violations)-10}개 프레임 더")
        sys.exit(1)
    else:
        print(f"\n[지지다각형] PASS — 접지프레임 {checked}개 전부 CoM이 지지다각형 안에 있음")


if __name__ == "__main__":
    main()
