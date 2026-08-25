"""STAGE 1 후보 MJCF 파라메트릭 생성기.

candidate_A.xml / candidate_B.xml을 손으로 만들며 잡았던 버그들(몸통에
<inertial>을 따로 줘서 머리·팔 질량이 무시되던 것, 롤오버 밑창 중앙/앞뒤
캡슐 높이가 뒤바뀌었던 것, 무릎 관절 기본각이 한계 경계와 겹치던 것)을
전부 여기 한 곳에 반영해뒀다. §4-B(몸 비율)·§4-C(다리·발 형상) 그리드를
손으로 복붙하면 이 버그들이 재발할 위험이 크므로, 이 생성기로만 MJCF를
만든다.

고정값(이번 스윕에서 안 바꾸는 축):
  - 전체 높이 40cm (STAGE 0 확정)
  - 다리 두께 = 다리 길이의 25% (§4-C1 그리드 중간값, §4-C에서 다시 열림)
  - 발 길이 = 다리 길이의 35%, 발 너비 = 발 길이의 70% (§4-C2 중간값)
  - 롤오버 밑창 곡률 반경 = 다리 길이의 0.2727배 (기존 40cm 기준 0.03m와 동일 비율)
  - 질량 배분(각 링크 kg)은 절대값 고정 — 이번 스윕은 "길이 비율"만 바꾸는
    것이라 질량까지 같이 바꾸면 두 변수를 동시에 바꾸는 셈이 된다
    (작업 규칙 1: 한 번에 한 변수만).

바뀌는 축: head_frac (torso_frac == leg_frac == (1-head_frac)/2)
"""

import numpy as np

TOTAL_HEIGHT = 0.40
LEG_THICKNESS_FRAC = 0.25
FOOT_LENGTH_FRAC = 0.35
FOOT_WIDTH_TO_LENGTH = 0.70
FOOT_THICKNESS = 0.005
ROLLOVER_R_FRAC = 0.03 / 0.11  # 원래 40cm/head45% 기준값에서 뽑은 비율

# 원래 40cm 버전에서 뽑은 종횡비 (몸통/머리가 비율이 바뀌어도 뚱뚱해지거나
# 납작해지지 않게 유지한다)
TORSO_Y_TO_H = 0.04 / 0.055
TORSO_X_TO_H = 0.03 / 0.055
HEAD_Y_TO_H = 0.07 / 0.18
HEAD_X_TO_H = 0.05 / 0.18
ARM_GAP = 0.02          # 팔이 몸통 옆면에서 떨어진 거리 (0.06 - 0.04)
ARM_Z_TO_TORSO_H = 0.02 / 0.11

JOINT_RANGES = dict(
    hip_yaw=(-0.5236, 0.5236),
    hip_roll=(-0.5236, 0.5236),
    hip_pitch=(-1.5708, 0.7854),
    knee=(-0.1745, 2.0944),
    ank_pitch=(-0.5236, 0.5236),
)

MASSES = dict(
    torso=0.53, head=0.45, arm=0.03,
    hip_yaw_link=0.02, hip_roll_link=0.03, thigh=0.08, shank=0.06,
    # 후보 A 발 (롤오버, 캡슐 3개 합)
    foot_a_heel=0.013, foot_a_mid=0.014, foot_a_toe=0.013,
    # 후보 B 발 (평면 + 챔퍼)
    foot_b_sole=0.035, foot_b_chamfer=0.0025,
)


def _leg_side_xml(side, sign, leg_h, leg_radius, ankle, foot_length, foot_width, R, hip_y, torso_half_z):
    thigh_len = shank_len = leg_h / 2
    lo, hi = JOINT_RANGES["knee"]

    if ankle == "none":
        half_len = foot_width / 2
        r_s = 0.003
        foot_half_len = foot_length / 2
        x_off = foot_half_len * (0.015 / 0.01925)  # 원래 비율 유지 (앞/뒤 위치)

        def z_arc(x):
            return R - np.sqrt(max(R * R - x * x, 0.0))

        z_mid = -FOOT_THICKNESS + r_s
        z_edge = -FOOT_THICKNESS + z_arc(x_off) + r_s

        foot_xml = f"""
              <body name="{side}_foot_link" pos="0 0 -{shank_len:.6f}">
                <geom class="foot_collision" name="{side}_foot_heel" type="capsule" pos="-{x_off:.6f} 0 {z_edge:.6f}" axisangle="1 0 0 1.5708" size="{r_s:.6f} {half_len:.6f}" mass="{MASSES['foot_a_heel']}" rgba="0.15 0.15 0.15 1"/>
                <geom class="foot_collision" name="{side}_foot_mid"  type="capsule" pos="0 0 {z_mid:.6f}" axisangle="1 0 0 1.5708" size="{r_s:.6f} {half_len:.6f}" mass="{MASSES['foot_a_mid']}" rgba="0.15 0.15 0.15 1"/>
                <geom class="foot_collision" name="{side}_foot_toe"  type="capsule" pos="{x_off:.6f} 0 {z_edge:.6f}" axisangle="1 0 0 1.5708" size="{r_s:.6f} {half_len:.6f}" mass="{MASSES['foot_a_toe']}" rgba="0.15 0.15 0.15 1"/>
              </body>"""
        ankle_joint_xml = ""
        knee_body_close = "            </body>"
    else:  # "pitch"
        half_len_x = foot_length / 2
        half_len_y = foot_width / 2
        half_len_z = FOOT_THICKNESS / 2
        chamfer_r = 0.0015
        foot_xml = f"""
              <body name="{side}_ank_pitch_link" pos="0 0 -{shank_len:.6f}">
                <joint name="{side}_ank_pitch" axis="0 1 0" range="{JOINT_RANGES['ank_pitch'][0]} {JOINT_RANGES['ank_pitch'][1]}"/>
                <geom class="foot_collision" name="{side}_foot_sole" type="box" pos="0 0 -{half_len_z:.6f}" size="{half_len_x:.6f} {half_len_y:.6f} {half_len_z:.6f}" mass="{MASSES['foot_b_sole']}" rgba="0.15 0.15 0.15 1"/>
                <geom class="foot_collision" name="{side}_foot_heel_chamfer" type="capsule" pos="-{half_len_x:.6f} 0 -0.002" axisangle="1 0 0 1.5708" size="{chamfer_r} {half_len_y:.6f}" mass="{MASSES['foot_b_chamfer']}" rgba="0.15 0.15 0.15 1"/>
                <geom class="foot_collision" name="{side}_foot_toe_chamfer" type="capsule" pos="{half_len_x:.6f} 0 -0.002" axisangle="1 0 0 1.5708" size="{chamfer_r} {half_len_y:.6f}" mass="{MASSES['foot_b_chamfer']}" rgba="0.15 0.15 0.15 1"/>
              </body>"""
        knee_body_close = "            </body>"

    return f"""
      <body name="{side}_hip_yaw_link" pos="0 {sign}{hip_y:.6f} -{torso_half_z:.6f}">
        <joint name="{side}_hip_yaw" axis="0 0 1" range="{JOINT_RANGES['hip_yaw'][0]} {JOINT_RANGES['hip_yaw'][1]}"/>
        <geom type="sphere" size="0.008" mass="{MASSES['hip_yaw_link']}" rgba="0.5 0.5 0.5 1"/>
        <body name="{side}_hip_roll_link">
          <joint name="{side}_hip_roll" axis="1 0 0" range="{JOINT_RANGES['hip_roll'][0]} {JOINT_RANGES['hip_roll'][1]}"/>
          <geom type="sphere" size="0.008" mass="{MASSES['hip_roll_link']}" rgba="0.5 0.5 0.5 1"/>
          <body name="{side}_hip_pitch_link">
            <joint name="{side}_hip_pitch" axis="0 1 0" range="{JOINT_RANGES['hip_pitch'][0]} {JOINT_RANGES['hip_pitch'][1]}"/>
            <geom name="{side}_thigh_geom" type="capsule" fromto="0 0 0 0 0 -{thigh_len:.6f}" size="{leg_radius:.6f}" mass="{MASSES['thigh']}" rgba="0.35 0.35 0.4 1"/>
            <body name="{side}_knee_link" pos="0 0 -{thigh_len:.6f}">
              <joint name="{side}_knee" axis="0 1 0" range="{lo} {hi}"/>
              <geom name="{side}_shank_geom" type="capsule" fromto="0 0 0 0 0 -{shank_len:.6f}" size="{leg_radius:.6f}" mass="{MASSES['shank']}" rgba="0.35 0.35 0.4 1"/>{foot_xml}
{knee_body_close}
          </body>
        </body>
      </body>"""


def generate(head_frac, torso_frac, leg_frac, ankle, model_name,
             leg_thickness_frac=None, foot_length_frac=None,
             torso_width_m=None, leg_width_m=None, total_height_m=None,
             leg_protrusion_m=None):
    """leg_thickness_frac/foot_length_frac을 넘기면 §4-C(다리두께·발면적) 스윕에
    쓸 수 있다 — 생략하면 §4-B 스윕에서 쓴 고정 중간값(LEG_THICKNESS_FRAC,
    FOOT_LENGTH_FRAC)을 그대로 쓴다.

    torso_width_m/leg_width_m(단위: m, 좌우 폭/지름)을 넘기면 기존 종횡비 상수
    (TORSO_Y_TO_H 등) 대신 이 값을 직접 쓴다 — docs/robot_spec.md(머리=1 기준
    비율 체계)처럼 폭이 높이의 고정 비율이 아니라 독립적으로 주어지는 경우용.
    전후(x축) 깊이는 기존 종횡비(TORSO_X_TO_H/TORSO_Y_TO_H)를 유지해 역산한다.
    total_height_m을 생략하면 TOTAL_HEIGHT(0.40m, STAGE 0 확정 스케일)를 쓴다.

    leg_protrusion_m(단위: m)을 leg_width_m과 함께 넘기면, 다리 중심(hip_y)을
    "다리 바깥쪽 면이 몸통 옆면에서 이만큼만 벗어난다"는 식으로 역산한다:
    hip_y = torso_half_y + leg_protrusion_m - leg_radius (docs/robot_spec.md
    2026-08-23 확정 규칙). 생략하면 기존 동작(hip_y = torso_half_y, 다리 중심이
    몸통 옆면에 바로 붙음)을 그대로 유지한다 — §4-B/§4-C 스윕 하위호환."""
    assert ankle in ("none", "pitch")
    assert abs(head_frac + torso_frac + leg_frac - 1.0) < 1e-9
    total_height_m = TOTAL_HEIGHT if total_height_m is None else total_height_m
    leg_thickness_frac = LEG_THICKNESS_FRAC if leg_thickness_frac is None else leg_thickness_frac
    foot_length_frac = FOOT_LENGTH_FRAC if foot_length_frac is None else foot_length_frac

    head_h = total_height_m * head_frac
    torso_h = total_height_m * torso_frac
    leg_h = total_height_m * leg_frac

    if leg_width_m is None:
        leg_radius = (leg_h * leg_thickness_frac) / 2
    else:
        leg_radius = leg_width_m / 2
    foot_length = leg_h * foot_length_frac
    foot_width = foot_length * FOOT_WIDTH_TO_LENGTH
    R = leg_h * ROLLOVER_R_FRAC

    torso_half_z = torso_h / 2
    if torso_width_m is None:
        torso_half_y = torso_h * TORSO_Y_TO_H
        torso_half_x = torso_h * TORSO_X_TO_H
    else:
        torso_half_y = torso_width_m / 2
        torso_half_x = torso_half_y * (TORSO_X_TO_H / TORSO_Y_TO_H)
    head_half_z = head_h / 2
    head_half_y = head_h * HEAD_Y_TO_H
    head_half_x = head_h * HEAD_X_TO_H

    if leg_protrusion_m is None:
        hip_y = torso_half_y
    else:
        hip_y = torso_half_y + leg_protrusion_m - leg_radius
    arm_y = torso_half_y + ARM_GAP
    arm_z = torso_h * ARM_Z_TO_TORSO_H

    root_z = leg_h + FOOT_THICKNESS + torso_half_z
    head_pos_z = torso_half_z + head_half_z

    l_leg = _leg_side_xml("l", "", leg_h, leg_radius, ankle, foot_length, foot_width, R, hip_y, torso_half_z)
    r_leg = _leg_side_xml("r", "-", leg_h, leg_radius, ankle, foot_length, foot_width, R, hip_y, torso_half_z)

    nu = 8 if ankle == "none" else 10
    joints = ["l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee"]
    if ankle == "pitch":
        joints.append("l_ank_pitch")
    joints += ["r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee"]
    if ankle == "pitch":
        joints.append("r_ank_pitch")
    actuators = "\n".join(
        f'    <position name="{j}_act" joint="{j}"/>' for j in joints
    )

    if ankle == "none":
        keyframe_joints = "0 0 0 0  0 0 0 0"
    else:
        keyframe_joints = "0 0 0 0 0  0 0 0 0 0"

    xml = f"""<!--
STAGE 1 §4-B 몸 비율 그리드 자동 생성 파일. scripts/gen_candidate_mjcf.py로
생성됨 — 직접 수정하지 말고 생성기를 고친 뒤 다시 만들 것.

head_frac={head_frac} torso_frac={torso_frac} leg_frac={leg_frac} ankle={ankle}
전체 높이={TOTAL_HEIGHT}m, 머리={head_h:.4f}m, 몸통={torso_h:.4f}m, 다리={leg_h:.4f}m
-->
<mujoco model="{model_name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" gravity="0 0 -9.81"/>

  <default>
    <joint type="hinge" damping="0.3" armature="0.001"/>
    <position kp="12" ctrlrange="-3.1416 3.1416"/>
    <geom contype="0" conaffinity="0" friction="0.9 0.02 0.01"/>
    <default class="foot_collision">
      <geom contype="1" conaffinity="1"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="0 0 0.01" contype="1" conaffinity="0" rgba="0.8 0.8 0.8 1"/>

    <body name="body_link" pos="0 0 {root_z:.6f}">
      <freejoint/>
      <site name="imu" pos="0 0 0"/>
      <geom name="torso_geom" type="box" size="{torso_half_x:.6f} {torso_half_y:.6f} {torso_half_z:.6f}" mass="{MASSES['torso']}" rgba="0.3 0.3 0.35 1"/>
      <geom name="head_geom" type="box" pos="0 0 {head_pos_z:.6f}" size="{head_half_x:.6f} {head_half_y:.6f} {head_half_z:.6f}" mass="{MASSES['head']}" rgba="0.2 0.6 0.9 1"/>
      <geom name="l_arm_geom" type="box" pos="0 {arm_y:.6f} {arm_z:.6f}" size="0.01 0.01 0.03" mass="{MASSES['arm']}" rgba="0.3 0.3 0.35 1"/>
      <geom name="r_arm_geom" type="box" pos="0 -{arm_y:.6f} {arm_z:.6f}" size="0.01 0.01 0.03" mass="{MASSES['arm']}" rgba="0.3 0.3 0.35 1"/>
{l_leg}
{r_leg}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <sensor>
    <gyro site="imu" name="gyro"/>
    <velocimeter site="imu" name="local_linvel"/>
    <accelerometer site="imu" name="accelerometer"/>
    <framezaxis objtype="site" objname="imu" name="upvector"/>
    <framelinvel objtype="site" objname="imu" name="global_linvel"/>
    <frameangvel objtype="site" objname="imu" name="global_angvel"/>
    <framepos objtype="site" objname="imu" name="position"/>
  </sensor>

  <keyframe>
    <key name="stand" qpos="0 0 {root_z:.6f} 1 0 0 0  {keyframe_joints}"/>
  </keyframe>
</mujoco>
"""
    return xml


def generate_from_head_unit_ratios(torso_h_ratio, torso_w_ratio, leg_h_ratio, leg_w_ratio,
                                    ankle, model_name, total_height_m=0.40,
                                    leg_protrusion_ratio=0.01):
    """docs/robot_spec.md 신규 비율 체계(머리 높이·폭 = 1 기준 단위, 몸통/다리의
    높이·폭이 각각 머리 대비 배수로 주어짐 — head_frac+torso_frac+leg_frac=1을
    가정하는 기존 generate()와 다른 파라미터화)를 위한 변환 헬퍼.

    STAGE 1 스모크테스트 가정(요약, 상세는 docs/assumptions.md):
      - 전체 높이 스케일은 total_height_m(기본 0.40m, STAGE 0 확정)을 앵커로 쓰고
        머리 높이(m) = total_height_m / (1 + torso_h_ratio + leg_h_ratio)로 역산한다.
      - 발 길이/폭 비율은 이 문서에 없으므로 기존 §4-C2 확정값(FOOT_LENGTH_FRAC,
        FOOT_WIDTH_TO_LENGTH)을 임시로 재사용한다.
      - "폭"은 좌우(y축) 폭으로 해석하고, 전후 깊이는 generate()의 기존 종횡비로
        역산한다(신규 문서가 깊이를 별도로 안 주므로).
      - leg_protrusion_ratio(기본 0.01, 머리=1 기준)는 docs/robot_spec.md
        2026-08-23 확정 규칙: 다리 바깥쪽 면이 몸통 옆면에서 이 비율만큼만
        벗어날 수 있다. generate()의 hip_y 공식으로 그대로 전달된다.
    """
    head_h_m = total_height_m / (1.0 + torso_h_ratio + leg_h_ratio)
    torso_h_m = head_h_m * torso_h_ratio
    leg_h_m = head_h_m * leg_h_ratio
    torso_width_m = head_h_m * torso_w_ratio
    leg_width_m = head_h_m * leg_w_ratio
    leg_protrusion_m = head_h_m * leg_protrusion_ratio

    head_frac = head_h_m / total_height_m
    torso_frac = torso_h_m / total_height_m
    leg_frac = leg_h_m / total_height_m

    return generate(head_frac, torso_frac, leg_frac, ankle, model_name,
                     torso_width_m=torso_width_m, leg_width_m=leg_width_m,
                     total_height_m=total_height_m, leg_protrusion_m=leg_protrusion_m)


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sweep_4b")
    os.makedirs(out_dir, exist_ok=True)

    for head_pct in (30, 40, 50, 60):
        head_frac = head_pct / 100.0
        rest = (1.0 - head_frac) / 2.0
        for ankle, label in (("none", "A"), ("pitch", "B")):
            xml = generate(head_frac, rest, rest, ankle, f"sweep4b_head{head_pct}_{label}")
            path = os.path.join(out_dir, f"head{head_pct}_{label}.xml")
            with open(path, "w") as f:
                f.write(xml)
            print("wrote", path)
