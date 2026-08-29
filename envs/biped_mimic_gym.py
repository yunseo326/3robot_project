"""STAGE 2 — 모방학습 환경 (로컬 CPU, Gymnasium).

envs/biped_rl_gym.py(STAGE1)의 관측/스텝 골격을 그대로 쓰고, references/stage1_walk.npz
(scripts/record_policy.py로 녹화한 case1_short 롤아웃)를 레퍼런스로 추종하도록
imitation 보상을 추가한다. CLAUDE.md §보상 구조 4항목(imitation + regularization +
limits + impact)을 전부 한 번에 투입한다(STAGE1과 달리 튜닝 순서 적용 안 함 —
STAGE2 지시사항).

2robot_project/envs/ant_mimic.py의 검증된 패턴(RSI, DeepMimic류 exp(-k·err) 보상)을
포팅했다. 다만:
  - ET는 2robot의 쿼터니언 tilt 계산 대신, STAGE1에서 이미 쓰던 upvector 센서 +
    낙상 geom 높이 기준을 그대로 재사용한다(코드베이스 일관성).
  - limits(관절한계 CBF, 발-발충돌)는 STAGE1 코드에서 그대로 가져왔다 — 2robot
    ant_mimic에는 없던 항목.
  - impact(발 Δv_z saturate)는 신규 작성 — 2robot에도 없던 항목. 발 body(l_foot_link/
    r_foot_link)의 z위치를 스텝간 유한차분해 속도를 추정하고, 발-바닥 접촉이 새로
    생기는 순간(rising edge)의 Δv_z²에 상한(clip)까지만 벌점을 준다.
  - 몸통 xy 위치도 추종 대상에 포함한다(2robot은 RSI로 시작위치가 매번 달라 아예
    뺐지만, CLAUDE.md가 "몸통 xy 위치"를 imitation 항목으로 명시하고 있고 RSI 프레임의
    실제 좌표에서 시작하므로 추종 자체는 의미가 있다).
  - 발 접촉 일치(𝟙[c=ĉ])는 record_policy.py가 저장한 foot_contact 배열과 비교해 신규
    구현 — 2robot에 선례 없음.
  - 목(neck) 관련 imitation 항목은 아직 목 관절이 없어(STAGE3에서 도입) 제외.
"""
import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

# envs/biped_rl_gym.py와 반드시 동일하게 유지 (STAGE1과 물리 스텝 해상도를 맞춰야
# references/stage1_walk.npz의 프레임 간격과 일치한다).
CTRL_DT = 0.02
SIM_DT = 0.004
ACTION_SCALE = 0.3
FALL_HEIGHT = 0.03
MIN_UPVECTOR_Z = 0.5
JOINT_LIMIT_MARGIN = 0.1
JOINT_LIMIT_GAMMA = 20.0

# regularization/limits 스케일 — STAGE1(envs/biped_rl_gym.py SCALES)과 동일 값 재사용.
REG_SCALES = dict(torque=-0.0005, action_rate=-0.02, action_acc=-0.01)
LIMIT_SCALES = dict(foot_collision=-1.0, joint_limit=-1.0)

# imitation exp(-k·err) 커널 계수 — 2robot ant_mimic.py의 pose(2.0)/velocity(0.1)/
# root(10.0) 값을 그대로 가져오고, root_pos/lin_vel/ang_vel은 신규 추정치.
K_LEG_POSE = 2.0
K_LEG_VEL = 0.1
# 팔(어깨·팔꿈치)은 leg_pose와 별도 항목 — CLAUDE.md STAGE3 "다리와 목의 가중치를
# 반드시 분리한다" 원칙을 팔에도 동일 적용(모델에 shoulder/elbow 관절이 있을 때만
# 활성화됨, __init__ 참고). 다리(±30°대)용으로 튜닝된 K_LEG_POSE=2.0을 어깨 스윙
# (최대 ~90°대)에 그대로 쓰면 오차 제곱합이 빠르게 커져 보상이 0으로 죽는다 — 실측
# (handmove_arm_full, 2026-08-29): imitation_leg_pose만 학습 내내 ~0으로 정체.
K_ARM_POSE = 0.4
K_ARM_VEL = 0.1
K_ROOT_POS = 20.0
K_ROOT_ORI = 10.0
K_LIN_VEL = 1.0
K_ANG_VEL = 1.0

# imitation 가중합 — 다리 비중을 가장 크게(CLAUDE.md: "다리와 목의 가중치를 반드시
# 분리한다"). survival은 정규화된 가중치 풀과 별개로 매 스텝 고정 가산(STAGE1의
# s.survival=0.5와 동일 취지). arm_pose/arm_vel은 팔 관절이 있는 모델에서만 0이
# 아닌 값을 갖는다(다리와 대칭적으로 동일 비중 부여).
IMITATION_WEIGHTS = dict(
    leg_pose=0.35, leg_vel=0.10, arm_pose=0.35, arm_vel=0.10,
    root_pos=0.15, root_ori=0.15,
    lin_vel_xy=0.05, lin_vel_z=0.05, ang_vel_xy=0.05, ang_vel_z=0.05,
    foot_contact=0.05,
)
SURVIVAL = 0.5

IMPACT_SCALE = -0.01
IMPACT_CLIP = 4.0  # (m/s)^2 상한 — saturate

FLOOR_GEOM_ID = 0


def _quat_to_upvector(quat):
    """(w,x,y,z) -> world-frame z축 단위벡터. mujoco framezaxis 센서와 동일 공식."""
    w, x, y, z = quat
    return np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])


class BipedMimicGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reference_path, model_path="models/character.xml",
                 min_episode_len=30, use_rsi=True, fall_geom_names=None,
                 action_scale=ACTION_SCALE, arm_action_scale=None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(CTRL_DT / SIM_DT))
        self.nu = self.model.nu
        self.use_rsi = use_rsi
        self.min_episode_len = min_episode_len

        ref = np.load(reference_path)
        self.ref_qpos = ref["qpos"]
        self.ref_qvel = ref["qvel"]
        self.ref_foot = ref["foot_contact"]
        self.n_frames = self.ref_qpos.shape[0]
        # 참조 몸통 upvector를 미리 전부 계산해둔다 (매 스텝 재계산 불필요).
        self.ref_upvec = np.array([_quat_to_upvector(q[3:7]) for q in self.ref_qpos])

        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.default_pose = self.model.key_qpos[kid][7:].copy()
        self.joint_lo = self.model.jnt_range[1:, 0].copy()
        self.joint_hi = self.model.jnt_range[1:, 1].copy()

        # 다리 vs 팔 DOF 분리 (관절 이름 기반, 모델에 무관하게 동작).
        # character.xml(STAGE2, 다리만)에서는 arm 인덱스가 빈 배열이 되어 arm_pose/
        # arm_vel 보상이 항상 0으로 꺼진다 — 기존 STAGE2 walk 학습 결과에 영향 없음.
        leg_qpos_idx, arm_qpos_idx = [], []
        leg_dof_idx, arm_dof_idx = [], []
        for j in range(1, self.model.njnt):  # 0번(root free joint) 제외
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            qadr = self.model.jnt_qposadr[j] - 7
            dadr = self.model.jnt_dofadr[j] - 6
            if "shoulder" in name or "elbow" in name:
                arm_qpos_idx.append(qadr)
                arm_dof_idx.append(dadr)
            else:
                leg_qpos_idx.append(qadr)
                leg_dof_idx.append(dadr)
        self._leg_qpos_idx = np.array(leg_qpos_idx, dtype=int)
        self._arm_qpos_idx = np.array(arm_qpos_idx, dtype=int)
        self._leg_dof_idx = np.array(leg_dof_idx, dtype=int)
        self._arm_dof_idx = np.array(arm_dof_idx, dtype=int)

        # ctrl = default_pose + action*action_scale로 매 스텝 절대 목표각을 정하므로,
        # default_pose 대비 action_scale보다 큰 편차는 policy가 action=±1을 내도 원리적으로
        # 낼 수 없다. STAGE2 걷기(기본값 0.3rad≈17°)는 다리 range로 충분했지만, 팔 실험
        # (handmove_arm_full_v2)에서 어깨 yaw가 목표 77°까지 못 가고 정확히 17.19°(=0.3rad)
        # 에서 얼어붙는 문제를 발견 — 원인이 보상(K_ARM_POSE)이 아니라 이 캡이었다. 다리는
        # 이미 검증된 0.3rad을 그대로 유지하고 팔에만 arm_action_scale을 별도로 줄 수 있게
        # 관절 인덱스(qpos[7:] 인덱스 == 이 코드베이스에서 actuator 인덱스와 이미 동일하게
        # 취급됨, default_pose/action이 전부 이 순서로 브로드캐스트되므로) 기준으로 벡터화한다.
        self.action_scale = np.full(self.nu, action_scale, dtype=float)
        if arm_action_scale is not None and len(self._arm_qpos_idx):
            self.action_scale[self._arm_qpos_idx] = arm_action_scale

        def gid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

        if fall_geom_names is None:
            fall_geom_names = [
                "head_geom", "torso_geom", "l_arm_geom", "r_arm_geom",
                "l_thigh_geom", "r_thigh_geom",
            ]
        self.fall_geom_ids = np.array([gid(n) for n in fall_geom_names])

        all_geom_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            for i in range(self.model.ngeom)
        ]
        foot_names = [n for n in all_geom_names if n and "foot" in n]
        self.l_foot_ids = set(gid(n) for n in foot_names if n.startswith("l_"))
        self.r_foot_ids = set(gid(n) for n in foot_names if n.startswith("r_"))

        self.l_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "l_foot_link")
        self.r_foot_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r_foot_link")

        self._gyro = self._sensor_slice("gyro")
        self._linvel = self._sensor_slice("local_linvel")
        self._upvec = self._sensor_slice("upvector")

        obs_dim = 3 + 3 + 3 + self.nu + self.nu + self.nu + self.nu + self.nu + 1 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)

        self.last_act = np.zeros(self.nu)
        self.last_last_act = np.zeros(self.nu)
        self.frame_idx = 0
        self.steps_in_episode = 0
        self._prev_foot_z = np.zeros(2)
        self._prev_foot_contact = np.zeros(2, dtype=np.int8)

    def _sensor_slice(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return adr, dim

    def _sensor(self, key):
        adr, dim = key
        return self.data.sensordata[adr:adr + dim]

    def _foot_contact(self):
        l_c = r_c = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist >= 0:
                continue
            g1, g2 = c.geom1, c.geom2
            other = g2 if g1 == FLOOR_GEOM_ID else (g1 if g2 == FLOOR_GEOM_ID else None)
            if other is None:
                continue
            if other in self.l_foot_ids:
                l_c = 1
            elif other in self.r_foot_ids:
                r_c = 1
        return np.array([l_c, r_c], dtype=np.int8)

    def _target(self, frame_idx):
        idx = min(frame_idx, self.n_frames - 1)
        return self.ref_qpos[idx], self.ref_qvel[idx], self.ref_foot[idx], self.ref_upvec[idx]

    def _get_obs(self):
        target_qpos, _, _, target_upvec = self._target(self.frame_idx + 1)
        return np.concatenate([
            self._sensor(self._gyro), self._sensor(self._linvel), self._sensor(self._upvec),
            self.data.qpos[7:] - self.default_pose, self.data.qvel[6:],
            self.last_act, self.last_last_act,
            target_qpos[7:],           # 다음 목표 다리 관절각
            [target_qpos[2]],          # 다음 목표 몸통 높이
            target_upvec,              # 다음 목표 몸통 방향(upvector)
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.use_rsi:
            start_max = max(self.n_frames - self.min_episode_len - 1, 1)
            self.frame_idx = int(self.np_random.integers(0, start_max))
        else:
            self.frame_idx = 0

        self.data.qpos[:] = self.ref_qpos[self.frame_idx]
        self.data.qvel[:] = self.ref_qvel[self.frame_idx]
        self.data.ctrl[:] = self.default_pose
        mujoco.mj_forward(self.model, self.data)

        self.last_act = np.zeros(self.nu)
        self.last_last_act = np.zeros(self.nu)
        self.steps_in_episode = 0
        self._prev_foot_contact = self._foot_contact()
        self._prev_foot_z = np.array([
            self.data.xpos[self.l_foot_bid][2], self.data.xpos[self.r_foot_bid][2],
        ])
        return self._get_obs(), {}

    def _terminated(self):
        min_h = float(np.min(self.data.geom_xpos[self.fall_geom_ids, 2]))
        fell = min_h < FALL_HEIGHT
        upvec = self._sensor(self._upvec)
        tipped = upvec[2] < MIN_UPVECTOR_Z
        return fell or tipped

    def _joint_limit_penalty(self):
        q = self.data.qpos[7:]
        qd = self.data.qvel[6:]
        dist_lo = q - self.joint_lo
        dist_hi = self.joint_hi - q
        into_lo = np.clip(JOINT_LIMIT_MARGIN - dist_lo, 0, None) * np.clip(-qd, 0, None)
        into_hi = np.clip(JOINT_LIMIT_MARGIN - dist_hi, 0, None) * np.clip(qd, 0, None)
        return float(np.sum(into_lo + into_hi))

    def _foot_foot_collision(self):
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist >= 0:
                continue
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.l_foot_ids and g2 in self.r_foot_ids) or \
               (g2 in self.l_foot_ids and g1 in self.r_foot_ids):
                n += 1
        return float(n)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.default_pose + action * self.action_scale
        ctrl = np.clip(ctrl, self.joint_lo, self.joint_hi)
        self.data.ctrl[:] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.frame_idx += 1
        self.steps_in_episode += 1
        target_qpos, target_qvel, target_foot, target_upvec = self._target(self.frame_idx)

        # ---- imitation ----
        cur_pose = self.data.qpos[7:]
        tgt_pose = target_qpos[7:]
        cur_vel = self.data.qvel[6:]
        tgt_vel = target_qvel[6:]
        leg_pose_err = float(np.sum((cur_pose[self._leg_qpos_idx] - tgt_pose[self._leg_qpos_idx]) ** 2))
        leg_vel_err = float(np.sum((cur_vel[self._leg_dof_idx] - tgt_vel[self._leg_dof_idx]) ** 2))
        if len(self._arm_qpos_idx):
            arm_pose_err = float(np.sum((cur_pose[self._arm_qpos_idx] - tgt_pose[self._arm_qpos_idx]) ** 2))
            arm_vel_err = float(np.sum((cur_vel[self._arm_dof_idx] - tgt_vel[self._arm_dof_idx]) ** 2))
        else:
            arm_pose_err = arm_vel_err = 0.0
        root_pos_err = float(np.sum((self.data.qpos[0:2] - target_qpos[0:2]) ** 2))
        quat_align = np.clip(abs(np.dot(self.data.qpos[3:7], target_qpos[3:7])), 0.0, 1.0)
        root_ori_err = 1.0 - quat_align ** 2
        lin_vel_xy_err = float(np.sum((self.data.qvel[0:2] - target_qvel[0:2]) ** 2))
        lin_vel_z_err = float((self.data.qvel[2] - target_qvel[2]) ** 2)
        ang_vel_xy_err = float(np.sum((self.data.qvel[3:5] - target_qvel[3:5]) ** 2))
        ang_vel_z_err = float((self.data.qvel[5] - target_qvel[5]) ** 2)

        cur_foot = self._foot_contact()
        foot_match = float(np.mean(cur_foot == target_foot))

        w = IMITATION_WEIGHTS
        r_leg_pose = w["leg_pose"] * np.exp(-K_LEG_POSE * leg_pose_err)
        r_leg_vel = w["leg_vel"] * np.exp(-K_LEG_VEL * leg_vel_err)
        if len(self._arm_qpos_idx):
            r_arm_pose = w["arm_pose"] * np.exp(-K_ARM_POSE * arm_pose_err)
            r_arm_vel = w["arm_vel"] * np.exp(-K_ARM_VEL * arm_vel_err)
        else:
            r_arm_pose = r_arm_vel = 0.0
        r_root_pos = w["root_pos"] * np.exp(-K_ROOT_POS * root_pos_err)
        r_root_ori = w["root_ori"] * np.exp(-K_ROOT_ORI * root_ori_err)
        r_lin_vel_xy = w["lin_vel_xy"] * np.exp(-K_LIN_VEL * lin_vel_xy_err)
        r_lin_vel_z = w["lin_vel_z"] * np.exp(-K_LIN_VEL * lin_vel_z_err)
        r_ang_vel_xy = w["ang_vel_xy"] * np.exp(-K_ANG_VEL * ang_vel_xy_err)
        r_ang_vel_z = w["ang_vel_z"] * np.exp(-K_ANG_VEL * ang_vel_z_err)
        r_foot_contact = w["foot_contact"] * foot_match
        r_survival = SURVIVAL

        # ---- regularization (STAGE1과 동일 스케일) ----
        action_rate = float(np.sum((action - self.last_act) ** 2))
        action_acc = float(np.sum((action - 2 * self.last_act + self.last_last_act) ** 2))
        r_torque = REG_SCALES["torque"] * float(np.sum(self.data.actuator_force ** 2))
        r_action_rate = REG_SCALES["action_rate"] * action_rate
        r_action_acc = REG_SCALES["action_acc"] * action_acc

        # ---- limits (STAGE1과 동일 로직) ----
        r_joint_limit = LIMIT_SCALES["joint_limit"] * self._joint_limit_penalty()
        r_foot_collision = LIMIT_SCALES["foot_collision"] * self._foot_foot_collision()

        # ---- impact: 발 접촉이 새로 생기는 순간의 Δv_z^2, saturate ----
        foot_z = np.array([
            self.data.xpos[self.l_foot_bid][2], self.data.xpos[self.r_foot_bid][2],
        ])
        foot_vz = (foot_z - self._prev_foot_z) / CTRL_DT
        new_contact = (cur_foot == 1) & (self._prev_foot_contact == 0)
        impact_sq = np.where(new_contact, foot_vz ** 2, 0.0)
        impact_sq = np.clip(impact_sq, 0.0, IMPACT_CLIP)
        r_impact = IMPACT_SCALE * float(np.sum(impact_sq))
        self._prev_foot_z = foot_z
        self._prev_foot_contact = cur_foot

        reward = (r_leg_pose + r_leg_vel + r_arm_pose + r_arm_vel
                  + r_root_pos + r_root_ori + r_lin_vel_xy
                  + r_lin_vel_z + r_ang_vel_xy + r_ang_vel_z + r_foot_contact + r_survival
                  + r_torque + r_action_rate + r_action_acc
                  + r_joint_limit + r_foot_collision + r_impact)

        self.last_last_act = self.last_act
        self.last_act = action

        terminated = self._terminated()
        truncated = self.frame_idx >= self.n_frames - 1

        info = {
            "reward_imitation_leg_pose": float(r_leg_pose),
            "reward_imitation_leg_vel": float(r_leg_vel),
            "reward_imitation_arm_pose": float(r_arm_pose),
            "reward_imitation_arm_vel": float(r_arm_vel),
            "reward_imitation_root_pos": float(r_root_pos),
            "reward_imitation_root_ori": float(r_root_ori),
            "reward_imitation_lin_vel_xy": float(r_lin_vel_xy),
            "reward_imitation_lin_vel_z": float(r_lin_vel_z),
            "reward_imitation_ang_vel_xy": float(r_ang_vel_xy),
            "reward_imitation_ang_vel_z": float(r_ang_vel_z),
            "reward_imitation_foot_contact": float(r_foot_contact),
            "reward_imitation_survival": float(r_survival),
            "reward_regularization": float(r_torque + r_action_rate + r_action_acc),
            "reward_limits_joint": float(r_joint_limit),
            "reward_limits_foot_collision": float(r_foot_collision),
            "reward_impact": float(r_impact),
        }
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info


class RewardComponentWrapper(gym.Wrapper):
    """envs/biped_rl_gym.py의 동명 클래스와 동일 패턴 — 에피소드 종료 시 항목별
    누적값+길이를 info에 담는다."""

    REWARD_KEYS = (
        "imitation_leg_pose", "imitation_leg_vel",
        "imitation_arm_pose", "imitation_arm_vel",
        "imitation_root_pos", "imitation_root_ori",
        "imitation_lin_vel_xy", "imitation_lin_vel_z", "imitation_ang_vel_xy",
        "imitation_ang_vel_z", "imitation_foot_contact", "imitation_survival",
        "regularization", "limits_joint", "limits_foot_collision", "impact",
    )

    def __init__(self, env):
        super().__init__(env)
        self._sums = {f"reward_{k}": 0.0 for k in self.REWARD_KEYS}
        self._len = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sums = {f"reward_{k}": 0.0 for k in self.REWARD_KEYS}
        self._len = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self._sums:
            if k in info:
                self._sums[k] += info[k]
        self._len += 1
        if terminated or truncated:
            info["episode_components"] = dict(self._sums)
            info["episode_components"]["length"] = self._len
        return obs, reward, terminated, truncated, info


def make_biped_mimic_env(reference_path, model_path="models/character.xml",
                          min_episode_len=30, use_rsi=True, fall_geom_names=None,
                          action_scale=ACTION_SCALE, arm_action_scale=None):
    def _make():
        env = BipedMimicGym(reference_path, model_path=model_path,
                             min_episode_len=min_episode_len, use_rsi=use_rsi,
                             fall_geom_names=fall_geom_names, action_scale=action_scale,
                             arm_action_scale=arm_action_scale)
        env = RewardComponentWrapper(env)
        return env
    return _make
