"""STAGE 1 순수 RL 환경 — 로컬 CPU(Gymnasium + stable-baselines3)용.

envs/biped_rl.py(MJX/brax용)와 보상·종료조건·관측 공식을 완전히 동일하게
포팅했다 — Colab GPU가 막혔을 때 "로컬로 전환"하면서 비교 조건까지 바뀌면
안 되기 때문이다 (2026-08-11, 사용자 지시: GPU 못 쓰면 로컬로, 단 비교가
섞이면 안 되니 공정하게). 순정 mujoco(단일 인스턴스, jax 아님)로 동작하고,
`stable_baselines3.common.env_util.make_vec_env`로 여러 개를 프로세스
병렬화해서 쓴다 (2robot_project의 make_ant_env 패턴).
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

REWARD_KEYS = (
    "forward", "survival", "lin_vel_z", "orientation",
    "foot_collision", "torque", "action_rate", "action_acc", "joint_limit",
)

# envs/biped_rl.py의 default_config()와 반드시 동일하게 유지한다.
CTRL_DT = 0.02
SIM_DT = 0.004
EPISODE_LENGTH = 1000
ACTION_SCALE = 0.3
RESET_NOISE_SCALE = 0.02
FALL_HEIGHT = 0.03
MIN_UPVECTOR_Z = 0.5
FORWARD_VEL_TARGET = 0.3
SCALES = dict(
    forward=1.5, survival=0.5, lin_vel_z=-2.0, orientation=-1.0,
    foot_collision=-1.0, torque=-0.0005, action_rate=-0.02,
    action_acc=-0.01, joint_limit=-1.0,
)
JOINT_LIMIT_MARGIN = 0.1
JOINT_LIMIT_GAMMA = 20.0


def _sensor_slice(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return adr, dim


class BipedRLGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, xml_path, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(CTRL_DT / SIM_DT))
        self.nu = self.model.nu

        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.init_qpos = self.model.key_qpos[kid].copy()
        self.default_pose = self.init_qpos[7:].copy()
        self.joint_lo = self.model.jnt_range[1:, 0].copy()
        self.joint_hi = self.model.jnt_range[1:, 1].copy()

        def gid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

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

        self._gyro = _sensor_slice(self.model, "gyro")
        self._linvel = _sensor_slice(self.model, "local_linvel")
        self._upvec = _sensor_slice(self.model, "upvector")

        obs_dim = 3 + 3 + 3 + self.nu + self.nu + self.nu + self.nu
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)

        self.last_act = np.zeros(self.nu)
        self.last_last_act = np.zeros(self.nu)
        self.step_count = 0
        self.render_mode = render_mode

    def _sensor(self, key):
        adr, dim = key
        return self.data.sensordata[adr:adr + dim]

    def _get_obs(self):
        return np.concatenate([
            self._sensor(self._gyro), self._sensor(self._linvel), self._sensor(self._upvec),
            self.data.qpos[7:] - self.default_pose, self.data.qvel[6:],
            self.last_act, self.last_last_act,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        noise = RESET_NOISE_SCALE
        qpos = self.init_qpos.copy()
        qpos[7:] += self.np_random.uniform(-noise, noise, size=self.nu)
        qvel = np.zeros(self.model.nv)
        qvel[6:] += self.np_random.uniform(-noise, noise, size=self.nu)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = self.default_pose
        mujoco.mj_forward(self.model, self.data)

        self.last_act = np.zeros(self.nu)
        self.last_last_act = np.zeros(self.nu)
        self.step_count = 0
        return self._get_obs(), {}

    def _foot_collision(self):
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if c.dist >= 0:
                continue
            if (g1 in self.l_foot_ids and g2 in self.r_foot_ids) or \
               (g2 in self.l_foot_ids and g1 in self.r_foot_ids):
                n += 1
        return float(n)

    def _joint_limit_penalty(self):
        q = self.data.qpos[7:]
        qd = self.data.qvel[6:]
        dist_lo = q - self.joint_lo
        dist_hi = self.joint_hi - q
        into_lo = np.clip(JOINT_LIMIT_MARGIN - dist_lo, 0, None) * np.clip(-qd, 0, None)
        into_hi = np.clip(JOINT_LIMIT_MARGIN - dist_hi, 0, None) * np.clip(qd, 0, None)
        return float(np.sum(into_lo + into_hi))

    def _terminated(self):
        min_h = float(np.min(self.data.geom_xpos[self.fall_geom_ids, 2]))
        fell = min_h < FALL_HEIGHT
        upvec = self._sensor(self._upvec)
        tipped = upvec[2] < MIN_UPVECTOR_Z
        return fell or tipped

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.default_pose + action * ACTION_SCALE
        ctrl = np.clip(ctrl, self.joint_lo, self.joint_hi)
        self.data.ctrl[:] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        local_linvel = self._sensor(self._linvel)
        forward = min(local_linvel[0], FORWARD_VEL_TARGET)
        r_forward = SCALES["forward"] * forward
        r_survival = SCALES["survival"]
        r_lin_vel_z = SCALES["lin_vel_z"] * local_linvel[2] ** 2
        upvec = self._sensor(self._upvec)
        r_orientation = SCALES["orientation"] * float(np.sum(upvec[:2] ** 2))
        foot_collision = self._foot_collision()
        r_foot_collision = SCALES["foot_collision"] * foot_collision
        r_torque = SCALES["torque"] * float(np.sum(self.data.actuator_force ** 2))
        action_rate = float(np.sum((action - self.last_act) ** 2))
        action_acc = float(np.sum((action - 2 * self.last_act + self.last_last_act) ** 2))
        r_action_rate = SCALES["action_rate"] * action_rate
        r_action_acc = SCALES["action_acc"] * action_acc
        r_joint_limit = SCALES["joint_limit"] * self._joint_limit_penalty()

        reward = (r_forward + r_survival + r_lin_vel_z + r_orientation
                  + r_foot_collision + r_torque + r_action_rate + r_action_acc
                  + r_joint_limit)

        self.last_last_act = self.last_act
        self.last_act = action
        self.step_count += 1

        terminated = self._terminated()
        truncated = self.step_count >= EPISODE_LENGTH

        info = {
            "reward_forward": r_forward, "reward_survival": r_survival,
            "reward_lin_vel_z": r_lin_vel_z, "reward_orientation": r_orientation,
            "reward_foot_collision": r_foot_collision, "reward_torque": r_torque,
            "reward_action_rate": r_action_rate, "reward_action_acc": r_action_acc,
            "reward_joint_limit": r_joint_limit,
        }
        return self._get_obs(), reward, terminated, truncated, info


class RewardComponentWrapper(gym.Wrapper):
    """2robot_project의 AntRewardComponentWrapper와 동일한 패턴 —
    에피소드가 끝날 때 항목별 누적값 + 길이를 info에 담는다."""

    def __init__(self, env):
        super().__init__(env)
        self._sums = {f"reward_{k}": 0.0 for k in REWARD_KEYS}
        self._len = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sums = {f"reward_{k}": 0.0 for k in REWARD_KEYS}
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


def make_biped_env(xml_path, render_mode=None):
    def _make():
        env = BipedRLGym(xml_path, render_mode=render_mode)
        env = RewardComponentWrapper(env)
        return env
    return _make
