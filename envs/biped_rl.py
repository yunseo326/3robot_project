"""STAGE 1 — 순수 RL 걷기 환경 (레퍼런스 모방 없음).

CLAUDE.md STAGE 1 / §보상 구조 "튜닝 순서" 1~5번까지만 쓴다:
  1. 과제 보상만 (전진 + 생존)
  2. 종료 조건 조이기
  3. 큰 정규화 (수직 속도, 롤/피치, 충돌)
  4. 토크/파워
  5. action rate -> action acc
추종/스타일 보상(6번)은 STAGE 2부터. 이 파일에는 없다.

서브스테이지 1-A(관절 유무), 1-B(몸 비율), 1-C(다리·발 형상) 전부 이 하나의
환경 클래스를 쓴다 — 차이는 어떤 MJCF(xml_path)를 넘기느냐뿐이다. "동일 조건
비교"(CLAUDE.md 작업 규칙 1)를 지키기 위해 보상 함수·하이퍼파라미터는 후보마다
절대 안 바꾼다.

mujoco_playground의 MjxEnv를 상속해 STAGE 0에서 이미 검증된
`learning.train_jax_ppo` 스크립트와 그대로 호환되게 만든다 (scripts/train_stage1.py
가 registry 대신 이 클래스를 직접 brax ppo.train에 넘긴다).
"""

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from ml_collections import config_dict

from mujoco_playground._src import mjx_env


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        action_scale=0.3,
        # 시작점 관측 노이즈 (도메인 무작위화는 STAGE 1에서 하지 않는다 — 나중 단계)
        reset_noise_scale=0.02,
        # ---- termination (튜닝 순서 2) ----
        # 머리/몸통/허벅지/팔 중 하나라도 이 높이 아래로 내려가면 "지면에 닿은 것"으로 보고 종료.
        # 각 지오메트리의 half-extent를 감안한 여유값 (전부 이 이상 높이에서 정상 보행함).
        fall_height=0.03,
        # 몸통이 이만큼 기울면(upvector의 z성분) 종료 — cos(60deg)=0.5
        min_upvector_z=0.5,
        # ---- 과제 보상 (튜닝 순서 1) ----
        forward_vel_target=0.3,  # m/s, STAGE0 OP3 측정값(1.107 m/s)보다 훨씬 보수적인 시작값
        scales=config_dict.create(
            forward=1.5,
            survival=0.5,
            lin_vel_z=-2.0,
            orientation=-1.0,
            foot_collision=-1.0,
            torque=-0.0005,
            action_rate=-0.02,
            action_acc=-0.01,
            joint_limit=-1.0,
        ),
        # ---- CBF 스타일 관절한계 페널티 (§보상 구조 limits) ----
        joint_limit_margin=0.1,   # q_m [rad]
        joint_limit_gamma=20.0,   # gamma_q
    )


class BipedRL(mjx_env.MjxEnv):
    """STAGE 1 다리 골격 실험용 순수 RL 이족 보행 환경."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict = None,
        config_overrides=None,
    ):
        config = config or default_config()
        super().__init__(config, config_overrides)

        self._xml_path = xml_path
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = config.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)

        self._nu = self._mj_model.nu
        m = self._mj_model

        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self._init_qpos = jp.array(m.key_qpos[kid])
        # 관절각 부분만 (자유joint 7개 제외) — action의 기준(default pose)
        self._default_pose = jp.array(m.key_qpos[kid][7:])

        self._joint_lo = jp.array(m.jnt_range[1:, 0])  # index 0 = free joint
        self._joint_hi = jp.array(m.jnt_range[1:, 1])

        def gid(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

        # 종료 조건에 쓸 지오메트리: 머리/몸통/허벅지/팔.
        # geom_xpos는 접촉 판정(contype/conaffinity)과 무관하게 항상 forward kinematics로
        # 계산되므로, 발을 제외한 지오메트리의 충돌이 꺼져 있어도(모델 전체 성능을 위해)
        # 높이 기반 종료 조건은 그대로 쓸 수 있다.
        fall_geom_names = [
            "head_geom", "torso_geom", "l_arm_geom", "r_arm_geom",
            "l_thigh_geom", "r_thigh_geom",
        ]
        self._fall_geom_ids = jp.array([gid(n) for n in fall_geom_names])

        # 발-발 충돌 판정용 (STAGE1 §4-A 표준 발 지오메트리 이름은 후보 A/B가 다르다).
        foot_geom_names = [n for n in
                            [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
                            if n and "foot" in n]
        l_foot = [n for n in foot_geom_names if n.startswith("l_")]
        r_foot = [n for n in foot_geom_names if n.startswith("r_")]
        self._l_foot_ids = jp.array([gid(n) for n in l_foot])
        self._r_foot_ids = jp.array([gid(n) for n in r_foot])

        self._imu_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")

    # ------------------------------------------------------------------
    # MjxEnv 필수 프로퍼티
    # ------------------------------------------------------------------
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # ------------------------------------------------------------------
    # 센서 헬퍼
    # ------------------------------------------------------------------
    def _sensor(self, data: mjx.Data, name: str, dim: int) -> jax.Array:
        return mjx_env.get_sensor_data(self._mj_model, data, name)

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, qpos_rng, qvel_rng = jax.random.split(rng, 3)

        noise = self._config.reset_noise_scale
        qpos = self._init_qpos.at[7:].add(
            jax.random.uniform(qpos_rng, (self._nu,), minval=-noise, maxval=noise)
        )
        qvel = jp.zeros(self._mjx_model.nv).at[6:].add(
            jax.random.uniform(qvel_rng, (self._nu,), minval=-noise, maxval=noise)
        )

        data = mjx_env.make_data(
            self._mj_model, qpos=qpos, qvel=qvel, ctrl=self._default_pose,
        )
        data = mjx.forward(self._mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "last_last_act": jp.zeros(self._nu),
            "step": jp.array(0, dtype=jp.int32),
        }
        metrics = {
            "reward/forward": jp.zeros(()),
            "reward/survival": jp.zeros(()),
            "reward/lin_vel_z": jp.zeros(()),
            "reward/orientation": jp.zeros(()),
            "reward/foot_collision": jp.zeros(()),
            "reward/torque": jp.zeros(()),
            "reward/action_rate": jp.zeros(()),
            "reward/action_acc": jp.zeros(()),
            "reward/joint_limit": jp.zeros(()),
        }
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        ctrl = self._default_pose + action * self._config.action_scale
        ctrl = jp.clip(ctrl, self._joint_lo, self._joint_hi)

        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        obs = self._get_obs(data, state.info)
        reward, reward_terms = self._reward(data, action, state.info)
        terminated = self._terminated(data)

        info = dict(state.info)
        info["last_last_act"] = state.info["last_act"]
        info["last_act"] = action
        info["step"] = state.info["step"] + 1

        metrics = dict(state.metrics)
        metrics.update(reward_terms)

        done = jp.where(terminated, 1.0, 0.0)
        done = jp.where(info["step"] >= self._config.episode_length, 1.0, done)

        return state.replace(
            data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info
        )

    # ------------------------------------------------------------------
    # 관측
    # ------------------------------------------------------------------
    def _get_obs(self, data: mjx.Data, info) -> jax.Array:
        gyro = self._sensor(data, "gyro", 3)
        local_linvel = self._sensor(data, "local_linvel", 3)
        upvector = self._sensor(data, "upvector", 3)
        qpos = data.qpos[7:]
        qvel = data.qvel[6:]
        return jp.concatenate([
            gyro, local_linvel, upvector,
            qpos - self._default_pose, qvel,
            info["last_act"], info["last_last_act"],
        ])

    # ------------------------------------------------------------------
    # 종료 조건 (튜닝 순서 2) — 머리/몸통/허벅지/팔 접지 또는 과도한 기울어짐
    # ------------------------------------------------------------------
    def _terminated(self, data: mjx.Data) -> jax.Array:
        min_h = jp.min(data.geom_xpos[self._fall_geom_ids, 2])
        fell = min_h < self._config.fall_height
        upvector = self._sensor(data, "upvector", 3)
        tipped = upvector[2] < self._config.min_upvector_z
        return jp.logical_or(fell, tipped)

    # ------------------------------------------------------------------
    # 보상 (튜닝 순서 1, 3, 4, 5 / limits)
    # ------------------------------------------------------------------
    def _reward(self, data: mjx.Data, action: jax.Array, info):
        s = self._config.scales
        local_linvel = self._sensor(data, "local_linvel", 3)

        # 1. 과제 보상: 전진 + 생존
        forward = jp.minimum(local_linvel[0], self._config.forward_vel_target)
        r_forward = s.forward * forward
        r_survival = s.survival

        # 3. 큰 정규화: 수직 속도 / 롤·피치 / 발-발 충돌
        r_lin_vel_z = s.lin_vel_z * jp.square(local_linvel[2])
        upvector = self._sensor(data, "upvector", 3)
        r_orientation = s.orientation * jp.sum(jp.square(upvector[:2]))
        foot_collision = self._foot_collision(data)
        r_foot_collision = s.foot_collision * foot_collision

        # 4. 토크
        r_torque = s.torque * jp.sum(jp.square(data.actuator_force))

        # 5. action rate -> action acc
        action_rate = jp.sum(jp.square(action - info["last_act"]))
        action_acc = jp.sum(
            jp.square(action - 2 * info["last_act"] + info["last_last_act"])
        )
        r_action_rate = s.action_rate * action_rate
        r_action_acc = s.action_acc * action_acc

        # limits: CBF 스타일 관절한계 페널티
        r_joint_limit = s.joint_limit * self._joint_limit_penalty(data)

        total = (
            r_forward + r_survival + r_lin_vel_z + r_orientation
            + r_foot_collision + r_torque + r_action_rate + r_action_acc
            + r_joint_limit
        )
        terms = {
            "reward/forward": r_forward,
            "reward/survival": r_survival,
            "reward/lin_vel_z": r_lin_vel_z,
            "reward/orientation": r_orientation,
            "reward/foot_collision": r_foot_collision,
            "reward/torque": r_torque,
            "reward/action_rate": r_action_rate,
            "reward/action_acc": r_action_acc,
            "reward/joint_limit": r_joint_limit,
        }
        return total, terms

    def _foot_collision(self, data: mjx.Data) -> jax.Array:
        """왼발-오른발 지오메트리 쌍이 서로 얼마나 겹치는지(AABB 근사)를 벌점으로.

        scripts/measure_stage0.py에서 확인했듯, 발 외 지오메트리는 충돌이 꺼져
        있어도 발끼리는(contype=conaffinity=1) mjx가 실제 접촉을 계산한다 —
        여기서는 그 실제 접촉(ncon)을 발-발 쌍만 걸러서 쓴다.
        """
        l_ids = self._l_foot_ids
        r_ids = self._r_foot_ids

        def is_foot_pair(g1, g2):
            l1 = jp.any(l_ids == g1) & jp.any(r_ids == g2)
            l2 = jp.any(l_ids == g2) & jp.any(r_ids == g1)
            return l1 | l2

        pair_hit = jax.vmap(is_foot_pair)(data.contact.geom[:, 0], data.contact.geom[:, 1])
        in_contact = data.contact.dist < 0
        return jp.sum(jp.where(pair_hit & in_contact, 1.0, 0.0))

    def _joint_limit_penalty(self, data: mjx.Data) -> jax.Array:
        """CBF 형태: 한계에서 여유 q_m 안으로 들어오면, 그 방향으로 더 파고드는
        속도에 비례해 벌점을 준다 (§보상 구조 limits, q_m=0.1rad, gamma_q=20)."""
        q = data.qpos[7:]
        qd = data.qvel[6:]
        q_m = self._config.joint_limit_margin
        gamma = self._config.joint_limit_gamma

        dist_lo = q - self._joint_lo
        dist_hi = self._joint_hi - q

        into_lo = jp.clip(q_m - dist_lo, 0.0, None) * jp.clip(-qd, 0.0, None)
        into_hi = jp.clip(q_m - dist_hi, 0.0, None) * jp.clip(qd, 0.0, None)
        return gamma * jp.sum(into_lo + into_hi)
