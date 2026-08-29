"""STAGE 2 학습 — 로컬 CPU (envs/biped_mimic_gym.py, SB3 PPO).

STAGE1(scripts/train_stage1_local.py)과 최대한 동일한 하이퍼파라미터를 쓴다 —
동일 조건 비교 원칙(CLAUDE.md 작업 규칙 1) 및 2robot_project/scripts/train_stage2.py
구조를 조합했다.

통과 기준(CLAUDE.md STAGE2): reward가 정체 없이 상승한다.
실패 시: 레퍼런스를 의심하지 않는다. RSI/ET 설정, 보상 가중치, 관측 공간 매핑을
순서대로 점검한다.

사용법:
  python scripts/train_stage2_local.py --label stage2_walk_smoke --timesteps 500000
  python scripts/train_stage2_local.py --label stage2_walk --timesteps 10000000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from envs.biped_mimic_gym import ACTION_SCALE, make_biped_mimic_env

SAVE_FREQ_STEPS = 200_000


class ComponentLoggingCallback(BaseCallback):
    """envs/biped_mimic_gym.py의 RewardComponentWrapper가 채우는
    info["episode_components"]를 항목별로 쪼개 로깅한다 (train_stage1_local.py와 동일 패턴)."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            comp = info.get("episode_components")
            if comp is None:
                continue
            comp = dict(comp)
            length = comp.pop("length")
            self.logger.record_mean("episode_length", length)
            for k, v in comp.items():
                short = k.replace("reward_", "")
                self.logger.record_mean(f"reward/{short}", v)
        return True


class TimeBudgetCallback(BaseCallback):
    """STAGE1과 동일한 발열 방지 페이싱 — budget_seconds가 지나면 이번 learn() 호출을 끊는다."""

    def __init__(self, budget_seconds):
        super().__init__()
        self.budget_seconds = budget_seconds
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.monotonic()

    def _on_step(self) -> bool:
        return (time.monotonic() - self.start_time) < self.budget_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference", default="references/stage1_walk.npz")
    parser.add_argument("--model-path", default="models/character.xml")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--min-episode-len", type=int, default=30)
    parser.add_argument("--no-rsi", action="store_true", help="RSI를 끄고 항상 frame 0에서 시작")
    parser.add_argument("--fall-geom-names", default=None,
                         help="쉼표로 구분한 낙상판정 geom 이름 목록. 생략시 character.xml 기본값 사용"
                              "(다른 모델 XML을 쓸 때 필요, 예: _viz_arms_temp.xml)")
    parser.add_argument("--action-scale", type=float, default=None,
                         help="ctrl = default_pose + action*action_scale의 스케일(rad). "
                              "생략시 envs/biped_mimic_gym.ACTION_SCALE(0.3rad) 사용. "
                              "레퍼런스 관절각이 default_pose 대비 이 값보다 크게 벗어나면 "
                              "action=±1이어도 물리적으로 못 따라간다(예: 팔 어깨 스윙).")
    parser.add_argument("--arm-action-scale", type=float, default=None,
                         help="어깨/팔꿈치 관절에만 별도로 줄 action_scale(rad). 생략시 "
                              "--action-scale과 동일(다리와 같은 스케일).")
    parser.add_argument("--chunk-minutes", type=float, default=20.0)
    parser.add_argument("--rest-minutes", type=float, default=7.0)
    parser.add_argument("--resume-from", default=None,
                         help="이 체크포인트(.zip)를 불러와 이어서 학습한다(num_timesteps 포함 "
                              "그대로 복원). --timesteps는 '이어받은 이후'가 아니라 '누적 총합' "
                              "기준이다 — 예: 3M에서 --resume-from + --timesteps 10000000이면 "
                              "7M을 더 돈다.")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    reference_path = os.path.abspath(os.path.join(root, args.reference))
    model_path = os.path.abspath(os.path.join(root, args.model_path))
    save_dir = os.path.join(root, "logs", "checkpoints", "stage2_local", args.label)
    tb_dir = os.path.join(root, "logs", "tb", "stage2_local")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    fall_geom_names = args.fall_geom_names.split(",") if args.fall_geom_names else None
    action_scale = args.action_scale if args.action_scale is not None else ACTION_SCALE
    env_factory = make_biped_mimic_env(
        reference_path=reference_path,
        model_path=model_path,
        min_episode_len=args.min_episode_len,
        use_rsi=not args.no_rsi,
        fall_geom_names=fall_geom_names,
        action_scale=action_scale,
        arm_action_scale=args.arm_action_scale,
    )
    vec_env = make_vec_env(env_factory, n_envs=args.n_envs)
    vec_env = VecMonitor(vec_env)

    if args.resume_from:
        resume_path = os.path.abspath(os.path.join(root, args.resume_from))
        model = PPO.load(resume_path, env=vec_env, tensorboard_log=tb_dir)
        print(f"resumed from {resume_path} at {model.num_timesteps} steps", flush=True)
    else:
        # STAGE1(train_stage1_local.py)과 동일한 하이퍼파라미터 — 동일 조건 원칙.
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            gamma=0.97,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.3,
            n_epochs=4,
            batch_size=256,
            n_steps=2048,
            verbose=1,
            tensorboard_log=tb_dir,
            seed=1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ_STEPS // args.n_envs, 1),
        save_path=save_dir,
        name_prefix=f"ppo_mimic_{args.label}",
    )
    component_callback = ComponentLoggingCallback()

    print(f"=== STAGE 2 (local) — {args.label} ({model_path}, ref={reference_path}) ===", flush=True)
    print(f"RSI={not args.no_rsi}, action_scale={action_scale}, "
          f"arm_action_scale={args.arm_action_scale}, "
          f"pacing: {args.chunk_minutes}min work / {args.rest_minutes}min rest", flush=True)

    chunk_seconds = args.chunk_minutes * 60
    rest_seconds = args.rest_minutes * 60
    chunk_i = 0
    while model.num_timesteps < args.timesteps:
        chunk_i += 1
        remaining = args.timesteps - model.num_timesteps
        time_cb = TimeBudgetCallback(chunk_seconds)
        print(f"--- chunk {chunk_i}: total so far {model.num_timesteps}/{args.timesteps} ---", flush=True)
        model.learn(
            total_timesteps=remaining,
            callback=[checkpoint_callback, component_callback, time_cb],
            tb_log_name=args.label,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        if model.num_timesteps < args.timesteps:
            print(f"--- chunk {chunk_i} done at {model.num_timesteps} steps, "
                  f"resting {args.rest_minutes}min ---", flush=True)
            time.sleep(rest_seconds)

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"Done training. saved: {final_path}", flush=True)
    print("TRAIN_STAGE2_LOCAL_OK", flush=True)


if __name__ == "__main__":
    main()
