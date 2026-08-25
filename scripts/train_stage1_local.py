"""STAGE 1 학습 — 로컬 CPU 버전 (Colab GPU가 막혔을 때의 대체 경로).

2robot_project scripts/train_stage1.py와 완전히 같은 패턴(SB3 PPO, 항목별
reward 로깅, 20만 스텝 체크포인트 주기). envs/biped_rl_gym.py가 GPU용
envs/biped_rl.py와 보상·종료조건을 동일하게 유지하므로, 어느 백엔드로
돌렸든 결과를 같은 기준으로 비교할 수 있다.

scripts/train_stage1.py(GPU, brax ppo.train)의 TRAIN_KWARGS와 최대한 맞춘
하이퍼파라미터:
  - num_timesteps=10,000,000 (동일)
  - learning_rate=3e-4, discounting/gamma=0.97, entropy_cost=0.01, seed=1 (동일)
  - gae_lambda=0.95 (양쪽 다 기본값, 동일)
  - clip_range=0.3 (brax ppo.train 기본값에 맞춤 — SB3 기본값 0.2가 아님)
  - n_epochs=4 (brax의 num_updates_per_batch=4에 대응)
  - batch_size=256 (brax의 minibatch 크기와 동일한 숫자를 씀)

**하드웨어 제약으로 못 맞추는 것**: GPU는 num_envs=8192(병렬 시뮬레이션)로
반복당 8192*20=163,840 스텝을 모으지만, 이 컴퓨터는 CPU 8코어라 n_envs=8이
현실적 한계다. n_steps를 키워 반복당 수집량(16,384)을 최대한 늘렸지만
GPU의 1/10 수준이다 — 즉 "같은 총 스텝 수·같은 보상·같은 옵티마이저
설정"까지는 동일하지만, "한 번의 그래디언트 업데이트가 보는 데이터 다양성"은
GPU 쪽이 훨씬 크다. 이 차이는 8개 로컬 결과끼리 비교하는 데는 영향이 없다
(전부 같은 조건으로 로컬에서 도니까) — GPU로 돌린 1-A1 결과와 직접 수치
비교할 때만 참고할 것.

사용법:
  python scripts/train_stage1_local.py --model_path models/sweep_4b/head30_A.xml \\
      --label head30_A --timesteps 10000000
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

from envs.biped_rl_gym import make_biped_env

SAVE_FREQ_STEPS = 200_000


class ComponentLoggingCallback(BaseCallback):
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
    """컴퓨터 과열 방지 (2026-08-12 사용자 지시): 한 번에 너무 오래 붙어서
    돌리지 않고, 벽시계 기준 budget_seconds가 지나면 이번 learn() 호출을
    끊는다. 스텝 수가 아니라 실제 경과 시간 기준이라 fps 변동과 무관하게
    "30분 일하고 쉰다"를 정확히 지킨다."""

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
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--chunk-minutes", type=float, default=20.0,
                         help="한 번에 이만큼(분) 학습하고 쉰다 (과열 방지)")
    parser.add_argument("--rest-minutes", type=float, default=7.0,
                         help="chunk 사이 쉬는 시간(분)")
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    xml_path = os.path.abspath(args.model_path)
    save_dir = os.path.join(root, "logs", "checkpoints", "stage1_local", args.label)
    tb_dir = os.path.join(root, "logs", "tb", "stage1_local")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    vec_env = make_vec_env(make_biped_env(xml_path), n_envs=args.n_envs)
    vec_env = VecMonitor(vec_env)

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
        n_steps=2048,  # n_envs=8 기준 반복당 16,384 스텝 수집 (GPU 163,840의 1/10)
        verbose=1,
        tensorboard_log=tb_dir,
        seed=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ_STEPS // args.n_envs, 1),
        save_path=save_dir,
        name_prefix=f"ppo_biped_{args.label}",
    )
    component_callback = ComponentLoggingCallback()

    print(f"=== STAGE 1 (local) — {args.label} ({xml_path}) ===", flush=True)
    print(f"pacing: {args.chunk_minutes}min work / {args.rest_minutes}min rest "
          f"(과열 방지, 2026-08-12 지시)", flush=True)

    chunk_seconds = args.chunk_minutes * 60
    rest_seconds = args.rest_minutes * 60
    chunk_i = 0
    while model.num_timesteps < args.timesteps:
        chunk_i += 1
        remaining = args.timesteps - model.num_timesteps
        time_cb = TimeBudgetCallback(chunk_seconds)
        print(f"--- chunk {chunk_i}: total so far {model.num_timesteps}/{args.timesteps} ---",
              flush=True)
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
    print("TRAIN_STAGE1_LOCAL_OK", flush=True)


if __name__ == "__main__":
    main()
