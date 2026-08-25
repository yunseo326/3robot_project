"""STAGE 1 학습 스크립트 — 1-A1(발목 유무)뿐 아니라 4-B/4-C 그리드 스윕에도
그대로 쓴다. 어떤 MJCF를 학습하느냐만 --model_path(또는 --candidate 단축)로
바뀌고, 나머지 하이퍼파라미터는 TRAIN_KWARGS에 고정돼 있다 — CLAUDE.md 작업
규칙 "한 번에 한 변수만 바꾼다"를 지키려면 이 스크립트를 절대 조건별로
따로 고쳐 쓰면 안 된다.

envs/biped_rl.py의 BipedRL을 mujoco_playground의 브랙스 래퍼로 감싸
brax.training.agents.ppo.train에 직접 넘긴다 (기성 레지스트리를 쓰지 않는
커스텀 로봇이므로 STAGE 0의 learning.train_jax_ppo를 그대로 재사용할 수
없다 — 대신 그 스크립트와 같은 패턴을 따른다).

사용법 (Colab VM 위에서):
  python scripts/train_stage1.py --candidate A --logdir /content/stage1/candidate_A
  python scripts/train_stage1.py --model_path models/sweep_4b/head30_A.xml \\
      --label head30_A --logdir /content/stage1/sweep_4b/head30_A
"""

import argparse
import functools
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from envs.biped_rl import BipedRL

try:
    import tensorboardX
except ImportError:
    tensorboardX = None


# 두 후보에 절대 동일하게 적용되는 하이퍼파라미터. candidate만 바꿔서 두 번 돌린다.
TRAIN_KWARGS = dict(
    num_timesteps=10_000_000,
    num_envs=8192,
    episode_length=1000,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=0.01,
    num_evals=10,
    batch_size=256,
    normalize_observations=True,
    reward_scaling=1.0,
    seed=1,
    log_training_metrics=True,
    training_metrics_steps=1_000_000,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["A", "B"], default=None,
                         help="1-A1 단축: models/candidate_{A,B}.xml")
    parser.add_argument("--model_path", default=None,
                         help="임의 MJCF 경로 (4-B/4-C 스윕용)")
    parser.add_argument("--label", default=None,
                         help="로그에 찍을 이름. 생략하면 --candidate 또는 파일명에서 뽑는다")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--load_checkpoint_path", default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.model_path:
        xml_path = args.model_path
        label = args.label or os.path.splitext(os.path.basename(xml_path))[0]
    elif args.candidate:
        xml_path = os.path.join(root, "models", f"candidate_{args.candidate}.xml")
        label = args.label or args.candidate
    else:
        parser.error("--candidate 또는 --model_path 중 하나는 있어야 함")

    env = BipedRL(xml_path)

    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    ckpt_path = os.path.join(logdir, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    with open(os.path.join(ckpt_path, "config.json"), "w") as f:
        json.dump({"label": label, "xml_path": xml_path, **TRAIN_KWARGS}, f, indent=2, default=str)

    writer = tensorboardX.SummaryWriter(logdir) if tensorboardX else None

    times = [time.monotonic()]

    def progress(num_steps, metrics):
        times.append(time.monotonic())
        if writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(key, value, num_steps)
            writer.flush()
        ep_reward = metrics.get("eval/episode_reward", float("nan"))
        ep_len = metrics.get("eval/avg_episode_length", float("nan"))
        print(f"[{label}] {num_steps}: "
              f"reward={ep_reward:.3f} episode_length={ep_len:.1f}", flush=True)

    train_fn = functools.partial(
        ppo.train,
        **TRAIN_KWARGS,
        network_factory=ppo_networks.make_ppo_networks,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        save_checkpoint_path=ckpt_path,
        restore_checkpoint_path=args.load_checkpoint_path,
        progress_fn=progress,
    )

    print(f"=== STAGE 1 — {label} ({xml_path}) ===", flush=True)
    print(f"logdir: {logdir}", flush=True)
    make_inference_fn, params, _ = train_fn(environment=env)

    print(f"Done training. total wall time: {times[-1] - times[0]:.1f}s", flush=True)
    print("TRAIN_STAGE1_OK", flush=True)


if __name__ == "__main__":
    main()
