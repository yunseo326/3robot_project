"""학습된 SB3 정책을 실제로 굴려서 영상으로 저장한다.

사용법:
  python scripts/render_policy.py --model logs/checkpoints/stage1_local/head30_B/final_model.zip \\
      --xml models/sweep_4b/head30_B.xml --out logs/result/stage1/head30_B_rollout.mp4
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mediapy as media
from stable_baselines3 import PPO

from envs.biped_rl_gym import BipedRLGym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = PPO.load(args.model)
    env = BipedRLGym(args.xml)

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    render_every = max(int(round((1.0 / args.fps) / env.model.opt.timestep / env.n_substeps)), 1)

    torso_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance = env.model.stat.extent * 2.2  # 걷는 동안 몸 전체가 프레임에 들어오게 더 멀리
    cam.azimuth = 90
    cam.elevation = -20

    obs, _ = env.reset(seed=0)
    frames = []
    n_control_steps = int(args.seconds / (env.n_substeps * env.model.opt.timestep))
    for i in range(n_control_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if i % render_every == 0:
            cam.lookat = env.data.xpos[torso_bid].copy()  # 로봇을 계속 따라가는 트래킹 카메라
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
        if terminated or truncated:
            print(f"episode ended at step {i} (terminated={terminated}, truncated={truncated})")
            break

    media.write_video(args.out, frames, fps=args.fps)
    print(f"saved {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
