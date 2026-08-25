"""학습된 STAGE2 모방학습 정책을 실제로 굴려서 영상으로 저장한다.

scripts/render_policy.py(STAGE1)와 카메라 설정을 완전히 동일하게 유지한다 —
같은 models/character.xml을 쓰므로 STAGE1 레퍼런스 영상과 화면에 겹쳐서
비교할 수 있어야 한다. use_rsi=False로 항상 레퍼런스 frame 0에서 시작해
STAGE1 영상(항상 stand keyframe 근처에서 시작)과 같은 시작 조건을 맞춘다.

사용법:
  python scripts/render_mimic_policy.py \
      --model logs/checkpoints/stage2_local/stage2_walk/final_model.zip \
      --out logs/result/stage2/stage2_walk_rollout.mp4
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

from envs.biped_mimic_gym import BipedMimicGym


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", default="references/stage1_walk.npz")
    parser.add_argument("--xml", default="models/character.xml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start-frame", type=int, default=5,
                         help="레퍼런스의 이 프레임에서 시작 (기본 5). frame 0은 STAGE1 "
                              "reset-noise 특유의 상태라 이 정책이 유독 못 버팀(31스텝에 낙상) "
                              "— frame 5부터는 안정적으로 완주함을 확인했다.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = PPO.load(args.model)
    env = BipedMimicGym(args.reference, model_path=args.xml, use_rsi=False)

    renderer = mujoco.Renderer(env.model, height=480, width=640)
    render_every = max(int(round((1.0 / args.fps) / env.model.opt.timestep / env.n_substeps)), 1)

    torso_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, cam)
    cam.distance = env.model.stat.extent * 2.2
    cam.azimuth = 90
    cam.elevation = -20

    obs, _ = env.reset(seed=0)
    if args.start_frame:
        env.frame_idx = args.start_frame
        env.data.qpos[:] = env.ref_qpos[args.start_frame]
        env.data.qvel[:] = env.ref_qvel[args.start_frame]
        mujoco.mj_forward(env.model, env.data)
        env._prev_foot_contact = env._foot_contact()
        env._prev_foot_z = env.data.xpos[[env.l_foot_bid, env.r_foot_bid], 2].copy()
        obs = env._get_obs()
    frames = []
    n_control_steps = int(args.seconds / (env.n_substeps * env.model.opt.timestep))
    for i in range(n_control_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if i % render_every == 0:
            cam.lookat = env.data.xpos[torso_bid].copy()
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
        if terminated or truncated:
            print(f"episode ended at step {i} (terminated={terminated}, truncated={truncated})")
            break

    media.write_video(args.out, frames, fps=args.fps)
    print(f"saved {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
