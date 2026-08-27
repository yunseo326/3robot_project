"""레퍼런스 npz(qpos)를 물리 계산 없이 그대로 재생해 영상으로 저장한다.

check_feasibility.py가 아직 없어 자기충돌/한계 검증 전이거나, 검증에서 걸린
레퍼런스라도 "그 좌표로 실제 이동하는지" 자체는 확인하고 싶을 때 쓴다.
mj_step(물리 시뮬레이션)을 전혀 안 돌리고 매 프레임 mj_forward(순수 정기구학)만
호출하므로 자기충돌·관절한계 위반이 있어도 발산하거나 죽지 않고 그대로 재생된다
— 대신 물리적으로 이 동작이 안전한지는 이 영상만으로 알 수 없다(별도 검증 필요).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import mediapy as media
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--xml", default="models/character.xml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ctrl-dt", type=float, default=0.02,
                         help="레퍼런스 프레임 간격(초). biped_mimic_gym.CTRL_DT와 일치")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    ref = np.load(args.reference)
    qpos = ref["qpos"]
    n = qpos.shape[0]

    renderer = mujoco.Renderer(model, height=480, width=640)
    torso_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "body_link")
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = model.stat.extent * 2.2
    cam.azimuth = 90
    cam.elevation = -20

    render_every = max(int(round((1.0 / args.fps) / args.ctrl_dt)), 1)

    frames = []
    for t in range(n):
        data.qpos[:] = qpos[t]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)  # 물리 스텝 없음 — 순수 정기구학
        if t % render_every == 0:
            cam.lookat = data.xpos[torso_bid].copy()
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    media.write_video(args.out, frames, fps=args.fps)
    print(f"saved {len(frames)} frames ({n} qpos frames, every {render_every}th rendered) -> {args.out}")


if __name__ == "__main__":
    main()
