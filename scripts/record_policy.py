"""STAGE 2 준비: STAGE 1(case1_short)에서 학습된 정책을 실행해 매 프레임 상태를
녹화한다 → references/stage1_walk.npz.

STAGE 1 결과는 물리 시뮬레이터 안에서 이미 검증된 동작(안 넘어지고 걸음)이므로,
이 npz를 레퍼런스로 쓰면 "현실성" 변수가 제거된 상태로 모방학습 파이프라인만
검증할 수 있다 (2robot_project/scripts/record_policy.py와 동일한 취지, 포팅).

주의: case1_short의 최종 episode_length는 ~222~296/1000으로 최대치에 못 미친다
(docs/assumptions.md §13). 2robot처럼 "min-length 이상인 롤아웃을 찾을 때까지
재시도"하면 못 찾을 수 있으므로, 여러 시드 중 **가장 오래 버틴 롤아웃**을
채택하는 방식으로 바꿨다 — min-length는 경고 기준일 뿐, 하드 조건이 아니다.

foot_contact도 함께 저장한다(2robot에는 없던 필드) — STAGE2 imitation 보상의
발 접촉 일치 항(𝟙[c=ĉ])에 쓴다. envs/biped_rl_gym.py의 l_foot_ids/r_foot_ids와
같은 방식으로 발-바닥(geom id 0) 접촉 쌍을 매 스텝 판정한다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import numpy as np
from stable_baselines3 import PPO

from envs.biped_rl_gym import BipedRLGym

FLOOR_GEOM_ID = 0  # models/character.xml 첫 geom이 floor


def _foot_contact(env):
    """(l_contact, r_contact) — 발-바닥 접촉 이진값."""
    l_c = r_c = 0
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        if c.dist >= 0:
            continue
        g1, g2 = c.geom1, c.geom2
        other = g2 if g1 == FLOOR_GEOM_ID else (g1 if g2 == FLOOR_GEOM_ID else None)
        if other is None:
            continue
        if other in env.l_foot_ids:
            l_c = 1
        elif other in env.r_foot_ids:
            r_c = 1
    return l_c, r_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                         default="logs/checkpoints/stage1_local/case1_short/final_model.zip")
    parser.add_argument("--xml", type=str, default="models/character.xml")
    parser.add_argument("--out", type=str, default="references/stage1_walk.npz")
    parser.add_argument("--min-length", type=int, default=150,
                         help="이 길이 미만이면 경고만 출력(하드 실패 아님) — case1_short가 "
                              "최대치(1000)에 못 미치는 정책이라 2robot과 달리 강제 안 함")
    parser.add_argument("--tries", type=int, default=20)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = BipedRLGym(args.xml)

    best = None  # (length, qpos, qvel, foot_contact, seed)
    for seed in range(args.tries):
        obs, _ = env.reset(seed=seed)
        qpos_list = [env.data.qpos.copy()]
        qvel_list = [env.data.qvel.copy()]
        foot_list = [_foot_contact(env)]
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            qpos_list.append(env.data.qpos.copy())
            qvel_list.append(env.data.qvel.copy())
            foot_list.append(_foot_contact(env))
        length = len(qpos_list)
        print(f"seed={seed} length={length} terminated={terminated} truncated={truncated}")
        if best is None or length > best[0]:
            best = (length, np.array(qpos_list), np.array(qvel_list),
                    np.array(foot_list, dtype=np.int8), seed)

    length, chosen_qpos, chosen_qvel, chosen_foot, seed = best
    if length < args.min_length:
        print(f"경고: 최장 롤아웃(seed={seed})도 length={length} < min_length={args.min_length}. "
              f"그래도 이걸로 저장한다 — STAGE1 정책이 원래 이 수준(docs/assumptions.md §13).")
    else:
        print(f"-> seed={seed} 채택 (length={length})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, qpos=chosen_qpos, qvel=chosen_qvel, foot_contact=chosen_foot)
    print(f"저장 완료: {args.out} (qpos {chosen_qpos.shape}, qvel {chosen_qvel.shape}, "
          f"foot_contact {chosen_foot.shape})")

    env.close()


if __name__ == "__main__":
    main()
