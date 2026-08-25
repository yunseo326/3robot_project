"""STAGE 0 — Robotis OP3를 mujoco_playground(MJX)에서 무수정으로 학습.

Colab 세션 위에서 실행한다: `colab exec -s <session> -f scripts/colab_launch_stage0.py`
(또는 이 파일 내용을 그대로 `colab exec -s <session>`의 stdin으로 넣어도 동일하다).

MJCF·보상 함수는 건드리지 않는다 — mujoco_playground가 제공하는 공식 학습 스크립트
(`learning.train_jax_ppo`)와 Op3 공식 프리셋(`locomotion_params.brax_ppo_config`)을
그대로 쓰고, 체크포인트 주기·총 스텝 수 같은 운영 파라미터만 CLI 오버라이드한다.

체크포인트는 Colab VM 로컬 디스크(/content)에 쓴다. Drive는 쓰지 않는다 — 세션마다
Drive 마운트 브라우저 재승인이 필요해 번거롭다는 사용자 판단(2026-08-10)에 따른 것이다.
대신 학습 도중 주기적으로 `colab download`로 체크포인트를 로컬 머신
(logs/checkpoints/stage0/op3_stock/)으로 직접 내려받아 세션 유실에 대비한다.
"""

import os
import subprocess

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

LOGDIR = "/content/stage0_op3/checkpoints"
LOG_FILE = f"{LOGDIR}/train_stdout.log"
os.makedirs(LOGDIR, exist_ok=True)

cmd = [
    "python", "-u", "-m", "learning.train_jax_ppo",
    "--env_name=Op3Joystick",
    "--impl=jax",
    f"--logdir={LOGDIR}",
    "--num_timesteps=20000000",
    "--num_evals=20",
    "--num_videos=3",
    "--use_tb=True",
]

with open(LOG_FILE, "w") as f:
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach so `colab exec` can return immediately
    )

print("launched pid:", proc.pid)
print("log file:", LOG_FILE)
print("logdir:", LOGDIR)
