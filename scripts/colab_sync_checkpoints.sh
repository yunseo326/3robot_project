#!/usr/bin/env bash
# VM(Colab) 로컬 디스크의 체크포인트를 이 컴퓨터로 내려받는다.
# Drive를 쓰지 않기로 한 결정(2026-08-10, 사용자 판단)에 따라, 체크포인트는 학습 중
# VM에만 있다가 이 스크립트를 주기적으로 실행해야 로컬로 넘어온다 — 세션이 죽으면
# 그 사이 못 받은 분량은 사라진다.
#
# 사용법:
#   scripts/colab_sync_checkpoints.sh <session> <vm_checkpoints_dir> <local_dest_dir>
# 예:
#   scripts/colab_sync_checkpoints.sh stage0_op3 \
#       /content/stage0_op3/checkpoints/Op3Joystick-20260810-033042/checkpoints \
#       logs/checkpoints/stage0/op3_stock

set -euo pipefail

SESSION="$1"
VM_DIR="$2"
LOCAL_DIR="$3"
TMP_TAR="/content/_ckpt_sync_$(date +%s).tar.gz"
LOCAL_TMP="$(mktemp -u).tar.gz"

mkdir -p "$LOCAL_DIR"

colab exec -s "$SESSION" --timeout 60 <<PYEOF
import subprocess
r = subprocess.run(["bash", "-c", "cd '$VM_DIR' && tar -czf '$TMP_TAR' ."], capture_output=True, text=True)
print(r.returncode, r.stdout, r.stderr)
PYEOF

colab download -s "$SESSION" "$TMP_TAR" "$LOCAL_TMP"
tar -xzf "$LOCAL_TMP" -C "$LOCAL_DIR"
rm -f "$LOCAL_TMP"

colab exec -s "$SESSION" --timeout 30 <<PYEOF
import os
os.remove("$TMP_TAR")
PYEOF

echo "synced: $VM_DIR -> $LOCAL_DIR"
