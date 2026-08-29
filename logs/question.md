# question.md — 사용자 답변이 필요한 것

## 요약
- **무릎 굽힘 축 의도 확인**: `rotation_value_test(_ver2).csv`에서 무릎(shin이 thigh 대비 상대회전하는 방향)이 로봇의 실제 무릎축(Y, 앞뒤/sagittal)이 아니라 X축(좌우) 방향으로 나온다. 진짜로 hip_roll류(다리를 옆으로 벌리는) 동작을 테스트한 것인지, 원래 sagittal 무릎을 의도했는데 Blender 쪽에서 축이 잘못 잡힌 것인지 확인 필요.

---

## 본문

**날짜**: 2026-08-29
**관련**: `blender/export_bones_with_rotation.py`, `rotation_value_test_ver2.csv`, 메모리 `project-blender-npz-pipeline`

`rotation_value_test.csv`(quaternion 없는 버전)에서 다리 관절각을 추출했더니, 초반 프레임(t=0~25 부근)에서 hip_yaw가 -180°에 가까운 값으로 나왔다. round-trip 검증(계산한 qpos를 MuJoCo에 넣고 되읽어 원본과 대조)은 0.0000° 오차로 완전히 통과했으므로 코드 버그는 아니다 — 문제는 이 다리 동작 자체가 우리 로봇의 실제 무릎 관절(Y축 1개짜리 힌지)로는 표현 불가능한 방향으로 굽혀지고 있다는 것.

`export_bones_with_rotation.py`로 quaternion까지 받아서(`rotation_value_test_ver2.csv`) 직접 확인한 결과:
- `thigh.R`의 rest(t=0) 자세에서 로컬 X축은 world X축과 정확히 일치한다 — 이건 우리 로봇의 **hip_roll 축(axis="1 0 0")과 같은 방향**이다.
- `thigh.inv() @ shin`으로 계산한 "진짜 무릎 상대회전"이 초반 프레임에서 axis ≈ (-1,0,0), 즉 **로컬 X축 회전**이었다. 로봇 무릎(axis="0 1 0", 로컬 Y=Blender 로컬 Z 방향에 대응)이 아니다.

**확인하고 싶은 것**: 이 테스트에서 실제로 Blender에서 thigh/shin에 넣은 회전이 어느 축·어느 값이었는지(가능하면 Blender 쪽 rotation_euler/rotation_quaternion 원본 기록이나 스크립트를 알려주시면 가장 확실합니다). 만약 진짜 목적이 hip_roll(다리 옆으로 벌리기) 테스트였다면 지금 결과가 맞는 것이고, sagittal 무릎(앞뒤 굽힘)을 테스트하려던 거였다면 Blender 쪽 축 설정을 다시 봐야 합니다.
