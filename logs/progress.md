# progress.md — 진행사항 + STAGE 0 측정값

## 현재 단계

```
현재 STAGE : 2 (통과, 2026-08-23)
다음 목표   : 사용자 판단 대기 — STAGE 3(목 정책입력화 + 팔·LCD 분리) 착수 여부
차단 요인   : 없음
```

**팔 모방학습 떨림 원인 진단 + 재검증 (2026-08-28, STAGE 진행상황과 무관)**: 사용자가 "이전 팔 모방학습(2026-08-28 앞 항목의 `handmove_arm_smoke`) 결과 로봇이 진동하며 제대로 못 움직였다"고 보고, 쿼터니언이 포함된 새 CSV(`npz_handmove_test_ver2_with_r.csv`, `upper_arm.*/forearm.*` 등 본마다 world quaternion 4컬럼 추가)를 제공해 3가지 확인을 요청했다.

1. **팔 관절 존재 확인(PASS)**: `models/_viz_arms_temp.xml`에서 팔 hinge 관절 8개(양팔 각 shoulder_yaw/roll/pitch+elbow) 정상 확인.
2. **중력 안정성 확인(PASS)**: stand keyframe에서 제어입력을 그 자세로 고정하고 8초 물리 롤아웃 — 위치 이탈 0.38mm, 기울어짐 0.0000°로 쓰러지거나 흔들리지 않음.
3. **떨림 원인 규명**: 기존 `references/handmove_test.npz`(v1, `convert_handmove_to_npz.py`)의 `r_elbow` 각도를 프레임별로 분석하니 실제 값은 거의 0인데도 413스텝 중 104번 부호가 뒤집히는 미세 진동이 있었다. 원인: 이 테스트 동작은 팔꿈치를 편 채(elbow≈0) 어깨만 움직이는데, v1의 팔꿈치 굽힘축 추정이 `cross(위팔방향, 아래팔방향)`(다리의 무릎축 추정 로직을 그대로 재사용)이라 두 벡터가 거의 평행해지는 이 동작에서 축 추정이 구조적으로 불안정해진다 — 정확히는 hip_yaw와 같은 "자기축 self-twist는 방향벡터만으로 복원 불가" 문제의 팔 버전(다리 회전값 검증 실험, 위 2026-08-29 항목과 동일 계열의 근본 원인).

   **해결**: `scripts/convert_handmove_to_npz_v2.py`(신규) — 쿼터니언이 있으면 외적 추정이 필요 없다. 어깨는 `R_joint(t) = R_bone_rootlocal(0)^T @ R_bone_rootlocal(t)`(rest 대비 순수 관절회전, `hip_euler_from_matrix` 재사용)로, 팔꿈치는 `rel(t)=upper_arm_quat(t)^-1 · forearm_quat(t)`를 rest로 보정한 뒤 `2·arccos(|w|)`로 직접 계산 — 축 추정 자체가 없어 팔이 펴진 구간에서도 안정적이다. `references/handmove_test_v2.npz` 생성 후 검증: round-trip 방향벡터 오차 0.027°(v1과 동급으로 정확), r_elbow max|diff|가 0.00984rad(v1, 실질적 진동) → 0.00000rad(v2, 사실상 완전히 평평)으로 개선.

   **스모크 재학습(30만 스텝, PYTHONIOENCODING=utf-8 필요 — cp949 콘솔에서 `—` UnicodeEncodeError 발생)**: `handmove_arm_smoke_v2` — reward -1.75→13.3으로 정체 없이 상승(v1의 -0.9→16.1과 같은 패턴, PASS).

   **남은 문제(참조 데이터 버그와 별개)**: 학습된 정책을 실제로 굴려보면(렌더링 + 액추에이터 명령 시계열 직접 비교) v1·v2 둘 다 30스텝 근처에서 낙상하고, 액추에이터 명령 자체가 여전히 들쭉날쭉하다(부호가 스텝마다 자주 뒤집힘, ctrl 범위 ±0.3 포화 빈번) — 이건 참조 데이터가 아니라 **정책이 아직 수렴 전(30만 스텝=19 PPO iteration)이라 학습 곡선의 std(≈0.8)가 아직 높은 상태**이기 때문으로 보인다(STAGE2 본학습 10M스텝에서는 이 흔들림이 실제로 해소된 전례가 있음 — 위 "본 학습(10M스텝)" 항목 참고). 즉 사용자가 본 "떨림"에는 최소 두 가지 원인이 섞여 있었다: ① 참조 데이터의 구조적 노이즈(고침) ② 스모크 수준 학습의 미수렴(스모크 테스트의 목적상 원래 기대 범위 밖 — 본학습 여부는 사용자 판단 대기).

   렌더링 노트: 이 환경에서 `MUJOCO_GL=egl`이 실패해 `MUJOCO_GL=wgl`로 대체해야 했고, ffmpeg가 PATH에 없어 `envs/3robot/Library/bin`을 PATH에 추가해야 `mediapy`가 동작했다(다음에도 필요).

**본학습 착수 + 완료 (2026-08-28→29, 사용자 승인)**: 스모크 통과 후 사용자가 본학습(10M스텝) 진행을 요청, 기존 20분/7분 발열 페이싱 유지 지시. 실측 fps(≈1875)로 예상 소요시간 안내(순수 연산 ~89분, 페이싱 포함 ~2시간). `handmove_arm_full` 라벨로 백그라운드 detached 실행(PID: `logs/train_handmove_full.pid`, 로그: `logs/train_handmove_full.log`) — 세션이 끝나도 프로세스는 독립적으로 계속 진행되어 실제로 10M스텝까지 정상 완주(TRAIN_STAGE2_LOCAL_OK). 청크 로그 기준(20분 작업×4회완료+7분 휴식×4회+마지막 부분청크 ~19분) 총 소요 약 2시간 — 사전 예상(~2시간)과 일치.

ep_rew_mean 추이(0%→25%→50%→75%→100%): -1.75 → 56.5 → 67.9 → 71.6 → **74.0**. 정체 없이 상승은 했지만 마지막 10%(9M→10M) 구간에서 73.8→74.0으로 사실상 평평해짐 — STAGE2 통과기준("정체 없이 상승")을 엄밀하게는 못 채운 것에 가깝다. 렌더링 확인 결과 실제로 시작 프레임과 무관하게 약 65~68스텝(≈1.3~1.4초, 레퍼런스 전체 8.3초 중 16%)에서 낙상 후 종료 — STAGE2 걷기 본학습(ep_len_mean 최종 293/1000)과 비교해 훨씬 이른 실패.

**원인 진단**: reward 항목별 시계열을 뜯어보니 `imitation_root_pos`(0.15 만점 근접), `imitation_root_ori`, `imitation_survival`, `limits_joint`(≈0, 위반 없음), `regularization`(≈-0.95, 매끈함)은 전부 50% 지점 이후 사실상 포화(최댓값 근접)됐는데, **`imitation_leg_pose`만 유일하게 0.48~0.55 근처에서 전혀 개선 없이 정체**돼 있다(참고로 이 항목의 스텝당 최댓값은 0.35인데 68스텝 누적합이 0.5 수준이면 스텝당 평균이 거의 0에 가깝다는 뜻 — "낮은 게 아니라 거의 완전히 실패"). `envs/biped_mimic_gym.py:247`의 `leg_pose_err = sum((qpos[7:]-target_qpos[7:])**2)`가 원인으로 보인다 — 이 코드는 `character.xml`(다리 8DOF만) 기준으로 짜여 "나머지 관절 전부"를 하나의 오차항으로 묶는데, `_viz_arms_temp.xml`(23차원 qpos, 다리8+팔8=16DOF)에 그대로 재사용되면서 **다리뿐 아니라 팔까지 이 하나의 "leg_pose" 항에 합쳐져 버린다.** 팔 어깨 yaw가 최대 ~77°(≈1.34rad)까지 크게 움직이는데, 이 항의 감쇠상수 `K_LEG_POSE=2.0`은 원래 다리(관절범위 대략 ±30°=0.52rad 수준)에 맞춰진 값이라 팔이 크게 움직이는 구간의 오차 제곱합이 순식간에 커져 `exp(-2.0×err)`가 거의 0으로 죽어버린다 — 팔을 크게 움직이라는 유효한 학습 신호가 사실상 없었던 것으로 보인다. CLAUDE.md가 STAGE3에서 "다리와 목의 가중치를 반드시 분리한다"고 못박은 것과 동일한 종류의 문제가, 팔 버전에서 미리 재현된 셈.

**주의 — 이 실험은 STAGE3 확정 전 임시 검증(character.xml·STAGE 상태 불변)이라 `biped_mimic_gym.py`를 이 시점에 고치지 않았다.** 고치려면(leg_pose와 arm_pose를 별도 가중치 항으로 분리) 재학습이 필요해 사용자 판단 대기로 남겨둔다 — 상세: `logs/question.md`.

렌더링/체크포인트: `logs/checkpoints/stage2_local/handmove_arm_full/final_model.zip`, `logs/result/handmove_v2/handmove_v2_full_policy_rollout.mp4`.

**leg_pose/arm_pose 분리 수정 + 재학습 착수 (2026-08-29, 사용자 승인, STAGE 진행상황과 무관)**: 위에서 진단한 `imitation_leg_pose` 항목이 다리+팔 오차를 한데 섞어 죽는 문제를 수정. `envs/biped_mimic_gym.py`에 관절 이름 기반(`"shoulder"`/`"elbow"` 포함 여부) 다리/팔 DOF 인덱스 분리 로직을 `__init__`에 추가(`_leg_qpos_idx`/`_arm_qpos_idx`/`_leg_dof_idx`/`_arm_dof_idx`) — 모델에 무관하게 동작하며, `character.xml`(STAGE2 걷기, 팔 관절 없음)에서는 arm 인덱스가 항상 빈 배열이 되어 `arm_pose`/`arm_vel` 보상이 늘 0으로 꺼진다(기존 STAGE2 결과에 영향 없음, 재확인 완료). `_viz_arms_temp.xml`에서는 다리 8DOF/팔 8DOF로 정확히 분리됨(재확인 완료).

`imitation_leg_pose`(다리 전용, `K_LEG_POSE=2.0` 그대로 유지)와 별도로 `imitation_arm_pose`(신규, `K_ARM_POSE=0.4` — 어깨 스윙 범위가 다리보다 훨씬 커서 감쇠상수를 다리의 1/5로 완화) + `imitation_arm_vel`(신규, `K_ARM_VEL=0.1`)을 추가, `IMITATION_WEIGHTS`에 `arm_pose=0.35`/`arm_vel=0.10`을 다리와 대칭으로 부여(CLAUDE.md STAGE3 "다리와 목의 가중치를 반드시 분리한다" 원칙을 팔에 적용). 재시작 직후 첫 로그에서 `imitation_arm_pose`가 즉시 0이 아닌 값(에피소드 누적 5.71, 스텝당 최댓값 0.35에 근접)을 내는 것 확인 — 이전엔 이 신호가 사실상 없었다.

`handmove_arm_full_v2` 라벨로 10M스텝 재학습 착수(기존 20분작업/7분휴식 페이싱 유지, `fall-geom-names`에 `l_upper_arm_geom`/`r_upper_arm_geom` 추가 — 팔이 몸통에 닿아도 낙상 판정되도록). PID: `logs/train_handmove_full_v2.pid`, 로그: `logs/train_handmove_full_v2.log`.

**재학습 완료 + 결과 (2026-08-29)**: `TRAIN_STAGE2_LOCAL_OK`로 10M스텝 정상 완주(청크 5회 + 마지막 부분청크, 총 소요 이전과 비슷한 약 2시간). ep_rew_mean 추이(0%→25%→50%→75%→100%): 12 → 58.6 → 125 → 347 → **331**(마지막 25%에서 347→331로 소폭 하락, PPO 탐험 노이즈 범위 — STAGE1 case1_short 때도 비슷한 흔들림이 있었고 그때도 실제 deterministic 정책은 안정적이었음). ep_len_mean: 28.6 → 46.8 → 83.9 → 217 → **206**. `imitation_leg_pose`(7.81→42.8→32.6)와 `imitation_arm_pose`(5.54→40.5→40.1)가 이제 **거의 같은 스케일로 나란히 개선** — 수정 전 arm_pose가 leg_pose에 흡수돼 죽어있던 것과 달리 둘 다 유효한 학습 신호를 받고 있음을 확인.

**렌더링 검증(4개 시작 프레임)**: start-frame 0/5/50/100/200 전부 `terminated=False, truncated=True`로 **레퍼런스 끝(413~415/415프레임)까지 완주, 한 번도 낙상하지 않음** — 수정 전(시작점 무관하게 65~68/415=16%에서 낙상)과 정반대 결과. 원인 진단(leg_pose·arm_pose 보상 혼합)이 맞았음이 실측으로 확인됨. 체크포인트: `logs/checkpoints/stage2_local/handmove_arm_full_v2/final_model.zip`, 영상: `logs/result/handmove_v2/handmove_v2_full_v2_policy_rollout.mp4`.

이 실험은 여전히 STAGE3 확정 전 임시 검증(character.xml·현재 STAGE 상태 불변)이다. 코드 수정(`envs/biped_mimic_gym.py`의 leg/arm 분리)은 이름 기반이라 STAGE2 걷기(`character.xml`, 팔 관절 없음)에는 영향 없음을 재확인 완료(수정 직후 arm_pose=0.0 확인).

**"레퍼런스와 학습결과가 다르다" 재확인 요청 → 진짜 원인 발견: action_scale 캡 (2026-08-29, 사용자 지적)**: 사용자가 `handmove_arm_full_v2` 결과 영상이 이상하다고 재검토 요청. 재검토 과정에서 두 가지를 발견했다.

1. **원래 비교 아티팩트의 카메라 문제**: 렌더 스크립트들이 STAGE1/2 걷기용으로 고정해둔 정면 카메라(`azimuth=90`)는 이 팔 동작(어깨 yaw = 수평면 회전)을 카메라 시선 방향과 거의 평행하게 만들어 스윙이 거의 안 보였다 — 레퍼런스도 정책 결과도 둘 다 "가만히 있는 것처럼" 보여서 비교 자체가 무의미했다. 측면(`azimuth=0`)으로 다시 찍으니 스윙이 뚜렷이 보였다(임시 스크립트로만 확인, `scripts/render_*.py` 원본 카메라값은 안 건드림 — STAGE1/2 걷기 레퍼런스와 카메라 규약 공유).
2. **진짜 원인**: 측면 카메라로도 정책이 팔을 거의 안 움직이는 게 뚜렷해서 실제 관절각을 직접 뽑아봤다. `handmove_arm_full_v2` 정책의 `r_shoulder_yaw`는 스텝 5 이후 **정확히 17.19°에서 고정**된 채 에피소드 끝(413스텝)까지 전혀 안 움직였다(레퍼런스는 0°→77°). 17.19°=0.3rad — `envs/biped_mimic_gym.py`의 `ctrl = default_pose + action·ACTION_SCALE`에서 `ACTION_SCALE=0.3rad`이 **전 관절 공용**이었던 게 원인. `action∈[-1,1]`이므로 어떤 관절이든 `default_pose`(어깨 yaw 기본값 0°) 대비 최대 0.3rad=17.19°까지만 명령 가능 — 정책이 action=+1로 포화됐어도(올바르게 "최대한 움직이라"고 배웠어도) 애초에 그 이상 낼 방법이 없었다. 다리는 걷기 동작 자체가 정지자세 대비 ±20~30° 안쪽이라 0.3rad로 충분했지만, 이 팔 동작(목표 77°)은 처음부터 이 상한에 걸릴 수밖에 없었다 — `K_ARM_POSE`(보상) 문제가 전혀 아니라 **행동공간(action space) 캡** 문제였다.

**수정 + 재학습 착수**: `BipedMimicGym.__init__`에 `action_scale`(스칼라, 기본값 유지)과 `arm_action_scale`(신규, 어깨/팔꿈치 관절에만 적용) 파라미터 추가 — 앞서 leg/arm 보상 분리에 쓴 관절 인덱스를 재사용해 관절별 action_scale 벡터를 구성한다. 다리는 검증된 0.3rad 그대로, 팔만 1.6rad(목표 최대 77°=1.34rad에 여유)로 분리. `character.xml`(STAGE2 걷기)은 `arm_action_scale` 미지정시 전 관절이 기존 0.3rad 그대로라 영향 없음(재확인 완료). `scripts/train_stage2_local.py`·`scripts/render_mimic_policy.py`에 `--action-scale`/`--arm-action-scale` CLI 옵션 추가(생략시 기존 동작과 동일).

`handmove_arm_full_v3` 라벨로 10M스텝 재학습 착수(동일 페이싱). 비교 아티팩트(측면 카메라 영상 + 실측 관절각 그래프): 진단 결과 정리, 사용자에게 공유함.

**사용자 지시로 3M스텝으로 축소 재실행 (2026-08-29)**: "10M도 필요없을 것 같다"는 사용자 판단에 따라 진행 중이던 10M(chunk1, ~220K스텝)을 정지하고 동일 설정(`arm_action_scale=1.6`)으로 3M스텝 재시작 — `TRAIN_STAGE2_LOCAL_OK` 정상 완료(~9분 청크 1회 + 이어서, 총 약 17분).

**결과: action_scale 수정 자체는 확인됐지만 3M은 미수렴**. 관절각 실측: r_shoulder_yaw가 더 이상 17.19°에 갇히지 않고 스텝 0~5 사이에 이미 29°→79°까지 크게 움직인다(수정이 유효함을 증명 — 이전엔 물리적으로 불가능했던 범위). 다만 **타이밍이 전혀 안 맞는다** — 레퍼런스 목표가 아직 0°~5° 근처인 스텝 0~40 구간에서 정책이 먼저 65~80°까지 팔을 크게 휘둘러버리고, 그 반동으로 균형이 무너져 **41/415스텝(10%)에서 낙상**(`terminated=True`) — v2(얼어붙은 팔, 그래서 오히려 안 넘어짐)의 206스텝보다도 짧다. reward/ep_len_mean은 3M 끝까지 정체 없이 계속 상승 중이었다(0%→100%: reward 1.14→56, ep_len 28.2→44.7, arm_pose 1.81→11.6, leg_pose와 거의 같은 속도로 동반 상승) — 즉 **학습 자체는 올바른 방향으로 가고 있으나 3M으로는 "팔을 크게 움직이면서도 안 넘어지는" 조합까지 못 배웠다.** 체크포인트: `logs/checkpoints/stage2_local/handmove_arm_full_v3/final_model.zip`, 영상: `logs/result/handmove_v2/handmove_v3_policy_rollout.mp4`(21프레임, 41스텝에서 낙상해 조기 종료).

**사용자 결정: 이어서 누적 10M까지 학습**. `scripts/train_stage2_local.py`에 `--resume-from`(체크포인트 zip을 `PPO.load(path, env=vec_env)`로 불러와 `num_timesteps` 포함 그대로 이어감) 옵션을 신규 추가 — 기존엔 매번 새 PPO를 생성해 처음부터만 돌릴 수 있었다. 3M 체크포인트를 `final_model_at_3M.zip`으로 백업(이어학습이 끝나면 `final_model.zip`이 10M 결과로 덮어써지므로) 후 동일 라벨(`handmove_arm_full_v3`)로 재개 — 로그에서 `resumed from ... at 3014664 steps` 확인, `total_timesteps=3031048`부터 재시작해 이전 3M 종료 시점 지표(ep_len_mean 44.4, ep_rew_mean 55.8)와 정확히 이어짐을 확인. `TRAIN_STAGE2_LOCAL_OK`로 10007496스텝 정상 완료.

**10M 결과: 3M 대비 개선은 있으나 완만한 정체, 낙상 시점은 크게 안 늘어남**. ep_rew_mean/ep_len_mean 추이(3M/약45%/약65%/약83%/10M): reward 57.5→61.8→67.9→**71.4**(약 8.3M 지점 피크)→**68.7**(10M, 소폭 하락), ep_len 45.8→45.6→48.6→**50.1**(피크)→**47.9**(10M). 즉 8.3M 근방에서 사실상 정체(마지막 1.7M스텝은 개선이 아니라 소폭 퇴보) — 3M 때 지적된 "정체 없이 상승"이 10M 전 구간에서는 더 이상 유지되지 않는다.

**관절각 실측 재확인**: 여전히 스텝 0~5 사이에 target이 0°~1°대인데 실제값이 50~67°까지 먼저 튀어오르는 "조기 오버슈트" 패턴이 그대로 남아있다 — 3M 때(41스텝 낙상)보다 아주 조금 늦게(46스텝) 낙상하는 정도만 개선. `upvector_z`가 스텝 20 이후(0.999→0.997→0.989→0.958→0.854→0.562) 서서히 무너지다 0.5 임계값 아래로 떨어져 `tipped` 판정으로 종료 — 팔을 조기에 크게 휘두른 반동이 누적되어 균형이 무너지는 패턴으로 보인다.

**해석**: `action_scale` 수정(팔이 물리적으로 목표까지 갈 수 있게 함)은 여전히 유효하고 확인됐지만, 이번엔 **보상이 "이르든 늦든 목표에 가까우면 보상"만 줄 뿐 "아직 움직일 때가 아닌데 미리 움직이면 감점"은 약하다**는 두 번째 문제가 드러났다. `K_ARM_POSE=0.4`가 다리(`K_LEG_POSE=2.0`) 대비 소프트하게 설정돼 있어(원래 의도: 큰 팔 스윙 범위에서 학습 초반 그래디언트가 죽지 않게), 60° 조기 오버슈트에도 `exp(-0.4×(60°)²_rad)≈0.65`로 꽤 높은 부분점수를 주고 있다 — 팔을 "제때" 움직이도록 강제하는 압력이 부족하다. `K_ARM_POSE`를 다리 쪽에 가깝게 올리거나(타이밍 정밀도↑, 다만 학습 초반 그래디언트 소실 재발 위험), action_rate/acc 정규화를 팔에 더 강하게 주는 것(급격한 스윙 자체에 직접 페널티) 등이 다음 후보로 제시됐으나, **사용자가 여기서 임시 검증을 종료하기로 결정** — 10M 결과(`handmove_arm_full_v3`)를 이번 실험의 최종 상태로 확정한다. 체크포인트: `logs/checkpoints/stage2_local/handmove_arm_full_v3/final_model.zip`(10M, 최종), `final_model_at_3M.zip`(3M, 참고용 백업), 영상: `logs/result/handmove_v2/handmove_v3_10M_policy_rollout.mp4`(24프레임, 46스텝에서 낙상).

**이 임시 검증 실험의 최종 요약**: 원래 사용자가 보고한 "떨림"에는 세 가지 서로 다른 원인이 섞여 있었던 것으로 정리된다 — ① 참조 데이터(v1 변환 스크립트)의 팔꿈치 축 추정 노이즈(해결, v2 쿼터니언 방식), ② `leg_pose`/`arm_pose` 보상이 하나로 섞여 팔 학습 신호가 죽던 문제(해결, 보상 항목 분리), ③ `ACTION_SCALE`이 전 관절 공용이라 팔이 물리적으로 목표각까지 명령될 수 없던 문제(해결, 관절별 action_scale 분리). 세 가지 모두 코드/파이프라인 차원에서는 고쳤고 재사용 가능한 형태(`arm_action_scale`, leg/arm 보상 분리, `--resume-from`)로 남겼다. 다만 K_ARM_POSE의 타이밍 민감도 부족이라는 네 번째 이슈는 발견만 하고 손대지 않은 채 남겨뒀다 — STAGE3에서 목 관절 도입 시 유사한 보상 튜닝이 어차피 필요하므로 그때 함께 다룰 수 있다. STAGE 진행상황(현재 STAGE 2)은 이 실험으로 변경되지 않는다.

**Blender 회전값 검증 + quaternion export 도입 (2026-08-29, STAGE 진행상황과 무관)**: `rotation_value_test.csv`(head/tail만)로 다리·팔 각도추출을 검증 — round-trip 오차는 0.0000°로 코드는 정확했지만, 결과값 자체가 두 가지 구조적 한계를 드러냈다: ① 몸통(spine) 자기축 트위스트(+13.311°, 양쪽 어깨가 정확히 동일 각도로 원운동)가 head/tail엔 전혀 안 남고 통째로 어깨 yaw에 흡수됨(자기축 회전은 위치 데이터로 원리적 복원 불가). ② 다리 무릎 굽힘이 로봇의 실제 무릎축(Y)이 아닌 X축 방향이라 hip_yaw가 150°~180°/−84°까지 튐(character.xml 한계 ±30° 대폭 초과) — 코드 버그 아니라 그 동작 자체가 지금 로봇 골격으로 불가능한 자세.

`blender/export_bones_with_rotation.py`(신규)로 본마다 world-space rotation quaternion 4컬럼을 추가 export하도록 변경, `rotation_value_test_ver2.csv`로 재검증 — 위 두 finding 모두 quaternion ground truth로 **정확히(13.311° 소수점까지) 확인됨**. 무릎축 불일치의 실제 의도(진짜 hip_roll 테스트인지, sagittal 무릎을 의도했는데 축이 잘못 잡힌 건지)는 아직 사용자 확인 대기. 상세: 메모리 `project-blender-npz-pipeline`.

**팔 모방학습 파이프라인 검증 실험 (2026-08-28, STAGE 진행상황과 무관)**: 사용자가 `t-pose.csv`/`npz_handmove_test.csv`(head=1단위 rig, 실척 배율 k=0.258065 — 다리용 `npz_test.csv`와는 스케일 관례가 다른 별도 rig)를 제공, 3단계로 검증 요청.

1. **CSV pos ↔ XML pos 일치 검증(PASS)**: `t-pose.csv`(250프레임 전부 static 확인) 기반으로 `models/_viz_arms_temp.xml`의 어깨 위치·팔 길이가 정확히 역산됐는지 재계산 대조 — 오차 1e-6 이하로 전부 일치.
2. **이론적 관절각 계산 ↔ MuJoCo 측정값 round-trip 검증**: `scripts/convert_csv_to_npz.py`의 다리 각도추출 로직을 팔(어깨3축+팔꿈치)로 확장하는 과정에서 버그 발견 — `r_upper_arm_geom`/`r_forearm_geom`이 로컬 -Y(왼팔은 +Y, 거울대칭)인데 다리 코드를 그대로 베껴 양팔 다 +Y로 가정했었다. `side_sign`으로 수정 후 재검증하니 250프레임 전체에서 방향벡터 오차 0.0000°(완전 일치)로 PASS(`scripts/convert_handmove_to_npz.py`에 반영).
3. **모방학습 파이프라인 검증(PASS)**: `_viz_arms_temp.xml`(시각화 전용 임시 모델, character.xml과 무관)에 팔 액추에이터 8개 + stand keyframe을 추가하고, `envs/biped_mimic_gym.py`에 `fall_geom_names` 파라미터를 추가(기본값 유지, 하위호환)해 다른 모델도 재사용 가능하게 했다. `references/handmove_test.npz`(오른팔 어깨 yaw 스윙)로 `train_stage2_local.py`를 그대로 재사용해 30만 스텝 스모크 학습 — reward가 -0.9→16.1로 정체 없이 상승(STAGE2와 동일한 통과 기준 충족). 체크포인트: `logs/checkpoints/stage2_local/handmove_arm_smoke/`.

**주의**: 이 실험은 사용자가 명시적으로 "임시 검증용, character.xml·STAGE 상태 불변"으로 범위를 한정해 진행했다 — CLAUDE.md의 "팔·LCD를 RL 정책에 넣지 않는다(STAGE3 기준)" 규칙과 겹칠 수 있어 먼저 확인받았다. STAGE 진행상황(현재 STAGE 2)은 이 실험으로 변경되지 않는다.

**STAGE 1 확정 (2026-08-23, 사용자 판단)**: case1_short를 주력으로 채택, `models/character.xml`로 확정(내용은 `models/smoke_case1.xml`과 동일 — 다리폭0.175, hip_y 돌출캡0.01, 발목없음). case2_long은 폐기하지 않고 `models/smoke_case2.xml` + 체크포인트를 그대로 보류(나중에 재검토 가능).

**주의 — CLAUDE.md 엄격 기준 미달 상태에서 확정**: STAGE1 통과 기준("에피소드 길이가 최대치에 도달해 유지된다")은 학습 곡선(stochastic rollout) 기준으로는 충족되지 않았다 — case1_short 최종 episode_length는 약 222~296/1000. 사용자가 이 표본실험 수준에서 STAGE1을 마무리하고 STAGE2로 넘어가기로 명시적으로 결정했다. **후속 확인(아래 레퍼런스 녹화 항목)**: 다행히 deterministic(탐험 노이즈 없이) 정책은 실제로 20/20 시드 전부 최대 길이(1001스텝)까지 안 넘어지고 완주했다 — 학습곡선의 낮은 episode_length는 PPO 탐험 노이즈 때문이었고, 정책 자체는 이미 충분히 안정적이었다. STAGE1 엄격 기준 미달 우려는 기우였던 것으로 보인다.

**STAGE 2 착수 및 통과 (2026-08-23)**: CLAUDE.md STAGE2("모방학습 파이프라인 검증") 진행. 자매 프로젝트 `2robot_project`(사족보행, STAGE2 기완료)의 `envs/ant_mimic.py`·`scripts/record_policy.py`·`scripts/train_stage2.py` 패턴을 포팅했다. 신규 파일:
- `scripts/record_policy.py` — case1_short 정책(20시드 중 최장 롤아웃, 전부 1001스텝 완주해 seed=0 채택)을 `references/stage1_walk.npz`(qpos/qvel/foot_contact)로 녹화.
- `envs/biped_mimic_gym.py` — RSI(레퍼런스 랜덤 프레임 시작) + imitation 9항목(다리 관절각·관절속도, 몸통 xy위치·자세, 선속도xy/z, 각속도xy/z, 발접촉일치, 생존) + regularization(토크·action rate·acc, STAGE1 재사용) + limits(관절한계CBF·발-발충돌, STAGE1 재사용) + impact(발 Δv_z saturate, 신규) 전체 구현. ET는 STAGE1의 upvector/낙상geom 기준 그대로 재사용(2robot의 쿼터니언 tilt 대신).
- `scripts/train_stage2_local.py` — STAGE1과 동일 PPO 하이퍼파라미터, 20분/7분 페이싱.

**스모크 학습(30만 스텝)** 통과 — 전 imitation 항목이 정체 없이 상승하는 걸 확인 후 본 학습 진행. **본 학습(10M스텝)** 결과, 전체 구간에서 정체 없이 상승(텐서보드 `logs/tb/stage2_local/stage2_walk_0` 기준, 0→10M스텝):

| 지표 | 시작(16k스텝) | 25% | 50% | 75% | 끝(10M스텝) |
| --- | --- | --- | --- | --- | --- |
| ep_rew_mean | 12.2 | 57.1 | 115.7 | 238.1 | **317.2** |
| ep_len_mean | 22.6 | 51.2 | 106.7 | 225.8 | **293.1** |
| imitation_leg_pose | 2.64 | 15.2 | 32.3 | 65.1 | **96.5** |
| imitation_root_pos | 3.12 | 7.10 | 10.46 | 12.96 | **16.54** |
| imitation_root_ori | 1.74 | 5.06 | 8.67 | 13.11 | **26.34** |
| imitation_foot_contact | 0.60 | 1.67 | 3.70 | 7.73 | **11.53** |
| limits_joint 페널티 | −0.775 | −0.002 | −0.007 | −0.013 | **0.000** |
| limits_foot_collision | 0 | 0 | 0 | 0 | 0 |

마지막 25%(7.5M→10M) 구간에서도 여전히 가파르게 상승 중 — 정체(plateau) 없음, **STAGE2 통과 기준 충족**. regularization은 −8.93→−18.48로 더 음수가 됐는데(정책이 더 활발하게 움직이며 자연히 커진 토크/액션변화 비용), tracking 보상 상승폭이 이를 압도해 전체 reward는 계속 순증가.

**참고**: STAGE1 때 겪은 텐서보드 디렉토리 이벤트파일 섞임 문제는 이번엔 재발 안 함 — `stage2_walk`/`stage2_walk_smoke`가 서로 다른 라벨이라 별도 폴더에 깔끔하게 기록됨.

**오버레이 비교 영상 + frame-0 취약점 발견 (2026-08-23)**: 사용자 요청으로 STAGE1 레퍼런스(`case1_short_rollout.mp4`)와 STAGE2 결과를 아티팩트에서 겹쳐 보이게(불투명도 슬라이더) 비교했다. `scripts/render_mimic_policy.py`(신규, `render_policy.py` 포팅)로 STAGE2 정책을 렌더링하는 과정에서 발견: **레퍼런스의 정확히 frame 0에서 시작하면 stage2_walk 정책이 31스텝 만에 낙상**하는데, frame 5 이상 어디서 시작해도 970프레임 가까이 안정적으로 완주한다(같은 정책, 시작점만 다름 — 프레임 0,5,10,20,30,50,80,100,150,200에서 직접 확인). RSI로 넓게(거의 모든 프레임에서) 학습했는데도 레퍼런스의 맨 첫 순간 하나만 유독 못 버티는 것으로 보인다. 원인 미확인 — STAGE3 착수 전 재확인할 만한 포인트로 남겨둔다. 영상은 frame 5부터 렌더링해 우회했다(`logs/result/stage2/stage2_walk_rollout.mp4`).

**다리 부착 규칙 정정 + 다리폭 변경 (2026-08-23)**: 아티팩트에서 "몸통 폭 (=다리 간격)"으로 잘못 표기한 것을 사용자가 정정 — 몸통 폭(자체 가로)과 다리 부착 위치(hip_y)는 별개 개념이며, `gen_candidate_mjcf.py`가 `hip_y = torso_half_y`로 고정해 우연히 같았을 뿐이다. 사용자가 확정한 새 규칙: **다리 1개의 바깥쪽 면은 몸통 옆면에서 최대 0.01(머리=1 기준)까지만 벗어날 수 있다** + **다리 폭(지름) 0.15→0.175**로 변경. `docs/robot_spec.md`·`docs/assumptions.md` §12에 반영. `scripts/gen_candidate_mjcf.py`의 `generate()`에 `leg_protrusion_m` 파라미터를 추가해 `hip_y = torso_half_y + leg_protrusion_m - leg_radius`로 계산하도록 수정(기존 §4-B/§4-C 스윕 호출부는 하위호환 유지). `models/smoke_case1.xml`·`models/smoke_case2.xml`을 새 규칙으로 재생성:

| | case1_short | case2_long |
| --- | --- | --- |
| 다리 지름 (0.15→0.175) | 4.52 cm | 3.41 cm |
| hip_y (5.16→3.16 / 3.90→2.39) | 3.16 cm | 2.39 cm |
| 다리-다리 간격(gap) | 1.81 cm | 1.37 cm |

스탠스가 좁아지고 다리가 두꺼워졌다. 재생성 후 `scripts/smoke_test_stage1.py` 재실행 — 처음엔 case2_long에서 자기충돌 FAIL이 떴으나, 원인은 지오메트리가 아니라 스크립트의 `geom_world_aabb()`가 capsule geom_size(반지름, 반길이, 0)를 box half-extent로 착각해 반길이를 y폭으로 잘못 쓰던 **버그**였다(스탠스가 넓을 때는 우연히 안 걸렸다). `_local_half_extent()`로 geom 타입별 분기 처리해 수정 후 재검사하니 두 후보 모두 컴파일/자기충돌/8초 stand-hold 안정성 전부 실제로 PASS.

**재학습 완료 (2026-08-23)**: 다리부착 규칙 정정 후 지오메트리(다리폭 0.175, hip_y 돌출캡 0.01)로 case1_short/case2_long을 동일 조건(lr 3e-4, seed 1, 10M스텝, `scripts/train_stage1_local.py`)으로 재학습. **순위가 역전됐다** — 옛 지오메트리(다리폭 0.15)에서는 case1_short가 모든 지표에서 앞섰는데, 새 지오메트리에서는 case2_long이 앞선다:

| 항목(최종 10% 평균) | case1_short | case2_long |
| --- | --- | --- |
| episode_length (평균/최댓값) | 222.0 / 296 | **281.5 / 534** |
| ep_rew_mean | 164.1 | **203.5** |
| forward reward(누적) | 77.7 | **98.1** |
| foot_collision | 0 | ≈0 |
| joint_limit 페널티 | **−0.66** | −1.24 |

둘 다 최대 에피소드 길이(1000) 미도달, 미수렴. **지오메트리를 한 축(다리 부착 방식)만 바꿔도 몸 비율 비교의 승자가 바뀐다** — 몸 비율(case1 vs case2)과 다리 부착 방식이 서로 독립적이지 않다는 뜻. CLAUDE.md 작업 규칙 1("한 번에 한 변수만 바꾼다")을 엄밀히 지키려면 추후 이 두 축을 분리해서 재검토할 필요가 있음 — 지금은 결과만 기록하고 판단은 사용자 대기.

**추출 시 주의사항**: `logs/tb/stage1_local/{case1_short,case2_long}_0`에는 2026-08-22(옛 지오메트리) 이벤트파일과 2026-08-23(재학습) 이벤트파일이 **같은 디렉토리에 섞여 있다** — SB3의 tb_log_name 자동증가(`_1`, `_2`...)가 기대와 달리 작동하지 않아 새 실행도 `_0`에 그대로 씀. pid로 구분해야 한다(case1: `*.308060.*`가 최신, case2: `*.337545.*`가 최신). EventAccumulator로 디렉토리 전체를 그냥 읽으면 두 실행의 스텝이 뒤섞여 곡선이 톱니파처럼 잘못 나오므로, 앞으로 이 디렉토리에서 곡선을 뽑을 땐 항상 최신 pid 파일만 골라서 읽을 것.

영상: `logs/result/stage1/case1_short_rollout.mp4`, `case2_long_rollout.mp4` (재학습본으로 덮어씀). 체크포인트: `logs/checkpoints/stage1_local/{case1_short,case2_long}/final_model.zip`(재학습본으로 덮어씀).

**STAGE 1 방향 전환 (2026-08-22)**: 사용자 지시로 `docs/robot_spec.md`를 새 비율 체계(머리 높이·폭=1 기준 단위, 몸통/다리가 머리 대비 배수)로 교체. **이전 §4-B(머리35%)·§4-C1(다리두께25%)·§4-C2(발면적40%) 확정치와 `models/character.xml`은 폐기**하고 STAGE 1 몸 비율 탐색을 새 체계로 처음부터 재시작한다 (상세: `docs/assumptions.md` §10). 관절 구성(발목 없음, 2026-08-13 비용 트랙 결정)은 비율 체계와 무관한 별개 축이라 폐기 대상이 아니고 그대로 재사용한다.

**STAGE 1 스모크테스트 (2026-08-22)**: CLAUDE.md STAGE 1 원칙("가장 먼저 실행할 것 — robot spec의 기본 조건을 먼저 테스트")에 따라, 새 문서의 1번/2번 "선제적으로 진행할 실험의 값"으로 후보 2개를 생성해 물리 검증만 실행 (RL 학습 없음). `scripts/gen_candidate_mjcf.py`에 신규 비율 체계 변환 헬퍼(`generate_from_head_unit_ratios`)를 추가하고, `scripts/smoke_test_stage1.py`(신규)로 컴파일/자기충돌(AABB)/8초 stand-hold 물리 롤아웃 안정성을 확인했다. 절대 크기 스케일·발 형상·관절 구성 등 새 문서에 없는 축은 임시 가정을 세워 진행했다(`docs/assumptions.md` §11).

| 후보 | 비율(몸통 높이/폭, 다리 높이/폭 — 머리=1 기준) | 절대 크기(전체 0.40m 앵커) | 컴파일 | 자기충돌 | 8초 롤아웃 안정성 |
| --- | --- | --- | --- | --- | --- |
| case1 (숏다리형) | 0.35/0.4, 0.2/0.15 | 머리25.8cm·몸통9.0cm·다리5.2cm | PASS | PASS (겹침 없음) | PASS (낙차 0.4mm, 발산 없음) |
| case2 (롱다리형) | 0.55/0.4, 0.5/0.15 | 머리19.5cm·몸통10.7cm·다리9.8cm | PASS | PASS (겹침 없음) | PASS (낙차 0.5mm, 발산 없음) |

두 후보 모두 통과. 이 롤아웃은 외란(push)을 주지 않은 "제자리 서기" 검증이라 RL 학습 단계의 통과 기준(넘어지지 않고 전진)과는 다르다 — 실제 걷기 가능 여부는 그리드 스윕/순수 RL 비교에서 확인해야 한다. 0.05 간격 그리드 스윕은 새 문서 지시대로 사용자가 요청할 때만 진행한다.

**case1_short vs case2_long 순수 RL 학습 (2026-08-22)**: STAGE 1 절차 3번("완전히 동일한 조건으로 학습")에 따라 `scripts/train_stage1_local.py`(로컬 CPU, SB3 PPO, lr=3e-4/gamma=0.97/gae_lambda=0.95/ent_coef=0.01/clip_range=0.3/n_epochs=4/batch_size=256/n_steps=2048/n_envs=8/seed=1, 20분작업·7분휴식 페이싱 — §4-B 8개 변형과 완전히 동일한 설정)로 두 후보를 순차 10M스텝 학습. 코드 수정 없음 — `envs/biped_rl_gym.py`의 fall/foot geom 이름 규칙이 두 XML과 이미 일치. 둘 다 실측 fps 2870~2910으로 완주(`TRAIN_STAGE1_LOCAL_OK`). 로컬(Colab 아님) 사용 이유: 이 세션이 백그라운드 작업이라 Colab의 불안정한 대화형 세션(`logs/colab_session_log.md`)을 감독하기 어렵고, 후보가 2개뿐이라 로컬로 충분.

텐서보드 로그(`logs/tb/stage1_local/{label}_0`) 전체 시계열 기준 비교:

| 항목 | case1_short (숏다리형) | case2_long (롱다리형) |
| --- | --- | --- |
| episode_length — 10%/50%/90%/100% 지점 | 137 → 197 → 252 → 239 | 41 → 70 → 161 → 199 |
| episode_length — 최종 10% 구간 평균 (최댓값) | **368.6** (최대 596.7) | 247.4 (최대 439.8) |
| ep_rew_mean — 최종 10% 구간 평균 | **268.3** | 174.4 |
| forward reward — 최종 10% 구간 평균 (환산 평균 전진속도 ≈0.22~0.23 m/s, 둘 다 forward_vel_target=0.3m/s 미포화) | **123.4** | 76.0 |
| foot_collision(발-발 충돌) | 전 구간 0 | 전 구간 ~0 |
| joint_limit 페널티 — 최종 10% 구간 평균 | -0.70 (여유 있음) | **-4.34** (관절 한계 근접 빈번 — 긴 다리가 스윙 범위를 더 많이 씀) |

**해석**: case1_short(숏다리형)이 case2_long(롱다리형)보다 10M스텝 시점에 명확히 더 잘 걷는다 — 에피소드 길이·보상 모두 우위이고, 관절 한계 페널티도 훨씬 작아 여유 있게 움직인다. 다만 둘 다 최대 에피소드 길이(1000)에 도달하지 못했고 case1도 곡선이 완전히 평평해지지 않아(7500K→10000K 구간 251→239로 소폭 하락) 추가 학습 여지가 있다 — "승리 확정"이 아니라 "10M스텝 기준 case1 우위" 정도로 읽어야 한다. 승자 확정, 스텝 수 연장, 0.05 그리드 스윕 등 다음 절차는 사용자 판단 대기.

영상: `logs/result/stage1/case1_short_rollout.mp4`, `logs/result/stage1/case2_long_rollout.mp4` (각 학습된 정책 결정적(deterministic) 롤아웃, 최대 10초).
체크포인트: `logs/checkpoints/stage1_local/{case1_short,case2_long}/final_model.zip`.

**STAGE 0 종료 (2026-08-10)**: 통과 기준 3개(걷는 영상 / 측정값 5개 기록 / 체크포인트 세션 유실 후 생존) 전부 충족.

**1-A1 종료 (2026-08-11)**: 후보 A(발목 없음+롤오버밑창) vs 후보 B(발목 pitch) 순수 RL 10M-step PPO 비교. **승자: 후보 B.** 피크 episode_length 979/1000(B) vs 500/1000(A), 최종 reward 355.8(B) vs 103.2(A) — B가 더 빠르고 더 높게 수렴. 상세: `docs/robot_spec.md` §4-A1. 디버깅 중 무릎 관절 기본각이 관절 한계 경계와 겹쳐 CBF 페널티가 상시 발동하던 버그를 발견해 관절 범위를 넓혀 수정했다(자세 자체는 원래 편 다리 유지 — 구부린 자세는 오히려 더 불안정해서 되돌림).

**절차 변경 (2026-08-11)**: 사용자 지시로 이후 4-B(몸 비율)·4-C(다리·발 형상)는 1-A1 승자만 쓰지 않고 **후보 A/B 둘 다** 스윕한다 — 관절 구성과 비율이 서로 독립적인지 직접 검증하기 위함. 실험 횟수가 이 지점부터 약 2배로 늘어난다.

**4-B 진행 상황 (2026-08-11)**: `scripts/gen_candidate_mjcf.py`로 머리 비율 4단계(30/40/50/60%, 몸통:다리 1:1 고정) × 발목 구성 2개 = 8개 MJCF를 `models/sweep_4b/`에 생성·검증 완료(질량·안정성 확인). `scripts/train_stage1.py`를 `--model_path`/`--label`로 일반화해 임의 MJCF를 학습할 수 있게 함.
- head30_A: 완료 (피크 reward 762.4/episode_length 879 @8.0M, 최종 172.5/266)
- head30_B: 2.3M 스텝까지 진행 후 Colab 세션 사망 (체크포인트 로컬 보존)
- head40_A~head60_B: 미착수

**후보 A 전용 트랙 + 4-B 최종 확정 (2026-08-13)**: 하드웨어 비용 때문에 이후 STAGE 1은 후보 A(발목 없음)만 진행하기로 결정(상세: `docs/robot_spec.md` §4-A1 "하드웨어 트랙 결정"). 5개 지점 전체 결과(A): 30%=500, **35%=660(최고)**, 40%=375, 45%=127(최저), 50%=523 — 비단조 패턴 확인. **§4-B 확정값: 머리 35%.** §4-C1(다리두께) 완료: 15%=161/117, 25%=660/569(최고), 35%=645/502. **확정값: 다리두께 25%.** §4-C2(발면적) 완료: 20%=158/110, 30%=476/409, 40%=781/678(최고), 50%=450/362 — 역U자형, 너무 크면 오히려 나쁨. **확정값: 발면적 40%.**

**STAGE 1 (후보 A 트랙) 잠정 확정 (2026-08-14)**: 발목 없음 / hip yaw+roll+pitch(3축) + knee / 롤오버 밑창 / 머리35% / 몸통:다리 32.5:32.5(1:1, 세부스윕 미실행) / 다리두께25% / 발면적40%. `models/character.xml`로 통합 생성 및 안정성 검증 완료. 상세: `docs/robot_spec.md` §5. 1-A2(hip축 실험)와 §4-B 3단계(몸통:다리 세부)는 의도적으로 건너뜀 — 필요시 재검토.

**다음 계획**: 사용자가 처음 제시한 대로, 이 최종 형태(A)에 발목(B)을 붙인 버전과 애니메이션/움직임 품질을 비교해서 발목 추가 여부 최종 판단.

**GPU 세션 반복 실패 (2026-08-11)**: `colab new`가 "Service Unavailable"로 3회 연속 실패. 로컬 CPU 대체 경로(`envs/biped_rl_gym.py` + `scripts/train_stage1_local.py`, SB3 PPO, 2robot_project 패턴 포팅, GPU와 최대한 동일한 하이퍼파라미터)를 만들어 검증까지 마쳤으나, 실측 처리량이 8코어로도 초당 ~1080스텝 수준이라 10M 스텝 1개에 약 2.6시간, 8개면 20시간+ 소요 예상. 사용자가 처음엔 GPU 유지를 선택해 5시간마다 재시도(cron)했으나, 35시간(7회 연속 실패) 뒤 로컬로 전환 결정.

**4-B 완료 (2026-08-13)**: 로컬 CPU로 8개(머리비율 4단계 × 발목구성 2개) 전부 학습 완료. 컴퓨터 과열 방지를 위해 20분 작업/7분 휴식 페이싱 적용(실측 fps 1889~2858, 처음 추정보다 훨씬 빠름 — 변형당 대략 20분~1시간). 결과 요약(episode_length/reward, SB3 rolling mean):

| 머리% | A(발목없음) | B(발목pitch) |
|---|---|---|
| 30 | 500/407 | **956/873** (전체 1위) |
| 40 | 375/308 | 690/601 |
| 50 | **523/428** | 930/824 |
| 60 | 137/97.3 (최악) | 803/682 |

발목 유무 우위(1-A1 결론)가 비율과 무관하게 유지됨을 재확인. 머리 60%는 양쪽 다 최악(무게중심/관성 문제로 추정). 후보 A 최적 비율(50%)과 후보 B 최적 비율(30%)이 다름 — 상세 해석과 다음 절차는 `docs/robot_spec.md` §4-B 참고.

---

## STAGE 0 측정값

**측정 대상**: Robotis OP3 (기성 모델). "우리 몸"이 아직 없어 OP3로 방법론을 검증했다 — 근거는 `logs/question.md` 참고. 스크립트: `scripts/measure_stage0.py` (순정 mujoco, CPU, GPU/Colab 불필요).

**방법론 주의사항**: 이 OP3 MJX 자산은 발(foot1/foot2) 지오메트리만 `contype/conaffinity=1`이고 나머지 전부(허벅지, 머리 포함) `0`으로 꺼져있다 — MJX GPU 학습 속도를 위한 표준 최적화다. 그래서 `mj_forward`의 접촉 목록(`data.ncon`)으로는 허벅지-허벅지, 머리-바닥 접촉을 감지할 수 없다 (첫 시도에서 "허벅지 접촉 4.5도"로 나온 값은 사실 발끼리의 접촉이었다). 이후 지오메트리를 world-frame AABB로 직접 계산해 겹침을 판정하는 방식으로 다시 측정했다. **STAGE 1에서 우리 몸 MJCF를 만들 때는 collision group을 우리가 직접 설계하므로, 필요한 지오메트리 쌍의 self-collision을 명시적으로 켜거나(권장) 이 스크립트의 AABB 방식을 계속 쓸지 미리 정해야 한다.**

| # | 측정 항목 | 값 | 방법/비고 |
| --- | --- | --- | --- |
| ① | 역진자 시간상수 τ = √(무게중심높이 / g) | h_com = 0.2677 m → **τ = 0.1652 s** | stand keyframe 기준 subtree CoM |
| ② | 정책 제어 주파수 | ctrl_dt = 0.02 s (50 Hz) → τ 안에 **8.3회** → 통과 (≥5) | Op3Joystick 기본 설정 |
| ③ | 고관절 최대 벌림각 | **외전(벌림) 방향: 89도까지 허벅지 접촉 없음** (참고: 내전/교차 방향은 2.5도에서 접촉 — 이건 "벌림각"이 아니라 별개 현상) | 발 지오메트리는 제외하고 순수 허벅지-종아리만 AABB 겹침 판정 |
| ④ | 명령 속도 상한 | 다리 길이 L = 0.2499 m. ③이 89도까지 안 걸려 허벅지 충돌이 binding 제약이 아님 → v_fall_limit = √(0.5·g·L) = **1.107 m/s**가 사실상의 상한 | f_step = 1/(2τ) ≈ 3.03 Hz 가정 |
| ⑤ | 머리 접지 각도 | 전방(pitch+) **80.0도**, 후방(pitch-) **89도까지 접촉 없음**, 좌측(roll+) **84.0도**, 우측(roll-) **84.0도** | stand 자세에서 두 발 중점을 피벗으로 강체 회전, AABB 최하단점이 z≤0이 되는 각도 |

**해석**: OP3는 다리가 상대적으로 얇아 ③·④에서 "허벅지 충돌"이 걷기 성능의 제약으로 작동하지 않는다. `docs/robot_spec.md`가 예상한 "다리가 두꺼우면 벌림각이 줄어든다"는 문제는 OP3가 아니라 우리가 설계할 두꺼운 다리에서만 나타날 것으로 보인다 — 이는 OP3 측정으로는 이 특정 제약을 미리 볼 수 없다는 뜻이고, STAGE 1 §4-C1(다리 두께)에서 반드시 직접 재확인해야 한다.

**영상/체크포인트**: `logs/result/stage0/rollout{0,1,2}.mp4` (Colab T4, mujoco_playground OP3Joystick, 20M 스텝 PPO, 무수정). 프레임 확인 결과 실제 보행 사이클(스윙-지지 교대) 확인, 중간에 비틀거림 후 회복하는 구간 있음 — reward 5~7대로 완전 수렴 전 상태와 일치. 체크포인트 19개 `logs/checkpoints/stage0/op3_stock/`에 로컬 저장, 마지막 체크포인트 로드 검증 완료(`orbax` restore 성공). Colab 세션이 중간에 한 번 실제로 죽었다가(`logs/colab_session_log.md`) 재생성 후 이어서 완주 — 체크포인트는 로컬 sync 덕에 유실 없음.

## 절대 크기 (§robot_spec.md §9)

**확정 (2026-08-10, STAGE 0 임시값 — §4-B·§4-C 그리드 스윕에서 재검증 가능)**. "캐릭터 디자인 금지"는 형태(모양)에 대한 금지이지 숫자(크기)에 대한 금지가 아니라는 사용자 확인에 따라 채운다.

```
전체 높이      : 40 cm
머리 높이      : 18 cm    (비율 45%,  §4-B 그리드 시작값)
몸통 높이      : 11 cm    (비율 27.5%)
다리 길이      : 11 cm    (비율 27.5%)
다리 두께      : 2.75 cm  (다리 길이 대비 25%,  §4-C1 그리드 시작값 — 범위 15~35% 중간)
발 길이 × 너비 × 두께 : 3.85 × 2.7 × 0.5 cm  (발 길이는 다리 길이 대비 35%,  §4-C2 그리드 시작값 — 범위 20~50% 중간, 너비는 발 길이의 70%)
총 질량        : 1.5 kg
머리 질량      : 0.45 kg  (비율 30% — 무거운 LCD 헤드 컨셉 반영, 균등 스케일링보다 높게 잡음)
팔 질량 합     : 0.06 kg  (비율 4% — STAGE 3의 "5% 미만이면 정책 밖" 조건을 만족하도록 목표한 값. 실제 형상 확정 후 재확인 필요)
결정 이유      : 시뮬레이션 전용 — 실기 제작 가능성은 열어두지 않는다(자유 설계). 절대 크기 자체는 STAGE 0에서 측정한 OP3(51cm급, τ=0.165s)보다 작지만, robot_spec.md §9가 경고한 "크기가 작을수록 τ가 짧아져 학습이 어려워진다"는 점을 고려해 극단적으로 작게(예: 20~25cm) 가지 않고 40cm로 완충했다.
```

## 로그

- 2026-08-10: 프로젝트 폴더 구조 초기화 (docs/, models/, references/, scripts/, envs/, policies/, logs/)
