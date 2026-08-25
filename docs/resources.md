# 자료 재구성

날짜: 8월 9
상태: 진행중
날짜 형식2: 2026-08-09
주요태그: 구현 시도

# resources.md — 외부 자료 조사

> 확인된 사실만 기록한다. 해석과 적용 판단은 `assumptions.md`에 있다.
**코드 상태 표기**: ✅공개 확인 / ❓미확인 / ❌없음
> 
> 
> 최종 갱신: 2026-08-08
> 

---

## 1. 캐릭터 이족 로봇 논문

### 《Olaf: Bringing an Animated Character to Life in the Physical World》 ★1순위 참조

Müller, Knoop, Mylonopoulos, Serifi, Hopkins, Grandia, Bächer.
【Disney Research Imagineering】. arXiv:2512.16705 (2025-12-18). 코드 ❌

**하드웨어**

- 88.7 cm (머리카락 제외) / 14.9 kg / 총 25 DoF
- 다리당 6, 어깨당 2, 목 3, 턱 1, 눈썹 1, 기계 눈 4
- 비대칭 6-DoF 다리 설계 (한쪽 다리를 뒤집어 좌우 액추에이터 충돌 회피)
- 다리를 하체 안에 숨기고 PU 폼 스커트로 가림
- Unitree + Dynamixel 액추에이터, 온보드 컴퓨터 3대

**제어 구조**

- 서기 정책과 걷기 정책을 **분리**
- articulation backbone (다리+목) = RL / show functions (팔·입·눈·눈썹) = 고전 제어
- path frame `(x, y, ψ)` 으로 전역 자세 불변성 확보
- 액션 = 관절 PD 목표값
- 관측 `s_t = (path frame 기준 루트 자세, 루트 속도, 관절각, 관절속도, a_{t-1}, a_{t-2}, 온도, 게이트 위상)`
- 제어 입력 `g_t`: 서기 `(q̂_neck, θ̂, p̂_z)` / 걷기 `(q̂_neck, v̂_PF)`. 학습 중 전 범위 무작위화
- 정책 50 Hz
- 게이트 생성 도구로 **heel-toe 모션이 있는 스타일 워크 사이클** 설계, 중요성을 §VIII-A에서 실증

**설계 워크플로**

> 캐릭터의 envelope 안에서 어디에 자유도를 둘지 빠르게 탐색하기 위해, 서기·걷기 정책을 반복 학습시키며 캐릭터 골격의 표현력을 시뮬레이션에서 평가했다.
애니메이션 저작을 위해, 로봇과 동일한 자유도를 가진 애니메이션 리그와 애니메이션 레퍼런스를 유지했다.
> 

**보상 전체 목록**

| 분류 | 항목 |
| --- | --- |
| Imitation | 몸통 xy 위치, 몸통 자세, 선속도 xy, 선속도 z, 각속도 xy, 각속도 z, 다리 관절각, 목 관절각, 다리 관절속도, 목 관절속도, 발 접촉 일치, 생존 |
| Regularization | 토크, 관절 가속도, 다리/목 action rate, 다리/목 action acc |
| Limits | 목 온도(CBF), 관절 한계 하한/상한(CBF), 발-발 충돌 |
| Impact | 발의 중력방향 속도변화 `Δv_z` 페널티 (saturate) |
- 다리/목 가중치 분리 이유: **반사 관성이 크게 다름**
- 관절 한계 여유 `q_m = 0.1 rad`, `γ_q = 20`
- 종료 조건: 머리·몸통·허벅지·팔의 지면 접촉
- impact saturate 이유: 접촉 해석이 만드는 큰 속도 변화가 critic 학습을 불안정하게 함

---

### 《Design and Control of a Bipedal Robotic Character》 (BDX)

Grandia, Knoop, Hopkins, Wiedebach, Bishop, Pickles, Müller, Bächer.
【Disney Research】+【Walt Disney Imagineering】. RSS 2024. arXiv:2501.05204. 코드 ❌

- **다리당 5 DoF**, 목·머리 4 DoF
- 액추에이터를 관절에 직접 배치 (ANYmal 방식)
- **발목 액추에이터를 발에 직접 배치. ankle roll 액추에이터는 없음**
- **수동 롤 허용을 위해 발바닥 두 개를 둥글게 깎음**
- 우레탄 폼 성형으로 착지 충격 완화
- 무릎이 뒤로 굽는 구조 (창작 의도)
- 정책 3분할: perpetual / periodic / episodic
- path frame 개념의 출처

**우리 형태와의 차이**: 디지티그레이드 다리, 무릎 역방향. **형태 참조 대상이 아니다.** 제어 구조만 참조한다.

---

### 《Robot Motion Diffusion Model》 (RobotMDM)

Serifi, Grandia, Knoop, Gross, Bächer.【Disney Research】+【ETH Zürich】. SIGGRAPH Asia 2024. 코드 ❓

- 평가 대상: **20 DoF, 0.84 m, 16.2 kg 이족 로봇** — 우리와 크기·형태가 가까움
- 사전학습 MDM + 물리 기반 제어의 critic을 surrogate 보상으로 써서 MDM 파인튜닝
- 텍스트 조건부 운동학 diffusion + RL 추종 제어기 결합

**판정**: 개념만 참고. 공개 코드 미확인이라 재현 불가. 우리가 흉내낼 수 있는 부분("생성된 운동학 모션을 RL 추종 정책으로 거른다")은 이미 파이프라인에 있다.

---

## 2. 실행 가능한 코드 자원

### ▣MuJoCo Menagerie / MuJoCo Playground

Menagerie 수록 휴머노이드 (DoF):

| 모델 | DoF | 라이선스 |
| --- | --- | --- |
| **Robotis OP3** | 20 | Apache-2.0 |
| **Berkeley Humanoid** | 12 | BSD-3-Clause |
| Booster T1 | 23 | Apache-2.0 |
| Unitree G1 | 29 | BSD-3-Clause |
| Fourier N1 | 23 | Apache-2.0 |
| PNDbotics Adam_lite | 25 | MIT |
| ToddlerBot 2XC / 2XM | 44 | MIT |
| Apptronik Apollo | 32 | Apache-2.0 |
| TALOS | 44 | Apache-2.0 |
| Agility Cassie (biped) | 28 | MIT |

MuJoCo Playground에 **locomotion 환경이 구현된 것**:
Berkeley Humanoid, Unitree H1, Unitree G1, Booster T1, Robotis OP3

→ STAGE 0 출발점. Robotis OP3가 소형 휴머노이드로 가장 적합.
→ Booster T1은 118cm로 큼. 스택 검증용으로만.

---

### ⌘Open Duck Mini v2

`github.com/apirrone/Open_Duck_Mini` — BDX 드로이드의 35~42cm 미니 버전 오픈소스

| 저장소 | 내용 |
| --- | --- |
| `Open_Duck_Mini` | 본체. MJCF/URDF, 문서, 사전학습 정책(ONNX) |
| `Open_Duck_Playground` | MuJoCo Playground(MJX) RL 환경. `joystick.py`, `rewards.py`, `poly_reference_motion.py`, `export_onnx.py`, MJCF(머리 有/無), flat/rough terrain 씬 |
| `Open_Duck_reference_motion_generator` | 파라메트릭 워크 엔진. `auto_waddle.py` → `polynomial_coefficients.pkl` |
| `Open_Duck_Mini_Runtime` | 실기 구동 (우리에겐 불필요) |

저장소 문서 확인 내용:

> BDX 논문에서 Disney가 기술한 모방 보상을 구현해 좋은 결과를 얻었다. 이 보상을 쓰려면 레퍼런스 모션이 필요하다. 파라메트릭 워크 엔진으로 그런 모션을 생성하는 저장소를 만들었다.
> 
- 새 로봇 추가 절차 문서화됨 (`playground` 아래 디렉토리를 만들고 `open_duck_mini_v2`를 복사)
- 액추에이터 파라미터 식별에 ▣BAM 사용 (`bam.to_mujoco`)

**우리에게 전이되는 것**

| 항목 | 전이 |
| --- | --- |
| MJX 환경 골격 (`joystick.py`, `runner.py`, 씬 구성) | ✅ 그대로 |
| 보상 구현 (`rewards.py`, BDX 모방 보상) | ✅ 거의 그대로 |
| ONNX 익스포트, 도메인 무작위화 | ✅ 그대로 |
| 새 로봇 추가 절차 | ✅ 그대로 |
| MJCF 모델 | ❌ 다리 형상 불일치 (디지티그레이드) |
| `auto_waddle.py` 레퍼런스 생성기 | ❌ 오리 다리 기준으로 파라미터화됨. 평발+짧은다리로 재작성 필요 |

⚠️ **라이선스 미확인.** 코드를 가져오기 전에 확인할 것.

---

### ▣LocoMuJoCo (v1.1)

`github.com/robfiras/loco-mujoco`

- MuJoCo(단일) + MJX/MJWarp(병렬) 지원
- 휴머노이드 12개 + 4족 4개 환경, 생체역학 인체 모델 4개
- **PPO / GAIL / AMP / DeepMimic**의 단일 파일 JAX 구현
- **AMASS·LAFAN1·자체 데이터셋 22,000개 이상을 각 휴머노이드용으로 리타게팅해서 제공**
- **robot-to-robot 리타게팅** — 자기 휴머노이드를 추가하면 전체 데이터셋에 즉시 접근
- DTW·Fréchet distance 등 궤적 비교 지표 (JAX)
- Gymnasium 인터페이스, 도메인·지형 무작위화 내장
- LAFAN1과 기본 데이터셋은 HuggingFace에서 자동 다운로드. **AMASS는 라이선스 때문에 별도 다운로드 필요**

→ STAGE 5의 1순위 도구. 우리 스택(MuJoCo/MJX)과 정확히 일치.

---

### ⌘GMR — General Motion Retargeting

`github.com/YanjieZe/GMR`, ICRA 2026

- ▣mink + ▣MuJoCo 기반 IK 솔버. CPU에서 실시간
- 입력: AMASS, OMOMO, LAFAN1
- 지원 로봇에 HighTorque Hi, PND Adam Lite, Booster T1, Unitree H1/H1-2, Kuavo 등 포함
- 인간 프레임 = `(body_name, 3D translation + rotation)` dict
- 로봇 프레임 = `(base translation, base rotation, joint positions)` 튜플

→ 관절각 복사가 아니라 **IK로 링크 목표를 맞추는 방식**이라 비율 차이에 상대적으로 강하다. STAGE 5의 2순위.

---

### 기타 리타게팅 도구 (참고)

| 도구 | 내용 |
| --- | --- |
| ⌘OmniRetarget | 인간·로봇 메시의 Laplacian deformation 최소화 + 접촉 보존. LAFAN1/OMOMO 대상 9시간 이상 궤적 생성 |
| ⌘PHUMA (DAVIAN Robotics) | 물리적 타당성 기준으로 인간 모션을 걸러낸 데이터셋. G1/H1-2 지원, 커스텀 로봇 리타게팅 가능. LocoMuJoCo와 연계 |

---

## 3. 사용 불가로 확인된 것

| 항목 | 상태 | 확인 내용 |
| --- | --- | --- |
| 《Humanoid-R0》 | 코드 ❌ | ICLR 2026 Withdrawn Submission |
| 《RLPF》 | 코드 ❌ | ICLR 2026 Withdrawn Submission |
| 《SafeFlow》 | 코드 ❌ | 휴머노이드 전용, 공개 없음 |
| 《RobotKeyframing》 | 코드 ❓ | 프로젝트 사이트·논문만 확인 |
| 《NIL》 | 코드 ❌ | "coming soon" 상태, 개인 재현 사례 없음 |
| STMR | 코드 ❓ | 저자가 공개를 주장하나 실행 가능한 저장소 미검증. Isaac Gym 기반 |

---

## 4. 이족 전환으로 사용 가능해진 도구 (미도입)

STAGE 조건을 만족하기 전에는 착수하지 않는다. CLAUDE.md §금지 사항 참조.

| 도구 | 코드 | 조건 |
| --- | --- | --- |
| ⌘MDM | ✅ | STAGE 5 완료 후. 인간(HumanML3D) 골격 출력 |
| ⌘MoMask | ✅ | 위와 동일. WebUI/Colab/Blender 애드온 있음 |
| ⌘T2M-GPT | ✅ | 위와 동일 |
| QuickMagic | 서비스 | 위와 동일. 프롬프트→FBX. 무료 한도 존재 |
| 〈AMP〉 | ✅ (LocoMuJoCo 내장) | 리타게팅 모션 30분 이상 + 사용자 허가 |
| 〈z공간/VAE〉 | — | 동작 10개 이상 + 사용자 허가 |
| ⌘AnyTop | ✅ MIT | 백업. 임의 골격용이지만 인간 골격 자원이 더 풍부 |

---

## 5. 조사 방법 메모

- 조사 시점: 2026-08-08
- 확인 경로: arXiv 원문(Olaf §III~VII 직접 확인), GitHub README, MuJoCo Menagerie 저장소, MuJoCo Playground 논문
- **읽지 않은 것**: 《Olaf》 §VIII 정량 결과, 《BDX》 전문, Open Duck Mini 관절 사양 원문