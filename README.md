# 3robot_project

시뮬레이션 전용 **소형 이족보행 캐릭터 로봇** RL 프로젝트.

- 목표: 소형 이족 캐릭터 로봇이 MuJoCo 시뮬레이션 안에서 자연스럽고 캐릭터답게 걷고, 앉기·인사 같은 트리거 동작을 수행하게 한다.
- 형태: 큰 LCD 머리 + 작은 몸통 + 짧고 두꺼운 평발 다리 + 뭉뚝한 팔.
- 시뮬레이터: MuJoCo / MJX / MuJoCo Playground (Colab T4, 무료 티어).
- 실기 배포 없음 — 시뮬레이션 전용.

## 진행 규칙

프로젝트는 STAGE 0 → 6의 단계적 파이프라인으로 진행된다(기성 스택 검증 → 순수 RL 걷기 → 모방학습 → 표현 채널 분리 → 트리거 동작 → 모캡 리타게팅 → 통합). 전체 규칙·통과 기준·금지 사항은 [`CLAUDE.md`](./CLAUDE.md)에 정의되어 있다.

## 현재 상태

- **STAGE 0** 통과 — 기성 모델 스택 검증 + 몸 치수 측정.
- **STAGE 1** 통과 — 순수 RL로 다리 골격·몸 비율 확정 (`models/character.xml`).
- **STAGE 2** 통과 (2026-08-23) — STAGE 1 정책을 레퍼런스로 모방학습 파이프라인 검증 완료.
- **STAGE 3** 착수 여부 사용자 판단 대기 (목을 정책 입력으로, 팔·LCD를 정책 밖으로 분리).

세부 진행 로그는 [`logs/progress.md`](./logs/progress.md), 미확정 사항은 [`docs/assumptions.md`](./docs/assumptions.md)를 참고.

## 디렉토리 구조

```
docs/       로봇 형태·관절 명세, 추정·근거, 외부 자료 조사
envs/       RL 환경 (biped_rl, biped_mimic)
models/     MJCF 로봇 모델 (candidate_A/B, character.xml 확정본)
references/ 모방학습용 레퍼런스 모션 (.npz)
scripts/    학습·녹화·리타게팅·사전검사 스크립트
policies/   동작별 학습된 정책 (stand/walk/sit)
logs/       진행 기록, 결과 영상. 학습 체크포인트(logs/checkpoints/)와
            텐서보드 로그(logs/tb/)는 용량 문제로 저장소에 포함하지 않는다.
```

## 참고

`logs/checkpoints/`, `logs/tb/`는 로컬에는 존재하지만 재생성 가능한 학습 산출물이라 `.gitignore`로 제외했다. 결과 확인용 영상은 `logs/result/`에 mp4로 포함되어 있다.
