# ContextAE: Train-only clustering 기반 K-context AutoEncoder

이 실험은 기존 HourAE에서 **고정 24개 시간대 분할만 data-driven context clustering으로 교체**합니다.

## 핵심 목적

HourAE의 24개 고정 모델이 아니라, TRAIN 데이터만 사용해 하루의 288개 5분 슬롯을 유사한 정상 패턴끼리 묶고,
선택된 K개의 context마다 별도의 AutoEncoder를 학습합니다.

이번 단계에서는 **direction-aware scoring은 아직 넣지 않습니다.**
고정 24시간 분할을 K-context clustering으로 바꾸는 효과만 분리해서 보기 위함입니다.

## Context discovery

1. TRAIN만 사용
2. 각 5분 time slot(00:00, 00:05, ..., 23:55)에 대해 여러 날의 값을 모음
3. feature별 across-day median과 MAD로 robust profile 생성
4. train-only RobustScaler로 profile 정규화
5. KMeans로 K=2..24 후보 clustering
6. 최소 cluster slot 수 조건을 만족하는 후보 중 silhouette score가 가장 큰 K 자동 선택
7. validation/test는 이미 정해진 288-slot -> context mapping만 사용

즉 K 선택과 context assignment에 validation label은 사용하지 않습니다.

## Context-specific AE

각 context에 대해 HourAE와 동일한 recipe를 사용합니다.

1. context train points 전체로 1차 AE 학습
2. reconstruction MSE 상위 5% 제거
3. retained subset에서 scaler와 fresh AE 재학습
4. final clean-train MSE mean/std로 z-score calibration
5. z > 3을 기본 anomaly threshold로 사용

기본 구조:

    Input F -> 32 -> 8 -> 32 -> F

## 실행

repository root에서:

    cd ContextAE
    ../.venv/bin/python run_plmn.py --plmn P0480

자동 K 선택 결과와 모든 후보 silhouette score가 콘솔과 JSON에 저장됩니다.

결과:

    data/ib_data/predictions/P0480_context_ae_valid.json

checkpoint:

    ContextAE/model/P0480/paperib/context_ae.pt

## K를 고정해서 비교

예:

    ../.venv/bin/python run_plmn.py --plmn P0480 --k 6

기본 자동 선택 범위:

    --k_min 2
    --k_max 24
    --min_cluster_slots 3

median만 이용해 profile을 만들고 싶으면:

    --median_only_profile

## 주의

현재 버전은 HourAE와 공정하게 비교하기 위해 다음을 그대로 유지합니다.

- stage1 removal 5%
- final reference contamination 0.01%
- threshold mean + 3 sigma
- AE architecture
- epoch / optimizer 설정
- stage-2 scaler 재학습

따라서 이번 결과에서는 **context partition 방식의 차이**를 먼저 해석해야 합니다.

## Test

방법과 threshold를 validation에서 고정한 후에만:

    ../.venv/bin/python run_plmn.py --plmn P0480 --eval_test
