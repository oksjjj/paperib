# ContextAE: 연속 시간구간 기반 K-context AutoEncoder

이 실험은 기존 HourAE의 **고정 24개 시간대 분할**을
TRAIN 데이터에서 자동으로 찾은 **K개의 연속 시간구간**으로 교체합니다.

## 핵심 목적

하루의 288개 5분 슬롯을 일반 KMeans로 섞어서 묶지 않고,
시간 순서 00:00 -> 23:55를 유지한 채 연속된 temporal regime으로 나눕니다.

예:

    Context 0: 00:00 ~ 05:35
    Context 1: 05:40 ~ 09:10
    Context 2: 09:15 ~ 17:45
    Context 3: 17:50 ~ 22:30
    Context 4: 22:35 ~ 23:55

각 구간마다 별도의 AutoEncoder를 학습합니다.

이번 단계에서는 **direction-aware scoring은 아직 포함하지 않습니다.**
고정 24시간 분할을 adaptive temporal segmentation으로 바꾸는 효과만 분리해서 확인합니다.

## Train-only temporal context discovery

1. TRAIN만 사용합니다.
2. 하루를 288개의 5분 time slot으로 표현합니다.
3. 각 slot에 대해 여러 train day의 값을 모읍니다.
4. 각 feature의 across-day median + MAD로 robust slot profile을 만듭니다.
5. profile은 TRAIN-only RobustScaler로 정규화합니다.
6. K=2..24 각각에 대해 연속 구간 partition을 찾습니다.
7. 각 K에서 구간 내부 profile SSE가 최소가 되도록 dynamic programming으로 경계를 최적화합니다.
8. 후보 K 중 TRAIN-only silhouette score가 가장 높은 K를 선택합니다.
9. validation/test는 이미 고정된 slot -> context mapping만 사용합니다.

따라서 다음에는 validation/test label이 사용되지 않습니다.

- K 선택
- 시간 경계 선택
- context assignment
- AE 학습
- decontamination

## 연속성 제약

일반 clustering과 달리 서로 떨어진 시간대는 같은 context가 될 수 없습니다.

각 context는 반드시 다음 형태입니다.

    [start_slot, end_slot]

즉 모든 context는 연속된 시간구간입니다.

현재 버전은 00:00 ~ 23:55의 선형 하루를 먼저 분할합니다.
23:55와 00:00을 하나의 circular segment로 합치는 것은 이후 별도 확장으로 둡니다.

## K 선택

기본값:

    --k_min 2
    --k_max 24
    --min_segment_slots 3

각 K에서:

    objective = sum of within-segment SSE

를 최소화하는 연속 분할을 먼저 찾습니다.

그 후 각 K의 분할 결과에 대해 silhouette score를 계산하고
TRAIN-only 기준으로 최고 silhouette의 K를 선택합니다.

K를 직접 고정할 수도 있습니다.

    ../.venv/bin/python run_plmn.py --plmn P0480 --k 6

이 경우에도 해당 K 안에서 시간 경계는 DP로 자동 결정됩니다.

## Context-specific AE

각 context에 대해서는 기존 HourAE와 같은 학습 recipe를 유지합니다.

1. context train point 전체로 stage-1 AE 학습
2. reconstruction MSE 상위 5% 제거
3. retained subset에서 scaler와 fresh AE 재학습
4. final clean-train MSE mean/std로 score calibration
5. z > 3을 기본 anomaly threshold로 사용

기본 AE:

    Input F -> 32 -> 8 -> 32 -> F

## 실행

repository root:

    cd ContextAE
    ../.venv/bin/python run_plmn.py --plmn P0480

실행하면 다음이 출력됩니다.

    K candidates (TRAIN-only contiguous segmentation):
       K=02 sil=... SSE=...
       K=03 sil=... SSE=...
     * K=05 sil=... SSE=...

    discovered contiguous contexts:
      context=0 00:00 ~ 05:35 ...
      context=1 05:40 ~ 09:10 ...
      ...

결과:

    data/ib_data/predictions/P0480_context_ae_valid.json

checkpoint:

    ContextAE/model/P0480/paperib/context_ae.pt

## Profile ablation

기본은 median + MAD입니다.

median만 사용하려면:

    --median_only_profile

## 현재 고정한 요소

HourAE와의 비교를 위해 다음은 그대로 유지합니다.

- stage1 removal = 5%
- final reference contamination = 0.01%
- threshold = mean + 3 sigma
- AE architecture
- epoch / optimizer
- stage-2 scaler 재학습

따라서 이번 실험에서는 우선 **시간 context를 어떻게 나누는가**의 효과를 해석합니다.

## Test

방법과 threshold를 validation에서 고정한 뒤에만:

    ../.venv/bin/python run_plmn.py --plmn P0480 --eval_test
