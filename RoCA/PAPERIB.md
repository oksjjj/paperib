# RoCA × paperib 학습 가이드

> 실제 구현 위치: `models/roca/`  
> `RoCA/` 아래 Python 파일은 이전 import/실행 경로를 위한 호환 wrapper입니다.

이 디렉터리는 공개된 RoCA 구현을 paperib의 인바운드 로밍 데이터에 적용하기 위한 코드다.

원본 구현은 `ruiking04/RoCA`이며 MIT License를 따른다. 원본 출처는 `official/ORIGIN.txt`, 라이선스는 `official/LICENSE`에서 확인할 수 있다.

이 저장소에서는 원본의 핵심 network, loss, augmentation 로직을 `RoCA/official/`에 유지하고, paperib 데이터 로딩과 평가를 연결하는 wrapper를 별도로 작성했다.

## 1. 먼저 볼 파일

추천 순서는 다음과 같다.

```text
models/roca/config.py
  ↓
models/roca/dataset.py
  ↓
models/roca/official/augmentations.py
  ↓
models/roca/official/network.py
  ↓
models/roca/official/loss.py
  ↓
models/roca/run_plmn.py
  ↓
export_predictions.py
```

처음부터 `models/roca/run_plmn.py`만 읽기보다 입력 window가 어떤 shape으로 바뀌고, backbone과 loss가 어떤 값을 만드는지 먼저 이해하는 편이 쉽다.

## 2. 전체 흐름

```text
5분 단위 다변량 시계열 (T, F)
  ↓
TRAIN-only MinMaxScaler
  ↓
sliding window 생성
  ↓
(N, F, window) 형태로 변환
  ↓
원본 / weak augmentation / strong augmentation
  ↓
CNN + LSTM Encoder/Decoder
  ↓
project / rec_project
  ↓
hypersphere center와 cosine distance 계산
  ↓
loss_n
  ↓
warmup 이후 score 상위 nu 비율을 잠재 anomaly로 분리
  ↓
contaminated-training loss
  ↓
학습 완료
  ↓
stride=1로 dense scoring
  ↓
TRAIN score quantile threshold
  ↓
VALID label과 평가
```

## 3. 입력 전처리와 MinMax 정규화

RoCA는 `common/data.py::prepare_plmn()`을 통해 데이터를 준비한다.

먼저 11개 core feature를 가져오고 다음 처리를 수행한다.

```text
NaN  → 0
+inf → 0
-inf → 0
```

그 다음 시간순으로:

```text
TRAIN 60%
VALID 20%
TEST  20%
```

로 분할한다.

이후 **TRAIN 데이터에만** `MinMaxScaler`를 fit한다.

feature별 변환은 개념적으로 다음과 같다.

$
x_f'
=
\frac{x_f-x_{f,\min}^{train}}
{x_{f,\max}^{train}-x_{f,\min}^{train}}
$

VALID와 TEST에는 TRAIN에서 이미 fit된 같은 scaler의 `transform()`만 적용한다.

따라서 validation/test의 min/max 정보가 train 정규화에 사용되지 않는다.

또한 기본 MinMaxScaler는 결과를 [0,1]에 강제로 clip하지 않으므로,
VALID/TEST 값이 TRAIN 범위를 넘어가면 0보다 작거나 1보다 큰 scaled 값이 나올 수 있다.

이 정규화가 끝난 뒤에 sliding window를 생성한다.

```text
raw feature
→ TRAIN-only MinMax scaling
→ window 생성
→ RoCA backbone
```

---

## 4. 입력 window

기본 설정:

```text
window_size = 32
time_step = 16
```

5분 데이터이므로 window 32개는 약 160분 구간이다.

학습 시 첫 window가 0번 point에서 시작했다면 다음 window는 16 point 뒤에서 시작한다.

```text
Window 0: 0 ~ 31
Window 1: 16 ~ 47
Window 2: 32 ~ 63
...
```

관련 코드:

```python
models/roca/dataset.py
subsequences()
block_to_nct()
```

원래 DataFrame/NumPy 시계열 shape는:

```text
(T, F)
```

이고 window를 만든 직후:

```text
(N, window, F)
```

가 된다.

Conv1d 입력을 위해 축을 바꾸면:

```text
(N, F, window)
```

가 된다.

여기서 F는 현재 paperib core feature 수인 11이다.

## 5. augmentation

파일:

```text
models/roca/official/augmentations.py
```

현재 실제 실행에서 사용하는 증강은 두 가지다.

```text
weak augmentation  = scaling
strong augmentation = jitter
```

`RoCAWindowDataset`의 한 item은 다음 네 값을 반환한다.

```text
원본 window
dummy label
weak augmented window
strong augmented window
```

학습 loop에서는 원본/weak/strong을 batch 축으로 합친다.

```python
all_data = torch.cat((data, aug1, aug2), dim=0)
```

따라서 같은 backbone이 세 종류의 view를 모두 처리한다.

## 6. network 구조

파일:

```text
models/roca/official/network.py
```

큰 흐름은 다음과 같다.

```text
(B, F, T)
  ↓
Conv1d
  ↓
Conv1d
  ↓
Conv1d
  ↓
(B, C, T')
  ↓
축 변경
  ↓
(B, T', C)
  ↓
LSTM Encoder
  ↓
LSTM Decoder
  ↓
원본 hidden sequence / 복원 output sequence
  ↓
Flatten
  ↓
Projection Head
  ↓
project / rec_project
```

중요한 점은 일반 AutoEncoder처럼 원본 feature 공간의 reconstruction MSE를 직접 anomaly score로 쓰지 않는다는 것이다.

RoCA는 projection space에서 다음 두 vector를 만든다.

```text
project
rec_project
```

그리고 이들이 hypersphere center에서 얼마나 떨어져 있는지를 anomaly score 계산에 사용한다.

## 7. hypersphere center

파일:

```text
models/roca/official/loss.py
```

함수:

```python
center_c()
```

TRAIN의 원본/증강 representation을 forward한 뒤 평균하여 center를 만든다.

개념적으로:

$$
c = \frac{1}{N}\sum_i z_i
$$

와 비슷하게 이해하면 된다.

실제 코드에서는 원본 representation과 decoder representation을 모두 합쳐 center를 계산한다.

이 center는 "정상 representation이 모여야 하는 중심" 역할을 한다.

## 8. loss_n

RoCA에서 가장 중요한 값이다.

`roca_loss()`에서 먼저 representation을 L2 normalize한다.

그리고 center와의 cosine distance를 계산한다.

$$
d(z,c)=1-\cos(z,c)
$$

원본 representation과 reconstruction representation 각각에 대해 거리를 구한 뒤 더한다.

$$
loss_n
=
d(z,c)
+
d(\hat z,c)
$$

따라서:

```text
loss_n 작음
→ center에 가까움
→ normal에 가까움

loss_n 큼
→ center에서 멂
→ anomaly에 가까움
```

paperib에서 RoCA anomaly score로 저장하는 값이 바로 이 `loss_n`이다.

## 9. warmup

기본값:

```text
warmup = 10
```

warmup 동안에는 train에 anomaly가 섞여 있을 수 있다는 것을 아직 고려하지 않고 모든 sample에:

```text
loss_n을 작게 만들기
```

를 적용한다.

즉 모든 sample을 우선 center 주변으로 모은다.

## 10. contaminated training

warmup 이후 `train_method == "contaminated"`이면 한 batch의 sample을 현재 score에 따라 나눈다.

기본값:

```text
nu = 0.01
```

즉 score가 가장 높은 약 1%를 잠재 anomaly candidate로 본다.

```text
낮은 score 99%
→ normal candidate

높은 score 1%
→ anomaly candidate
```

이때 실제 anomaly label은 사용하지 않는다.

코드:

```python
idx_n = score가 낮은 n_n개
idx_a = score가 높은 n_a개
```

normal candidate에는:

$$
loss_n
$$

을 최소화한다.

잠재 anomaly candidate에는:

$$
loss_a = 4-loss_n
$$

을 적용한다.

`loss_a`를 최소화하면 반대로 `loss_n`은 커지는 방향이므로 anomaly candidate가 center에서 멀어지도록 유도한다.

최종 one-class loss:

$$
L_{OC}
=
\operatorname{mean}
\left(
loss_n(normal)
\cup
\mu\,loss_a(anomaly)
\right)
$$

여기서 기본:

```text
mu = 0.85
```

다.

이 부분이 RoCA가 contaminated train을 직접 다루는 핵심이다.

## 11. variance regularizer

`loss_sigam`은 representation의 각 차원 분산이 지나치게 작아지는 것을 막는다.

즉 모든 sample representation이 하나의 동일한 점으로 붕괴하는 것을 방지하기 위한 항이다.

최종 학습 loss는:

$$
L = L_{OC} + \omega L_{sigma}
$$

기본값:

```text
omega = 0.02
```

## 12. 학습과 평가의 stride 차이

학습 시 기본:

```text
time_step = 16
```

이지만 평가 시에는:

```text
time_step = 1
```

로 다시 score를 계산한다.

이유는 VUS-PR 등 모델 비교를 위해 가능한 모든 시점에 촘촘하게 anomaly score를 부여하기 위해서다.

따라서:

```text
학습 효율을 위한 sparse windows
≠
최종 평가용 dense score series
```

다.

## 13. window score를 point score로 바꾸는 방법

한 window에서 `loss_n` 하나가 나온다.

이 score를 해당 window의 마지막 point에 기록한다.

예:

```text
window = t=10 ... 41
score = 0.8
→ point 41의 score = 0.8
```

함수:

```python
score_block()
```

초기 `window_size-1` point는 완전한 window가 없기 때문에 NaN이 된다.

## 14. threshold

현재 paperib wrapper의 기본 RoCA threshold는 TRAIN score quantile이다.

기본:

```text
q = 0.99
```

즉:

$$
\tau = Q_{0.99}(S_{train})
$$

validation label을 이용해 threshold를 고르는 것이 아니다.

함수:

```python
train_quantile_threshold()
```

## 15. models/roca/run_plmn.py 읽는 순서

```text
_load_prepare_plmn()
_device()
train_quantile_threshold()
score_block()
train_and_eval()
main()
```

특히 `train_and_eval()` 안에서는 다음 블록을 구분해서 읽으면 된다.

```text
1. 데이터 준비
2. config 덮어쓰기
3. train window 생성
4. Dataset / DataLoader 생성
5. model / optimizer / center 생성
6. epoch 학습
7. dense train/valid scoring
8. train quantile threshold
9. validation 평가
10. checkpoint 저장
11. UI JSON 저장
```

## 16. ContextAE와 비교

RoCA:

```text
긴 window 입력
→ CNN/LSTM representation
→ hypersphere center 기반 score
→ contaminated loss 안에서 잠재 anomaly 분리
```

ContextAE:

```text
한 시점의 11개 feature 입력
→ 시간 context별 별도 AE
→ reconstruction MSE
→ context별 Stage-1 상위 5% 제거
→ fresh AE 재학습
```

즉 두 모델은 contaminated train을 다루는 방식 자체가 다르다.

RoCA는 학습 loss 안에서 잠재 anomaly를 구분하고,
ContextAE는 Stage-1 reconstruction error로 sample을 실제 train set에서 제거한 뒤 재학습한다.

## 17. 공부하면서 반드시 답할 수 있어야 하는 질문

- RoCA 입력 shape가 왜 (B, C, T)인가?
- window_size와 time_step의 차이는 무엇인가?
- train_y가 모두 0인 이유는 무엇인가?
- weak augmentation과 strong augmentation은 무엇인가?
- CNN과 LSTM이 각각 어떤 역할을 하는가?
- project와 rec_project는 무엇인가?
- center_c()는 무엇을 계산하는가?
- loss_n이 왜 anomaly score가 되는가?
- warmup 동안에는 왜 contaminated 분리를 하지 않는가?
- nu=0.01은 정확히 무엇을 의미하는가?
- idx_n과 idx_a는 실제 label로 나누는가?
- loss_a = 4 - loss_n을 최소화하면 왜 anomaly가 center에서 멀어지는가?
- loss_sigam은 왜 필요한가?
- 학습 stride는 16인데 평가 stride는 왜 1인가?
- window score를 왜 window 마지막 point에 붙이는가?
- threshold는 validation label을 이용해서 계산하는가?
- ContextAE의 decontamination과 RoCA의 contaminated loss는 무엇이 다른가?

## 18. 실행

한 사업자:

```bash
.venv/bin/python -m models.roca.run_plmn --plmn P0480 --epochs 20
```

결과:

```text
Prediction:
data/ib_data/predictions/P0480_roca.json

Checkpoint:
RoCA/model/P0480/paperib/roca.pt
```

UI에서는 prediction source를 RoCA로 선택하여 score와 예측 구간을 확인할 수 있다.

## 19. 기본 설정 요약

```text
window_size = 32
train time_step = 16
evaluation time_step = 1
epochs = 20
train_method = contaminated
warmup = 10
nu = 0.01
mu = 0.85
threshold = TRAIN score q0.99
```

최종적으로 RoCA를 이해할 때 가장 중요한 연결은 다음이다.

```text
window
→ augmentation
→ CNN/LSTM
→ project / rec_project
→ center distance
→ loss_n
→ warmup 후 top-nu 잠재 anomaly 분리
→ contaminated loss
→ dense anomaly score
→ TRAIN quantile threshold
→ VALID 평가
```
