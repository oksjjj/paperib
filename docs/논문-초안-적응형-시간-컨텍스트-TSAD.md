# 오염된 비지도 다변량 시계열을 위한 적응형 시간 컨텍스트 기반 이상 탐지

**영문 제목(안):** Adaptive Temporal Context-Based Anomaly Detection for Unlabeled Contaminated Multivariate Time Series

> 본 문서는 현재까지 확정된 연구 방향과 P0480 development 결과를 반영한 석사논문 초안이다.  
> 나머지 9개 사업자에 대한 전문가 레이블링과 최종 10개 사업자 실험은 진행 예정이며, 해당 결과가 필요한 부분은 `[추후 작성]`으로 표시하였다.

---

# 초록

실제 이동통신 운영 환경의 다변량 시계열에서는 학습 데이터에 이상 구간이 포함되어 있음에도 이상 레이블을 학습 단계에서 사용할 수 없는 경우가 많다. 또한 정상 상태의 분포가 하루 전체에서 일정하지 않고 시간대에 따라 달라질 수 있으며, 일부 지표는 값의 증가와 감소가 모두 중요하지만 다른 지표는 특정 방향의 변화만 운영상 이상으로 해석된다. 이러한 특성은 clean-normal training data를 가정하거나 하나의 전역 정상 분포를 학습하는 기존 시계열 이상 탐지 방법의 적용을 어렵게 한다.

본 연구는 anomaly label을 사용하지 않는 contaminated training 환경에서 시간에 따른 정상성의 변화를 자동으로 발견하고, 각 시간 컨텍스트에 적합한 정상 모델을 학습하는 적응형 시간 컨텍스트 기반 이상 탐지 방법을 제안한다. 먼저 학습 데이터의 각 5분 time slot에 대해 여러 날짜의 관측치를 이용하여 median과 MAD 기반의 robust temporal profile을 구성한다. 이후 시간 순서를 유지하는 연속 분할을 적용하여 후보 컨텍스트 수 $K$별 최적 경계를 탐색하고, 학습 데이터만을 이용한 silhouette score를 기준으로 $K$를 선택한다. 각 컨텍스트에서는 AutoEncoder를 이용해 정상 패턴을 모델링하며, 1차 학습 후 높은 reconstruction error를 갖는 학습 샘플을 제거하고 새로운 AutoEncoder를 재학습하는 iterative decontamination을 적용한다.

개발용 사업자 P0480의 validation set에서 제안한 적응형 컨텍스트 모델은 고정된 24개 시간대별 AutoEncoder와 비교하여 VUS-PR을 0.1202에서 0.1662로 향상시키고, PR-AUC를 0.4456에서 0.4638로 향상시켰다. 동시에 false alarms per day는 24.51에서 12.79로 감소하였다. 반면 Event Recall은 0.667에서 0.500으로 감소하여, 시간 컨텍스트의 세분화 정도와 정상 모델의 안정성 사이에 trade-off가 존재함을 확인하였다.

최종 연구에서는 서로 다른 국가의 10개 인바운드 로밍 사업자를 대상으로 전문가가 운영 관점의 anomaly interval을 레이블링하고, Global AE, 고정 24시간 모델, 적응형 시간 컨텍스트 모델 및 decontamination ablation을 동일한 평가 프로토콜에서 비교한다. 또한 feature별 이상 방향성을 고려하는 direction-aware anomaly scoring을 확장 모듈로 검증할 예정이다.

**주요어:** Time Series Anomaly Detection, Multivariate Time Series, Contaminated Training, Temporal Context, AutoEncoder, Decontamination, Inbound Roaming

---

# 1. 서론

## 1.1 연구 배경

시계열 이상 탐지(Time-Series Anomaly Detection, TSAD)는 시스템 장애, 품질 저하, 센서 이상, 네트워크 장애와 같이 시간에 따라 발생하는 비정상 상태를 자동으로 탐지하는 문제이다. 이동통신 네트워크에서는 다수의 성능 지표와 제어면 통계가 지속적으로 수집되기 때문에, 사람이 모든 사업자와 모든 지표를 실시간으로 확인하기 어렵다. 이에 따라 정상 패턴을 자동으로 학습하고 비정상 구간을 탐지하는 TSAD는 실제 운영 관점에서 중요한 기술이다.

그러나 실제 운영 데이터는 공개 benchmark와 다른 특성을 가진다. 첫째, 학습 구간 전체가 정상이라고 보장하기 어렵다. 장애 또는 품질 이상이 과거 학습 데이터에 이미 포함되어 있을 수 있으며, 모든 학습 데이터를 전문가가 사전에 레이블링하는 것은 비용이 크다. 따라서 실제 환경에서는 다음과 같은 contaminated training setting이 자연스럽다.

$$
p_{\mathrm{train}}(X)
=
(1-\epsilon)p_N(X)+\epsilon p_A(X)
$$

여기서 $p_N$은 정상 분포, $p_A$는 이상 분포이며, $\epsilon$은 학습 데이터에 포함된 anomaly contamination 비율이다.

둘째, 정상 상태 자체가 하나의 고정된 분포를 따르지 않을 수 있다. 특히 로밍 트래픽은 시간대에 따라 트래픽 규모와 실패/성공 패턴이 달라질 수 있다. 따라서 하나의 전역 분포 $p(X)$를 학습하는 것보다 시간 컨텍스트 $C$를 조건으로 하는 다음 분포가 실제 정상성을 더 잘 표현할 수 있다.

$$
p_N(X\mid C)
$$

셋째, 통계적으로 드문 변화가 모두 운영상 중요한 이상은 아니다. 일부 지표에서는 증가와 감소가 모두 중요하지만, 다른 지표에서는 특정 방향의 변화만 장애 또는 품질 저하와 관련될 수 있다. 따라서 단순한 reconstruction error만으로는 운영자의 판단 기준과 모델의 anomaly score 사이에 차이가 발생할 수 있다.

본 연구는 이러한 실제 이동통신 데이터의 특성을 바탕으로, **anomaly label을 사용하지 않는 contaminated training 환경에서 시간에 따라 변화하는 정상성을 적응적으로 학습하는 방법**을 연구한다.

## 1.2 기존 접근의 한계와 연구 동기

본 연구 초기에는 contaminated training에 강건한 기존 TSAD 방법을 실제 인바운드 로밍 데이터에 적용하였다. 그러나 개발용 사업자 P0480에서 기존 방법은 일부 anomaly를 높은 점수로 평가했지만 많은 false alarm을 생성하거나, 일부 운영상 중요한 anomaly를 낮은 순위로 평가하는 문제가 관찰되었다.

추가적인 분석을 통해 단순히 더 복잡한 sequence model을 적용하는 것보다 정상 상태의 시간적 이질성을 먼저 분리하는 것이 중요할 가능성을 확인하였다. 실제로 하루를 24개의 시간대로 고정 분할하여 각 시간대별 AutoEncoder를 학습한 HourAE는 기존 실험들보다 높은 anomaly ranking 성능을 보였다. 이는 하나의 전역 정상 모델보다 시간대별 정상 모델이 실제 데이터에 더 적합할 수 있음을 시사한다.

특히 P0480의 초기 실험에서는 긴 시간 구간의 패턴과 temporal dependency를 직접 모델링하는 window/sequence 기반 TSAD보다, 각 시점의 다변량 관측값이 해당 시간대의 정상 상태에서 얼마나 벗어났는지를 평가하는 point-wise 방식이 operational anomaly와 더 잘 정렬될 가능성이 관찰되었다. 본 연구에서 point-wise 방식은 시간 정보를 사용하지 않는 독립적인 point anomaly detection을 의미하지 않는다. 시간 $t$의 관측값 $x_t$를 그대로 평가하되, 해당 시점이 속한 temporal context $c_t$를 조건으로 정상 분포를 정의하는 **context-aware point-wise TSAD**를 의미한다.

즉 기존의 window 기반 접근이 다음과 같이 긴 시퀀스를 입력으로 사용하는 반면,

$$
[x_{t-L+1},\ldots,x_t]
\rightarrow
\text{Temporal Model}
\rightarrow
s_t
$$

본 연구는 다음과 같이 현재 시점의 다변량 상태와 시간 컨텍스트의 관계를 중심으로 anomaly score를 정의한다.

$$
s_t=f(x_t\mid c_t)
$$

따라서 본 연구의 가설은 SKT 인바운드 로밍 데이터에서 모든 이상이 복잡한 subsequence pattern으로 나타난다고 가정하기보다, **시간 컨텍스트에 따라 달라지는 정상 상태로부터의 다변량 point deviation이 더 중요한 이상 단서일 수 있다**는 것이다. 현재 이 관찰은 P0480 development 결과에 기반한 가설이며, 최종적으로는 10개 사업자 실험을 통해 일반적으로 성립하는지 검증한다.

하지만 고정된 24개 시간 분할에는 두 가지 문제가 있다. 첫째, 1시간이라는 구간은 데이터로부터 얻어진 것이 아니라 사람이 임의로 정한 단위이다. 둘째, 각 시간대의 학습 샘플 수가 감소하면서 정상 분포 추정이 불안정해질 수 있다. 이에 본 연구에서는 정상 패턴이 실제로 변하는 지점을 학습 데이터로부터 탐색하여, 필요한 수만큼의 연속적인 temporal context를 자동으로 구성하는 방향으로 연구를 확장하였다.

## 1.3 연구 문제

본 연구의 핵심 문제는 다음과 같이 정의한다.

$$
\boxed{
\text{Unlabeled Contaminated Training}
+
\text{Context-dependent Normality}
}
$$

사업자 $j$의 시간 $t$에서 관측되는 $F$차원 다변량 시계열을

$$
x_t\in\mathbb{R}^{F}
$$

라고 하자. 시간 $t$가 속하는 temporal context를 $c_t$라고 하면 학습 데이터는 다음과 같은 conditional contamination model로 표현할 수 있다.

$$
p_{\mathrm{train}}(x\mid c)
=
(1-\epsilon_c)p_N(x\mid c)
+
\epsilon_c p_A(x\mid c)
$$

본 연구의 목표는 anomaly label을 사용하지 않고 다음을 동시에 추정하는 것이다.

1. 시간에 따라 달라지는 정상 context
2. 각 context에서의 정상 데이터 분포
3. contaminated training data에서 정상 모델 학습을 방해하는 high-anomaly candidate

최종적으로 다음과 같은 anomaly score를 생성하는 것을 목표로 한다.

$$
s_t = f(x_t,c_t)
$$

여기서 $c_t$ 역시 사람이 고정적으로 정의하는 것이 아니라 학습 데이터에서 자동으로 결정한다.

## 1.4 연구 질문

본 연구는 다음 연구 질문에 답하고자 한다.

**RQ1.** 긴 window/sequence의 temporal dependency를 직접 모델링하는 방식과 비교했을 때, 시간 컨텍스트에 따른 정상 상태로부터의 point-wise deviation을 모델링하는 방식이 인바운드 로밍 이상 탐지에 더 적합한가?

**RQ2.** 하나의 전역 정상 모델보다 시간 컨텍스트를 분리한 정상 모델이 실제 이동통신 시계열 이상 탐지에 유리한가?

**RQ3.** 고정된 24시간 분할보다 학습 데이터에서 자동으로 발견한 적응형 temporal context가 anomaly ranking과 false alarm 측면에서 유리한가?

**RQ4.** 각 temporal context 내부에서 iterative decontamination을 수행하면 contaminated training의 영향을 줄일 수 있는가?

**RQ5.** 제안 방법의 효과가 특정 개발용 사업자에만 국한되지 않고 서로 다른 10개 로밍 사업자에서 반복적으로 나타나는가?

**RQ6.** feature별 운영 의미에 따른 방향성을 anomaly scoring에 반영할 경우 false alarm을 추가로 감소시킬 수 있는가? `[확장 실험 예정]`

## 1.5 연구의 기여

본 연구의 기여는 다음 세 가지로 정리한다.

1. **Data-driven Adaptive Temporal Context Discovery**  
   고정된 시간 단위를 사전에 설정하지 않고, anomaly label을 사용하지 않은 contaminated train data의 robust temporal profile로부터 정상 패턴이 유사한 연속 시간 구간과 context 수 $K$를 자동으로 결정한다.

2. **Context-wise Iterative Decontamination**  
   각 temporal context에서 1차 정상 모델을 학습한 뒤 높은 anomaly score를 갖는 train sample을 제거하고 새로운 정상 모델을 재학습함으로써, contaminated training 환경에서 label 없이 정상 모델을 구성한다.

3. **Operational Evaluation on Real Telecom Data**  
   서로 다른 10개 인바운드 로밍 사업자를 대상으로 전문가가 작성한 operational ground truth를 이용하여 VUS-PR, PR-AUC, Event-F1, Event Recall, false alarms per day 등 ranking 성능과 실제 alert burden을 함께 평가한다.

추가적으로 feature별 이상 방향성을 반영하는 direction-aware scoring을 확장하여 operational false alarm 감소 효과를 검증할 예정이다.

---

# 2. 관련 연구

본 연구의 관련 연구는 단순히 기존 TSAD 모델을 나열하는 방식이 아니라, 현재 연구의 문제 설정과 방법 선택을 뒷받침하는 관점으로 구성한다. 구체적으로 (1) TSAD의 대표 접근, (2) contaminated training, (3) temporal context와 seasonality, (4) point-wise와 subsequence 기반 이상 탐지의 차이, (5) robust statistics와 decontamination, (6) TSAD 평가 지표, (7) 실제 통신 운영 환경과 전문가 ground truth의 일곱 관점에서 선행연구를 검토한다.

## 2.1 시계열 이상 탐지의 대표 접근

TSAD 방법은 reconstruction-based, prediction-based, density/proximity-based, representation learning 기반 방법 등으로 구분할 수 있다. Reconstruction-based 방법은 정상 패턴을 복원하도록 모델을 학습한 뒤 입력과 복원값의 차이를 anomaly score로 사용하며, prediction-based 방법은 과거 구간으로부터 미래 값을 예측한 뒤 prediction error를 이용한다. 최근에는 attention, graph neural network, Transformer 등으로 긴 temporal dependency와 변수 간 관계를 함께 모델링하는 방법들이 제안되고 있다.

본 연구에서는 새로운 deep backbone 자체를 제안하는 것을 목표로 하지 않는다. 오히려 실제 인바운드 로밍 데이터에서 긴 sequence를 복잡하게 모델링하는 것이 항상 operational anomaly와의 정렬을 보장하지 않는다는 preliminary observation에서 출발한다.

따라서 이 절에서는 다음 질문을 중심으로 대표 문헌을 검토한다.

- 기존 TSAD는 어떤 단위(point, window, subsequence)로 anomaly를 정의하는가?
- reconstruction, prediction, density 기반 방법은 각각 어떤 가정을 갖는가?
- 긴 temporal dependency를 모델링하는 것이 어떤 상황에서 유리한가?
- 실제 benchmark에서 복잡한 deep model이 항상 우수한가?

현재 TSB-AD benchmark는 다양한 TSAD 알고리즘과 평가 환경을 통합적으로 비교하였으며, 복잡한 모델이 모든 데이터셋에서 우수한 것은 아니라는 점과 평가 프로토콜의 중요성을 강조한다 [4].

**추가 문헌 검토 예정:** 대표 reconstruction-based, prediction-based, Transformer/attention 기반 TSAD 4~6편.

## 2.2 Contaminated Training을 고려한 TSAD

많은 비지도 TSAD 방법은 학습 데이터가 대부분 정상이라는 가정을 사용한다. 그러나 실제 운영 데이터에서는 과거 장애나 품질 저하가 train period에 이미 포함되어 있을 수 있으며, 모든 학습 구간을 전문가가 사전에 레이블링하고 제거하는 것은 현실적으로 어렵다.

본 연구는 다음과 같은 contaminated training setting을 기본 문제로 둔다.

$$
p_{\mathrm{train}}(X)
=
(1-\epsilon)p_N(X)
+
\epsilon p_A(X)
$$

Ho와 Armanfard는 contaminated multivariate time series에서 anomaly label 없이 학습하는 TSAD-C를 제안하였다 [1]. TSAD-C는 decontamination과 spatio-temporal modeling을 결합하여 contaminated training 자체를 독립적인 문제로 다룬다.

본 연구 역시 train anomaly label을 사용하지 않는다는 점에서 동일한 현실적 문제를 다루지만, 핵심 관심은 복잡한 backbone보다 **시간에 따라 달라지는 정상 분포를 어떤 context로 분리하고 학습할 것인가**에 있다.

이 절에서는 다음을 검토한다.

- clean-normal training assumption의 한계
- contaminated training에서 성능이 저하되는 원인
- label-free decontamination 및 robust training 방법
- iterative filtering 또는 sample reweighting 방식
- contamination ratio에 대한 민감도

**추가 문헌 검토 예정:** RoCA, RiAD 및 contaminated-training/robust TSAD 관련 4~6편.

## 2.3 Temporal Context, Seasonality와 Context-dependent Normality

시계열의 정상 분포는 하루 전체에서 동일하지 않을 수 있다. 특히 통신 트래픽은 시간대, 요일, 생활 패턴 등의 영향을 받아 일정한 주기성을 보이며, 동일한 값이라도 어느 시간대에 발생했는지에 따라 정상 또는 이상으로 해석될 수 있다.

이를 확률적으로 표현하면 전역 정상 분포

$$
p_N(x)
$$

보다 context를 조건으로 하는 분포

$$
p_N(x\mid c)
$$

가 더 적합할 수 있다.

Porcelli 등은 seasonal time series에서 timestamp-derived condition을 이용하는 conditional VAE 기반 이상 탐지 방법을 제안하였다 [2]. hour-of-day 또는 day-of-week와 같은 시간 정보를 condition으로 사용한다는 점에서 본 연구와 관련이 있다.

그러나 기존 방법들이 사전에 정의된 hour/day category를 사용하는 경우가 많은 반면, 본 연구는 다음과 같이 context 자체를 train data에서 자동으로 발견하는 것을 목표로 한다.

$$
\boxed{
\text{Predefined Temporal Context}
\rightarrow
\text{Data-driven Adaptive Temporal Context}
}
$$

이 절에서는 다음을 검토한다.

- seasonality를 anomaly detection에 반영하는 방법
- hour-of-day, day-of-week 등을 condition으로 사용하는 방법
- regime switching 또는 context-dependent normality
- temporal segmentation 및 change/regime discovery
- 고정 context와 adaptive context의 차이

**추가 문헌 검토 예정:** seasonal anomaly detection, conditional AE/VAE, temporal regime/context discovery 관련 4~6편.

## 2.4 Point-wise와 Window/Subsequence 기반 이상 탐지

TSAD에서는 anomaly를 개별 시점의 이상으로 볼 것인지, 일정 길이의 subsequence 또는 range anomaly로 볼 것인지에 따라 문제 정의가 달라진다.

window/sequence 기반 방법은 일반적으로

$$
[x_{t-L+1},\ldots,x_t]
\rightarrow
\text{Temporal Model}
\rightarrow
s_t
$$

와 같이 연속된 관측 구간의 temporal dependency를 모델링한다.

반면 본 연구에서 검토하는 context-aware point-wise 방식은 현재 시점의 다변량 상태를 해당 시간 context의 정상 상태와 비교한다.

$$
s_t=f(x_t\mid c_t)
$$

중요한 점은 본 연구의 point-wise 방식이 시간을 무시하는 정적 anomaly detection이 아니라는 것이다. 시간 정보는 temporal context $c_t$의 형태로 명시적으로 사용된다.

P0480의 preliminary experiment에서는 sequence/window 기반 모델보다 context-aware point-wise reconstruction이 operational GT anomaly의 ranking과 정렬에 더 유리할 가능성이 관찰되었다. 그러나 이는 아직 하나의 development operator에 대한 관찰이므로, 10개 사업자 실험을 통해 검증해야 할 가설로 둔다.

이 절에서는 다음을 검토한다.

- point anomaly, contextual anomaly, collective/subsequence anomaly의 정의
- point-wise와 subsequence 기반 evaluation의 차이
- contextual anomaly detection의 개념
- 긴 temporal dependency가 반드시 필요한 경우와 그렇지 않은 경우

**추가 문헌 검토 예정:** point/contextual/subsequence anomaly 정의 및 비교 관련 3~5편.

## 2.5 Robust Statistics와 Decontamination

본 연구에서는 contaminated train data에서 anomaly의 영향을 줄이기 위해 median, MAD, high-score filtering 및 retraining을 사용한다.

평균과 표준편차는 극단값에 민감할 수 있으므로 temporal profile을 구성할 때 robust location/scale estimator인 median과 MAD를 사용한다.

$$
m_{s,f}
=
\operatorname{median}_{d}(x_{d,s,f})
$$

$$
MAD_{s,f}
=
\operatorname{median}_{d}
\left|
x_{d,s,f}-m_{s,f}
\right|
$$

또한 각 temporal context에서 1차 AutoEncoder를 학습한 후 reconstruction error가 높은 train sample을 제거하고 fresh model을 다시 학습한다.

$$
D_k
\rightarrow
AE_k^{(1)}
\rightarrow
\text{High-score filtering}
\rightarrow
D'_k
\rightarrow
AE_k^{(2)}
$$

이 절에서는 다음을 검토한다.

- median/MAD와 robust estimation의 이론적 배경
- trimming 또는 high-loss sample filtering
- iterative cleaning/retraining
- reconstruction error 기반 decontamination
- robust loss, reweighting 방식과의 차이

**추가 문헌 검토 예정:** robust statistics 및 iterative decontamination 관련 3~5편.

## 2.6 시계열 이상 탐지 평가 지표

TSAD에서는 anomaly가 단일 point뿐 아니라 일정 시간 동안 지속되는 range 형태로 발생할 수 있기 때문에, 평가 지표 선택이 결과 해석에 큰 영향을 미친다.

### 2.6.1 Point-wise metric과 Point Adjustment

기본적인 Precision, Recall, F1은 각 point를 독립적으로 평가한다. 그러나 range anomaly에서 일부 point만 탐지하더라도 anomaly event 자체를 탐지했다는 의미가 있을 수 있기 때문에, 기존 연구에서는 Point Adjustment(PA)가 널리 사용되어 왔다.

하지만 PA는 anomaly range 내 한 점만 탐지해도 전체 range를 탐지한 것으로 보정하기 때문에 모델의 성능을 과대평가할 가능성이 있다. 따라서 본 연구에서는 PA-F1을 primary metric으로 사용하지 않는다.

**추가 문헌 검토 예정:** Point Adjustment의 한계와 평가 왜곡을 분석한 논문 2~3편.

### 2.6.2 Range/Event 기반 평가

실제 운영에서는 anomaly point 하나보다 anomaly event를 탐지했는지가 중요하다. 따라서 range-based Precision/Recall, event-level F1, affiliation metric 등 point 독립 평가를 보완하기 위한 평가 방법들이 제안되어 왔다.

본 연구에서는 GT anomaly interval과 predicted anomaly segment의 overlap을 기준으로 Event Recall과 Event-F1을 함께 보고한다. 또한 운영자가 실제로 확인해야 하는 잘못된 alert 수를 표현하기 위해 FA/day를 사용한다.

**추가 문헌 검토 예정:** range-based precision/recall, affiliation, event-level metric 관련 2~4편.

### 2.6.3 VUS 기반 평가

Boniol 등은 range anomaly에서 tolerance window에 따라 metric이 달라지는 문제를 완화하기 위해 Volume Under the Surface(VUS)를 제안하였다 [3]. VUS는 여러 tolerance 범위에 대한 성능을 통합하여 threshold 또는 단일 tolerance 값에 지나치게 의존하지 않는 평가를 가능하게 한다.

본 연구에서는 다음을 primary ranking metric으로 사용한다.

$$
\boxed{VUS\text{-}PR}
$$

특히 anomaly class imbalance가 매우 큰 실제 로밍 데이터에서는 ROC 계열보다 Precision-Recall 기반 지표가 실제 anomaly ranking의 품질을 해석하는 데 더 적합할 수 있다.

TSB-AD 역시 다양한 TSAD metric을 비교하며 VUS-PR을 신뢰성 있는 평가 지표 중 하나로 사용한다 [4].

**추가 문헌 검토 예정:** VUS 원 논문, range-aware AUC/VUS 후속 연구 및 benchmark 연구.

### 2.6.4 PR-AUC와 Class Imbalance

본 연구 데이터에서는 anomaly point의 비율이 매우 낮기 때문에 AUROC만으로 성능을 평가하면 많은 true negative의 영향으로 실제 탐지 품질이 과도하게 좋아 보일 수 있다.

PR-AUC는 anomaly class를 중심으로 Precision과 Recall 사이의 trade-off를 평가하기 때문에 severe class imbalance 상황에서 중요한 보조 ranking metric으로 사용한다.

이 절에서는 다음을 검토한다.

- ROC-AUC와 PR-AUC의 차이
- severe imbalance에서 PR curve가 갖는 장점
- random PR-AUC baseline과 anomaly prevalence의 관계

**추가 문헌 검토 예정:** ROC vs PR curve, imbalanced anomaly detection evaluation 관련 2~3편.

## 2.7 실제 통신 운영 환경과 전문가 Ground Truth

실제 이동통신 환경에서 통계적으로 희귀한 변화가 모두 운영상 중요한 anomaly는 아니다.

$$
\text{Statistical Anomaly}
\neq
\text{Operational Anomaly}
$$

예를 들어 count가 매우 낮은 지표가 0에서 1로 변하는 현상은 통계적으로 높은 anomaly score를 받을 수 있지만 실제 서비스 영향은 거의 없을 수 있다. 반대로 여러 주요 지표가 동시에 변화하거나 성공률이 하락하는 현상은 운영자가 확인할 가치가 높은 anomaly일 수 있다.

따라서 본 연구에서는 전문가가 실제 운영 관점에서 anomaly interval을 정의한 hand-labeled ground truth를 사용한다.

이 절에서는 다음을 검토한다.

- cellular/network KPI anomaly detection
- telecom fault/anomaly monitoring
- statistical anomaly와 operational alert의 차이
- false alarm burden과 alert fatigue
- domain expert labeling 또는 human-in-the-loop anomaly detection

**추가 문헌 검토 예정:** telecom/network anomaly detection 및 expert labeling 관련 3~5편.

## 2.8 관련 연구 구성 요약

최종 참고문헌은 다음과 같은 비중으로 구성하는 것을 목표로 한다.

| 관점 | 주요 역할 | 목표 문헌 수 |
|---|---|---:|
| TSAD 대표 방법 | 기존 접근 및 backbone 배경 | 4~6 |
| Contaminated Training | clean-normal assumption의 한계 | 4~6 |
| Temporal Context / Seasonality | $p(x\mid c)$의 필요성 | 4~6 |
| Point vs Subsequence | point-wise 문제 설정의 근거 | 3~5 |
| Robust / Decontamination | median, MAD, filtering, retraining | 3~5 |
| Evaluation Metrics | PA, Event, VUS, PR-AUC | 5~7 |
| Telecom / Expert GT | 실제 운영 환경과 전문가 레이블 | 3~5 |

중복되는 문헌을 고려하면 최종적으로 약 25~35편 수준의 핵심 참고문헌을 구성하는 것을 목표로 한다. 특히 본 연구의 핵심 문제 설정과 직접 연결되는 **Contaminated Training**, **Temporal Context**, **Evaluation Metrics** 문헌을 우선적으로 검토한다.

---

# 3. 데이터 및 문제 설정

## 3.1 인바운드 로밍 다변량 시계열

본 연구는 실제 이동통신사의 inbound roaming 운영 데이터에서 수집된 다변량 통계 시계열을 사용한다. 각 시계열은 5분 간격으로 기록되며, 사업자별 약 37,720개의 timestamp를 포함한다.

현재 primary input은 약 46개의 raw counter feature로 구성된다. 이들 feature는 접속 시도, 성공, 실패, timeout 등 roaming signaling 및 품질 상태와 관련된 다변량 통계로 구성된다.

기업 데이터의 기밀성을 고려하여 논문에서는 실제 사업자명, PLMN 및 원본 metric 의미를 공개하지 않고 각각 `Operator 1`부터 `Operator 10`, `Feature 1` 등의 익명화된 표기를 사용한다.

## 3.2 사업자 선정

전체 inbound roaming 사업자 중 트래픽 규모가 충분하고 운영 중요도가 높은 사업자를 우선적으로 고려하였다. 동일 국가의 사업자가 중복되지 않도록 하여 서로 다른 10개 국가의 사업자를 선정하였다.

사업자들은 traffic volume, periodicity, baseline level, failure pattern 및 변동성이 서로 다르기 때문에, 하나의 사업자에 특화된 결과가 다른 사업자에서도 재현되는지 검증하는 데 사용한다.

개발 과정에서는 `Operator 1`에 해당하는 P0480을 development operator로 사용하였다. 나머지 9개 사업자는 최종 방법을 고정한 이후 전문가 레이블링을 수행하고 독립 검증에 사용한다.

## 3.3 Operational Ground Truth

본 연구의 anomaly label은 단순한 statistical outlier가 아니라 실제 운영자가 확인할 가치가 있는 이상 구간을 의미한다.

본 연구에서 operational anomaly는 다음과 같이 정의한다.

> 해당 사업자의 일반적인 운용 패턴에서 벗어나며, 이동통신 서비스 또는 운영 관점에서 조사·확인이 필요하다고 전문가가 판단하는 시간 구간.

레이블링 시 다음 요소를 함께 고려한다.

- 성공 또는 수락률의 급격한 저하
- 접속 시도량의 급락 또는 급증
- 실패 및 timeout 계열 지표의 급증
- 변화의 지속 시간
- 주변 시간대의 정상 패턴과의 차이
- 동시에 변화하는 관련 지표의 수와 크기

단순히 희귀한 값이라는 이유만으로 anomaly로 정의하지 않으며, 가능한 경우 point label보다 range label을 우선한다.

현재 P0480 validation set에는 총 6개의 전문가 GT event가 존재한다. 나머지 9개 사업자는 동일한 원칙에 따라 레이블링을 진행할 예정이다.

## 3.4 데이터 분할

각 사업자의 전체 시계열은 시간 순서를 유지하여 다음과 같이 분할한다.

$$
Train:60\%,\qquad Validation:20\%,\qquad Test:20\%
$$

Train set에는 실제 anomaly가 포함될 수 있으며, train anomaly label은 모델 학습이나 context discovery 과정에서 사용하지 않는다.

$$
\boxed{\text{Train anomaly labels used}=0}
$$

Validation set은 development 단계의 분석과 method 설계 검증에 사용하며, Test set은 최종 방법을 확정한 후 최종 성능 평가에만 사용한다.

P0480은 연구 방법 개발에 사용되었기 때문에 validation 성능은 preliminary/development 결과로 해석한다. 나머지 9개 사업자에서는 동일한 방법을 가능한 한 수정 없이 적용하여 특정 사업자에 대한 과적합을 방지한다.

---

# 4. 제안 방법

## 4.1 전체 구조

제안 방법의 전체 과정은 다음과 같다.

```text
Unlabeled contaminated train data
        ↓
Train-only robust temporal profile construction
        ↓
Adaptive contiguous temporal segmentation
        ↓
K temporal contexts
        ↓
Context-specific stage-1 AutoEncoder
        ↓
High-score train sample filtering
        ↓
Fresh context-specific AutoEncoder
        ↓
Context-wise anomaly scoring
        ↓
Operational anomaly detection
```

핵심은 학습 데이터 전체에 하나의 정상 모델을 적용하는 대신, 시간에 따라 달라지는 정상 상태를 먼저 발견한 후 각 context에서 독립적으로 정상성을 학습하는 것이다.

## 4.2 Robust Temporal Profile Construction

5분 간격 시계열의 경우 하루는 총 288개의 time slot으로 구성된다.

$$
S=\{0,1,\ldots,287\}
$$

각 slot $s$에 대해 train period의 여러 날짜에서 동일한 시간대의 관측치를 수집한다.

$$
X_s
=
\{x_{d,s}\mid d\in D_{\mathrm{train}}\}
$$

학습 데이터에 anomaly가 포함될 수 있으므로 평균보다 outlier의 영향을 적게 받는 median과 MAD를 사용한다.

feature $f$에 대한 slot $s$의 중심은 다음과 같다.

$$
m_{s,f}
=
\operatorname{median}_{d}(x_{d,s,f})
$$

변동성은 다음과 같이 정의한다.

$$
MAD_{s,f}
=
\operatorname{median}_{d}
\left|
x_{d,s,f}-m_{s,f}
\right|
$$

최종 temporal profile은 다음과 같이 구성한다.

$$
p_s
=
[
m_{s,1},\ldots,m_{s,F},
MAD_{s,1},\ldots,MAD_{s,F}
]
$$

feature별 scale 차이가 segmentation을 지배하지 않도록 train-only RobustScaler를 적용한다.

## 4.3 Adaptive Contiguous Temporal Segmentation

일반 clustering을 사용하면 서로 멀리 떨어진 시간대가 하나의 context로 묶일 수 있다. 그러나 본 연구에서는 temporal context를 운영상 해석 가능한 **연속 시간 구간**으로 정의한다.

하루의 288개 profile을 $K$개의 연속 segment로 나눈다고 하자.

$$
\mathcal{S}
=
\{S_1,S_2,\ldots,S_K\}
$$

각 segment 내부의 profile이 최대한 유사하도록 다음 within-segment SSE를 최소화한다.

$$
J_K
=
\sum_{k=1}^{K}
\sum_{s\in S_k}
\|p_s-\mu_k\|_2^2
$$

여기서 $\mu_k$는 $k$번째 segment의 평균 profile이다.

각 후보 $K$에 대해 시간 순서를 보존하는 최적 partition을 dynamic programming으로 계산한다. 현재 후보 범위는 다음과 같다.

$$
K\in\{2,\ldots,24\}
$$

너무 짧은 segment가 생성되는 것을 방지하기 위해 최소 segment 길이를 설정한다. 현재 구현에서는 3개의 slot, 즉 15분을 최소 단위로 사용한다.

각 $K$에 대해 최적 contiguous partition을 얻은 후 train profile에 대한 silhouette score를 계산한다.

$$
K^*
=
\arg\max_K
Silhouette(K)
$$

이 과정에는 validation/test label이 사용되지 않는다.

현재 구현은 00:00부터 23:55까지의 하루를 선형 sequence로 분할하며, 23:55와 00:00을 하나의 segment로 연결하는 circular segmentation은 향후 확장으로 남긴다.

## 4.4 Context-specific AutoEncoder

선택된 $K^*$개의 temporal context 각각에 대해 별도의 point AutoEncoder를 학습한다.

현재 구조는 다음과 같다.

$$
F
\rightarrow 32
\rightarrow 8
\rightarrow 32
\rightarrow F
$$

P0480에서 $F=46$이며 ReLU activation과 Adam optimizer를 사용한다.

context $k$에 속하는 train sample 집합을

$$
D_k
=
\{x_t\mid c_t=k\}
$$

라고 하면 각 context마다 별도의 scaler와 AutoEncoder를 학습한다.

이와 같이 context를 분리하면 다음과 같은 조건부 정상 분포를 모델링하는 효과를 가진다.

$$
p_N(x\mid c=k)
$$

## 4.5 Context-wise Iterative Decontamination

각 context의 train data 역시 anomaly를 포함할 수 있으므로 두 단계 학습을 사용한다.

### 4.5.1 Stage 1

context $k$의 모든 train point를 이용해 첫 번째 AutoEncoder를 학습한다.

$$
AE_k^{(1)}
=
fit(D_k)
$$

각 train sample의 reconstruction MSE를 계산한다.

$$
e_{t}^{(1)}
=
\frac{1}{F}
\sum_{f=1}^{F}
(x_{t,f}-\hat{x}_{t,f}^{(1)})^2
$$

현재 구현에서는 reconstruction error 상위 5%를 contamination candidate로 간주한다.

$$
D'_k
=
\{x_t\in D_k
\mid
e_t^{(1)}
\le Q_{0.95}(e^{(1)}_k)\}
$$

중요한 점은 이 과정에서 실제 anomaly label을 사용하지 않는다는 것이다.

### 4.5.2 Stage 2

제거 후 남은 데이터 $D'_k$를 사용하여 새로운 scaler를 학습하고, 새로운 AutoEncoder를 처음부터 재학습한다.

$$
AE_k^{(2)}
=
fit(D'_k)
$$

Stage 2 모델은 최종 anomaly scoring에 사용된다.

현재 5% 제거율은 기존 내부 실험과 preliminary 분석을 바탕으로 한 기본값이며, 최종 논문에서는 제거율에 대한 sensitivity analysis를 수행할 예정이다.

## 4.6 Anomaly Score 및 Threshold

최종 AutoEncoder의 reconstruction MSE를

$$
e_t
=
\frac{1}{F}
\sum_{f=1}^{F}
(x_{t,f}-\hat{x}_{t,f})^2
$$

라고 한다.

각 context에서 decontamination 후 train reconstruction error의 평균과 표준편차를 이용해 z-score 형태로 정규화한다.

$$
s_t
=
\frac{e_t-\mu_{e,c_t}}
{\sigma_{e,c_t}+\epsilon}
$$

현재 기본 anomaly decision은 다음 기준을 사용한다.

$$
s_t>3
$$

다만 P0480 preliminary 결과에서 ranking metric에 비해 FA/day가 여전히 높게 나타났기 때문에, mean + standard deviation 기반 calibration은 최종 방법의 한계로 남아 있다. 최종 연구에서는 train-only empirical quantile 또는 robust calibration을 추가로 비교할 예정이다.

## 4.7 Direction-aware Anomaly Scoring `[확장 예정]`

실제 roaming metric은 feature에 따라 이상으로 해석되는 방향이 다르다. 일부 feature는 증가와 감소가 모두 중요하지만, 다른 feature는 증가 방향만 운영상 anomaly로 해석될 수 있다.

최종 reconstruction residual을 다음과 같이 정의한다.

$$
r_{t,f}=x_{t,f}-\hat{x}_{t,f}
$$

context와 feature별 train residual의 median과 MAD를 이용해 정규화하면

$$
z_{t,f}
=
\frac{
r_{t,f}
-
\operatorname{median}(r_{c_t,f}^{train})
}{
1.4826\cdot MAD(r_{c_t,f}^{train})+\epsilon
}
$$

으로 표현할 수 있다.

양방향 변화가 중요한 feature 집합을 $\mathcal{F}_{\pm}$, 증가 방향만 중요한 feature 집합을 $\mathcal{F}_{+}$라 하면 direction-aware evidence는 다음과 같이 정의할 수 있다.

$$
D_{t,f}
=
\begin{cases}
|z_{t,f}|, & f\in\mathcal{F}_{\pm}\\
\max(z_{t,f},0), & f\in\mathcal{F}_{+}
\end{cases}
$$

본 확장에서는 이 score를 최종 inference뿐 아니라 Stage 1 decontamination에도 동일하게 적용하여, 운영적으로 정상인 방향의 변화가 train에서 불필요하게 제거되는 것을 방지하고자 한다.

feature direction mapping은 validation anomaly 성능을 보고 결정하지 않고, metric의 사전 operational semantics에 따라 정의한다.

---

# 5. 실험 설계

## 5.1 실험 목적

최종 실험은 다음 네 가지 질문을 분리하여 검증한다.

1. 시간 context가 필요한가?
2. 고정 24시간 분할보다 adaptive context가 유리한가?
3. iterative decontamination이 실제로 도움이 되는가?
4. 결과가 여러 사업자에서 일관되게 나타나는가?

direction-aware scoring이 구현된 이후에는 다섯 번째 질문을 추가한다.

5. feature direction semantics가 false alarm을 감소시키는가?

## 5.2 비교 방법

최종 ablation은 다음 구성을 기본으로 한다.

| Method | Adaptive Context | Decontamination | Direction-aware |
|---|---:|---:|---:|
| Global AE | X | X | X |
| Global AE + Decontamination | X | O | X |
| HourAE-24 | Fixed 24 | O | X |
| Adaptive Context AE | O | X | X |
| Adaptive Context AE + Decontamination | O | O | X |
| Full Model | O | O | O |

가능한 경우 contaminated training을 직접 다루는 기존 TSAD 방법을 external baseline으로 추가한다.

P0480 개발 과정에서 사용한 RoCA, MSRS 등은 preliminary failure analysis와 연구 동기 설명에 활용하되, 최종 main comparison은 동일한 feature와 evaluation protocol에서 재실행한다.

## 5.3 평가 지표

본 연구는 다음 지표를 사용한다.

### VUS-PR

range anomaly와 tolerance window를 함께 고려하는 ranking metric으로 primary metric으로 사용한다.

### PR-AUC

class imbalance가 심한 anomaly detection에서 score ranking 품질을 보조적으로 평가한다.

### Event-F1 및 Event Recall

GT event와 prediction event가 겹치는지 여부를 기준으로 event detection 성능을 평가한다.

### Point-wise Precision / Recall / F1

threshold를 적용한 이후 point 수준의 오탐 및 미탐을 분석한다.

### False Alarms per Day

GT와 겹치지 않는 predicted event 수를 관측 일수로 나눈다.

$$
FA/day
=
\frac{N_{\mathrm{false\ alarm\ events}}}
{N_{\mathrm{observation\ days}}}
$$

운영 환경에서는 높은 recall뿐 아니라 false alarm burden을 낮추는 것이 중요하므로 핵심 operational metric으로 사용한다.

## 5.4 통계 보고

10개 사업자별 결과를 모두 보고하며 단순 macro mean에만 의존하지 않는다.

최종 표에는 다음 통계를 포함한다.

- Operator-wise result
- Mean
- Median
- Standard deviation 또는 IQR
- Adaptive vs baseline Win/Tie/Loss
- 가능한 경우 paired Wilcoxon signed-rank test
- effect size 또는 성능 차이

사업자 수가 10개로 크지 않기 때문에 p-value만으로 결론을 내리지 않고 사업자별 일관성을 함께 해석한다.

---

# 6. Preliminary Development Result: P0480

## 6.1 기존 TSAD 적용 결과

연구 초기에는 기존 contaminated TSAD 및 sequence modeling 기반 방법을 P0480에 적용하였다.

대표적으로 RoCA의 P0480 validation 결과는 다음과 같았다.

- VUS-PR: 약 0.052
- PR-AUC: 약 0.405
- Event Recall: 1/6
- FA/day: 약 1.61

이 결과는 false alarm의 수 자체는 적었지만 여러 operational GT anomaly의 ranking이 충분히 높지 않음을 보여주었다.

이후 multi-scale reconstruction/prediction 기반 internal pilot인 MSRS에서는 VUS-PR이 약 0.108, PR-AUC가 약 0.434로 상승했으나 FA/day가 약 4.37로 증가하였다.

이러한 실험은 복잡한 sequence 구조를 계속 추가하는 것보다 시간에 따라 정상 상태 자체를 분리하여 학습할 필요성을 제기하였다.

## 6.2 Fixed 24-hour Context

기존 내부 분석 방식에 따라 하루를 24개의 시간대로 나누고 각 시간마다 별도의 AutoEncoder를 학습하였다.

HourAE의 P0480 validation 결과는 다음과 같다.

| Metric | HourAE-24 |
|---|---:|
| VUS-PR | 0.1202 |
| PR-AUC | 0.4456 |
| Event Recall | 0.6667 (4/6) |
| Point F1 | 0.0564 |
| FA/day | 24.51 |
| Predicted events | 646 |

HourAE는 기존 sequence-based pilot보다 GT anomaly ranking을 향상시켰다. 특히 GT Event 1~4를 threshold 기준으로 검출하였다.

그러나 하루를 24개로 세분화하면서 false alarm이 크게 증가하였다. 이는 시간 context 분리 자체는 유효할 수 있지만 지나치게 세분화된 context가 정상 모델의 안정성을 감소시킬 가능성을 보여준다.

## 6.3 Adaptive Temporal Context

train-only robust profile에 대해 $K=2,\ldots,24$의 contiguous segmentation을 수행한 결과 P0480에서는 silhouette score 기준으로 다음 $K$가 선택되었다.

$$
K^*=2
$$

선택된 temporal context는 다음과 같다.

$$
C_1=00{:}00\sim07{:}20
$$

$$
C_2=07{:}25\sim23{:}55
$$

선택된 silhouette score는 약 0.332였다.

Adaptive ContextAE와 HourAE-24를 동일한 AutoEncoder hyperparameter와 train/validation split에서 비교한 결과는 다음과 같다.

| Metric | Adaptive $K=2$ | HourAE-24 | 변화 |
|---|---:|---:|---:|
| VUS-PR | **0.1662** | 0.1202 | +0.0461 |
| PR-AUC | **0.4638** | 0.4456 | +0.0183 |
| R-AUC-PR | **0.2342** | 0.1755 | +0.0587 |
| Event-F1 | **0.0174** | 0.0123 | +0.0052 |
| Event Recall | 0.5000 | **0.6667** | -0.1667 |
| Point Precision | **0.0463** | 0.0294 | +0.0169 |
| Point Recall | 0.5676 | **0.7027** | -0.1351 |
| Point F1 | **0.0855** | 0.0564 | +0.0291 |
| FA/day | **12.79** | 24.51 | -11.72 |
| Predicted events | **338** | 646 | -308 |

Adaptive context는 고정 24시간 모델보다 VUS-PR을 약 38% 상대 향상시켰으며, FA/day를 약 48% 감소시켰다. 또한 PR-AUC와 Point F1도 향상되었다.

반면 Event Recall은 4/6에서 3/6으로 감소하였다. 특히 GT Event 4는 HourAE에서는 높은 점수를 받았지만 Adaptive ContextAE에서는 threshold 아래로 내려갔다.

이 결과는 temporal context를 지나치게 세분화할 경우 false alarm이 증가하고, 반대로 context를 지나치게 넓게 정의할 경우 특정 anomaly type이 정상 분포에 희석될 가능성이 있음을 보여준다.

즉 본 문제에서는 다음 trade-off가 존재한다.

$$
\boxed{
\text{Context granularity}
\leftrightarrow
\text{Normal-model stability}
}
$$

## 6.4 Event-wise Ranking

P0480의 6개 GT event에 대한 Adaptive ContextAE의 최대 anomaly score percentile은 다음과 같다.

| Event | Max score percentile | Threshold detection |
|---|---:|---:|
| Event 1 | 99.43% | O |
| Event 2 | 100.00% | O |
| Event 3 | 99.64% | O |
| Event 4 | 90.22% | X |
| Event 5 | 81.57% | X |
| Event 6 | 92.38% | X |

Event 1~3은 validation 전체에서 매우 높은 anomaly ranking을 보였다. Event 5는 Adaptive와 HourAE 모두 낮은 score를 보여 현재 reconstruction-based point model이 충분히 설명하지 못하는 anomaly type일 가능성이 있다.

본 결과는 threshold 성능만 보는 것보다 continuous score ranking과 event-level detection을 분리하여 분석할 필요성을 보여준다.

---

# 7. 10개 사업자 최종 평가 계획

## 7.1 레이블링

현재 P0480을 제외한 9개 사업자에 대해 동일한 operational anomaly definition을 적용하여 전문가 레이블링을 진행한다.

레이블링 과정에서는 가능하면 특정 모델의 prediction overlay를 보지 않고 원본 시계열 및 운영 metric을 기준으로 GT를 작성한다. 이후 missed anomaly와 false alarm 분석 단계에서 모델 prediction을 활용한다.

각 사업자에 대해 최소한 다음 정보를 기록한다.

- anomaly start time
- anomaly end time
- anomaly type/reason
- 필요 시 confidence 또는 운영 중요도
- 관련 feature group

## 7.2 최종 데이터셋 통계 `[추후 작성]`

| Operator | # Points | # Features | # Valid GT Events | # Test GT Events | Test Anomaly Ratio | Selected K |
|---|---:|---:|---:|---:|---:|---:|
| Operator 1 | 약 37,720 | 46 | 6 | `[TBD]` | `[TBD]` | 2 |
| Operator 2 | `[TBD]` | 46 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 3 | `[TBD]` | 46 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 4 | `[TBD]` | 46 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 5 | `[TBD]` | 46 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 6 | `[TBD]` | 46 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 7 | `[TBD]` | 46 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 8 | `[TBD]` | 46 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 9 | `[TBD]` | 46 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 10 | `[TBD]` | 46 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

## 7.3 최종 Main Result Table `[추후 작성]`

Primary metric은 VUS-PR로 한다.

| Operator | Global AE | HourAE-24 | Adaptive Context AE | Adaptive + Decontam | Full Model |
|---|---:|---:|---:|---:|---:|
| Operator 1 | `[TBD]` | 0.1202* | `[TBD]` | 0.1662* | `[TBD]` |
| Operator 2 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 3 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 4 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 5 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 6 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 7 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 8 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 9 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Operator 10 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **Mean** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| **Median** | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

\* 현재 P0480 validation development 결과이며 최종 Test 결과가 아니다.

## 7.4 Operational Result Table `[추후 작성]`

| Method | Event-F1 | Event Recall | PR-AUC | FA/day | Detection Delay |
|---|---:|---:|---:|---:|---:|
| Global AE | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| HourAE-24 | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Adaptive Context | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Adaptive + Decontam | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |
| Full Model | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` | `[TBD]` |

## 7.5 사업자별 Context 분석 `[추후 작성]`

사업자별로 train-only segmentation에서 선택된 $K$와 시간 경계를 보고한다.

| Operator | Selected K | Temporal Contexts |
|---|---:|---|
| Operator 1 | 2 | 00:00–07:20 / 07:25–23:55 |
| Operator 2 | `[TBD]` | `[TBD]` |
| Operator 3 | `[TBD]` | `[TBD]` |
| ... | ... | ... |
| Operator 10 | `[TBD]` | `[TBD]` |

사업자별로 서로 다른 $K$가 선택된다면 fixed 24-hour partition보다 adaptive context가 필요한 근거가 된다. 반대로 대부분의 사업자에서 유사한 $K$와 경계가 반복된다면 inbound roaming traffic이 공통적인 시간 regime을 갖는다는 실증적 관찰로 해석할 수 있다.

---

# 8. 논의

## 8.1 Sequence 기반 TSAD와 Context-aware Point-wise 방식

P0480의 preliminary 결과는 SKT 인바운드 로밍 데이터에서 긴 window의 temporal dependency를 직접 모델링하는 것보다, 현재 시점의 다변량 상태를 해당 시간대의 정상 상태와 비교하는 point-wise 방식이 더 효과적일 가능성을 보여준다. 이는 본 데이터의 anomaly가 반드시 긴 subsequence의 형태로만 나타나는 것이 아니라, 특정 시간 컨텍스트에서 기대되는 정상 상태로부터의 순간적인 다변량 편차로도 충분히 표현될 수 있음을 의미한다.

다만 본 연구의 point-wise 접근은 시간 정보를 제거한 정적 이상 탐지가 아니다. 각 시점 $x_t$는 학습 데이터에서 자동으로 발견한 temporal context $c_t$에 조건화되어 평가되므로, 하루 주기와 시간대별 정상 분포의 차이를 명시적으로 사용한다.

$$
s_t=f(x_t\mid c_t)
$$

따라서 본 연구의 방법은 **context-aware point-wise TSAD**로 해석할 수 있다. 현재 단계에서 sequence 기반 TSAD보다 point-wise 방식이 일반적으로 우수하다고 단정하지 않으며, P0480에서 관찰된 hypothesis를 나머지 9개 사업자의 독립 평가를 통해 검증한다. 최종 10개 사업자 결과에서 동일한 경향이 반복될 경우, 인바운드 로밍 데이터의 anomaly 특성이 복잡한 subsequence modeling보다 context-dependent multivariate deviation으로 더 잘 설명된다는 근거로 해석할 수 있다.

## 8.2 왜 Adaptive Context가 필요한가

P0480 결과에서 24개의 독립 AutoEncoder는 anomaly recall을 높였지만 매우 많은 false alarm을 발생시켰다. 이는 시간대를 세분화할수록 각 모델이 사용할 수 있는 train sample 수가 감소하고, 해당 시간대의 정상 분포 추정이 불안정해질 수 있기 때문이다.

반면 $K=2$ adaptive context는 더 많은 train sample을 하나의 context에서 공유함으로써 normal model을 안정화하였고, 그 결과 VUS-PR과 PR-AUC를 향상시키면서 FA/day를 크게 줄였다.

따라서 temporal context는 단순히 많이 나눌수록 좋은 것이 아니라, 서로 다른 정상 regime을 충분히 분리하면서도 각 model이 정상 분포를 안정적으로 학습할 수 있는 수준으로 설정되어야 한다.

## 8.3 Silhouette와 Anomaly Detection 성능의 차이

현재 $K$ 선택에는 train profile의 silhouette score를 사용한다. 이는 anomaly label 없이 $K$를 선택할 수 있다는 장점이 있지만, silhouette는 정상 time profile의 분리도를 평가할 뿐 anomaly detection 성능을 직접 최적화하지 않는다.

실제로 P0480에서는 $K=2$가 높은 ranking 및 낮은 FA/day를 보였지만 HourAE-24가 추가로 하나의 GT event를 검출하였다.

따라서 다음 두 목표는 동일하지 않다.

$$
\text{Temporal profile separation}
\neq
\text{Anomaly detection optimum}
$$

최종 연구에서는 여러 사업자에서 silhouette 기반 $K$가 안정적으로 성능을 향상시키는지 검증하여 train-only context selection 기준의 타당성을 평가한다.

## 8.4 Ranking과 Threshold Calibration

P0480 Adaptive ContextAE의 PR-AUC는 약 0.464로 anomaly prevalence에 비해 높은 ranking 성능을 보였지만 FA/day는 여전히 12.79로 높았다.

이는 다음 두 문제가 구분되어야 함을 의미한다.

$$
\boxed{
\text{Anomaly Ranking}
\neq
\text{Operational Decision Calibration}
}
$$

현재 mean + $3\sigma$ threshold는 train과 validation score distribution shift에 민감하다. 따라서 최종 시스템에서는 robust train-only calibration을 별도 검토할 필요가 있다.

## 8.5 Direction-aware Scoring의 필요성

운영 metric의 의미를 고려하면 모든 reconstruction residual을 대칭적으로 처리하는 것이 적절하지 않을 수 있다. 운영상 의미 없는 방향의 deviation까지 anomaly evidence에 포함되면 정상적인 변동이 false alarm으로 연결될 수 있다.

direction-aware scoring은 모델 입력값을 임의로 clipping하는 대신 원본 분포와 AutoEncoder 학습은 유지하고 scoring 단계에서 operational semantics를 반영한다는 장점이 있다.

다만 이 요소의 효과는 현재 수치적으로 검증되지 않았으므로 최종 논문에서는 반드시 ablation을 통해 별도로 입증해야 한다.

---

# 9. 한계

본 연구의 현재 한계는 다음과 같다.

첫째, 방법 개발 과정이 P0480 한 사업자의 validation 결과를 중심으로 이루어졌다. 따라서 해당 사업자에 대한 development overfitting 가능성을 배제할 수 없다. 이를 줄이기 위해 최종 방법을 고정한 뒤 나머지 9개 사업자에서 독립적으로 검증한다.

둘째, 현재 temporal segmentation은 하루를 00:00부터 23:55까지 선형적으로 처리한다. 실제 daily cycle에서는 23:55와 00:00이 인접하므로 circular segmentation이 더 자연스러울 수 있다.

셋째, 현재 iterative decontamination에서 상위 5% reconstruction error를 제거하는 고정 비율을 사용한다. 여러 사업자에서 contamination 정도가 다를 수 있으므로 제거율 sensitivity 또는 train-only robust threshold 방식에 대한 추가 분석이 필요하다.

넷째, 현재 anomaly decision은 context별 reconstruction error의 mean + $3\sigma$ 형태에 기반한다. score distribution이 heavy-tailed하거나 train-validation shift가 존재할 경우 false alarm이 증가할 수 있다.

다섯째, 10개 사업자는 고트래픽 사업자 중심으로 선정되므로 모든 저트래픽 로밍 사업자까지 결과를 일반화하기 어렵다.

여섯째, 실제 기업 데이터의 보안 문제로 raw time series와 사업자/feature 실명을 공개하기 어렵기 때문에 완전한 데이터셋 공개에는 제약이 있다.

---

# 10. 결론

본 연구는 anomaly label 없이 contaminated multivariate time series를 학습해야 하는 실제 이동통신 환경에서, 정상 상태가 시간에 따라 달라진다는 점에 주목하였다.

초기 실험에서 기존 sequence/window 기반 TSAD 및 global normal modeling만으로는 operational GT와 충분한 정렬을 얻기 어렵거나 많은 false alarm이 발생하였다. 반면 temporal context를 조건으로 각 시점의 다변량 deviation을 평가하는 point-wise 접근은 P0480에서 더 높은 anomaly ranking 성능을 보여, 본 인바운드 로밍 데이터에서는 긴 temporal dependency 자체보다 context-dependent normality가 중요한 단서일 가능성을 확인하였다. 고정된 24시간 context 모델은 이러한 가설을 뒷받침했지만 지나친 세분화로 false alarm이 증가하였다.

이를 해결하기 위해 본 연구는 train data에서 robust temporal profile을 구성하고 시간 순서를 유지하는 adaptive contiguous segmentation을 통해 temporal context의 수와 경계를 자동으로 결정하는 방법을 제안하였다. 또한 각 context 내부에서 iterative decontamination을 수행하여 anomaly label 없이 정상 모델을 재학습하였다.

P0480 development 결과에서 제안한 Adaptive ContextAE는 고정 24시간 모델보다 VUS-PR과 PR-AUC를 향상시키고 FA/day를 약 절반 수준으로 감소시켰다. 이는 실제 데이터의 정상성을 하나의 global distribution 또는 고정 시간 단위로 정의하기보다 데이터에 적응적으로 temporal context를 발견하는 것이 유효할 가능성을 보여준다.

향후 나머지 9개 사업자에 대한 전문가 레이블링을 완료한 뒤 총 10개 사업자에서 동일한 방법을 평가하고, adaptive context, decontamination, direction-aware scoring의 개별 효과를 ablation을 통해 검증한다. 최종적으로는 여러 사업자에서 일관된 성능 향상이 관찰되는지를 통해 제안 방법의 일반성을 평가하고자 한다.

---

# 참고문헌

## 참고문헌 구성 및 읽기 우선순위

본 논문의 참고문헌은 단순히 TSAD 관련 논문을 나열하는 방식이 아니라, 현재 연구의 문제 정의와 방법 선택을 뒷받침하는 관점별로 구성한다. 우선순위는 현재 연구에서 필요한 개념의 중요도와 실험 설계와의 직접적인 연관성을 기준으로 정하였다.

| 우선순위 | 관점 | 논문에서의 역할 | 목표 문헌 수 |
|---:|---|---|---:|
| 1 | TSAD 평가 지표 | PA의 한계, Range/Event 평가, VUS-PR, PR-AUC 선택 근거 | 5~7 |
| 2 | Contaminated Training / Robust TSAD | clean-normal assumption의 한계와 label-free contaminated training 근거 | 4~6 |
| 3 | Temporal Context / Seasonality | 시간에 따라 정상 분포가 달라지는 문제와 $p(x\mid c)$의 근거 | 4~6 |
| 4 | Point-wise vs Subsequence | point/contextual/subsequence anomaly의 차이와 현재 문제 설정의 근거 | 3~5 |
| 5 | Robust Statistics / Decontamination | median, MAD, filtering, retraining의 근거 | 3~5 |
| 6 | Telecom / Expert Ground Truth | 실제 통신 운영 환경, false alarm, 전문가 레이블의 의미 | 3~5 |
| 7 | 대표 TSAD 방법 및 Benchmark | 기존 reconstruction/prediction/deep TSAD의 전체 배경 | 4~6 |

## 1순위 — TSAD 평가 지표

현재 연구에서 VUS-PR을 primary metric으로 사용하고 있으므로 가장 먼저 읽는다. 특히 Point Adjustment의 문제점, point-based metric과 range/event-based metric의 차이, VUS의 도입 배경, severe class imbalance에서 PR-AUC가 갖는 의미를 정리한다.

### 반드시 읽을 논문

[3] P. Boniol, A. K. Krishna, M. Bruel, Q. Liu, M. Huang, T. Palpanas, R. S. Tsay, A. Elmore, M. J. Franklin, and J. Paparrizos,  
**“VUS: effective and efficient accuracy measures for time-series anomaly detection,”** *The VLDB Journal*, 2025.  
- 역할: VUS-ROC / VUS-PR의 정의와 tolerance 개념
- 주소: https://doi.org/10.1007/s00778-025-00907-x

[4] Q. Liu and J. Paparrizos,  
**“The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark,”** *NeurIPS 2024, Datasets and Benchmarks Track*, 2024.  
- 역할: TSB-AD, TSAD benchmark 문제, VUS-PR의 평가 신뢰성
- 주소: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html

[5] S. Kim, K. Choi, H.-S. Choi, B. Lee, and S. Yoon,  
**“Towards a Rigorous Evaluation of Time-Series Anomaly Detection,”** *AAAI*, 2022.  
- 역할: Point Adjustment가 성능을 과대평가할 수 있다는 근거
- 주소: https://doi.org/10.1609/aaai.v36i7.20680

[6] N. Tatbul, T. J. Lee, S. Zdonik, M. Alam, and J. Gottschlich,  
**“Precision and Recall for Time Series,”** *NeurIPS*, 2018.  
- 역할: point anomaly가 아닌 range anomaly를 평가하기 위한 Precision/Recall 확장
- 주소: https://proceedings.neurips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html

[7] A. Huet, J. M. Navarro, and D. Rossi,  
**“Local Evaluation of Time Series Anomaly Detection Algorithms,”** *KDD*, 2022.  
- 역할: Affiliation Precision/Recall, event-local evaluation
- 주소: https://doi.org/10.1145/3534678.3539339
- arXiv: https://arxiv.org/abs/2206.13167

[8] T. Saito and M. Rehmsmeier,  
**“The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,”** *PLOS ONE*, 2015.  
- 역할: severe class imbalance에서 PR-AUC를 사용하는 근거
- 주소: https://doi.org/10.1371/journal.pone.0118432

### 읽을 때 확인할 항목

- Point Adjustment가 왜 문제가 되는가?
- range detection과 existence detection은 어떻게 다른가?
- anomaly cardinality와 tolerance가 metric에 어떤 영향을 주는가?
- VUS-PR은 단일 threshold 또는 단일 tolerance metric과 무엇이 다른가?
- PR-AUC의 random baseline이 anomaly prevalence와 어떤 관계가 있는가?

---

## 2순위 — Contaminated Training / Robust TSAD

본 연구의 핵심 설정인 **anomaly label을 사용하지 않는 contaminated training**을 직접 다루는 문헌이다.

[1] T. K. K. Ho and N. Armanfard,  
**“Contaminated Multivariate Time-Series Anomaly Detection with Spatio-Temporal Graph Conditional Diffusion Models,”** *UAI*, 2025.  
- 역할: TSAD-C, label-free contaminated training과 decontamination
- 주소: https://proceedings.mlr.press/v286/ho25a.html
- arXiv: https://arxiv.org/abs/2308.12563

[9] X. Mou, R. Wang, B. Li, T. Wo, J. Sun, H. Wang, and X. Liu,  
**“RoCA: Robust Contrastive One-Class Time Series Anomaly Detection With Contaminated Data,”** *IEEE Transactions on Neural Networks and Learning Systems*, 2026.  
- 역할: contaminated train에서 contrastive learning + one-class learning
- DOI: https://doi.org/10.1109/TNNLS.2026.3702237
- arXiv: https://arxiv.org/abs/2503.18385

[10] Z. Chen, Z. Li, X. Chen, X. Chen, H. Fan, and R. Hu,  
**“Rectifying inaccurate unsupervised learning for robust time series anomaly detection,”** *Information Sciences*, vol. 662, 120222, 2024.  
- 역할: RiAD, contaminated training에서 uncertainty와 adaptive reconstruction loss 활용
- 주소: https://doi.org/10.1016/j.ins.2024.120222

[11] W. Li, C. Feng, T. Chen, and J. Zhu,  
**“Robust Learning of Deep Time Series Anomaly Detection Models with Contaminated Training Data,”** 2022.  
- 역할: 여러 deep TSAD가 contamination에 얼마나 민감한지 분석하고 model-agnostic robust learning 제안
- 주소: https://arxiv.org/abs/2208.01841

[12] C. Qiu, A. Li, M. Kloft, M. Rudolph, and S. Mandt,  
**“Latent Outlier Exposure for Anomaly Detection with Contaminated Data,”** *ICML*, 2022.  
- 역할: latent label을 반복적으로 추정하는 contaminated anomaly detection
- 주소: https://proceedings.mlr.press/v162/qiu22b.html
- arXiv: https://arxiv.org/abs/2202.08088

[13] S. Zhang, Y. Lai, P. Zheng, B. Yu, X. Tang, Q. Fu, and P. He,  
**“CLEANet: Robust and Efficient Anomaly Detection in Contaminated Multivariate Time Series,”** 2025.  
- 역할: contaminated MTS에서 adaptive reconstruction weighting과 robust representation
- 주소: https://arxiv.org/abs/2510.22619

### 읽을 때 확인할 항목

- train이 clean-normal이라고 가정하는가?
- contamination ratio를 알고 있어야 하는가?
- anomaly label 없이 contamination을 어떻게 추정하는가?
- filtering, reweighting, robust loss 중 어떤 방식을 사용하는가?
- contamination을 제거한 뒤 모델을 재학습하는가?
- 본 연구의 context-wise decontamination과 어떤 차이가 있는가?

---

## 3순위 — Temporal Context / Seasonality / Context-dependent Normality

본 연구의 핵심 아이디어인 **시간에 따라 정상 분포가 달라진다**는 가정을 뒷받침하는 문헌이다.

[2] L. Porcelli, M. Trovati, and F. Palmieri,  
**“Real-time anomaly detection in seasonal time series with conditional variational autoencoder,”** *Applied Soft Computing*, vol. 184, 113761, 2025.  
- 역할: timestamp-derived context와 conditional VAE
- 주소: https://doi.org/10.1016/j.asoc.2025.113761

[14] Z. Li, W. Chen, and D. Pei,  
**“Robust and Unsupervised KPI Anomaly Detection Based on Conditional Variational Autoencoder,”** *IEEE IPCCC*, 2018.  
- 역할: Bagel, time information을 condition으로 사용하는 KPI anomaly detection
- 주소: https://doi.org/10.1109/PCCC.2018.8710885

[15] H. Xu, W. Chen, N. Zhao, Z. Li, J. Bu, Z. Li, Y. Liu, Y. Zhao, D. Pei, Y. Feng, J. Chen, Z. Wang, and H. Qiao,  
**“Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications,”** *The Web Conference (WWW)*, 2018.  
- 역할: Donut, seasonal KPI에 대한 unsupervised anomaly detection
- 주소: https://doi.org/10.1145/3178876.3185996
- arXiv: https://arxiv.org/abs/1802.03903

[16] H. Kim and H. Kim,  
**“Contextual anomaly detection for multivariate time series data,”** *Quality Engineering*, vol. 35, no. 4, pp. 686–695, 2023.  
- 역할: contextual variable에 따라 정상 패턴이 달라지는 multivariate time series
- 주소: https://doi.org/10.1080/08982112.2023.2179404

[17] C. U. Carmona, F.-X. Aubet, V. Flunkert, and J. Gasthaus,  
**“Neural Contextual Anomaly Detection for Time Series,”** *IJCAI*, 2022.  
- 역할: NCAD, contextual anomaly 개념과 time-series representation
- 주소: https://doi.org/10.24963/ijcai.2022/394

### 읽을 때 확인할 항목

- context를 hour/day처럼 고정적으로 정의하는가?
- context가 timestamp에서 직접 주어지는가, 데이터로부터 학습되는가?
- $p(x)$ 대신 $p(x\mid c)$를 어떻게 모델링하는가?
- point별 condition과 sequence context의 차이는 무엇인가?
- 본 연구의 adaptive contiguous context discovery와 어떤 차이가 있는가?

---

## 4순위 — Point-wise vs Window/Subsequence / Contextual Anomaly

본 연구가 sequence/window 기반 접근보다 **context-aware point-wise 방식이 더 적합할 가능성**을 검증하기 때문에 anomaly type 자체를 정리할 필요가 있다.

[18] V. Chandola, A. Banerjee, and V. Kumar,  
**“Anomaly Detection: A Survey,”** *ACM Computing Surveys*, vol. 41, no. 3, Article 15, 2009.  
- 역할: point anomaly, contextual anomaly, collective anomaly의 기본 정의
- 주소: https://doi.org/10.1145/1541880.1541882

[6] N. Tatbul et al.,  
**“Precision and Recall for Time Series,”** *NeurIPS*, 2018.  
- 역할: point anomaly와 range anomaly의 평가 단위 차이
- 주소: https://proceedings.neurips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html

[17] C. U. Carmona et al.,  
**“Neural Contextual Anomaly Detection for Time Series,”** *IJCAI*, 2022.  
- 역할: contextual anomaly detection 관점
- 주소: https://doi.org/10.24963/ijcai.2022/394

### 읽을 때 확인할 항목

- point anomaly / contextual anomaly / collective anomaly를 어떻게 정의하는가?
- subsequence anomaly에서 temporal dependency가 왜 필요한가?
- 현재 SKT 인바운드 anomaly가 어느 유형에 더 가까운가?
- point-wise detection이라도 temporal context를 사용하면 contextual anomaly로 볼 수 있는가?

---

## 5순위 — Robust Statistics / Decontamination

본 연구의 robust temporal profile, median/MAD, high-score filtering 및 retraining을 뒷받침하는 문헌이다.

[19] P. J. Rousseeuw and C. Croux,  
**“Alternatives to the Median Absolute Deviation,”** *Journal of the American Statistical Association*, vol. 88, no. 424, pp. 1273–1283, 1993.  
- 역할: MAD의 robustness, 1.4826 scaling, breakdown point
- 주소: https://doi.org/10.1080/01621459.1993.10476408
- PDF: https://wis.kuleuven.be/stat/robust/papers/publications-1993/rousseeuwcroux-alternativestomedianad-jasa-1993.pdf

[12] C. Qiu et al.,  
**“Latent Outlier Exposure for Anomaly Detection with Contaminated Data,”** *ICML*, 2022.  
- 역할: contaminated data에서 반복적 latent outlier 추정
- 주소: https://proceedings.mlr.press/v162/qiu22b.html

[11] W. Li et al.,  
**“Robust Learning of Deep Time Series Anomaly Detection Models with Contaminated Training Data,”** 2022.  
- 역할: robust learning / contamination mitigation
- 주소: https://arxiv.org/abs/2208.01841

[10] Z. Chen et al.,  
**“Rectifying inaccurate unsupervised learning for robust time series anomaly detection,”** *Information Sciences*, 2024.  
- 역할: adaptive reconstruction loss와 anomaly candidate 억제
- 주소: https://doi.org/10.1016/j.ins.2024.120222

### 읽을 때 확인할 항목

- median과 MAD를 쓰는 이론적 이유
- high-error sample을 제거하는 것이 trimming과 어떤 관계가 있는가?
- 고정 top-k% 제거와 robust cutoff 방식의 장단점
- scaler를 contamination 제거 전/후 어느 시점에 fit하는 것이 적절한가?

---

## 6순위 — Telecom / KPI / Expert Ground Truth

실제 통신 환경에서 statistical anomaly와 operational anomaly가 다를 수 있다는 점, 그리고 expert labeling과 false alarm burden이 중요하다는 근거를 확보하기 위한 문헌이다.

[20] Z. Li, W. Chen, and D. Pei,  
**“Robust and Unsupervised KPI Anomaly Detection Based on Conditional Variational Autoencoder,”** *IEEE IPCCC*, 2018.  
- 역할: 실제 KPI monitoring, 운영자 관점의 자동 anomaly detection
- 주소: https://doi.org/10.1109/PCCC.2018.8710885

[21] H. Xu et al.,  
**“Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications,”** *WWW*, 2018.  
- 역할: 대규모 KPI monitoring과 seasonal anomaly
- 주소: https://doi.org/10.1145/3178876.3185996

[22]  
**“A Framework for AI-Based Large-Scale Time-Series Anomaly Detection in 5G Core Networks,”** *IEEE ICTC*, 2025.  
- 역할: 실제 5G core KPI에서 large-scale TSAD, point-level anomaly amplification 및 threshold 문제
- 주소: https://doi.org/10.1109/ICTC66702.2025.11388910
- IEEE Xplore: https://ieeexplore.ieee.org/abstract/document/11388910

[23] D. Kim, Y. S. Sim, S. Yoon, S. Yoon, S. Yoa, S. Lee, and W. Lim,  
**“AEGIS: Toward Expert-in-the-loop Industrial Anomaly Detection,”** *AAAI*, 2026.  
- 역할: domain expert feedback과 anomaly detection system의 상호작용
- 주소: https://doi.org/10.1609/aaai.v40i48.42359

[24]  
**“Beyond the Norm: Unsupervised Anomaly Detection in Telecommunications with Mahalanobis Distance,”** *Computers*, vol. 14, no. 12, 561, 2025.  
- 역할: telecom KPI의 multivariate unsupervised anomaly detection
- 주소: https://doi.org/10.3390/computers14120561

### 읽을 때 확인할 항목

- 실제 통신/KPI 환경에서 false alarm을 어떻게 다루는가?
- anomaly GT는 누가 어떤 기준으로 정의하는가?
- threshold를 어떻게 운영적으로 정하는가?
- expert feedback이 모델 학습에 직접 들어가는가, 평가에만 쓰이는가?
- statistical anomaly와 operational anomaly를 구분하는가?

---

## 7순위 — 대표 TSAD 모델 및 Benchmark

최종적으로 기존 TSAD 분야의 대표적인 방법을 읽어 현재 연구가 어떤 계열과 다른지 정리한다.

[25] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei,  
**“Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network,”** *KDD*, 2019.  
- 모델: OmniAnomaly
- 역할: stochastic RNN 기반 multivariate sequence modeling
- 주소: https://www.kdd.org/kdd2019/accepted-papers/view/robust-anomaly-detection-for-multivariate-time-series-through-stochastic-re

[26] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga,  
**“USAD: UnSupervised Anomaly Detection on Multivariate Time Series,”** *KDD Workshop*, 2020.  
- 역할: adversarially trained AutoEncoder 기반 unsupervised MTS anomaly detection
- 주소: https://www.kdd.org/kdd2020/accepted-papers/view/usad-unsupervised-anomaly-detection-on-multivariate-time-series.html

[27] S. Tuli, G. Casale, and N. R. Jennings,  
**“TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data,”** *PVLDB*, vol. 15, no. 6, pp. 1201–1214, 2022.  
- 역할: Transformer 기반 broad temporal context modeling
- 주소: https://doi.org/10.14778/3514061.3514067
- PDF: https://vldb.org/pvldb/vol15/p1201-tuli.pdf

[28] J. Xu, H. Wu, J. Wang, and M. Long,  
**“Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy,”** *ICLR*, 2022.  
- 역할: attention association discrepancy 기반 sequence anomaly detection
- 주소: https://openreview.net/forum?id=LzQQ89U1qm_
- arXiv: https://arxiv.org/abs/2110.02642

[4] Q. Liu and J. Paparrizos,  
**“The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark,”** *NeurIPS*, 2024.  
- 역할: 40개 알고리즘을 포함한 benchmark와 단순/복잡 모델 비교
- 주소: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html

---

## 우선 읽을 핵심 논문

전체 문헌을 한 번에 읽기보다 현재 연구에 직접 필요한 순서로 다음 논문부터 우선 읽는다.

1. **VUS: effective and efficient accuracy measures for time-series anomaly detection**  
   https://doi.org/10.1007/s00778-025-00907-x

2. **The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark**  
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html

3. **Towards a Rigorous Evaluation of Time-Series Anomaly Detection**  
   https://doi.org/10.1609/aaai.v36i7.20680

4. **Precision and Recall for Time Series**  
   https://proceedings.neurips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html

5. **Contaminated Multivariate Time-Series Anomaly Detection with Spatio-Temporal Graph Conditional Diffusion Models**  
   https://proceedings.mlr.press/v286/ho25a.html

6. **RoCA: Robust Contrastive One-Class Time Series Anomaly Detection With Contaminated Data**  
   https://doi.org/10.1109/TNNLS.2026.3702237

7. **Rectifying inaccurate unsupervised learning for robust time series anomaly detection**  
   https://doi.org/10.1016/j.ins.2024.120222

8. **Real-time anomaly detection in seasonal time series with conditional variational autoencoder**  
   https://doi.org/10.1016/j.asoc.2025.113761

9. **Robust and Unsupervised KPI Anomaly Detection Based on Conditional Variational Autoencoder**  
   https://doi.org/10.1109/PCCC.2018.8710885

10. **Anomaly Detection: A Survey**  
    https://doi.org/10.1145/1541880.1541882

이 10편을 먼저 읽은 뒤, point/contextual anomaly, robust statistics, telecom operational evaluation 및 대표 TSAD 모델 문헌으로 확장한다.

---

# 참고문헌 읽기 및 정리 계획

앞으로 읽는 논문은 단순 요약이 아니라 아래 항목을 기준으로 정리하여 본문에 반영한다.

1. 논문이 정의하는 anomaly의 단위: point / contextual / subsequence / event
2. train data에 anomaly가 포함되는지 여부와 contamination assumption
3. 시간 context 또는 seasonality를 사용하는지
4. 정상 분포를 global하게 보는지 context-conditioned하게 보는지
5. decontamination 또는 robust training 방식
6. anomaly score와 threshold 결정 방식
7. 평가 지표: AUROC, PR-AUC, F1, PA-F1, range metric, VUS 등
8. 사용 데이터셋과 anomaly ratio
9. 본 연구와 직접적으로 겹치는 부분
10. 본 연구와 구분되는 차별점

우선 읽을 문헌의 순서는 다음과 같이 한다.

1. VUS 및 TSAD evaluation metric
2. Contaminated training / robust TSAD
3. Temporal context / seasonality / contextual anomaly detection
4. Point-wise vs subsequence anomaly definition
5. Robust statistics / iterative decontamination
6. Telecom operational anomaly / expert labeling
7. 대표 TSAD backbone 및 benchmark

---

# 작성 메모

- 현재 수치 결과는 P0480 validation development 결과이며 최종 Test 성능으로 표현하지 않는다.
- 나머지 9개 사업자 레이블링 후 최종 main table을 갱신한다.
- 최종 방법을 확정한 이후 P0480 Test와 나머지 사업자 Test를 평가한다.
- direction-aware scoring은 구현 및 ablation 완료 전까지 확정 contribution으로 과장하지 않는다.
- Global AE / Global AE + Decontamination ablation을 반드시 추가한다.
- 가능하면 contaminated-training 대표 baseline을 동일한 10개 사업자에서 재실행한다.
- 최종 논문에서는 `P0480`, `M###`와 같은 내부 ID를 제거하고 `Operator 1`, `Feature 1` 형식으로 익명화한다.