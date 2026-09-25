# OmniAnomaly 관찰 · 논문용 시사점

IB inbound roaming 시계열 + 사람 라벨(GT) 위에서 **순정 OmniAnomaly(재구성 기반)** 를 돌렸을 때 정리한 관찰이다.  
실행·run 이름은 [OmniAnomaly/PAPERIB.md](../OmniAnomaly/PAPERIB.md), 전체 설계는 [논문-설계-노트.md](./논문-설계-노트.md).

> **범위:** 현재까지는 주로 `paperib` run · PLMN `P0480` · POT 임계값 기준의 **관찰/가설**.  
> “OA는 항상 극단만 잡는다”처럼 **모델 일반 성질로 단정하지 않는다.**

---

## 1. 한 줄 요약

OmniAnomaly는 재구성 능력이 좋아 **mild한 운영 이상을 정상처럼 잘 복원**하고,  
점수(score)가 잘 안 떨어져 **탐지에 실패**하기 쉽다.  
남는 탐지는 주로 **다지표가 동시에 무너지는 severe**이며,  
그런 구간은 **단순한 규칙/통계만으로도 상당수 잡을 가능성이 크다** → 복잡한 재구성 모델의 **추가 이득이 제한적**일 수 있다.

---

## 2. 점수 · 탐지 동작 (이 파이프라인)

| 항목 | 내용 |
|------|------|
| 모델 | 비지도 VAE(+NF 등), reconstruction log-prob 계열 |
| 점수 | **낮을수록 이상** (`score < threshold`) |
| Primary thr | **POT** (label-free; train score 꼬리) |
| Secondary | valid **best-F1** (GT를 보는 oracle — primary 비교에 쓰지 않음) |
| 학습 | chronological train; 사람 라벨 ∩ train 구간은 학습에서 **제외** |

---

## 3. 관찰: mild vs severe (P0480 · `paperib` · valid)

### 3.1 모델이 강하게 잡은 예 — `oa_002048`

- 구간: 대략 2026-07-24 07:55–09:05 (사람 라벨의 rate 붕괴 + attempt 폭증 + fail 폭증과 일치).
- 특징: M971·실패 계열이 **수만 단위**로 치솟고, SUCCESS/ACCEPT rate가 거의 0에 가까움.
- score 최저 ≈ **-5944** (valid 중앙값 ≈ **+125**, POT thr ≈ **-183**).
- → **다변량 severe collapse**. 재구성이 크게 실패해 score가 thr 아래로 떨어짐.

### 3.2 놓친 사람 라벨 (같은 valid)

다수가 `attempt_drop` / `rate_degrade` 중심이고, fail 폭증이 약하거나 없음.

| 경향 | 내용 |
|------|------|
| score | 대략 **+77 ~ +111** 등 → **정상대** (중앙값과 비슷) |
| M971·fail | 평소 수준에 가깝거나, rate만 상대적으로 나쁨 |
| 결과 | POT(및 그 근처)으로도 **구간을 거의 못 넘김** |

즉 “임계값만 조금 풀면 mild가 다 잡힌다”기보다, **score 자체가 이상으로 안 나오는** 경우가 많다.

### 3.3 지표 해석 함정 (point vs event)

- valid anomaly **시점**이 한 severe 구간에 몰리면, **event는 1/N만 맞춰도 point recall은 높아 보일 수 있다.**
- 논문/보고에는 **point + event(segment)** 를 같이 쓰는 것을 권장.

---

## 4. 해석 (왜 이런가)

1. **재구성 inductive bias**  
   train에 흔한 “잔잔한 rate/attempt 변동”은 잘 복원됨 → mild 운영 이상의 likelihood가 크게 안 나빠짐.
2. **raw count 스케일 (`paperib`)**  
   큰 counter 폭증이 energy/log-prob를 지배 → **폭발형**만 두드러짐.
3. **GT 정의**  
   사람 라벨은 통계적 outlier만이 아니라 **운영상 의미 있는 mild**를 포함 ([라벨링-원칙.md](./라벨링-원칙.md)).  
   모델 목표와 GT가 어긋날 수 있음.
4. **희소성**  
   anomaly 시점이 전체의 매우 작은 비율 → 불균형·지표 분산이 큼. “데이터가 안정적”이라기보다 **희귀 운영 이상**에 가깝다.

---

## 5. 논문에서 쓸 문장 (초안 · 방어적)

**권장**

- “본 설정(`paperib` raw, POT)에서 OmniAnomaly는 **severe multivariate collapse**에는 반응했으나, **mild rate/attempt 이상**에서는 score가 정상 분포에 머무는 사례가 관찰되었다.”
- “재구성이 mild 패턴을 잘 복원한 결과로 해석할 수 있으며, 일반화는 추가 PLMN·feature run으로 검증한다.”
- “severe 구간은 **단순 baseline**(rate/attempt/fail 규칙 등)으로도 상당수 재현될 수 있어, 재구성 모델의 **추가 이득이 제한적**일 수 있다.”

**비권장**

- “OmniAnomaly는 극단 anomaly만 탐지한다.” (모델 보편 성질처럼 들림)
- “극단은 모델 없이 항상 잡는다.” (미검증 단정)

---

## 6. 논문 프레이밍과의 연결

목표는 “여러 모델 중 최강 하나”가 아니라:

> IB 운영 GT에서 **재구성 모델이 언제 필요한가 / 어디에 실패하는가**

| 축 | 역할 |
|----|------|
| OmniAnomaly | 재구성 baseline — mild 복원·severe 편중 **가설의 근거** |
| Attention 등 (예: AT) | 맥락·다른 inductive bias와 비교 |
| 단순 baseline | “severe는 모델 없이도?”를 **숫자로** 검증 |
| Feature (`comb` / `comb_share`) | raw 스케일 한계를 완화하면 mild score가 살아나는지 |

비교 시 **임계값 프로토콜을 맞출 것**: primary는 양측 모두 label-free (OA=POT, AT=train percentile 등). best-F1은 oracle 상한으로만.

---

## 7. 다음에 검증할 것 (체크리스트)

- [ ] severe ∩ OA 탐지 vs **규칙 baseline** overlap (P/R, event)
- [ ] mild 구간 score 분포 vs 정상 (PLMN 여러 개)
- [ ] `comb_share`에서 mild score가 thr 쪽으로 움직이는지
- [ ] reason별 (`fail_surge` vs `attempt_drop` only) strata 표
- [ ] point + event 동시 보고

---

## 8. 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-09-25 | P0480 `paperib`/POT 관찰·논문용 시사점 초안 |
