# COMB run 설계 (paperib × OmniAnomaly)

`paperib` run은 PLMN에 포함된 **모든 raw metric**으로 학습한다.  
운영 관점에서 의미 있는 신호만 쓰려면 **COMB 계열 run**을 따로 둔다.

라벨링 UI에서는 **모델 run** 드롭다운으로 `paperib` / `comb` / `comb_share` 예측 JSON을 전환한다.

> **보안:** masked metric ID(`M###`)와 원본 metric명의 대응표는 **git에 올리지 않는다**.  
> 로컬 `data/ib_data/raw/*_mapping.txt`만 사용한다. 이 문서·코드 주석에도 대응을 적지 않는다.

---

## 1. COMB run을 둔 이유 (`--run_name comb`, `comb_share`)

**모니터링에 필수적이지 않은 항목의 변동으로 잡힌 anomaly는 운용에 의미가 없다.**

- 전체 metric(`paperib`)에는 접속 품질과 직접 관련 없는 counter·지표가 많다.
- 그런 차원의 reconstruction 이상은 **트래픽/장애와 무관한 노이즈**로 FP가 늘거나, 사람이 보는 이상과 모델 점수가 어긋나기 쉽다.
- 접속 시도·성공/실패·타임아웃 등 **품질과 연결된 소수 feature**만 남겨 **운영상 해석 가능한 이상**에 집중한다.

고정 raw counter 목록은 `run_plmn.py`의 `COMB_RAW_COUNTERS`(masked ID 14개)이다.

---

## 2. run 두 가지: `comb` vs `comb_share`

| run | 입력 | 표준화 | 용도 |
|-----|------|--------|------|
| **`comb`** | 14 raw counter + `S_RATE` + `A_RATE` (16차원) | train **MinMax** (모든 차원 동일) | 기존 방식 유지·비교 baseline |
| **`comb_share`** | `M971` + 나머지 counter **÷ M971** (14차원) | `log1p(M971)` 후 train **MinMax**; 비율도 train MinMax | 시도량 대비 **비율** 중심 |

### 2.1 `comb` (기존)

- 데이터 프레임에 있는 **원시 카운트**와 UI와 동일한 **SUCCESS/ACCEPT rate** (`S_RATE`, `A_RATE`)를 그대로 넣는다.
- 모든 차원에 동일하게 MinMaxScaler (train fit, valid/test transform).

```bash
cd OmniAnomaly
../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10 --stable_train \
  --run_name comb
```

산출물 예:

- `model/P0480/comb/`, `data/ib_data/predictions/P0480_omnianomaly_comb.json`

### 2.2 `comb_share` (시도량 대비 비율 + 다른 표준화)

**개별 건수의 절대 변동보다, 전체 시도량(`M971`) 대비 해당 counter의 비율이 중요하기 때문**에 별도 run을 둔다.

- `M971`은 **규모(볼륨)** 정보로 남긴다.
- 나머지 COMB raw counter는 **÷ M971** 한 **share**로 넣는다 (`S_RATE`/`A_RATE`는 동일 비율이라 별도 입력 없음).
- 표준화:
  - **M971**: 꼬리가 긴 count → `log1p` 후 MinMax (train fit)
  - **share 차원**: train MinMax (동일 scaler, train만 fit)

```bash
../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10 --stable_train \
  --run_name comb_share
```

산출물 예:

- `model/P0480/comb_share/`, `data/ib_data/predictions/P0480_omnianomaly_comb_share.json`

메타/예측 JSON의 `feature_mode`는 각각 `comb`, `comb_share`이며 `comb_scaling` 필드에 적용 방식이 기록된다.

---

## 3. `paperib`와의 관계

- **`paperib`**: 변경 없음. 전체 raw metric, MinMax.
- **`comb` / `comb_share`**: 모델·체크포인트·예측 JSON 경로가 run 이름별로 **분리**되어 서로 덮어쓰지 않는다.
- **`window_length` ≠ 100**: 경로에 `_w{N}`이 붙는다 (예: `model/P0480/comb_w200/`, `P0480_omnianomaly_comb_w200.json`). 기본 win=100 산출물은 유지된다.

라벨링 UI의 metric 필터·그래프는 `comb`와 `comb_share` 모두 **동일한 표시 기준**(COMB raw counters + S_RATE/A_RATE)을 쓴다. `comb_share`는 **학습 입력·표준화**만 다르고, UI에서 토글하는 시리즈 집합은 바꾸지 않는다.

---

## 4. 참고

- 통합 사용법: [PAPERIB.md](./PAPERIB.md)
- feature 상수·행렬 변환: `run_plmn.py` (`COMB_LEGACY_COLUMNS`, `COMB_ENGINEERED_COLUMNS`, `_comb_matrix`, `_scale_comb_splits`)
