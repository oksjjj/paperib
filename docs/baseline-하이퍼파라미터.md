# Baseline 하이퍼파라미터 · valid 튜닝 가이드

논문 비교군(**Argos 제외**)의 **개선된 기본 하이퍼**, valid 전용 평가, 학습·스윕 명령을 한곳에 둔다.  
스토리·역할: [논문-스토리라인.md](./논문-스토리라인.md).  
파이프라인 공통: [논문-설계-노트.md](./논문-설계-노트.md).

> **범위:** LSTM-AD · OmniAnomaly · Anomaly Transformer.  
> **Classical:** 디렉터리 삭제됨 (원논문 없음 · Argos heuristic과 중복) — [baseline-참고논문.md](./baseline-참고논문.md).  
> **지금 단계:** 하이퍼파라미터 튜닝 → **valid만** 스코어/메트릭/UI export.  
> **UI:** labeling 앱의 Metrics valid|test 토글·`[VALID]`/`[TEST]` 태그는 유지. test 구간이 비어 있으면 test 메트릭만 비어 보임.  
> **최종 표:** 튜닝 확정 후 같은 설정으로 `--with_test` 한 번만 돌려 test 고정 보고.  
> **Primary thr:** label-free (train POT / percentile). valid **best-F1 oracle**은 상한·분석용.

공통:

```bash
cd /path/to/paperib
source .venv/bin/activate   # 또는 .venv/bin/python …
PLMN=P0480
```

Feature run: `paperib` | `comb` | `comb_share` (`M688` 제외는 prep 공통).

---

## 0. 우선순위 (valid만)

| 순위 | 모델 | 왜 | 빠른 레버 |
|------|------|-----|-----------|
| 1 | **AT / LSTM** | thr만으로도 재학습 없이 재평가 | `anormly_ratio` / `anomaly_ratio` |
| 2 | **전 모델** | raw 스케일·mild | `--run_name comb` / `comb_share` |
| 3 | **OA** | mild 복원은 bias; thr·window는 보조 | `--level`, `--window_length`, epoch + `--stable_train` |

기대: percentile thr는 “쓸 만하게” 가능. OA mild 약점은 **스토리상 남는 것**이 자연스러움.

---

## 1. Classical — 삭제됨

IQR+EWMA OR baseline은 **제거** (`Classical/` 삭제).  
이유: [baseline-참고논문.md](./baseline-참고논문.md) §3.

---

## 2. LSTM-AD (forecasting residual)

코드: `LSTM_AD/run_plmn.py`, `LSTM_AD/lstm_ad.py`.  
문서: `LSTM_AD/PAPERIB.md`.  
참고: Malhotra et al., ESANN 2015 — [baseline-참고논문.md](./baseline-참고논문.md).

**valid 관찰 (primary):** percentile thr가 알람을 남발 → **P 낮음**, R은 P보다 높지만 낮은 편.  
best-F1은 초극단 thr로 pred가 거의 안 나옴 → 비교용 아님.  
스토리: [논문-스토리라인.md](./논문-스토리라인.md) §4.3.

### 2.1 하이퍼파라미터 (개선 기본값)

| 이름 | CLI | 기본 | 의미 |
|------|-----|------|------|
| Window | `--window` | **`100`** | 입력 lookback (구 32; OA/AT와 동일) |
| Hidden | `--hidden` | **`128`** | LSTM hidden (구 64) |
| Layers | `--layers` | `2` | LSTM layers |
| Epochs | `--epochs` | **`10`** | OA/AT와 동일 (20은 선택 스윕) |
| Batch | `--batch_size` | `64` | |
| LR | `--lr` | `1e-3` | Adam |
| Anomaly ratio | `--anomaly_ratio` | `1.0` | train −MSE 하위 % → thr (label-free) |
| Score | (고정) | −MSE | 낮을수록 이상 |
| Feature run | `--run_name` | `paperib` | |
| Eval | (default) / `--with_test` | **valid only** | |

### 2.2 valid 개선 포인트

1. **`anomaly_ratio` 그리드** (0.5 / 2 / 5)  
2. **`window` 64 / 50** (짧을 때 residual 민감도)  
3. **`comb` / `comb_share`** — valid 초극단 spike 완화 여부  
4. mild는 residual이 thr 근처면 ratio만으로도 recall↑ 가능; severe spike는 thr를 극단으로 끌어올림(best-F1 PA와 충돌)

### 2.3 학습·평가 명령어

```bash
cd LSTM_AD

# 기본 학습+valid 평가 (window=100, hidden=128, epochs=10)
../.venv/bin/python run_plmn.py --plmn "$PLMN"

# thr 민감도
../.venv/bin/python run_plmn.py --plmn "$PLMN" --anomaly_ratio 0.5
../.venv/bin/python run_plmn.py --plmn "$PLMN" --anomaly_ratio 2.0
../.venv/bin/python run_plmn.py --plmn "$PLMN" --anomaly_ratio 5.0

# window / epoch 스윕
../.venv/bin/python run_plmn.py --plmn "$PLMN" --window 64
../.venv/bin/python run_plmn.py --plmn "$PLMN" --epochs 20

# feature
../.venv/bin/python run_plmn.py --plmn "$PLMN" --run_name comb
../.venv/bin/python run_plmn.py --plmn "$PLMN" --run_name comb_share

# 최종 표용
../.venv/bin/python run_plmn.py --plmn "$PLMN" --with_test
```

산출: `data/ib_data/predictions/{PLMN}_lstm_ad[_{run}].json`,  
`LSTM_AD/model/{PLMN}/{run}/lstm_ad/`.

---

## 3. OmniAnomaly (reconstruction)

코드: `OmniAnomaly/run_plmn.py`, `PlmnConfig`.  
문서: `OmniAnomaly/PAPERIB.md`, [OmniAnomaly-관찰.md](./OmniAnomaly-관찰.md).  
참고: Su et al., KDD 2019 — [baseline-참고논문.md](./baseline-참고논문.md).

### 3.1 하이퍼파라미터

#### CLI에서 자주 쓰는 것

| 이름 | CLI | 기본 | 의미 |
|------|-----|------|------|
| Epochs | `--max_epoch` | `10` | `0`이면 체크포인트만 재스코어 |
| Window | `--window_length` | `100` | 비기본이면 artifact에 `_w{N}` |
| Batch | `--batch_size` | `50` | |
| POT level | `--level` | dataset auto (`default_pot_level`) | SPOT 초기 꼬리 quantile |
| POT q | `--pot_q` | `1e-4` | 위험 수준 |
| Stable train | `--stable_train` | **권장 on** | lr=5e-4, ε=1e-3, global grad clip |
| Feature run | `--run_name` | `paperib` | |
| Split | `--train_ratio` / `--valid_ratio` | `0.6` / `0.2` | |
| Eval | (default) / `--with_test` | **valid only** | |

#### `PlmnConfig` 내부 (코드 기본 · CLI 미노출)

| 이름 | 기본 | 의미 |
|------|------|------|
| `z_dim` | 3 | latent dim |
| `rnn_num_hidden` / `dense_dim` | 500 | GRU / dense |
| `posterior_flow_type` / `nf_layers` | `nf` / 20 | normalizing flow |
| `initial_lr` | `0.001` (`stable`: 5e-4) | |
| `std_epsilon` | `1e-4` (`stable`: 1e-3) | |
| `l2_reg` | `1e-4` | |
| `gradient_clip_norm` | 10 (`stable`: 5, global) | |
| `early_stop` | False | train 내부 holdout |

Score: reconstruction log-prob **합** (높을수록 정상) → anomaly iff `score < thr`.

### 3.2 valid 개선 포인트

1. **POT만:** `--level 0.01/0.02/0.05`, `--pot_q 1e-3` — `--max_epoch 0`으로 재스코어  
2. **`comb` / `comb_share`** — mild score가 thr 쪽으로 움직이는지 ([관찰 문서](./OmniAnomaly-관찰.md))  
3. **`window_length` 50 / 200**  
4. **`max_epoch` 20–30 + `--stable_train`**  
5. mild 미탐은 재구성 bias → 하이퍼로 “고친다”기보다 **한계 서술**이 논문 포지션과 맞음

### 3.3 학습·평가 명령어

```bash
cd OmniAnomaly

# 기본 학습 + valid 평가
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 10 --stable_train

# 체크포인트만 재스코어 + POT 스윕
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 0 --stable_train --level 0.01
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 0 --stable_train --level 0.02
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 0 --stable_train --level 0.05
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 0 --stable_train --pot_q 1e-3

# window / epoch
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 20 --stable_train --window_length 50
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 20 --stable_train --window_length 200

# feature
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 10 --stable_train --run_name comb
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 10 --stable_train --run_name comb_share

# 최종 표용
../.venv/bin/python run_plmn.py --plmn "$PLMN" --max_epoch 0 --stable_train --with_test
```

산출: `data/ib_data/predictions/{PLMN}_omnianomaly[_{run}][_wN].json`,  
`OmniAnomaly/model|result/{PLMN}/{run}/`.

---

## 4. Anomaly Transformer

코드: `AnomalyTransformer/run_plmn.py`, `plmn_engine.py`.  
문서: `AnomalyTransformer/PAPERIB.md`.  
참고: Xu et al., ICLR 2022 — [baseline-참고논문.md](./baseline-참고논문.md).

### 4.1 하이퍼파라미터 (기본값)

| 이름 | CLI | 기본 | 의미 |
|------|-----|------|------|
| Epochs | `--num_epochs` | `10` | `0` = 재스코어만 |
| Window | `--win_size` | `100` | |
| Batch | `--batch_size` | `64` | |
| LR | `--lr` | `1e-4` | |
| Association loss weight | `--k` | `3.0` | 공식 AT `k` |
| Anomaly ratio | `--anormly_ratio` | `1.0` | train energy 상위 (100−r)% → thr |
| `d_model` | `--d_model` | `512` | |
| `e_layers` | `--e_layers` | `3` | encoder layers |
| Train / score step | `--train_step` / `--score_step` | `1` | 슬라이딩 stride (1=최밀·느림) |
| Feature run | `--run_name` | `paperib` | |
| Eval | (default) / `--with_test` | **valid only** | |

UI score: `log10(thr) − log10(energy)` (thr에서 0; anomaly iff score &lt; 0).

### 4.2 valid 개선 포인트

1. **`anormly_ratio` 0.5 / 1 / 2 / 5** — thr만; `--num_epochs 0`  
2. **`k` 1 / 5**  
3. **`win_size` 50 / 200**  
4. feature `comb`  
5. epoch 20

### 4.3 학습·평가 명령어

```bash
cd AnomalyTransformer

# 기본 학습 + valid 평가
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 10

# thr 스윕 (재학습 없음)
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 0 --anormly_ratio 0.5
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 0 --anormly_ratio 2.0
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 0 --anormly_ratio 5.0

# k / window
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 10 --k 1.0
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 10 --k 5.0
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 10 --win_size 50
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 20

# feature
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 10 --run_name comb

# 최종 표용
../.venv/bin/python run_plmn.py --plmn "$PLMN" --num_epochs 0 --with_test
```

산출: `data/ib_data/predictions/{PLMN}_anomalytransformer[_{run}].json`,  
`AnomalyTransformer/model|result/{PLMN}/{run}/`.

---

## 5. 공통 평가·UI 메모

| 항목 | 내용 |
|------|------|
| Split | train 60 / valid 20 / test 20 |
| Primary thr | train만 (POT / percentile) |
| Runner 기본 | **valid만** 스코어·메트릭·prediction JSON |
| Runner 최종 | `--with_test` → valid+test |
| UI Metrics | valid ↔ test 전환 유지; oracle thr는 **valid 라벨**로만 탐색 |
| 보고 | point + event; mild/severe(`reasons`) breakdown |
| Argos | 이 문서 범위 밖 — [Argos-IB.md](./Argos-IB.md); prediction 유지 |

valid 스윕 후 최종 표는 **같은 설정으로 `--with_test`** 한 번만 고정해 보고할 것.

---

## 6. 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-09-25 | 초안: 4 baseline 하이퍼·valid 개선 명령 |
| 2026-09-26 | 평가 산출물 리셋; runner 기본=valid only + `--with_test`; Classical/LSTM 개선 기본값 반영 |
| 2026-09-26 | Classical 실험 제외; 참고 논문 링크 (`baseline-참고논문.md`); LSTM window=100 |
| 2026-09-26 | `Classical/` 디렉터리 삭제 |
