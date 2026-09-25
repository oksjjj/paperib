# Argos × paperib (IB inbound roaming)

[Argos](https://arxiv.org/abs/2501.14170) ([microsoft/argos](https://github.com/microsoft/argos))는 LLM이 **설명 가능한 Python 이상 탐지 규칙**을 생성·수정하고, 배포 시에는 규칙만 실행하는 agentic AD 시스템이다.

이 폴더(`Argos_ib/`)는 그 파이프라인을 **paperib chronological valid 평가**에 맞게 붙인 어댑터이다.

---

## 1. 원본 Argos vs paperib 매핑

| Argos | paperib |
|-------|---------|
| 입력 | **단변량** CSV `value,label,index` |
| 학습 | 라벨을 보며 규칙 생성 (준지도) |
| 평가 fold (`test_df`) | 우리 **valid** |
| paperib **test** | CSV에 넣지 않음 (held-out) |

CSV = train+valid만 export하고, `train_test_split = n_train/(n_train+n_valid)` 로  
Argos의 test fold = paperib valid가 되게 맞춘다.

다변량 PLMN은 Argos가 한 시계열씩 다루므로, paperib 기본은 **multi-metric ensemble**:
핵심 축(`A_RATE`, `M971`, fail 후보)마다 단변량 규칙을 두고 **OR/sum/two_stage**로 융합한다
([논문-스토리라인.md](./논문-스토리라인.md) §3).

---

## 2. 설치

```bash
# 1) 공식 코드 (LLM 모드가 필요할 때만 — heuristic은 Argos_ib만으로 충분)
cd /path/to/paperib
git clone https://github.com/microsoft/argos.git Argos

# 2) Argos 의존성 (권장: 별도 venv 또는 paperib .venv에 추가)
cd Argos
../.venv/bin/pip install -r requirements.txt
```

LLM 키는 repo 루트 ``.env``에 둔다 (``.gitignore`` 처리됨).

**Standard OpenAI (권장):**

```bash
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o
```

**또는 Azure OpenAI:**

```bash
OPENAI_AZURE_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/
OPENAI_AZURE_API_KEY=...
OPENAI_AZURE_API_VERSION=2024-08-01-preview
```

---

## 3. 실행

기본은 **heuristic multi-metric ensemble** (LLM 불필요):

```bash
cd /path/to/paperib
.venv/bin/python Argos_ib/run_plmn.py --plmn P0480 --mode heuristic
```

커스텀 축·융합:

```bash
# OR (default): 어느 축이라도 fire
.venv/bin/python Argos_ib/run_plmn.py --plmn P0480 --mode heuristic \
  --metrics A_RATE M971 M520 --fuse or

# 합산 점수: 2개 이상 축이 동시에 fire
.venv/bin/python Argos_ib/run_plmn.py --plmn P0480 --mode heuristic \
  --metrics A_RATE M971 M520 M965 --fuse sum --score-threshold 2

# 2단계: (rate∪M971) AND fail
.venv/bin/python Argos_ib/run_plmn.py --plmn P0480 --mode heuristic --fuse two_stage
```

단일 metric (레거시):

```bash
.venv/bin/python Argos_ib/run_plmn.py --plmn P0480 --mode heuristic --metric A_RATE
```

LLM 학습 (`train-LLM-only`, API 키 필요):

```bash
.venv/bin/python Argos_ib/run_plmn.py --plmn P0480 --mode train-LLM-only --metric A_RATE
# 또는 ensemble (축마다 LLM — 비용 큼)
.venv/bin/python Argos_ib/run_plmn.py --plmn P0480 --mode train-LLM-only \
  --metrics A_RATE M971 --fuse or --max_iter 5
```

성공 시 `rule*.py` / ensemble 디렉터리와 **paperib valid** point/event 지표,
`predictions/{PLMN}_argos_heuristic_ensemble.json` 을 남긴다.

기본 ensemble 축: `A_RATE`, `M971`.  
`--with-fails`는 사람 라벨의 ``fail_metrics``(예: M855, M037, M841, M185, M965)만 추가한다.  
융합 score는 UI용으로 **−(# fire axes)** 로 저장 (낮을수록 이상).

---

## 4. 평가 지표

| 지표 | 의미 |
|------|------|
| point P/R/F1 | valid 시점 단위 |
| event recall | GT 세그먼트 중 한 점이라도 hit한 비율 |
| n_pred_events | 규칙이 만든 예측 세그먼트 수 |

OA와 공정 비교 시: Argos는 **라벨을 보고 규칙을 학습**하므로 unsupervised OA/POT와 정보량이 다르다. 표에 **준지도(규칙 학습)** 임을 명시할 것.

---

## 5. 산출물

| 경로 | 내용 |
|------|------|
| `data/ib_data/argos/{PLMN}/{metric}.csv` | Argos 입력 |
| `data/ib_data/argos/{PLMN}/{metric}_meta.json` | split·`train_test_split` |
| `data/ib_data/argos/results/...` | 규칙·지표 |
| `data/ib_data/predictions/{PLMN}_argos*.json` | 라벨링 UI용 (ensemble: `*_argos_heuristic_ensemble.json`) |
| `data/ib_data/argos/results/{PLMN}/ensemble/` | 축별 규칙·융합 메타 |

---

## 5. 한계 (문서화)

- 공식 Argos는 **단변량** 중심 → IB는 **metric별 규칙 + 융합(ensemble)** (`Argos_ib/ensemble.py`).
- LLM API 비용·비결정성 → 재현을 위해 규칙 파일·시드를 보관. Ensemble+LLM은 축 수만큼 비용 증가.
- `heuristic`은 Argos LLM 루프가 아니라 **같은 `inference()` 계약의 IB baseline** (앙상블 포함).

---

## 6. 인용

```bibtex
@article{argos,
  title={Argos: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via Large Language Models},
  author={Gu, Yile and Xiong, Yifan and Mace, Jonathan and Jiang, Yuting and Hu, Yigong and Kasikci, Baris and Cheng, Peng},
  journal={arXiv preprint arXiv:2501.14170},
  year={2025}
}
```
