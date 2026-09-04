# OmniAnomaly (PyTorch)

[NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) (KDD 2019) 공식 TensorFlow 구현을 **알고리즘 수준에서 그대로** PyTorch로 포팅한 버전입니다.

## 원본과의 대응

| 항목 | 공식 (TF 1.12 + tfsnippet) | 이 버전 (PyTorch) |
|------|---------------------------|-------------------|
| 프레임워크 | TensorFlow + tfsnippet + TFP | PyTorch 2.x |
| 학습 손실 | `mean(log q − log p(x\|z))` (prior 제외) | **`mean(log q − log p(x\|z) − log p(z))`** (GSSM prior 포함) |
| Posterior | RecurrentDistribution + Planar NF (`u_hat`) | 동일 |
| Prior | LinearGaussianStateSpaceModel | Identity GSSM |
| Early stop | TrainLoop: best valid 가중치만 복원 | 동일 (기본: full epoch + best 복원) |
| Grad clip | `tf.clip_by_norm` (텐서별) | 텐서별 clip (옵션: global) |
| 디바이스 | CUDA | MPS / CUDA / CPU |

의도적인 수정:
1. 원본 `RecurrentDistribution.log_prob_step`은 `[z_t, input_q]`로 concat 하는데, sampling은 `[input_q, z_{t-1}]`입니다. 그대로 두면 SGVB가 붕괴합니다. 이 포팅은 **sampling과 동일한 conditioning**으로 density를 계산합니다.
2. 원본 `OmniAnomaly.get_training_loss`는 `log p(z)`(GSSM)를 SGVB에서 빼지만, 논문의 Linear Gaussian State Space connection을 살리기 위해 이 포팅은 **`log_joint = log p(x|z) + log p(z)`** 로 학습합니다. (`--exclude_prior`로 TF와 동일하게 끌 수 있음)

## 디렉터리

```
OmniAnomaly/
├── main.py                 # 학습 / 평가 엔트리
├── requirements.txt
├── omni_anomaly/           # 모델 · 학습 · 평가 패키지
├── scripts/
│   ├── download_smd.py
│   ├── download_smap_msl.py
│   ├── preprocess_data.py
│   ├── eval_from_scores.py
│   └── viz_gt_anomalies.py
├── data/
│   ├── SMAP/               # 준비된 SMAP pickle
│   ├── MSL/
│   ├── SMD/                # machine-* pickle
│   └── raw/
│       ├── nasa/           # SMAP/MSL 원본
│       └── ServerMachineDataset/  # SMD 원본
├── model/ · result/ · log/
└── viz_gt/ · viz_pred/
```

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 준비

### SMD

```bash
python scripts/download_smd.py
python scripts/preprocess_data.py --dataset SMD
```

### SMAP / MSL

```bash
python scripts/download_smap_msl.py
python scripts/preprocess_data.py --dataset SMAP
python scripts/preprocess_data.py --dataset MSL
```

## 실행

```bash
# level은 데이터셋에 따라 자동 설정 (수동: --level 0.07)
# 디바이스 기본: MPS > CUDA > CPU (--device cuda 로 강제 가능)

python main.py --dataset machine-1-1 --max_epoch 20
python main.py --dataset SMAP --max_epoch 20
python main.py --dataset MSL --max_epoch 20
```

결과·점수는 `result/`, 체크포인트는 `model/{dataset}/`에 저장됩니다.

### TensorBoard

학습 중 scalar는 `log/{dataset}/{exp}/tensorboard/`에 기록됩니다.

```bash
tensorboard --logdir log/SMAP
```

브라우저에서 `http://localhost:6006` 을 여세요. 끄려면 `--no_tensorboard`.

## POT level 권장값 (논문 Appendix B)

| 데이터셋 | level (low quantile) | q |
|----------|----------------------|---|
| SMAP | **0.07** | **1e-4** |
| MSL | 0.01 | **1e-4** |
| SMD group 1 | 0.005 | **1e-4** |
| SMD group 2 | 0.0025 | **1e-4** |
| SMD group 3 | 0.0001 | **1e-4** |

```bash
# 저장된 score로 POT만 재평가 (재학습/재스코어링 없음)
python scripts/eval_from_scores.py --dataset SMAP

# 체크포인트에서 scoring+평가만 (학습 스킵)
python main.py --dataset SMAP --max_epoch 0 --restore_dir model/SMAP/<exp>

# GT / pred 시각화
python scripts/viz_gt_anomalies.py --dataset machine-1-1 --gt_only
python scripts/viz_gt_anomalies.py --dataset machine-1-1 --run_name noprior_noes_paper
```
