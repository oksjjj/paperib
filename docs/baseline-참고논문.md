# Baseline · 제안 방법 — 참고 논문

논문 실험에 넣는 방법은 **가급적 기존 논문(또는 공식 구현)에 대응**시킨다.  
스토리·역할: [논문-스토리라인.md](./논문-스토리라인.md).  
하이퍼·명령: [baseline-하이퍼파라미터.md](./baseline-하이퍼파라미터.md).

> **논문 스토리:** baseline = **딥러닝 비지도 TSAD**, 분류는 **탐지 원리(inductive bias) 3종**  
> (예측 잔차 · 재구성·우도 · 시간 연관성 괴리) — “Transformer”를 동급 카테고리로 두지 않음.  
> 통신 IB에서 한계를 보이고 **준지도 rule Argos**가 더 적합함을 주장.  
> (SOTA 레이스가 아니라 **적합성·설명·운영 GT 정합**이 전면.)

---

## 1. 실험 비교군 (권고)

1차 축 = **이상이 score로 번역되는 방식**. 아키텍처는 괄호/비고.

| 탐지 원리 | 방법 | 참고 논문 | Venue | 코드 |
|-----------|------|-----------|-------|------|
| **예측 잔차** | LSTM-AD | Malhotra et al., *Long Short Term Memory Networks for Anomaly Detection in Time Series* | ESANN 2015 | `LSTM_AD/` |
| **재구성·우도** | OmniAnomaly | Su et al., *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network* | KDD 2019 | `OmniAnomaly/` |
| **시간 연관성 괴리** | Anomaly Transformer | Xu et al., *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy* | ICLR 2022 | `AnomalyTransformer/` |
| **준지도 KPI 규칙** (제안) | Argos (+ multi-metric IB) | Gu et al., *Argos: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via Large Language Models* | arXiv:2501.14170 | `Argos_ib/` |

비권장 라벨링: “Forecasting / Reconstruction / **Transformer**” — Transformer만 아키텍처 축이라 MECE가 깨짐.

### BibTeX 스케치

```bibtex
@inproceedings{malhotra2015lstm,
  title={Long Short Term Memory Networks for Anomaly Detection in Time Series},
  author={Malhotra, Pankaj and Vig, Lovekesh and Shroff, Gautam and Agarwal, Puneet},
  booktitle={ESANN},
  year={2015}
}

@inproceedings{su2019omnianomaly,
  title={Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network},
  author={Su, Ya and Zhao, Youjian and Niu, Chenhao and Liu, Rong and Sun, Wei and Pei, Dan},
  booktitle={KDD},
  year={2019}
}

@inproceedings{xu2022anomalytransformer,
  title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
  author={Xu, Jiehui and Wu, Haixu and Wang, Jianmin and Long, Mingsheng},
  booktitle={ICLR},
  year={2022}
}

@article{gu2025argos,
  title={Argos: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via Large Language Models},
  author={Gu, Yile and Castro, Yifan and Dong, Jingchao and Song, Boyi and Geng, Tharuka and Han, Qizheng and others},
  journal={arXiv preprint arXiv:2501.14170},
  year={2025}
}
```

(저자 목록은 최종 제출 전 arXiv/공식 PDF로 재확인.)

---

## 2. 완전 지도 · 약지도 — **실험 제외, 포지셔닝용 인용**

완전 지도 윈도우 분류기는 **실험군이 아니다.**  
스토리에서 “비지도 한계 → 지도가 답” 비약을 막는 데 쓴다. 상세 논증: [논문-스토리라인.md](./논문-스토리라인.md) §2.

| 역할 | 참고 | 비고 |
|------|------|------|
| TSAD 학습 설정 정리 | Zamanzadeh et al., *Deep Learning for Time Series Anomaly Detection: A Survey* (arXiv:2211.05244; ACM CSUR) | supervised는 정상+이상 **완전 라벨** 가정; 실무에선 이상이 미지·미라벨인 경우가 많음 |
| 라벨이 불완전할 때 | Jiang et al., *Weakly Supervised Anomaly Detection: A Survey*, arXiv:2302.04549 | incomplete / inexact / inaccurate; 라벨 비용·불균형 |
| 딥 AD 리뷰 | Pang et al., *Deep Learning for Anomaly Detection: A Review*, ACM Computing Surveys 2021 | 다수는 비지도/준지도; 지도는 라벨 가용성에 종속 |
| (예시) 이상 인지 지도 | Ruff et al., *Deep Semi-Supervised Anomaly Detection* (Deep SAD), ICLR 2020 | 소량 이상 라벨을 쓰는 준지도 — **우리 실험군은 아님**; “라벨을 쓰되 완전 지도와는 다름” 관련 논의용 |

```bibtex
@article{zamanzadeh2024tsad,
  title={Deep Learning for Time Series Anomaly Detection: A Survey},
  author={Zamanzadeh Darban, Zahra and Webb, Geoffrey I. and Pan, Shirui and Aggarwal, Charu and Salehi, Mahsa},
  journal={ACM Computing Surveys},
  year={2024},
  note={arXiv:2211.05244}
}

@article{jiang2023wsad,
  title={Weakly Supervised Anomaly Detection: A Survey},
  author={Jiang, Minqi and Hou, Chaochuan and Zheng, Aobo and others},
  journal={arXiv preprint arXiv:2302.04549},
  year={2023}
}

@article{pang2021deeplr,
  title={Deep Learning for Anomaly Detection: A Review},
  author={Pang, Guansong and Shen, Chunhua and Cao, Longbing and Hengel, Anton Van Den},
  journal={ACM Computing Surveys},
  year={2021}
}
```

**IB에서 완전 지도가 특히 어려운 점(요약):** 미라벨 PLMN 커버리지, 희소·불균형, 신규 이상(열린 집합), mild 경계 노이즈, 설명·감사, “GBDT면 되지?”와의 목표 불일치 — 스토리라인 §2.3.

---

## 3. Classical (IQR + EWMA) — **삭제됨**

### 왜 넣지 않았나

한때 `Classical/`에 feature별 **IQR fence** + **EWMA residual z-score** OR 융합을 두었으나,  
TSAD 벤치에서 “Classical = 이 조합”으로 고정된 **단일 방법 논문이 없다.**

| 요소 | 고전 출처 (Related Work용) | 비고 |
|------|---------------------------|------|
| IQR fence | Tukey, *Exploratory Data Analysis*, 1977 | boxplot outlier fence |
| EWMA | Roberts, *Control Chart Tests Based on Geometric Moving Averages*, Technometrics 1959 | SPC |

“severe는 단순 규칙으로?” 시사점은 **Argos heuristic**이 담당하므로 실험군·코드 모두 제거했다 (`Classical/` 삭제, 2026-09-26).

---

## 4. 구현 vs 원논문 차이 (공정 서술용)

| 방법 | paperib에서 맞춘 것 | 의도적/실용 차이 |
|------|---------------------|------------------|
| LSTM-AD | next-step residual, train 정규 thr | 원논문은 다변량 Gaussian on errors; 우리는 −MSE + train percentile |
| OmniAnomaly | SRNN+NF 재구성, POT | IB chronological split, feature runs |
| Anomaly Transformer | association discrepancy, energy thr | IB prep 공유; POT 대신 train energy percentile |
| Argos | 단변량 `inference` 규칙 | **제안:** multi-metric ensemble (OR/sum/two_stage) |

---

## 5. 여유 시 ablation (아직 비권고)

| 방법 | 논문 | 비고 |
|------|------|------|
| USAD | Audibert et al., KDD 2020 | reconstruction |
| TranAD | Tuli et al., VLDB 2022 | transformer |
| InterFusion | Li et al., KDD 2021 | hierarchical VAE |

1픽(LSTM / OA / AT) + Argos가 안정되기 전에는 넣지 않는다.

---

## 6. 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-09-26 | 초안: 논문 대응 baseline 정리; Classical은 실험 제외 |
| 2026-09-26 | 스토리 한 줄: DL 비지도 → IB 한계 → 준지도 Argos가 더 적합 |
| 2026-09-26 | §2 추가: 완전 지도·약지도 포지셔닝 인용 (실험 제외) |
| 2026-09-26 | `Classical/` 코드 삭제 |
| 2026-09-26 | 비교군 축을 탐지 원리 3종으로 MECE 정리 (Transformer ≠ 동급 카테고리) |
