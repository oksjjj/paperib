# Baseline · 제안 방법 — 참고 논문

논문 실험에 넣는 방법은 **가급적 기존 논문(또는 공식 구현)에 대응**시킨다.  
스토리·역할: [논문-스토리라인.md](./논문-스토리라인.md).  
하이퍼·명령: [baseline-하이퍼파라미터.md](./baseline-하이퍼파라미터.md).

> **논문 스토리 (2026-11-03 재개정):** contamination은 기존 robust TSAD로 흡수 → **RiAD backbone** → Shared multi-operator → expert operational calibration.  
> Contamination 신규 해법·RiAD vs 다른 robust bake-off는 **비기여**.  
> 상세: [논문-스토리라인.md](./논문-스토리라인.md).

---

## 1. Backbone (메인)

| 역할 | 방법 | 참고 논문 | Venue | 비고 |
|------|------|-----------|-------|------|
| **Contamination-robust backbone** | RiAD | *Rectifying inaccurate unsupervised learning for robust time series anomaly detection* (Rectified Inaccurate Anomaly Detection) | Information Sciences, 2024 | 선정 기준 충족; SOTA/bake-off 주장 없음 |
| **비교 backbone (코드 공개)** | RoCA | *RoCA: Robust Contrastive One-class Time Series Anomaly Detection with Contaminated Data* | arXiv:2503.18385, 2025 | [ruiking04/RoCA](https://github.com/ruiking04/RoCA); paperib `RoCA/` · 기여≠bake-off |

```bibtex
@article{chen2024riad,
  title={Rectifying inaccurate unsupervised learning for robust time series anomaly detection},
  author={Chen, Zejian and Li, Zuoyong and Chen, Xinwei and Chen, Xiaobo and Fan, Haoyi and Hu, Rong},
  journal={Information Sciences},
  volume={662},
  pages={120222},
  year={2024},
  doi={10.1016/j.ins.2024.120222},
  note={RiAD}
}

@article{mou2025roca,
  title={RoCA: Robust Contrastive One-class Time Series Anomaly Detection with Contaminated Data},
  author={Mou, Xudong and Wang, Rui and Li, Bo and Wo, Tianyu and Sun, Jie and Wang, Hui and Liu, Xudong},
  journal={arXiv preprint arXiv:2503.18385},
  year={2025},
  note={RoCA; code https://github.com/ruiking04/RoCA}
}
```

---

## 2. 참고 비교군 · 사전 관찰 (선택)

사전 관찰·동기·(선택) 비robust 대비. 메인 실험의 필수 bake-off가 아님.  
1차 축 = 탐지 원리. “Transformer”를 동급 카테고리로 두지 않음.

| 탐지 원리 | 방법 | 참고 논문 | Venue | 코드 |
|-----------|------|-----------|-------|------|
| **예측 잔차** | LSTM-AD | Malhotra et al., *Long Short Term Memory Networks for Anomaly Detection in Time Series* | ESANN 2015 | `LSTM_AD/` |
| **재구성·우도** | OmniAnomaly | Su et al., *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network* | KDD 2019 | `OmniAnomaly/` |
| **시간 연관성 괴리** | Anomaly Transformer | Xu et al., *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy* | ICLR 2022 | `AnomalyTransformer/` |
| **Operational calibration 옵션** | Argos (+ multi-metric IB) | Gu et al., *Argos: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via Large Language Models* | arXiv:2501.14170 | `Argos_ib/` |

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

## 3. 완전 지도 · 약지도 — **실험 제외, 포지셔닝용 인용**

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

## 4. Classical (IQR + EWMA) — **삭제됨**

### 왜 넣지 않았나

한때 `Classical/`에 feature별 **IQR fence** + **EWMA residual z-score** OR 융합을 두었으나,  
TSAD 벤치에서 “Classical = 이 조합”으로 고정된 **단일 방법 논문이 없다.**

| 요소 | 고전 출처 (Related Work용) | 비고 |
|------|---------------------------|------|
| IQR fence | Tukey, *Exploratory Data Analysis*, 1977 | boxplot outlier fence |
| EWMA | Roberts, *Control Chart Tests Based on Geometric Moving Averages*, Technometrics 1959 | SPC |

“severe는 단순 규칙으로?” 시사점은 **Argos heuristic**이 담당하므로 실험군·코드 모두 제거했다 (`Classical/` 삭제, 2026-09-26).

---

## 5. 구현 vs 원논문 차이 (공정 서술용)

| 방법 | paperib에서 맞춘 것 | 의도적/실용 차이 |
|------|---------------------|------------------|
| LSTM-AD | next-step residual, train 정규 thr | 원논문은 다변량 Gaussian on errors; 우리는 −MSE + train percentile |
| OmniAnomaly | SRNN+NF 재구성, POT | IB chronological split, feature runs |
| Anomaly Transformer | association discrepancy, energy thr | IB prep 공유; POT 대신 train energy percentile |
| Argos | 단변량 `inference` 규칙 | **제안:** multi-metric ensemble (OR/sum/two_stage) |

---

## 6. 여유 시 ablation (아직 비권고)

| 방법 | 논문 | 비고 |
|------|------|------|
| USAD | Audibert et al., KDD 2020 | reconstruction |
| TranAD | Tuli et al., VLDB 2022 | transformer |
| InterFusion | Li et al., KDD 2021 | hierarchical VAE |

1픽(LSTM / OA / AT) + Argos가 안정되기 전에는 넣지 않는다.

---

## 7. 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-09-26 | 초안: 논문 대응 baseline 정리; Classical은 실험 제외 |
| 2026-09-26 | 스토리 한 줄: DL 비지도 → IB 한계 → 준지도 Argos가 더 적합 |
| 2026-09-26 | §2 추가: 완전 지도·약지도 포지셔닝 인용 (실험 제외) |
| 2026-09-26 | `Classical/` 코드 삭제 |
| 2026-09-26 | 비교군 축을 탐지 원리 3종으로 MECE 정리 (Transformer ≠ 동급 카테고리) |
| 2026-11-03 | (구) baseline=3단계 백본; Argos=옵션 |
| 2026-11-03 | 재개정: **RiAD**=메인 backbone; LSTM/OA/AT=참고·관찰; Argos=operational 옵션 |
| 2026-11-03 | 평가: VUS-PR primary · Event-F1 · operational; PA-F1 제외 |
| 2026-11-03 | 비교 backbone **RoCA** (`RoCA/`, 공식 코드 발췌) 추가 |
