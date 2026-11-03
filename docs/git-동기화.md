# Git 동기화 (Codex · 로컬 공용)

## 추적하는 것 (비식별 유지)

| 경로 | 내용 |
|------|------|
| `data/ib_data/raw/masked_*.csv` | paper 10 PLMN만 · `P####` / `M###` ID |
| `data/ib_data/labels/*_labels.json` | 사람 라벨 |
| `data/ib_data/predictions/*.json` | 모델 예측 + valid 메트릭 |
| `results/**/*.json`, `*.csv` | HP 스윕·요약 지표 |
| `docs/`, `RiAD/`, `RoCA/`, `labeling/` | 코드·문서 |

## 올리지 않는 것

| 경로 | 이유 |
|------|------|
| `*_mapping.txt` | masked ↔ 원본명 역매핑 |
| `data/ib_data/_local_raw_full/` | 전체 사업자 raw |
| `*.pt` / `RiAD/model/` / `RoCA/model/` | 가중치 (용량) |
| `labeling/cache/*.pkl` | 로컬 캐시 (preprocess로 재생성) |

paper raw 재생성:

```bash
.venv/bin/python scripts/export_paper_raw.py
```

## push 후 빌드가 도는 이유 · 끄는 방법

이 repo는 GitHub Pages가 `main`의 `/docs`를 배포하도록 켜져 있어,  
**push마다 `pages-build-deployment`가 실행**됩니다. (앱 웹 기능과 무관 · 문서 정적 배포)

더 이상 웹/Pages가 필요 없으면 **Pages를 끄면** push 후 그 빌드가 멈춥니다.

### UI

1. GitHub → `oksjjj/paperib` → **Settings** → **Pages**
2. Source를 **None** / Disable

### CLI

```bash
gh api -X DELETE repos/oksjjj/paperib/pages
```

확인:

```bash
gh api repos/oksjjj/paperib --jq .has_pages
# false 이면 OK
```

Actions에 다른 workflow가 있으면 **Actions → 해당 workflow → ⋯ → Disable** 도 함께 확인.
