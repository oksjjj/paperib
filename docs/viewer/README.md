# Anomaly Label Viewer (GitHub Pages)

웹 뷰어는 **두 탭**으로 구성됩니다.

| 탭 | 데이터 경로 | 내용 |
|----|-------------|------|
| **라벨링** | `data/` | 라벨 있는 사업자 · anomaly UI · 고해상도 (권장 ≤ 20개) |
| **Top 100** | `data-top100/` | `top100.txt` 100개 · 샘플링 시계열 · anomaly UI 없음 |

공통 기능: 이동 / 박스 줌 / **값 탐색** / 줌인·줌아웃 / 전체 · Y+ / Y- / Y 자동 · metric 필터 · hover 패널

## 로컬에서 보기

```bash
# 두 탭 모두 쓰려면 both export (처음 또는 데이터 갱신 후)
.venv/bin/python labeling/export_viewer.py --catalog both

cd docs/viewer && ../../.venv/bin/python -m http.server 8765
# → http://127.0.0.1:8765/
#    #labeled  /  #top100
```

export 옵션:

```bash
.venv/bin/python labeling/export_viewer.py                    # 라벨링 탭만 (data/)
.venv/bin/python labeling/export_viewer.py --catalog top100   # Top 100만 (data-top100/)
.venv/bin/python labeling/export_viewer.py --catalog both     # 둘 다 (git push 전 권장)
.venv/bin/python labeling/export_viewer.py --plmn P0480
.venv/bin/python labeling/export_viewer.py --top100-max-points 8000  # Top 100 용량 조절
```

## GitHub Pages에 반영

```bash
.venv/bin/python labeling/export_viewer.py --catalog both
git add docs/viewer/
git commit -m "Update viewer export"
git push
```

1. 저장소 **Settings → Pages** → Source: branch `main`, folder **/docs**
2. `https://<user>.github.io/<repo>/viewer/`

**git에 포함되는 것**

- `docs/viewer/app.js`, `index.html`, `style.css` — UI (탭·anomaly 숨김 등)
- `docs/viewer/data/` — 라벨링 탭 JSON
- `docs/viewer/data-top100/` — Top 100 탭 JSON (`--catalog top100` 또는 `both`로 생성)

`export_viewer.py`가 `app.js`/`style.css`에 `?v=타임스탬프`를 붙입니다.  
JSON은 `cache: "no-store"`로 로드합니다. mapping은 적용하지 않습니다.
