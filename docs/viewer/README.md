# Anomaly Label Viewer (GitHub Pages)

라벨링된 일부 사업자의 시계열과 anomaly 목록을 **정적**으로 보여 주는 뷰어입니다.

- 이동 / 박스 줌 / 줌인·줌아웃 / 전체 (X 변경 시 Y 자동)
- Y+ / Y- / Y 자동 (세로축 확대·축소·자동 맞춤)
- 선 hover 시 메트릭 이름·값
- anomaly 목록 클릭 시 강조 + 구간 이동
- 그래프 위 라벨 영역 클릭 시 선택
- export 시 라벨 주변은 원본 해상도 유지 (+ 전체 min/max 다운샘플)

## 로컬에서 보기

```bash
# 라벨이 있는 PLMN JSON 생성
.venv/bin/python labeling/export_viewer.py

# 정적 서버 (docs/viewer 기준)
cd docs/viewer && ../../.venv/bin/python -m http.server 8765
# → http://127.0.0.1:8765/
```

특정 사업자만:

```bash
.venv/bin/python labeling/export_viewer.py --plmn P0480 P0193
```

## GitHub Pages 설정

1. 이 저장소 **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` (또는 사용 중 브랜치), folder: **/docs**
4. 몇 분 후:
   - `https://<user>.github.io/<repo>/viewer/`

`docs/viewer/data/` 아래 JSON은 공개용으로보내진 데이터이며 **git에 포함**합니다.  
원본 `labeling/labels/*.json`은 gitignore입니다. 라벨을 바꾼 뒤 Pages에 반영하려면:

```bash
.venv/bin/python labeling/export_viewer.py
git add docs/viewer && git commit && git push
```

`export_viewer.py`가 `app.js`/`style.css`에 `?v=타임스탬프`를 붙여서, 푸시 후 캐시를 지우지 않아도 새 화면이 로드되도록 합니다.  
JSON 데이터는 `cache: "no-store"`로 가져옵니다.
메트릭·PLMN **mapping은 적용하지 않습니다**(마스킹 ID만 공개).

사업자 목록은 `labeling/labels/*_labels.json`이 있고 라벨이 1개 이상인 경우만 포함합니다.
라벨이 사라진 PLMN의 `docs/viewer/data/*.json`은 export 시 삭제됩니다.
