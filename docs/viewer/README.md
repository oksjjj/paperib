# Anomaly Label Viewer (GitHub Pages)

**라벨이 있는 사업자만** 정적 페이지로 보여 줍니다 (권장 ≤ 20개).  
시계열은 기본 **전체 5분 해상도**에 가깝게 export합니다.

- 이동 / 박스 줌 / **값 탐색**(클릭·←→, 드래그 이동 없음) / 줌인·줌아웃 / 전체
- Y+ / Y- / Y 자동 (세로축 확대·축소·자동 맞춤)
- **표시 metric** 체크리스트 (해제 시 숨김 · 전체 선택/해제 · 화면 합계 내림차순 · 개별 변경 시 Y 자동 · 전체 해제 시 Y 유지)
- 이 시점 특성값 패널 (표시 중인 metric만, 초기부터 표시)
- 빈 영역 시간 툴팁 + 파란 수직 점선
- 선 hover 시 메트릭 이름·값
- anomaly 목록 클릭 시 강조 + 구간 이동
- 라벨링 앱과 같은 mid-tone 선 색
- anomaly 구간/점은 shape 오버레이로 표시 (라벨 JSON 임베드)

## 로컬에서 보기

```bash
# 라벨 있는 PLMN만 (기본) · 고해상도
.venv/bin/python labeling/export_viewer.py

# 정적 서버 (docs/viewer 기준)
cd docs/viewer && ../../.venv/bin/python -m http.server 8765
# → http://127.0.0.1:8765/
```

옵션:

```bash
.venv/bin/python labeling/export_viewer.py --top-n 50      # 순위 상위 N (다운샘플 필요할 수 있음)
.venv/bin/python labeling/export_viewer.py --plmn P0480 P0193
.venv/bin/python labeling/export_viewer.py --max-points 5000  # 용량 줄이기
```

## GitHub Pages 설정

1. 이 저장소 **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` (또는 사용 중 브랜치), folder: **/docs**
4. 몇 분 후:
   - `https://<user>.github.io/<repo>/viewer/`

`docs/viewer/data/` 아래 JSON은 공개용으로보내진 데이터이며 **git에 포함**합니다.  
원본 `labeling/labels/*.json`·`/data/`·mapping은 gitignore입니다. 반영 절차:

```bash
.venv/bin/python labeling/export_viewer.py
git add docs/viewer && git commit && git push
```

`export_viewer.py`가 `app.js`/`style.css`에 `?v=타임스탬프`를 붙여서, 푸시 후 캐시를 지우지 않아도 새 화면이 로드되도록 합니다.  
JSON 데이터는 `cache: "no-store"`로 가져옵니다.  
메트릭·PLMN **mapping은 적용하지 않습니다**(마스킹 ID만 공개).
