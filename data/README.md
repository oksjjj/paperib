# data/

인바운드 로밍 사업자(IB) 이상 탐지 실험에 사용하는 데이터 디렉터리입니다.

```text
data/
└── ib_data/
    ├── raw/              # 비식별화 CSV 및 로컬 전용 매핑 파일
    ├── _local_raw_full/  # 전체 사업자 원본 데이터 (로컬 전용)
    ├── cache/            # 공통 PLMN 로딩 캐시 (Git 제외)
    ├── labels/           # 전문가 라벨 및 PLMN 순위
    └── predictions/      # 모델 예측 결과
```

## 데이터 준비 및 UI 실행

저장소 루트에서 실행합니다. 비식별화 CSV는 `data/ib_data/raw/`에 둡니다.

```bash
python scripts/build_ib_cache.py
python ui/app.py
```

전체 사업자 데이터에서 논문 대상 사업자만 추출하는 방법과 매핑 파일 보안 규칙은 [ib_data/README.md](ib_data/README.md)를 참고하세요.
