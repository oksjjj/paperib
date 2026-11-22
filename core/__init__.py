"""UI와 독립적으로 모델 학습/평가에서 사용하는 공통 연구 코드 패키지.

의존성 방향:
    models/      ─┐
    evaluation/  ├─> core/
    labeling/    ┘

core는 labeling UI를 import하지 않는다.
"""
