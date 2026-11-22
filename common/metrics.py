"""이전 import 경로 호환용 wrapper.

새 연구 코드는 core.metrics를 직접 사용한다.
"""

from core.metrics import *  # noqa: F401,F403
