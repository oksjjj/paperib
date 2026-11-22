"""이전 import 경로 호환용 wrapper.

새 연구 코드는 core.data를 직접 사용한다.
기존 스크립트가 common.data를 import해도 동일하게 동작하도록
core.data의 공개 함수를 다시 export한다.
"""

from core.data import *  # noqa: F401,F403
