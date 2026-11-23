"""TRAIN-only temporal context discovery 로직.

하루 288개 5분 slot의 robust profile을 만들고,
연속성 제약이 있는 Dynamic Programming 분할과 silhouette로 K를 선택한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

def local_slot_ids(local_times: pd.Series) -> np.ndarray:
    """로컬 시각을 5분 단위 slot 번호(00:00=0, ..., 23:55=287)로 변환한다."""
    hour = local_times.dt.hour.to_numpy(dtype=np.int16)
    minute = local_times.dt.minute.to_numpy(dtype=np.int16)
    return (hour * 12 + minute // 5).astype(np.int16)


def slot_label(slot: int) -> str:
    """0~287의 5분 slot 번호를 HH:MM 문자열로 변환한다."""
    hour = int(slot) // 12
    minute = (int(slot) % 12) * 5
    return f"{hour:02d}:{minute:02d}"


def robust_mad(x: np.ndarray, axis: int = 0) -> np.ndarray:
    # MAD는 평균이 아니라 중앙값으로부터의 절대 편차의 중앙값이다. axis=0이면 특성마다 계산한다.
    med = np.median(x, axis=axis)
    return np.median(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)


def build_train_slot_profiles(
    train_raw: np.ndarray,
    train_slots: np.ndarray,
    *,
    use_mad: bool,
) -> dict[str, Any]:
    """TRAIN만 사용해 하루의 각 5분 slot에 대한 robust profile을 만든다.

    먼저 TRAIN raw feature를 RobustScaler로 변환한 뒤,
    같은 slot에 속한 여러 날짜의 값을 median과 선택적으로 MAD로 요약한다.
    TRAIN에 특정 slot이 하나도 없으면 validation/test를 참고하지 않고,
    시계방향 거리상 가장 가까운 TRAIN 관측 slot의 profile을 복사한다.
    """
    # 첫 번째 scaling은 counter마다 다른 크기가 거리 계산에 과도하게 영향을 주지 않게 한다.
    clustering_input_scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    train_scaled = clustering_input_scaler.fit_transform(train_raw)

    observed_slots = np.unique(train_slots).astype(int)
    if observed_slots.size == 0:
        raise RuntimeError("no train time slots observed")

    observed_profiles: dict[int, np.ndarray] = {}
    observed_counts: dict[int, int] = {}

    # 같은 시각(예: 매일 09:00)에 기록된 TRAIN 행들을 묶는다. 날짜 순서를 학습하는 과정은 아니다.
    for slot in observed_slots:
        # block: (해당 slot에서 관측된 날짜 수, F). median profile은 (F,), MAD 포함 시 (2F,)이다.
        block = train_scaled[train_slots == slot]
        med = np.median(block, axis=0)
        if use_mad:
            mad = np.median(np.abs(block - med), axis=0)
            profile = np.concatenate([med, mad], axis=0)
        else:
            profile = med
        observed_profiles[int(slot)] = profile.astype(np.float64)
        observed_counts[int(slot)] = int(len(block))

    all_slots = np.arange(288, dtype=int)
    missing_slots: list[int] = []
    profiles: list[np.ndarray] = []
    counts: list[int] = []

    # 완성할 slot profile은 하루 288개의 고정 길이 표이다.
    for slot in all_slots:
        if int(slot) in observed_profiles:
            profiles.append(observed_profiles[int(slot)])
            counts.append(observed_counts[int(slot)])
            continue

        missing_slots.append(int(slot))
        # 23:55와 00:00도 인접하다고 판단하도록 원형 거리 min(|a-b|,288-|a-b|)를 사용한다.
        circular = np.minimum(
            np.abs(observed_slots - slot),
            288 - np.abs(observed_slots - slot),
        )
        nearest_slot = int(observed_slots[int(np.argmin(circular))])
        profiles.append(observed_profiles[nearest_slot].copy())
        counts.append(0)

    # (288,F 또는 2F)로 쌓은 profile의 열별 scale을 다시 조정하여 SSE와 silhouette에 사용한다.
    profile_matrix = np.vstack(profiles)
    profile_scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )
    profile_matrix_scaled = profile_scaler.fit_transform(profile_matrix)

    return {
        "slots": all_slots,
        "observed_slots": observed_slots,
        "missing_slots": missing_slots,
        "profiles": profile_matrix,
        "profiles_scaled": profile_matrix_scaled,
        "slot_counts": np.asarray(counts, dtype=int),
        "input_scaler": clustering_input_scaler,
        "profile_scaler": profile_scaler,
    }


def build_segment_cost_matrix(
    x: np.ndarray,
    *,
    min_segment_slots: int,
) -> np.ndarray:
    """모든 연속 시간구간 [start, end)에 대해 구간 내부 SSE cost를 계산한다."""
    n, _d = x.shape
    m = int(min_segment_slots)
    # cost[a,b]는 구간 [a,b)의 중심으로부터의 제곱거리 합. 최소 길이 미만은 무한대로 유지한다.
    cost = np.full((n + 1, n + 1), np.inf, dtype=np.float64)

    # prefix sum을 준비하면 모든 구간에 대해 매번 평균/제곱오차를 다시 계산하지 않아도 된다.
    cum_sum = np.vstack(
        [np.zeros((1, x.shape[1]), dtype=np.float64), np.cumsum(x, axis=0)]
    )
    cum_sq = np.concatenate(
        [[0.0], np.cumsum(np.sum(x * x, axis=1), dtype=np.float64)]
    )

    for start in range(n):
        ends = np.arange(start + m, n + 1, dtype=int)
        if ends.size == 0:
            continue
        sums = cum_sum[ends] - cum_sum[start]
        lengths = (ends - start).astype(np.float64)
        # SSE = sum(||x_i||²) - ||sum(x_i)||² / 구간 길이. 각 가능한 끝점을 벡터 연산한다.
        sse = (
            cum_sq[ends]
            - cum_sq[start]
            - np.sum(sums * sums, axis=1) / lengths
        )
        # 부동소수점 오차로 이론상 0인 SSE가 약간 음수가 되는 경우를 보정한다.
        cost[start, ends] = np.maximum(sse, 0.0)

    return cost


def optimal_contiguous_partition(
    cost: np.ndarray,
    *,
    k: int,
    min_segment_slots: int,
) -> tuple[np.ndarray, list[int], float]:
    """00:00~23:55의 288개 slot을 정확히 K개의 연속 구간으로 나눌 때 전체 SSE가 최소인 분할을 찾는다."""
    n = cost.shape[0] - 1
    m = int(min_segment_slots)

    if k < 1 or k * m > n:
        raise ValueError(
            f"infeasible K={k} for n_slots={n}, min_segment_slots={m}"
        )

    # dp[g,j]: 처음 j개 slot을 정확히 g개 연속 구간으로 나눌 때 최소 SSE.
    dp = np.full((k + 1, n + 1), np.inf, dtype=np.float64)
    # prev[g,j]: 최적해의 마지막 구간 시작 위치. 최종 경계를 역추적하기 위한 표이다.
    prev = np.full((k + 1, n + 1), -1, dtype=np.int32)
    # 0개의 slot을 0개 구간으로 분할한 비용은 0이며 나머지 상태는 아직 불가능하다.
    dp[0, 0] = 0.0

    for seg in range(1, k + 1):
        # 남은 구간에도 최소 m개의 slot을 남겨야 하므로 가능한 end만 탐색한다.
        j_min = seg * m
        j_max = n - (k - seg) * m

        for end in range(j_min, j_max + 1):
            start_min = (seg - 1) * m
            start_max = end - m
            starts = np.arange(start_min, start_max + 1, dtype=int)
            # 점화식: 마지막 구간 시작 a 후보마다 dp[g-1,a]+cost[a,j]를 구해 최소값을 선택한다.
            values = dp[seg - 1, starts] + cost[starts, end]
            best_pos = int(np.argmin(values))
            best_start = int(starts[best_pos])
            best_value = float(values[best_pos])

            if np.isfinite(best_value):
                dp[seg, end] = best_value
                prev[seg, end] = best_start

    if not np.isfinite(dp[k, n]):
        raise RuntimeError(f"failed to build contiguous partition for K={k}")

    # dp[K,288]에서 prev를 역추적하여 0과 288을 포함하는 K+1개 경계를 얻는다.
    boundaries = [n]
    end = n
    for seg in range(k, 0, -1):
        start = int(prev[seg, end])
        if start < 0:
            raise RuntimeError(f"invalid DP backtrack for K={k}, seg={seg}")
        boundaries.append(start)
        end = start
    boundaries = sorted(boundaries)

    # 각 연속 구간에 0부터 context ID를 부여한다. 같은 context의 slot은 서로 연속이다.
    labels = np.empty(n, dtype=np.int16)
    for context_id, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        labels[a:b] = context_id

    return labels, boundaries, float(dp[k, n])


def fit_context_segmentation(
    slot_info: dict[str, Any],
    cfg: Any,
) -> dict[str, Any]:
    """TRAIN slot profile만 사용해 K개의 연속 temporal context를 찾는다."""
    x = np.asarray(slot_info["profiles_scaled"], dtype=np.float64)
    n_slots = len(x)

    if n_slots != 288:
        raise RuntimeError(f"expected 288 daily slots, got {n_slots}")

    m = int(cfg.min_segment_slots)
    if m < 1:
        raise RuntimeError("min_segment_slots must be >= 1")

    max_feasible_k = n_slots // m

    # K가 명시되면 한 후보만 평가하고, 아니면 허용 범위 내 여러 K를 평가한다.
    if cfg.k > 0:
        candidate_ks = [int(cfg.k)]
    else:
        lower = max(int(cfg.k_min), 2)
        upper = min(int(cfg.k_max), max_feasible_k, n_slots - 1)
        if lower > upper:
            raise RuntimeError(
                f"invalid K range: {lower}..{upper} "
                f"for min_segment_slots={m}"
            )
        candidate_ks = list(range(lower, upper + 1))

    # K에 무관한 구간 비용표는 한 번만 만들고 모든 후보에서 재사용한다.
    cost = build_segment_cost_matrix(
        x,
        min_segment_slots=m,
    )

    candidates: list[dict[str, Any]] = []
    accepted: list[tuple[float, int, np.ndarray, list[int], float]] = []

    for k in candidate_ks:
        if k < 2 or k > max_feasible_k:
            continue

        labels, boundaries, within_sse = optimal_contiguous_partition(
            cost,
            k=int(k),
            min_segment_slots=m,
        )

        # silhouette는 구간 내부 거리와 다른 구간까지의 거리의 상대적 크기를 비교한다.
        sizes = np.bincount(labels, minlength=int(k))
        sil = float(silhouette_score(x, labels, metric="euclidean"))

        row = {
            "k": int(k),
            "silhouette": sil,
            "within_sse": float(within_sse),
            "min_segment_slots": int(sizes.min()),
            "max_segment_slots": int(sizes.max()),
            "boundaries": [int(v) for v in boundaries],
            "eligible": bool(int(sizes.min()) >= m),
        }
        candidates.append(row)
        accepted.append(
            (sil, int(k), labels.copy(), list(boundaries), float(within_sse))
        )

    if not accepted:
        raise RuntimeError("no feasible contiguous K segmentation")

    # --k를 명시하면 K를 고정하고, 그렇지 않으면 TRAIN profile의 silhouette가 가장 높은 K를 선택한다.
    accepted.sort(key=lambda item: (-item[0], item[1]))
    best_sil, selected_k, labels, boundaries, within_sse = accepted[0]

    segments: list[dict[str, Any]] = []
    for context_id, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segments.append(
            {
                "context_id": int(context_id),
                "start_slot": int(a),
                "end_slot": int(b - 1),
                "start": slot_label(int(a)),
                "end": slot_label(int(b - 1)),
                "n_slots": int(b - a),
            }
        )

    return {
        "selected_k": int(selected_k),
        "selected_silhouette": float(best_sil),
        "selected_within_sse": float(within_sse),
        "slot_to_context": labels.astype(np.int16),
        "segments": segments,
        "boundaries": [int(v) for v in boundaries],
        "missing_slots": list(slot_info.get("missing_slots") or []),
        "candidate_scores": sorted(candidates, key=lambda r: r["k"]),
    }


def context_ids_from_slots(
    slots: np.ndarray,
    slot_to_context: np.ndarray,
) -> np.ndarray:
    # TRAIN에서 확정한 288개 lookup table을 모든 split에 동일하게 적용한다.
    return slot_to_context[np.asarray(slots, dtype=int)].astype(np.int16)
