"""
recompute.py

순서가 이미 정해진 하루의 시각을 계산한다 — 사용자가 순서·머무는 시간·고정 시각을 손봤을 때 쓴다.

AI 일정 생성(service._build_day_timeline)은 순서까지 스스로 정하지만, 이 모듈은 순서를 건드리지 않고
"이어 붙이기"만 한다: 노드 시작 = 앞 노드 끝 + 이동 시간, 노드 끝 = 시작 + 머무는 시간.
거기에 규칙 몇 개를 얹는다.
  - 고정 시각(pinned_time): 도착이 더 이르면 그 시각까지 기다리고, 더 늦으면 경고
  - 식사 시간대: 기다리지 않고, 맛집이 시간대 밖이면 경고만 (결정은 사용자가 한다)
  - 종료 시간 초과 / 자정 넘김 / 도보로 너무 먼 구간: 경고
  - 머무는 시간이 비어 있으면 엔진의 카테고리·이름 기본값

외부 API를 부르지 않는 순수 계산이라 빠르다(ODsay 미사용 — 이동 시간은 직선거리 추정식).
"""

from features.schedule.utils import _travel_time_for_routing, parse_time_to_minutes
from features.schedule.service import _adjusted_stay, _find_meal_times, _clock_str

# 순서를 바꿀 수 없는 기준점 노드(숙소·출발지) — 머무는 시간이 정해져 있는 일반 노드로 계산에 참여하되 표시만 다르게 한다
ANCHOR_CATEGORIES = ("숙소", "출발지")

_PIN_TOLERANCE_MIN = 5          # 고정 시각보다 이 정도(분) 늦는 건 봐준다
_MISSING_COORD_TRAVEL_MIN = 15  # 좌표가 없는 노드가 낀 구간의 이동 시간 가정
_WALK_UNREALISTIC_MIN = 999     # utils가 "도보로는 불가(5km 초과)"를 나타내는 값


def _travel_minutes(prev: dict, cur: dict, mode: str) -> int:
    """두 노드 사이 이동 시간(분). 카테고리 없이 좌표만 넘겨서 '출발지' 카테고리용 고정값이 붙지 않게 한다."""
    if any(p.get("lat") is None or p.get("lng") is None for p in (prev, cur)):
        return _MISSING_COORD_TRAVEL_MIN
    a = {"lat": prev["lat"], "lng": prev["lng"]}
    b = {"lat": cur["lat"], "lng": cur["lng"]}
    return int(_travel_time_for_routing(a, b, mode))


def _warn(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def recompute_day(
    items: list[dict],
    start_time: str,
    end_time: str,
    transport_mode: str = "대중교통",
    lunch_time: str = "12:00",
    dinner_time: str = "18:30",
) -> dict:
    """
    items: 순서대로 [{name, category, lat, lng, stay_minutes|None, pinned_time|None}, ...]

    Returns:
        {
          "items": [{"index", "time", "end_time", "stay_minutes", "travel_minutes"(다음 노드까지),
                     "anchor", "warnings": [{code, message}]}],
          "end_time": 마지막 노드가 끝나는 시각, "is_over_time", "over_minutes",
          "day_warnings": [{code, message, ...}],
        }
    """
    start_min = parse_time_to_minutes(start_time)
    end_min = parse_time_to_minutes(end_time)
    meal_windows = _find_meal_times(start_min, end_min, {"lunch_time": lunch_time, "dinner_time": dinner_time})

    results: list[dict] = []
    cursor = start_min
    prev: dict | None = None

    for idx, item in enumerate(items):
        warnings: list[dict] = []

        travel_in = 0
        if prev is not None:
            travel_in = _travel_minutes(prev, item, transport_mode)
            # 앞 노드가 "다음 노드까지 이동 시간"을 갖도록 채운다
            results[-1]["travel_minutes"] = travel_in
            if travel_in >= _WALK_UNREALISTIC_MIN:
                warnings.append(_warn("travel_unrealistic", "걸어서 가기엔 너무 멀어요. 이동수단을 바꿔 보세요."))
        arrival = cursor + (0 if travel_in >= _WALK_UNREALISTIC_MIN else travel_in)

        stay = item.get("stay_minutes")
        if stay is None:
            stay = _adjusted_stay({"name": item.get("name", ""), "category": item.get("category", "관광지")}, "normal")
        stay = max(0, int(stay))

        start_i = arrival
        pin_raw = item.get("pinned_time")
        if pin_raw:
            pin = parse_time_to_minutes(pin_raw)
            if pin > arrival:
                start_i = pin  # 고정 시각까지 기다린다
            elif arrival - pin > _PIN_TOLERANCE_MIN:
                warnings.append(_warn("late_for_pin", f"고정 시각 {pin_raw}보다 {arrival - pin}분 늦게 도착해요"))

        if item.get("category") == "맛집" and meal_windows:
            if not any(w["window_start"] <= start_i <= w["window_end"] for w in meal_windows):
                spans = " / ".join(
                    f"{_clock_str(w['window_start'])}~{_clock_str(w['window_end'])}" for w in meal_windows
                )
                warnings.append(_warn("meal_outside_window", f"식사 시간대({spans})가 아니에요"))

        if start_i >= 24 * 60:
            warnings.append(_warn("after_midnight", "자정을 넘겨서 시작해요"))

        end_i = start_i + stay
        results.append({
            "index":          idx,
            "time":           _clock_str(start_i),
            "end_time":       _clock_str(end_i),
            "stay_minutes":   stay,
            "travel_minutes": 0,  # 다음 노드가 오면 위에서 채운다
            "anchor":         item.get("category") in ANCHOR_CATEGORIES,
            "warnings":       warnings,
        })
        cursor = end_i
        prev = item

    last_end = cursor if items else start_min
    over = max(0, last_end - end_min)
    day_warnings: list[dict] = []
    if over > 0:
        movable = [i for i, it in enumerate(items) if it.get("category") not in ANCHOR_CATEGORIES]
        day_warnings.append({
            **_warn("over_end", f"종료 시간({end_time})을 {over}분 넘겨요"),
            "over_minutes": over,
            # 화면이 "마지막 장소를 다른 날로 옮길까요?"를 물을 대상 (기준점 노드는 제외)
            "suggest_move_index": movable[-1] if movable else None,
        })

    return {
        "items":        results,
        "end_time":     _clock_str(last_end),
        "is_over_time": over > 0,
        "over_minutes": over,
        "day_warnings": day_warnings,
    }
