"""
service.py

generate_itinerary 를 역할별로 분리:
  1. _enrich_places         - 장소 데이터 보강 (체류시간 등)
  2. _build_pinned_info      - 핀 장소 인덱싱
  3. _assign_places_to_days  - 장소 → 날짜 배정
  4. _redistribute           - 하루 최대치 초과분 재분배
  5. _apply_pinned           - 핀 장소 통합 + 핀 시간 충돌 사전 검사
  6. _build_day_timeline     - 하루 타임라인 생성
  7. generate_itinerary      - 진입점 (조합만 담당)
"""

import logging
from collections import defaultdict

from features.schedule.utils import (
    calculate_travel_time, optimized_route, nearest_neighbor_route,
    parse_time_to_minutes, minutes_to_time_str, calculate_dates,
    check_duplicate_places, check_category_sequence,
    haversine_distance,
)
from features.schedule.models import UserPreferences, HotelStay, DeparturePoint

logger = logging.getLogger(__name__)

# ─── 상수 ──────────────────────────────────────────────────────

_BASE_STAY: dict[str, int] = {
    "관광지": 90,
    "맛집":   60,
    "카페":   40,
    "쇼핑":   60,
    "출발지": 120,
    "숙소":   0,
}
_LANDMARK_BONUS = 30
_UNIQUE_BONUS   = 20

PACE_MULTIPLIER: dict[str, float] = {"tight": 0.7, "normal": 1.0, "relaxed": 1.3}

HOTEL_MOVE_THRESHOLD_KM = 30


# ─── 다중 숙소 헬퍼 ────────────────────────────────────────────

def _get_hotel_for_night(hotels: list[HotelStay], day_idx: int) -> dict | None:
    """day_idx(0-based) 밤을 보낼 숙소 반환"""
    day_num = day_idx + 1
    for h in hotels:
        if h.check_in_day <= day_num < h.check_out_day:
            return {"name": h.name, "lat": h.lat, "lng": h.lng, "address": h.address or ""}
    return None


def _is_hotel_move_day(day_idx: int, hotels: list[HotelStay]) -> bool:
    if day_idx == 0:
        return False
    prev = _get_hotel_for_night(hotels, day_idx - 1)
    curr = _get_hotel_for_night(hotels, day_idx)
    if not prev or not curr or prev["name"] == curr["name"]:
        return False
    return haversine_distance(prev, curr) >= HOTEL_MOVE_THRESHOLD_KM


def _max_places_for_day(day_idx: int, hotels: list[HotelStay], default: int) -> int:
    return max(2, default // 2) if _is_hotel_move_day(day_idx, hotels) else default


# ─── 다중 출발지 헬퍼 ─────────────────────────────────────────

def _get_departure_for_day(
    day_idx: int,
    departure_points: list[DeparturePoint],
    hotels: list[HotelStay],
) -> dict | None:
    day_num = day_idx + 1
    for dp in departure_points:
        if dp.day == day_num:
            return {"name": dp.name, "lat": dp.lat, "lng": dp.lng,
                    "address": dp.address or "", "category": "출발지"}
    if day_idx > 0:
        return _get_hotel_for_night(hotels, day_idx - 1)
    return None


def _get_return_point(departure_points: list[DeparturePoint]) -> dict | None:
    candidates = [dp for dp in departure_points if dp.is_return_point]
    if candidates:
        dp = min(candidates, key=lambda x: x.day)
    else:
        day1 = [dp for dp in departure_points if dp.day == 1]
        dp = day1[0] if day1 else None
    if dp is None:
        return None
    return {"name": dp.name, "lat": dp.lat, "lng": dp.lng, "address": dp.address or ""}


# ─── 1. 장소 보강 ──────────────────────────────────────────────

def _adjusted_stay(place_dict: dict, pace: str) -> int:
    base = _BASE_STAY.get(place_dict.get("category", "관광지"), 60)
    if place_dict.get("is_landmark"):
        base += _LANDMARK_BONUS
    if place_dict.get("is_unique"):
        base += _UNIQUE_BONUS
    return int(base * PACE_MULTIPLIER.get(pace, 1.0))


def _enrich_places(places, pace: str) -> list[dict]:
    return [
        {
            "name":        p.name,
            "lat":         p.lat,
            "lng":         p.lng,
            "category":    p.category,
            "address":     p.address or "",
            "is_landmark": p.is_landmark,
            "is_unique":   p.is_unique,
            "pinned":      False,
            "stay": _adjusted_stay(
                {"category": p.category, "is_landmark": p.is_landmark, "is_unique": p.is_unique},
                pace,
            ),
        }
        for p in places
    ]


# ─── 2. 핀 정보 구성 ───────────────────────────────────────────

def _build_pinned_info(pinned_places, enriched: list[dict]) -> dict:
    pinned_info: dict = {}
    for pin in pinned_places:
        if 0 <= pin.place_index < len(enriched):
            pinned_info[pin.place_index] = {
                "day":   pin.day,
                "time":  pin.time,
                "place": enriched[pin.place_index].copy(),
            }
    return pinned_info


# ─── 3. 장소 → 날짜 배정 ─────────────────────────────────────

def _geographic_cluster(
    places: list[dict],
    n_days: int,
    max_iter: int = 10,
) -> list[list[dict]]:
    """
    앵커가 없을 때 사용하는 2D 거리 기반 k-means 클러스터링.

    위도+경도를 동시에 고려하므로 동서로 넓게 퍼진 지역에서도
    위도 단일 기준보다 훨씬 정확한 배정을 합니다.

    알고리즘:
      1. 위도+경도 정렬 후 균등 간격으로 초기 centroid n개 선택
      2. 각 장소를 가장 가까운 centroid에 배정
      3. 각 클러스터의 무게중심으로 centroid 갱신
      4. 2~3 수렴 (max_iter 상한)
      5. 날짜별 불균형이 크면 균등 보정
    """
    if not places:
        return [[] for _ in range(n_days)]

    if len(places) <= n_days:
        # 장소 수가 날짜 수 이하면 그냥 하루에 하나씩
        assignments: list[list[dict]] = [[] for _ in range(n_days)]
        for i, p in enumerate(places):
            assignments[i % n_days].append(p)
        return assignments

    # ── 1. 초기 centroid: 위도+경도 합산 정렬 후 균등 선택 ──
    sorted_places = sorted(places, key=lambda p: p["lat"] + p["lng"])
    step = max(1, len(sorted_places) // n_days)
    centroids: list[dict] = [
        {"lat": sorted_places[i * step]["lat"], "lng": sorted_places[i * step]["lng"]}
        for i in range(n_days)
    ]

    assignments = [[] for _ in range(n_days)]

    for _ in range(max_iter):
        new_assignments: list[list[dict]] = [[] for _ in range(n_days)]

        # ── 2. 각 장소를 가장 가까운 centroid로 배정 ──
        for place in places:
            best = min(
                range(n_days),
                key=lambda i: haversine_distance(centroids[i], place),
            )
            new_assignments[best].append(place)

        # ── 3. centroid 갱신 ──
        new_centroids: list[dict] = []
        for i in range(n_days):
            if new_assignments[i]:
                new_centroids.append({
                    "lat": sum(p["lat"] for p in new_assignments[i]) / len(new_assignments[i]),
                    "lng": sum(p["lng"] for p in new_assignments[i]) / len(new_assignments[i]),
                })
            else:
                new_centroids.append(centroids[i])  # 빈 클러스터는 기존 centroid 유지

        # ── 수렴 판단: centroid 이동 거리가 모두 10m 미만이면 종료 ──
        if all(
            haversine_distance(centroids[i], new_centroids[i]) < 0.01
            for i in range(n_days)
        ):
            assignments = new_assignments
            break

        centroids = new_centroids
        assignments = new_assignments

    # ── 4. 불균형 보정: 가장 많은 날에서 가장 적은 날로 이동 ──
    for _ in range(n_days * len(places)):
        counts = [len(assignments[i]) for i in range(n_days)]
        max_day = max(range(n_days), key=lambda i: counts[i])
        min_day = min(range(n_days), key=lambda i: counts[i])
        if counts[max_day] - counts[min_day] < 2:
            break
        # min_day centroid와 가장 가까운 장소를 이동
        if assignments[max_day]:
            move_idx = min(
                range(len(assignments[max_day])),
                key=lambda i: haversine_distance(centroids[min_day], assignments[max_day][i]),
            )
            assignments[min_day].append(assignments[max_day].pop(move_idx))

    return assignments


def _assign_places_to_days(
    enriched: list[dict],
    pinned_info: dict,
    n_days: int,
    hotels: list[HotelStay],
    departure_points: list[DeparturePoint],
) -> tuple[list[list[dict]], list[dict | None]]:
    """
    숙소·출발지 좌표를 기준으로 직접 배정.
    앵커가 전혀 없으면 지리적 클러스터링(위도 기반)으로 대체.
    """
    hotel_places_as_anchor = [p for p in enriched if p.get("category") == "숙소"]

    def _get_anchor_for_day_with_fallback(day_idx: int) -> dict | None:
        anchor = _get_departure_for_day(day_idx, departure_points, hotels)
        if anchor is None:
            anchor = _get_hotel_for_night(hotels, day_idx)
        if anchor is None and day_idx > 0:
            anchor = _get_hotel_for_night(hotels, day_idx - 1)
        if anchor is None and hotel_places_as_anchor:
            anchor = hotel_places_as_anchor[0]
        return anchor

    day_anchors: list[dict | None] = [
        _get_anchor_for_day_with_fallback(i) for i in range(n_days)
    ]

    pinned_names = {info["place"]["name"] for info in pinned_info.values()}
    hotel_names     = {h.name for h in hotels}
    departure_names = {dp.name for dp in departure_points}

    unassigned = [
        p for p in enriched
        if p["name"] not in pinned_names
        and p["name"] not in hotel_names
        and p["name"] not in departure_names
        and p.get("category") != "숙소"
    ]

    anchored_days = [i for i, a in enumerate(day_anchors) if a is not None]

    # ── 앵커가 아예 없으면 지리적 클러스터링으로 대체 ──────
    if not anchored_days:
        logger.info("[배정] 앵커 없음 → 위도 기반 지리적 클러스터링")
        assignments = _geographic_cluster(unassigned, n_days)
        logger.info("[배정] n_days=%d, 총 장소=%d개", n_days, len(unassigned))
        for i, day_places in enumerate(assignments):
            names = ", ".join(p["name"] for p in day_places) or "(없음)"
            logger.debug("  Day %d → %s", i + 1, names)
        return assignments, day_anchors

    # ── 앵커 좌표 기준 그룹화 ─────────────────────────────
    def _anchor_key(day_idx: int) -> tuple:
        a = day_anchors[day_idx]
        if a is None:
            return (None, None)
        return (round(a["lat"], 5), round(a["lng"], 5))

    anchor_groups: dict[tuple, list[int]] = {}
    for i in anchored_days:
        key = _anchor_key(i)
        anchor_groups.setdefault(key, []).append(i)

    assignments: list[list[dict]] = [[] for _ in range(n_days)]

    for place in unassigned:
        best_anchor_key = min(
            anchor_groups.keys(),
            key=lambda k: haversine_distance(
                {"lat": k[0], "lng": k[1]}, place
            ) if k[0] is not None else float("inf"),
        )
        group = anchor_groups[best_anchor_key]
        best_day = min(group, key=lambda i: len(assignments[i]))
        assignments[best_day].append(place)

    # ── 날짜별 균형 보정 ──────────────────────────────────
    for _ in range(n_days * len(unassigned)):
        counts = [len(assignments[i]) for i in range(n_days)]
        max_day = max(range(n_days), key=lambda i: counts[i])
        min_day = min(range(n_days), key=lambda i: counts[i])
        if counts[max_day] - counts[min_day] < 2:
            break
        min_anchor = day_anchors[min_day]
        if min_anchor and assignments[max_day]:
            move_idx = min(
                range(len(assignments[max_day])),
                key=lambda i: haversine_distance(min_anchor, assignments[max_day][i]),
            )
        else:
            move_idx = -1
        place_to_move = assignments[max_day].pop(move_idx)
        assignments[min_day].append(place_to_move)

    logger.info("[배정] n_days=%d, 총 장소=%d개", n_days, len(unassigned))
    for i, day_places in enumerate(assignments):
        anchor = day_anchors[i]
        anchor_name = anchor["name"] if anchor else "없음"
        names = ", ".join(p["name"] for p in day_places) or "(없음)"
        logger.debug("  Day %d (앵커: %s) → %s", i + 1, anchor_name, names)

    return assignments, day_anchors


# ─── 4. 재분배 ────────────────────────────────────────────────

def _redistribute(
    day_assignments: list[list],
    day_anchors: list[dict | None],
    max_per_day: int,
    hotels: list,
    transport_mode: str,
) -> list[list]:
    n_days = len(day_assignments)
    result = [list(d) for d in day_assignments]

    for day_idx in range(n_days):
        limit = _max_places_for_day(day_idx, hotels, max_per_day)
        if len(result[day_idx]) <= limit:
            continue

        anchor = day_anchors[day_idx]
        if anchor:
            result[day_idx].sort(
                key=lambda p: haversine_distance(anchor, p), reverse=True
            )

        overflow = result[day_idx][limit:]
        result[day_idx] = result[day_idx][:limit]

        for place in overflow:
            placed = False
            for target in range(day_idx + 1, n_days):
                target_limit = _max_places_for_day(target, hotels, max_per_day)
                if len(result[target]) < target_limit:
                    result[target].append(place)
                    logger.debug("  [재분배] '%s' Day %d → Day %d", place["name"], day_idx + 1, target + 1)
                    if day_anchors[target]:
                        result[target] = optimized_route(result[target], day_anchors[target], transport_mode)
                    placed = True
                    break
            if not placed:
                for target in range(day_idx - 1, -1, -1):
                    target_limit = _max_places_for_day(target, hotels, max_per_day)
                    if len(result[target]) < target_limit:
                        result[target].append(place)
                        logger.debug("  [재분배 역방향] '%s' Day %d → Day %d", place["name"], day_idx + 1, target + 1)
                        if day_anchors[target]:
                            result[target] = optimized_route(result[target], day_anchors[target], transport_mode)
                        placed = True
                        break
            if not placed:
                result[day_idx].append(place)
                logger.warning("  [재분배 실패] '%s' 넘길 날 없음, Day %d 유지", place["name"], day_idx + 1)

    return result


# ─── 5. 핀 장소 통합 ──────────────────────────────────────────

def _check_pin_time_conflicts(
    timed_pins: list[dict],
    regular_places: list[dict],
    current_min: int,
    transport_mode: str,
    pace: str,
) -> list[str]:
    warnings_list = []
    prev_time = current_min
    prev_place = None

    for pin in timed_pins:
        pin_min = parse_time_to_minutes(pin["time"])
        if prev_place is not None:
            travel = calculate_travel_time(prev_place, pin["place"], transport_mode)
            earliest_arrival = prev_time + int(travel)
            if earliest_arrival > pin_min:
                warnings_list.append(
                    f"'{pin['place']['name']}' {pin['time']} 도착은 "
                    f"이전 일정 기준 최소 {minutes_to_time_str(earliest_arrival)} 이후 가능 "
                    f"(이동시간 추정 {int(travel)}분 포함)"
                )
        prev_time  = pin_min + pin["place"].get("stay", 60)
        prev_place = pin["place"]

    return warnings_list


def _apply_pinned(
    day_places: list,
    pinned_info: dict,
    day_num: int,
    current_start_min: int,
    transport_mode: str,
    pace: str,
) -> tuple[list, list[str]]:
    pins = [info for info in pinned_info.values() if info["day"] == day_num]
    if not pins:
        return day_places, []

    pinned_names = {p["place"]["name"] for p in pins}
    regular      = [p for p in day_places if p["name"] not in pinned_names]

    timed = sorted(
        [p for p in pins if p.get("time")],
        key=lambda x: parse_time_to_minutes(x["time"]),
    )
    untimed = [p for p in pins if not p.get("time")]

    pin_warnings = _check_pin_time_conflicts(
        timed, regular, current_start_min, transport_mode, pace
    )

    untimed_places = []
    for pin in untimed:
        pl = pin["place"].copy()
        pl["pinned"] = True
        untimed_places.append(pl)

    if not timed:
        pool = regular + untimed_places
        if not pool:
            return [], pin_warnings
        start = pool[0]
        return optimized_route(pool, start, transport_mode), pin_warnings

    anchors = []
    for pin in timed:
        pl = pin["place"].copy()
        pl["pinned"]      = True
        pl["pinned_time"] = pin["time"]
        anchors.append(pl)

    remaining  = regular + untimed_places
    used_names: set[str] = set()
    result: list = []
    prev_anchor = None

    for anchor in anchors:
        available = [p for p in remaining if p["name"] not in used_names]
        if available and prev_anchor:
            segment = optimized_route(available, prev_anchor, transport_mode)
            for p in segment:
                result.append(p)
                used_names.add(p["name"])
        result.append(anchor)
        used_names.add(anchor["name"])
        prev_anchor = anchor

    leftover = [p for p in remaining if p["name"] not in used_names]
    if leftover and prev_anchor:
        result.extend(optimized_route(leftover, prev_anchor, transport_mode))

    return result, pin_warnings


# ─── 6. 하루 타임라인 생성 ────────────────────────────────────

def _find_meal_times(start_min: int, end_min: int, prefs: dict) -> list:
    lunch  = parse_time_to_minutes(prefs.get("lunch_time",  "12:00"))
    dinner = parse_time_to_minutes(prefs.get("dinner_time", "18:00"))
    result = []
    if 11*60+30 <= lunch  <= 14*60 and start_min <= lunch  <= end_min:
        result.append({"type": "lunch",  "time": lunch,
                       "time_str": minutes_to_time_str(lunch),
                       "window_start": lunch  - 30, "window_end": lunch  + 30})
    if 17*60+30 <= dinner <= 20*60 and start_min <= dinner <= end_min:
        result.append({"type": "dinner", "time": dinner,
                       "time_str": minutes_to_time_str(dinner),
                       "window_start": dinner - 30, "window_end": dinner + 30})
    return result


def _arrange_meals_in_places(
    places: list[dict],
    meal_times: list,
    current_start_min: int,
    avg_stay: int,
    avg_travel: int = 20,
) -> list[dict]:
    """
    정렬된 장소 리스트(sorted_places)에서 맛집을 식사 시간대에 맞게 재배치.

    타임라인 빌드 전(time 필드 없는 상태)에 호출합니다.
    각 장소의 예상 도착 시간을 current_start_min + 누적 (체류+이동)시간으로 추정하고,
    맛집이 식사 window 밖에 있으면 window에 가장 가까운 위치로 이동합니다.

    제약:
    - pinned 장소는 이동하지 않습니다.
    - 맛집이 없거나 meal_times가 비어있으면 그대로 반환합니다.
    - time 필드에 의존하지 않으므로 _build_day_timeline 어느 단계에서도
      안전하게 호출할 수 있습니다.
    """
    if not meal_times or not places:
        return places

    result = list(places)

    for meal in meal_times:
        # 각 장소의 추정 도착 분 계산 (체류 + 이동시간 모두 포함)
        def _estimated_minute(idx: int) -> int:
            return current_start_min + sum(
                result[j].get("stay", avg_stay) + avg_travel for j in range(idx)
            )

        # 이미 window 안에 맛집이 있으면 스킵
        in_window = any(
            result[i].get("category") == "맛집"
            and meal["window_start"] <= _estimated_minute(i) <= meal["window_end"]
            for i in range(len(result))
        )
        if in_window:
            continue

        # unpinned 맛집 후보 중 첫 번째 선택
        candidate_idx = next(
            (i for i, p in enumerate(result)
             if p.get("category") == "맛집" and not p.get("pinned")),
            None
        )
        if candidate_idx is None:
            continue

        # 식사 목표 시간에 가장 가까운 삽입 위치 탐색
        target_min = meal["time"]
        best_insert_idx = 1
        for i in range(len(result)):
            if _estimated_minute(i) <= target_min:
                best_insert_idx = i + 1

        best_insert_idx = min(best_insert_idx, len(result))

        # 위치 이동
        candidate = result.pop(candidate_idx)
        if candidate_idx < best_insert_idx:
            best_insert_idx -= 1
        result.insert(best_insert_idx, candidate)

        logger.debug(
            "[식사 배치] '%s' → 인덱스 %d (%s window)",
            candidate.get("name", "?"), best_insert_idx, meal["type"]
        )

    return result

def human_like_route(day_places):
    관광지 = [p for p in day_places if p["category"] == "관광지"]
    맛집   = [p for p in day_places if p["category"] == "맛집"]
    카페   = [p for p in day_places if p["category"] == "카페"]
    기타   = [p for p in day_places if p["category"] not in ["관광지", "맛집", "카페"]]

    # 랜드마크 먼저 (경복궁 같은 거)
    관광지.sort(key=lambda p: not p.get("is_landmark", False))

    result = []

    # 오전 관광
    result += 관광지[:2]

    # 점심
    if 맛집:
        result.append(맛집.pop(0))

    # 카페
    if 카페:
        result.append(카페.pop(0))

    # 오후 관광
    result += 관광지[2:4]

    # 저녁
    if 맛집:
        result.append(맛집.pop(0))

    # 남은 것들
    result += 관광지[4:] + 카페 + 맛집 + 기타

    return result

def _build_day_timeline(
    day_idx: int,
    day_places: list,
    hotels: list[HotelStay],
    departure_points: list[DeparturePoint],
    return_point: dict | None,
    n_days: int,
    transport_mode: str,
    daily_start_time: str,
    daily_end_time: str,
    prefs_dict: dict,
    pinned_info: dict,
    pace: str,
) -> dict:
    start_h, start_m = map(int, daily_start_time.split(":"))
    end_h,   end_m   = map(int, daily_end_time.split(":"))
    current_min = start_h * 60 + start_m

    tonight_hotel  = _get_hotel_for_night(hotels, day_idx)
    start_location = _get_departure_for_day(day_idx, departure_points, hotels)
    is_move_day    = _is_hotel_move_day(day_idx, hotels)
    explicit_departure = next(
        (dp for dp in departure_points if dp.day == day_idx + 1), None
    )

    day_places, pin_warnings = _apply_pinned(
        day_places, pinned_info, day_idx + 1,
        current_min, transport_mode, pace,
    )

    if not day_places:
        return _empty_day(day_idx, daily_start_time, daily_end_time, pin_warnings)

    timeline: list[dict] = []
    total_travel, total_stay = 0.0, 0
    has_departure = False
    sorted_places: list[dict] = []

    # ── 명시적 출발지가 있는 날 ────────────────────────────────
    if explicit_departure:
        dep_dict = {
            "name": explicit_departure.name,
            "lat":  explicit_departure.lat,
            "lng":  explicit_departure.lng,
            "address": explicit_departure.address or "",
        }
        # 공항/항구 등 대기가 긴 거점은 120분, 그 외(기차역·버스터미널 등)는 30분
        _LONG_WAIT_KEYWORDS = ["공항", "airport", "인천", "김포", "제주", "항구", "페리"]
        if day_idx == 0:
            stay_min = 120 if any(k in explicit_departure.name for k in _LONG_WAIT_KEYWORDS) else 30
        else:
            stay_min = 0

        sorted_places = human_like_route(day_places)
        first_travel = int(calculate_travel_time(dep_dict, sorted_places[0], transport_mode)) if sorted_places else 0

        timeline.append({
            "place":           explicit_departure.name,
            "category":        "출발지",
            "address":         explicit_departure.address or "",
            "lat":             explicit_departure.lat,
            "lng":             explicit_departure.lng,
            "stay_minutes":    stay_min,
            "travel_minutes":  first_travel,
            "time":            minutes_to_time_str(current_min),
            "type":            "arrival" if day_idx == 0 else "transfer_start",
            "pinned":          False,
        })
        total_stay  += stay_min
        current_min += stay_min
        has_departure = True

        if sorted_places:
            total_travel += first_travel
            current_min  += first_travel

    # ── 출발지 없음: 숙소 or 첫 장소 기준 ────────────────────
    else:
        if start_location:
            sorted_places = human_like_route(day_places)
            if is_move_day:
                prev_hotel = _get_hotel_for_night(hotels, day_idx - 1)
                if prev_hotel and sorted_places:
                    t = calculate_travel_time(prev_hotel, sorted_places[0], transport_mode)
                    total_travel += t
                    current_min  += int(t)
        else:
            sorted_places = human_like_route(day_places)

    # ── 식사 시간 배치 ────────────────────────────────────────
    # sorted_places(time 필드 없음) 상태에서 직접 호출.
    # 누적 체류시간 기반 추정으로 window를 판단하므로 time 필드 주입 불필요.
    meal_times = _find_meal_times(start_h * 60 + start_m, end_h * 60 + end_m, prefs_dict)
    avg_stay_for_meal = int(
        sum(p.get("stay", 75) for p in sorted_places) / len(sorted_places)
    ) if sorted_places else 75
    # avg_travel: 장소 간 평균 이동시간 추정 (식사 window 판단 정확도 개선)
    # avg_travel_for_meal = 20
    avg_travel_for_meal = int(total_travel / len(sorted_places)) if sorted_places else 30
    sorted_places = _arrange_meals_in_places(
        sorted_places, meal_times, current_min, avg_stay_for_meal, avg_travel_for_meal
    )

    # # ── 장소 방문 ───────────────────────────────────────────────
    # travel_minutes_list: list[int] = []
    # used_lunch = False
    # lunch_time = next((m["time"] for m in meal_times if m["type"] == "lunch"), None)

    # for i, place in enumerate(sorted_places):
    #     if (
    #         place["category"] == "맛집"
    #         and not used_lunch
    #         and lunch_time
    #         and abs(current_min - lunch_time) <= 90
    #         ):            
    #         if current_min < lunch_time:
    #             current_min = lunch_time
            
    #         # 점심 시간보다 너무 늦으면 앞으로 당김
    #         elif current_min > lunch_time + 60:
    #             current_min = lunch_time

    #         used_lunch = True

    #     if place.get("pinned") and place.get("pinned_time"):
    #         pm = parse_time_to_minutes(place["pinned_time"])
    #         if pm > current_min:
    #             current_min = pm

    #     # 다음 장소까지 이동 시간 미리 계산 (desc에 포함)
    #     next_travel = 0
    #     if i < len(sorted_places) - 1:
    #         next_travel = int(calculate_travel_time(place, sorted_places[i + 1], transport_mode))

    #     timeline.append({
    #         "place":           place["name"],
    #         "category":        place["category"],
    #         "address":         place.get("address", ""),
    #         "stay_minutes":    place["stay"],
    #         "travel_minutes":  next_travel,  # 다음 장소까지 이동시간 (추정)
    #         "time":            minutes_to_time_str(current_min),
    #         "type":            "visit",
    #         "pinned":          place.get("pinned", False),
    #     })
    #     total_stay  += place["stay"]
    #     current_min += place["stay"]

    #     if next_travel > 0:
    #         total_travel += next_travel
    #         current_min  += next_travel
    # ── 장소 방문 ───────────────────────────────────────────────
    travel_minutes_list: list[int] = []

    used_lunch = False
    lunch_time = next((m["time"] for m in meal_times if m["type"] == "lunch"), None)

    for i, place in enumerate(sorted_places):

        # 👉 다음 장소 이동시간 미리 계산
        next_travel = 0
        if i < len(sorted_places) - 1:
            next_travel = int(
                calculate_travel_time(place, sorted_places[i + 1], transport_mode)
            )

        # 👉 이전 장소 → 현재 장소 이동시간
        prev_travel = 0
        if i > 0:
            prev_travel = int(
                calculate_travel_time(sorted_places[i - 1], place, transport_mode)
            )

        # 👉 "도착 시간" 기준 계산 (핵심)
        arrival_time = current_min + prev_travel

        # 🔥 점심 처리 (위치 + 시간 기반)
        if (
            place["category"] == "맛집"
            and not used_lunch
            and lunch_time
            and (lunch_time - 60) <= arrival_time <= (lunch_time + 60)
        ):
            # 점심 시간에 맞춰 이동
            current_min = lunch_time
            used_lunch = True

        # 🔥 카페 시간 제한 (UX 개선)
        if place["category"] == "카페" and current_min < 14 * 60:
            continue  # 오전 카페 제거

        # 🔥 pinned 시간 우선
        if place.get("pinned") and place.get("pinned_time"):
            pm = parse_time_to_minutes(place["pinned_time"])
            if pm > current_min:
                current_min = pm

        # ── 장소 방문 기록 ──
        timeline.append({
            "place":           place["name"],
            "category":        place["category"],
            "address":         place.get("address", ""),
            "lat":             place.get("lat"),
            "lng":             place.get("lng"),
            "stay_minutes":    place["stay"],
            "travel_minutes":  next_travel,
            "time":            minutes_to_time_str(current_min),
            "type":            "visit",
            "pinned":          place.get("pinned", False),
        })

        total_stay  += place["stay"]
        current_min += place["stay"]

        # 이동시간 반영
        if next_travel > 0:
            total_travel += next_travel
            current_min  += next_travel


    # ── 마지막 날: 복귀 ───────────────────────────────────────
    if day_idx == n_days - 1 and return_point:
        return_travel = 0
        if sorted_places:
            return_travel = int(calculate_travel_time(sorted_places[-1], return_point, transport_mode))
            total_travel += return_travel
            current_min  += return_travel
            # 마지막 방문 장소의 travel_minutes를 복귀 이동시간으로 소급 수정
            if timeline:
                timeline[-1]["travel_minutes"] = return_travel
        timeline.append({
            "place":          return_point.get("name", "출발지"),
            "category":       "출발지",
            "address":        return_point.get("address", ""),
            "lat":            return_point.get("lat"),
            "lng":            return_point.get("lng"),
            "stay_minutes":   0,
            "travel_minutes": 0,
            "time":           minutes_to_time_str(current_min),
            "type":           "departure",
            "pinned":         False,
        })
        has_departure = True

    # ── 중간 날: 숙소 복귀 ────────────────────────────────────
    elif day_idx < n_days - 1 and tonight_hotel:
        hotel_travel = 0
        if sorted_places:
            hotel_travel = int(calculate_travel_time(sorted_places[-1], tonight_hotel, transport_mode))
            total_travel += hotel_travel
            current_min  += hotel_travel
            # 마지막 방문 장소의 travel_minutes를 숙소 이동시간으로 소급 수정
            if timeline:
                timeline[-1]["travel_minutes"] = hotel_travel

        is_checkin = any(
            h.check_in_day == day_idx + 1 for h in hotels
            if h.name == tonight_hotel["name"]
        )
        timeline.append({
            "place":          tonight_hotel["name"],
            "category":       "숙소",
            "address":        tonight_hotel.get("address", ""),
            "lat":            tonight_hotel.get("lat"),
            "lng":            tonight_hotel.get("lng"),
            "stay_minutes":   0,
            "travel_minutes": 0,  # 숙소는 최종 목적지이므로 다음 이동 없음
            "time":           minutes_to_time_str(current_min),
            "type":           "hotel_checkin" if is_checkin else "hotel",
            "pinned":         False,
            "hotel_info": {
                "is_checkin":    is_checkin,
                "check_in_day":  next(
                    (h.check_in_day  for h in hotels if h.name == tonight_hotel["name"]), None
                ),
                "check_out_day": next(
                    (h.check_out_day for h in hotels if h.name == tonight_hotel["name"]), None
                ),
            },
        })

    cat_warnings = check_category_sequence(timeline)
    is_over      = current_min > (end_h * 60 + end_m)
    total_min    = total_stay + int(total_travel)

    logger.info(
        "Day %d: %s ~ %s (%d분) %s",
        day_idx + 1, daily_start_time,
        minutes_to_time_str(current_min), total_min,
        "[초과]" if is_over else "",
    )

    return {
        "places":                timeline,
        "total_places":          len(timeline),
        "total_stay_minutes":    total_stay,
        "total_travel_minutes":  int(total_travel),
        "total_minutes":         total_min,
        "start_time":            daily_start_time,
        "end_time":              minutes_to_time_str(current_min),
        "planned_end_time":      daily_end_time,
        "is_over_time":          is_over,
        "has_departure_point":   has_departure,
        "is_hotel_move_day":     is_move_day,
        "hotel_info": {
            "tonight":    tonight_hotel,
            "start_from": start_location,
        } if (tonight_hotel or start_location) else None,
        "departure_info": {
            "name":            explicit_departure.name,
            "lat":             explicit_departure.lat,
            "lng":             explicit_departure.lng,
            "address":         explicit_departure.address or "",
            "is_return_point": explicit_departure.is_return_point,
        } if explicit_departure else None,
        "meal_times":        meal_times,
        "category_warnings": cat_warnings,
        "pin_warnings":      pin_warnings,
    }


def _empty_day(day_idx: int, start_time: str, end_time: str, pin_warnings: list = None) -> dict:
    return {
        "places": [], "total_places": 0,
        "total_stay_minutes": 0, "total_travel_minutes": 0, "total_minutes": 0,
        "start_time": start_time, "end_time": end_time, "planned_end_time": end_time,
        "is_over_time": False,
        "has_departure_point": False, "is_hotel_move_day": False,
        "hotel_info": None, "departure_info": None,
        "meal_times": [], "category_warnings": [],
        "pin_warnings": pin_warnings or [],
    }


# ─── 7. 가용 시간 기반 최대 장소 수 ───────────────────────────

def _calc_max_places_per_day(
    start_h: int, start_m: int,
    end_h: int,   end_m: int,
    avg_stay: int = 75,
    avg_travel: int = 30,
) -> int:
    available = (end_h * 60 + end_m) - (start_h * 60 + start_m)
    per_place = avg_stay + avg_travel
    return max(1, available // per_place)


# ─── 진입점 ───────────────────────────────────────────────────

def generate_itinerary(
    places,
    n_days: int,
    transport_mode: str = "대중교통",
    hotels: list = None,
    departure_points: list = None,
    start_date: str = None,
    daily_start_time: str = "09:00",
    daily_end_time: str = "18:00",
    pinned_places: list = None,
    user_preferences: UserPreferences = None,
) -> dict:
    if pinned_places    is None: pinned_places    = []
    if user_preferences is None: user_preferences = UserPreferences()
    if hotels           is None: hotels           = []
    if departure_points is None: departure_points = []

    pace       = user_preferences.pace
    prefs_dict = {
        "lunch_time":  user_preferences.lunch_time,
        "dinner_time": user_preferences.dinner_time,
    }

    logger.info(
        "일정 생성 시작: %d개 장소 / %d일 / pace=%s / 고정=%d개",
        len(places), n_days, pace, len(pinned_places),
    )

    dups = check_duplicate_places([
        {"name": p.name, "lat": p.lat, "lng": p.lng, "category": p.category}
        for p in places
    ])
    if dups:
        logger.warning("중복 장소 %d개 감지", len(dups))

    enriched    = _enrich_places(places, pace)
    pinned_info = _build_pinned_info(pinned_places, enriched)
    dates_info  = calculate_dates(start_date, n_days) if start_date else []
    return_point = _get_return_point(departure_points)

    start_h, start_m = map(int, daily_start_time.split(":"))
    end_h,   end_m   = map(int, daily_end_time.split(":"))

    day_assignments, day_anchors = _assign_places_to_days(
        enriched, pinned_info, n_days, hotels, departure_points
    )

    avg_stay    = int(sum(p["stay"] for p in enriched) / len(enriched)) if enriched else 75
    max_per_day = _calc_max_places_per_day(
        start_h, start_m, end_h, end_m,
        avg_stay=avg_stay, avg_travel=30,
    )
    logger.info("[재분배] 하루 최대 장소 수: %d개 (avg_stay=%d분)", max_per_day, avg_stay)

    day_assignments = _redistribute(
        day_assignments, day_anchors, max_per_day, hotels, transport_mode
    )

    result: dict = {}
    for day_idx in range(n_days):
        date_info = dates_info[day_idx] if dates_info else None
        day_data  = _build_day_timeline(
            day_idx          = day_idx,
            day_places       = day_assignments[day_idx],
            hotels           = hotels,
            departure_points = departure_points,
            return_point     = return_point,
            n_days           = n_days,
            transport_mode   = transport_mode,
            daily_start_time = daily_start_time,
            daily_end_time   = daily_end_time,
            prefs_dict       = prefs_dict,
            pinned_info      = pinned_info,
            pace             = pace,
        )
        day_key = f"day_{day_idx + 1}"
        result[day_key] = {
            "date":     date_info["formatted"] if date_info else f"Day {day_idx + 1}",
            "date_raw": date_info["date"]      if date_info else None,
            **day_data,
        }

    result["duplicates"] = dups
    return result