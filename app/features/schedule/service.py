"""
service.py

generate_itinerary 를 역할별로 분리:
  1. _enrich_places         - 장소 데이터 보강 (체류시간 등)
  2. _build_pinned_info      - 핀 장소 인덱싱
  3. _assign_places_to_days  - 장소 → 날짜 배정 (숙소/출발지 기반 직접 할당)
  4. _redistribute           - 하루 최대치 초과분 재분배
  5. _apply_pinned           - 핀 장소 통합 + 핀 시간 충돌 사전 검사
  6. _build_day_timeline     - 하루 타임라인 생성
  7. generate_itinerary      - 진입점 (조합만 담당)
"""

import warnings
from collections import defaultdict

from features.schedule.utils import (
    calculate_travel_time, optimized_route, nearest_neighbor_route,
    parse_time_to_minutes, minutes_to_time_str, calculate_dates,
    check_duplicate_places, check_category_sequence,
    haversine_distance,
)
from features.schedule.models import UserPreferences, HotelStay, DeparturePoint

warnings.filterwarnings("ignore")

# ─── 상수 ──────────────────────────────────────────────────────

# 기본 체류 시간 (분)
_BASE_STAY: dict[str, int] = {
    "관광지": 90,
    "맛집":   60,
    "카페":   40,
    "쇼핑":   60,
    "출발지": 120,
    "숙소":   0,
}
# 랜드마크/특색 장소 보정 (분)
_LANDMARK_BONUS = 30
_UNIQUE_BONUS   = 20

PACE_MULTIPLIER: dict[str, float] = {"tight": 0.7, "normal": 1.0, "relaxed": 1.3}

# 숙소 이동일 판단 기준 (km)
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
    """전날 숙소와 오늘 밤 숙소가 다르고 30km 이상 떨어진 날"""
    if day_idx == 0:
        return False
    prev = _get_hotel_for_night(hotels, day_idx - 1)
    curr = _get_hotel_for_night(hotels, day_idx)
    if not prev or not curr or prev["name"] == curr["name"]:
        return False
    return haversine_distance(prev, curr) >= HOTEL_MOVE_THRESHOLD_KM


def _max_places_for_day(day_idx: int, hotels: list[HotelStay], default: int) -> int:
    """숙소 이동일은 장소 수 자동 축소"""
    return max(2, default // 2) if _is_hotel_move_day(day_idx, hotels) else default


# ─── 다중 출발지 헬퍼 ─────────────────────────────────────────

def _get_departure_for_day(
    day_idx: int,
    departure_points: list[DeparturePoint],
    hotels: list[HotelStay],
) -> dict | None:
    """
    해당 날 아침 출발 위치 결정 (우선순위 순).
    1. 해당 일차에 명시된 DeparturePoint
    2. 전날 밤 잔 숙소
    3. None
    """
    day_num = day_idx + 1
    for dp in departure_points:
        if dp.day == day_num:
            return {"name": dp.name, "lat": dp.lat, "lng": dp.lng,
                    "address": dp.address or "", "category": "출발지"}
    if day_idx > 0:
        return _get_hotel_for_night(hotels, day_idx - 1)
    return None


def _get_return_point(departure_points: list[DeparturePoint]) -> dict | None:
    """마지막 날 복귀 기준점 (is_return_point=True 우선, 없으면 day=1)"""
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
    """
    카테고리 기본값 + 랜드마크/특색 보정 + pace 배율.
    랜드마크(is_landmark)는 +30분, 특색 장소(is_unique)는 +20분 추가.
    """
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
            # stay는 enrich 후 계산 (is_landmark 등 필요)
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
                "day":      pin.day,
                "time":     pin.time,
                "priority": pin.priority,
                "place":    enriched[pin.place_index].copy(),
            }
    return pinned_info


# ─── 3. 장소 → 날짜 배정 (숙소/출발지 기반 직접 할당) ────────

def _assign_places_to_days(
    enriched: list[dict],
    pinned_info: dict,
    n_days: int,
    hotels: list[HotelStay],
    departure_points: list[DeparturePoint],
) -> list[list[dict]]:
    """
    KMeans 대신 숙소·출발지 좌표를 기준으로 직접 배정.

    각 날의 기준 좌표(앵커)를 구한 뒤,
    핀되지 않은 장소 각각에 대해 "가장 가까운 앵커의 날"로 배정.
    앵커가 없는 날은 장소 수가 가장 적은 날로 균등 분배.
    """
    # 날짜별 앵커 좌표
    # hotels 파라미터가 비어있어도 places 중 category="숙소"인 것을 앵커로 활용
    hotel_places_as_anchor = [p for p in enriched if p.get("category") == "숙소"]

    def _get_anchor_for_day_with_fallback(day_idx: int) -> dict | None:
        anchor = _get_departure_for_day(day_idx, departure_points, hotels)
        if anchor is None:
            anchor = _get_hotel_for_night(hotels, day_idx)
        if anchor is None and day_idx > 0:
            anchor = _get_hotel_for_night(hotels, day_idx - 1)
        # hotels 파라미터가 비어있을 때 places의 숙소 카테고리 장소를 앵커로 사용
        if anchor is None and hotel_places_as_anchor:
            anchor = hotel_places_as_anchor[0]
        return anchor

    day_anchors: list[dict | None] = []
    for day_idx in range(n_days):
        day_anchors.append(_get_anchor_for_day_with_fallback(day_idx))

    pinned_names = {info["place"]["name"] for info in pinned_info.values()}

    # 숙소 이름(hotels 파라미터) + 카테고리가 숙소인 장소 + 출발지 모두 배정 대상에서 제외
    hotel_names     = {h.name for h in hotels}
    departure_names = {dp.name for dp in departure_points}

    unassigned = [
        p for p in enriched
        if p["name"] not in pinned_names
        and p["name"] not in hotel_names
        and p["name"] not in departure_names
        and p.get("category") != "숙소"
    ]

    assignments: list[list[dict]] = [[] for _ in range(n_days)]
    anchored_days = [i for i, a in enumerate(day_anchors) if a is not None]

    # 앵커가 동일한 좌표(같은 숙소)인 날이 여러 개일 때 거리만으로 배정하면
    # 한 날로 쏠리므로, 동일 앵커 그룹 내에서는 균등 분배로 전환
    def _anchor_key(day_idx: int) -> tuple:
        a = day_anchors[day_idx]
        if a is None:
            return (None, None)
        return (round(a["lat"], 5), round(a["lng"], 5))

    # 앵커 좌표 기준으로 날짜 그룹화
    anchor_groups: dict[tuple, list[int]] = {}
    for i in anchored_days:
        key = _anchor_key(i)
        anchor_groups.setdefault(key, []).append(i)

    for place in unassigned:
        if anchored_days:
            # 1단계: 가장 가까운 앵커 좌표 찾기
            best_anchor_key = min(
                anchor_groups.keys(),
                key=lambda k: haversine_distance(
                    {"lat": k[0], "lng": k[1]}, place
                ) if k[0] is not None else float("inf"),
            )
            # 2단계: 동일 앵커 그룹 내에서 장소 수가 가장 적은 날로 균등 분배
            group = anchor_groups[best_anchor_key]
            best_day = min(group, key=lambda i: len(assignments[i]))
        else:
            best_day = min(range(n_days), key=lambda i: len(assignments[i]))
        assignments[best_day].append(place)

    # ── 날짜별 균형 보정 ──────────────────────────────────────
    # 앵커가 달라도 장소가 한 날에 몰릴 수 있음.
    # 빈 날이 있으면 가장 많은 날에서 장소를 하나씩 가져와 균형을 맞춤.
    min_per_day = len(unassigned) // n_days  # 각 날이 가져야 할 최소 장소 수
    for _ in range(n_days * len(unassigned)):  # 최대 반복 상한
        counts = [len(assignments[i]) for i in range(n_days)]
        max_day = max(range(n_days), key=lambda i: counts[i])
        min_day = min(range(n_days), key=lambda i: counts[i])
        # 가장 많은 날과 가장 적은 날의 차이가 2 이상일 때만 이동
        if counts[max_day] - counts[min_day] < 2:
            break
        # 가장 많은 날에서 min_day 앵커와 가장 가까운 장소를 이동
        min_anchor = day_anchors[min_day]
        if min_anchor and assignments[max_day]:
            move_idx = min(
                range(len(assignments[max_day])),
                key=lambda i: haversine_distance(min_anchor, assignments[max_day][i]),
            )
        else:
            move_idx = -1  # 앵커 없으면 마지막 장소 이동
        place_to_move = assignments[max_day].pop(move_idx)
        assignments[min_day].append(place_to_move)

    # ── 배정 결과 테이블 출력 ─────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[배정] n_days={n_days}, 총 장소={len(unassigned)}개")
    print(f"{'장소':<20} {'lat':>10} {'lng':>10} {'day':>6}")
    print("-" * 60)
    for day_idx, day_places in enumerate(assignments):
        for p in day_places:
            print(f"{p['name']:<20} {p['lat']:>10.6f} {p['lng']:>10.6f} {day_idx+1:>6}")
    print("=" * 60 + "\n")

    # ── 날짜별 배정 요약 ──────────────────────────────────────
    print("[날짜별 배정]")
    for day_idx, day_places in enumerate(assignments):
        anchor = day_anchors[day_idx]
        anchor_name = anchor["name"] if anchor else "없음"
        place_names = ", ".join(p["name"] for p in day_places) or "(없음)"
        print(f"  Day {day_idx+1} (앵커: {anchor_name}) → {place_names}")
    print()

    return assignments, day_anchors


# ─── 4. 재분배 ────────────────────────────────────────────────

def _redistribute(
    day_assignments: list[list],
    day_anchors: list[dict | None],
    max_per_day: int,
    hotels: list,
    transport_mode: str,
) -> list[list]:
    """
    초과분을 인접한 날로 넘김.
    재분배 후 해당 날의 경로를 재최적화해 동선 품질 유지.
    """
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
            # 다음 날 우선
            for target in range(day_idx + 1, n_days):
                target_limit = _max_places_for_day(target, hotels, max_per_day)
                if len(result[target]) < target_limit:
                    result[target].append(place)
                    print(f"  [재분배] '{place['name']}' Day {day_idx+1} → Day {target+1}")
                    # 재분배 받은 날 경로 재최적화
                    if day_anchors[target]:
                        result[target] = optimized_route(result[target], day_anchors[target], transport_mode)
                    placed = True
                    break
            # 이전 날 시도
            if not placed:
                for target in range(day_idx - 1, -1, -1):
                    target_limit = _max_places_for_day(target, hotels, max_per_day)
                    if len(result[target]) < target_limit:
                        result[target].append(place)
                        print(f"  [재분배] '{place['name']}' Day {day_idx+1} → Day {target+1} (역방향)")
                        if day_anchors[target]:
                            result[target] = optimized_route(result[target], day_anchors[target], transport_mode)
                        placed = True
                        break
            if not placed:
                result[day_idx].append(place)
                print(f"  [재분배 실패] '{place['name']}' 넘길 날 없음, Day {day_idx+1} 유지")

    return result


# ─── 5. 핀 장소 통합 ──────────────────────────────────────────

def _check_pin_time_conflicts(
    timed_pins: list[dict],
    regular_places: list[dict],
    current_min: int,
    transport_mode: str,
    pace: str,
) -> list[str]:
    """
    timed 핀들이 현실적으로 도달 가능한지 사전 검사.
    이전 핀(또는 시작시간)에서 다음 핀까지 필요한 최소 시간을 계산해
    불가능한 경우 경고 메시지 반환.
    """
    warnings_list = []
    prev_time = current_min
    prev_place = None

    for pin in timed_pins:
        pin_min = parse_time_to_minutes(pin["time"])
        if prev_place is not None:
            travel = calculate_travel_time(prev_place, pin["place"], transport_mode)
            stay   = pin["place"].get("stay", 60)
            earliest_arrival = prev_time + int(travel)
            if earliest_arrival > pin_min:
                warnings_list.append(
                    f"'{pin['place']['name']}' {pin['time']} 도착은 "
                    f"이전 일정 기준 최소 {minutes_to_time_str(earliest_arrival)} 이후 가능"
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
    """
    핀 장소를 경로에 통합하고, timed 핀 충돌 경고를 반환.
    반환값: (정렬된 장소 리스트, 경고 메시지 리스트)
    """
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

    # timed 핀 충돌 사전 검사
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


def _arrange_meals(timeline: list, meal_times: list) -> list:
    """
    타임라인 내 맛집이 식사 윈도우 안에 없으면 suggested_time 태그 부여.
    순서 변경 없이 태그만 추가 (동선 최적화 결과 유지).
    """
    if not meal_times:
        return timeline
    restaurants = [p for p in timeline if p.get("category") == "맛집" and not p.get("pinned")]
    if not restaurants:
        return timeline

    arranged, used = timeline.copy(), set()
    for meal in meal_times:
        in_window = any(
            p.get("category") == "맛집" and
            meal["window_start"] <= parse_time_to_minutes(p.get("time", "00:00")) <= meal["window_end"]
            for p in arranged
        )
        if in_window:
            continue
        available = [r for r in restaurants if r["place"] not in used]
        if available:
            sel = available[0]
            used.add(sel["place"])
            for p in arranged:
                if p["place"] == sel["place"]:
                    p["meal_time"]      = meal["type"]
                    p["suggested_time"] = meal["time_str"]
                    break
    return arranged


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
    """하루치 타임라인 딕셔너리 반환"""
    start_h, start_m = map(int, daily_start_time.split(":"))
    end_h,   end_m   = map(int, daily_end_time.split(":"))
    current_min = start_h * 60 + start_m

    tonight_hotel  = _get_hotel_for_night(hotels, day_idx)
    start_location = _get_departure_for_day(day_idx, departure_points, hotels)
    is_move_day    = _is_hotel_move_day(day_idx, hotels)
    explicit_departure = next(
        (dp for dp in departure_points if dp.day == day_idx + 1), None
    )

    # 핀 통합 (시간 충돌 사전 검사 포함)
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
        stay_min = 120 if day_idx == 0 else 0
        timeline.append({
            "place":        explicit_departure.name,
            "category":     "출발지",
            "address":      explicit_departure.address or "",
            "stay_minutes": stay_min,
            "time":         minutes_to_time_str(current_min),
            "type":         "arrival" if day_idx == 0 else "transfer_start",
            "pinned":       False,
        })
        total_stay  += stay_min
        current_min += stay_min
        has_departure = True

        sorted_places = optimized_route(day_places, dep_dict, transport_mode)
        if sorted_places:
            t = calculate_travel_time(dep_dict, sorted_places[0], transport_mode)
            total_travel += t
            current_min  += int(t)

    # ── 출발지 없음: 숙소 or 첫 장소 기준 ────────────────────
    else:
        if start_location:
            sorted_places = optimized_route(day_places, start_location, transport_mode)
            if is_move_day:
                prev_hotel = _get_hotel_for_night(hotels, day_idx - 1)
                if prev_hotel and sorted_places:
                    t = calculate_travel_time(prev_hotel, sorted_places[0], transport_mode)
                    total_travel += t
                    current_min  += int(t)
        else:
            sorted_places = optimized_route(day_places, day_places[0], transport_mode)

    # ── 장소 방문 ───────────────────────────────────────────────
    for i, place in enumerate(sorted_places):
        if place.get("pinned") and place.get("pinned_time"):
            pm = parse_time_to_minutes(place["pinned_time"])
            if pm > current_min:
                current_min = pm

        timeline.append({
            "place":        place["name"],
            "category":     place["category"],
            "address":      place.get("address", ""),
            "stay_minutes": place["stay"],
            "time":         minutes_to_time_str(current_min),
            "type":         "visit",
            "pinned":       place.get("pinned", False),
        })
        total_stay  += place["stay"]
        current_min += place["stay"]

        if i < len(sorted_places) - 1:
            t = calculate_travel_time(place, sorted_places[i + 1], transport_mode)
            total_travel += t
            current_min  += int(t)

    # ── 마지막 날: 복귀 ───────────────────────────────────────
    if day_idx == n_days - 1 and return_point:
        if sorted_places:
            t = calculate_travel_time(sorted_places[-1], return_point, transport_mode)
            total_travel += t
            current_min  += int(t)
        timeline.append({
            "place":        return_point.get("name", "출발지"),
            "category":     "출발지",
            "address":      return_point.get("address", ""),
            "stay_minutes": 0,
            "time":         minutes_to_time_str(current_min),
            "type":         "departure",
            "pinned":       False,
        })
        has_departure = True

    # ── 중간 날: 숙소 복귀 ────────────────────────────────────
    elif day_idx < n_days - 1 and tonight_hotel:
        if sorted_places:
            t = calculate_travel_time(sorted_places[-1], tonight_hotel, transport_mode)
            total_travel += t
            current_min  += int(t)

        is_checkin = any(
            h.check_in_day == day_idx + 1 for h in hotels
            if h.name == tonight_hotel["name"]
        )
        timeline.append({
            "place":        tonight_hotel["name"],
            "category":     "숙소",
            "address":      tonight_hotel.get("address", ""),
            "stay_minutes": 0,
            "time":         minutes_to_time_str(current_min),
            "type":         "hotel_checkin" if is_checkin else "hotel",
            "pinned":       False,
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

    meal_times   = _find_meal_times(start_h * 60 + start_m, current_min, prefs_dict)
    timeline     = _arrange_meals(timeline, meal_times)
    cat_warnings = check_category_sequence(timeline)
    is_over      = current_min > (end_h * 60 + end_m)
    total_min    = total_stay + int(total_travel)

    move_tag = " [숙소 이동일]" if is_move_day else ""
    dep_tag  = f" [출발지: {explicit_departure.name}]" if explicit_departure else ""
    print(f"Day {day_idx+1}{move_tag}{dep_tag}: {daily_start_time} ~ "
          f"{minutes_to_time_str(current_min)} ({total_min}분) "
          f"{'[초과]' if is_over else ''}")

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
    # deprecated 파라미터 (models.py validator에서 이미 변환됨)
    hotel: dict = None,
    departure_point: dict = None,
) -> dict:
    """
    여행 일정 생성 진입점.
    하위 호환 변환(hotel, departure_point)은 models.py validator에서 처리.
    service 레벨에서는 hotels/departure_points만 사용.
    """
    if pinned_places    is None: pinned_places    = []
    if user_preferences is None: user_preferences = UserPreferences()
    if hotels           is None: hotels           = []
    if departure_points is None: departure_points = []

    pace       = user_preferences.pace
    prefs_dict = {"lunch_time": user_preferences.lunch_time,
                  "dinner_time": user_preferences.dinner_time}

    # ── 로그 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"일정 생성 시작: {len(places)}개 장소 / {n_days}일 / pace={pace} / 고정={len(pinned_places)}개")
    dups = check_duplicate_places([
        {"name": p.name, "lat": p.lat, "lng": p.lng, "category": p.category}
        for p in places
    ])
    print(f"\n중복 체크: {len(dups)}개")
    if hotels:
        print(f"숙소 {len(hotels)}개:")
        for h in hotels:
            print(f"  - {h.name} ({h.check_in_day}일차 → {h.check_out_day}일차)")
    if departure_points:
        print(f"출발지 {len(departure_points)}개:")
        for dp in departure_points:
            mark = " [복귀 기준]" if dp.is_return_point else ""
            print(f"  - {dp.name} ({dp.day}일차{mark})")
    print("=" * 60 + "\n")

    # ── 데이터 준비 ────────────────────────────────────────────
    enriched    = _enrich_places(places, pace)
    pinned_info = _build_pinned_info(pinned_places, enriched)
    dates_info  = calculate_dates(start_date, n_days) if start_date else []
    return_point = _get_return_point(departure_points)

    start_h, start_m = map(int, daily_start_time.split(":"))
    end_h,   end_m   = map(int, daily_end_time.split(":"))

    # ── 장소 배정 ──────────────────────────────────────────────
    day_assignments, day_anchors = _assign_places_to_days(
        enriched, pinned_info, n_days, hotels, departure_points
    )

    avg_stay    = int(sum(p["stay"] for p in enriched) / len(enriched)) if enriched else 75
    max_per_day = _calc_max_places_per_day(start_h, start_m, end_h, end_m,
                                           avg_stay=avg_stay, avg_travel=30)
    print(f"[재분배] 하루 최대 장소 수: {max_per_day}개 (avg_stay={avg_stay}분)")

    day_assignments = _redistribute(
        day_assignments, day_anchors, max_per_day, hotels, transport_mode
    )

    # ── 일별 타임라인 ──────────────────────────────────────────
    result: dict = {}
    for day_idx in range(n_days):
        date_info  = dates_info[day_idx] if dates_info else None
        day_data   = _build_day_timeline(
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

    return result