import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict

from features.schedule.utils import (
    calculate_travel_time, nearest_neighbor_route,
    parse_time_to_minutes, minutes_to_time_str, calculate_dates,
    check_duplicate_places, check_category_sequence,
)
from features.schedule.models import UserPreferences

warnings.filterwarnings("ignore")

STAY_TIME_MAP = {
    "관광지": 90,
    "맛집":   60,
    "카페":   40,
    "쇼핑":   60,
    "출발지": 120,
    "숙소":   0,
}
PACE_MULTIPLIER = {"tight": 0.7, "normal": 1.0, "relaxed": 1.3}


# ─── 보조 함수 ─────────────────────────────────────────────────

def _adjusted_stay(category: str, pace: str) -> int:
    return int(STAY_TIME_MAP.get(category, 60) * PACE_MULTIPLIER.get(pace, 1.0))


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
                    p["meal_time"] = meal["type"]   # [FIX 3] meal_type → meal_time (프론트엔드와 통일)
                    p["suggested_time"] = meal["time_str"]
                    break
    return arranged


def _apply_pinned(day_places: list, pinned_info: dict, day_num: int) -> list:
    pins = [info for info in pinned_info.values() if info["day"] == day_num]
    if not pins:
        return day_places
    pinned_names = {p["place"]["name"] for p in pins}
    regular = [p for p in day_places if p["name"] not in pinned_names]
    timed   = sorted([p for p in pins if p.get("time")],
                     key=lambda x: parse_time_to_minutes(x["time"]))
    untimed = sorted([p for p in pins if not p.get("time")],
                     key=lambda x: {"must": 0, "high": 1, "medium": 2}.get(x["priority"], 3))
    result = []
    for pin in timed:
        pl = pin["place"].copy(); pl["pinned"] = True; pl["pinned_time"] = pin["time"]
        result.append(pl)
    result.extend(regular)
    for pin in untimed:
        pl = pin["place"].copy(); pl["pinned"] = True
        result.append(pl)
    return result


def _empty_day(day_idx: int, date_info, start_time: str, end_time: str) -> dict:
    return {
        "date": date_info["formatted"] if date_info else f"Day {day_idx + 1}",
        "date_raw": date_info["date"] if date_info else None,
        "places": [], "total_places": 0,
        "total_stay_minutes": 0, "total_travel_minutes": 0, "total_minutes": 0,
        "start_time": start_time, "end_time": end_time,
        "has_departure_point": False, "has_hotel": False,
    }


# ─── 메인 서비스 함수 ──────────────────────────────────────────

def generate_itinerary(
    places,
    n_days: int,
    transport_mode: str = "대중교통",
    hotel: dict = None,
    departure_point: dict = None,
    start_date: str = None,
    daily_start_time: str = "09:00",
    daily_end_time: str = "18:00",
    pinned_places: list = None,
    user_preferences: UserPreferences = None,
) -> dict:

    if pinned_places is None:
        pinned_places = []
    if user_preferences is None:
        user_preferences = UserPreferences()

    pace = user_preferences.pace
    prefs_dict = {"lunch_time": user_preferences.lunch_time,
                  "dinner_time": user_preferences.dinner_time}

    # 중복 체크
    dups = check_duplicate_places([
        {"name": p.name, "lat": p.lat, "lng": p.lng, "category": p.category}
        for p in places
    ])
    print(f"\n중복 체크: {str(len(dups)) + '개' if dups else '없음'}")

    # 장소 enrichment
    enriched = [
        {
            "name": p.name, "lat": p.lat, "lng": p.lng,
            "category": p.category,
            "stay": _adjusted_stay(p.category, pace),
            "address": p.address or "",
            "open_time": p.open_time, "close_time": p.close_time,
            "is_landmark": p.is_landmark, "is_unique": p.is_unique,
            "pinned": False,
        }
        for p in places
    ]

    # pinned_info 구성
    pinned_info: dict = {}
    for pin in pinned_places:
        if 0 <= pin.place_index < len(enriched):
            pinned_info[pin.place_index] = {
                "day": pin.day, "time": pin.time, "priority": pin.priority,
                "place": enriched[pin.place_index].copy(),
            }

    print(f"일정 생성: {len(enriched)}개 / {n_days}일 / pace={pace} / 고정={len(pinned_places)}개")

    dates_info = calculate_dates(start_date, n_days) if start_date else []
    start_h, start_m = map(int, daily_start_time.split(":"))
    end_h,   end_m   = map(int, daily_end_time.split(":"))

    # KMeans 클러스터링
    n_clusters = min(len(enriched), n_days)
    scaled  = StandardScaler().fit_transform([[p["lat"], p["lng"]] for p in enriched])
    labels  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)

    pinned_names = {info["place"]["name"] for info in pinned_info.values()}
    clusters: dict = defaultdict(list)
    for place, label in zip(enriched, labels):
        if place["name"] not in pinned_names:
            clusters[label].append(place)

    day_assignments = [[] for _ in range(n_days)]
    for i, (_, cluster_places) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1]))):
        day_assignments[i % n_days].extend(cluster_places)

    for day_idx in range(n_days):
        day_assignments[day_idx] = _apply_pinned(
            day_assignments[day_idx], pinned_info, day_idx + 1
        )

    # 일별 타임라인
    result: dict = {}

    for day_idx in range(n_days):
        day_key    = f"day_{day_idx + 1}"
        date_info  = dates_info[day_idx] if dates_info else None
        day_places = day_assignments[day_idx]

        if not day_places:
            result[day_key] = _empty_day(day_idx, date_info, daily_start_time, daily_end_time)
            continue

        timeline = []
        total_travel, total_stay = 0, 0
        current_min   = start_h * 60 + start_m
        has_departure = False
        has_hotel_    = False
        sorted_places = []  # [FIX 2] NameError 방지: 항상 명시적으로 초기화

        # [FIX 1] _apply_pinned 이후 day_places가 비어있을 수 있으므로 재확인
        if not day_places:
            result[day_key] = _empty_day(day_idx, date_info, daily_start_time, daily_end_time)
            continue

        # 첫날 출발지 도착
        if day_idx == 0 and departure_point:
            timeline.append({
                "place": departure_point.get("name", "출발지"), "category": "출발지",
                "address": departure_point.get("address", ""), "stay_minutes": 120,
                "time": minutes_to_time_str(current_min), "type": "arrival", "pinned": False,
            })
            total_stay  += 120; current_min += 120; has_departure = True
            sorted_places = nearest_neighbor_route(day_places, departure_point)
            # [FIX 1] sorted_places가 비어있을 경우 이동 시간 계산 건너뜀
            if sorted_places:
                t = calculate_travel_time(departure_point, sorted_places[0], transport_mode)
                total_travel += t; current_min += int(t)
        else:
            start_loc = (
                {"name": hotel.get("name", "숙소"), "lat": hotel["lat"], "lng": hotel["lng"]}
                if hotel else day_places[0]  # [FIX 1] 위 재확인으로 IndexError 방지됨
            )
            sorted_places = nearest_neighbor_route(day_places, start_loc)

        # 장소 방문
        for i, place in enumerate(sorted_places):
            if place.get("pinned") and place.get("pinned_time"):
                pm = parse_time_to_minutes(place["pinned_time"])
                if pm > current_min:
                    current_min = pm
            timeline.append({
                "place": place["name"], "category": place["category"],
                "address": place.get("address", ""), "stay_minutes": place["stay"],
                "time": minutes_to_time_str(current_min),
                "type": "visit", "pinned": place.get("pinned", False),
            })
            total_stay  += place["stay"]; current_min += place["stay"]
            if i < len(sorted_places) - 1:
                t = calculate_travel_time(place, sorted_places[i+1], transport_mode)
                total_travel += t; current_min += int(t)

        # 마지막 날 출발지 복귀
        if day_idx == n_days - 1 and departure_point:
            if sorted_places:
                t = calculate_travel_time(sorted_places[-1], departure_point, transport_mode)
                total_travel += t; current_min += int(t)
            timeline.append({
                "place": departure_point.get("name", "출발지"), "category": "출발지",
                "address": departure_point.get("address", ""), "stay_minutes": 90,
                "time": minutes_to_time_str(current_min), "type": "departure", "pinned": False,
            })
            total_stay += 90; current_min += 90; has_departure = True

        # 중간 날 숙소 복귀
        elif day_idx < n_days - 1 and hotel:
            if sorted_places:
                t = calculate_travel_time(sorted_places[-1], hotel, transport_mode)
                total_travel += t; current_min += int(t)
            timeline.append({
                "place": hotel.get("name", "숙소"), "category": "숙소",
                "address": hotel.get("address", ""), "stay_minutes": 0,
                "time": minutes_to_time_str(current_min), "type": "hotel", "pinned": False,
            })
            has_hotel_ = True

        meal_times   = _find_meal_times(start_h*60 + start_m, current_min, prefs_dict)
        timeline     = _arrange_meals(timeline, meal_times)
        cat_warnings = check_category_sequence(timeline)
        is_over      = current_min > (end_h * 60 + end_m)
        total_min    = total_stay + int(total_travel)

        print(f"Day {day_idx+1}: {daily_start_time} ~ {minutes_to_time_str(current_min)} "
              f"({total_min}분) {'초과' if is_over else ''}")

        result[day_key] = {
            "date":     date_info["formatted"] if date_info else f"Day {day_idx + 1}",
            "date_raw": date_info["date"]      if date_info else None,
            "places":   timeline,
            "total_places":          len(timeline),
            "total_stay_minutes":    total_stay,
            "total_travel_minutes":  int(total_travel),
            "total_minutes":         total_min,
            "start_time":            daily_start_time,
            "end_time":              minutes_to_time_str(current_min),
            "planned_end_time":      daily_end_time,
            "is_over_time":          is_over,
            "has_departure_point":   has_departure,
            "has_hotel":             has_hotel_,
            "meal_times":            meal_times,
            "category_warnings":     cat_warnings,
        }

    return result