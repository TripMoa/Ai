import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict

from features.schedule.utils import (
    calculate_travel_time, nearest_neighbor_route,
    parse_time_to_minutes, minutes_to_time_str, calculate_dates,
    check_duplicate_places, check_category_sequence,
    haversine_distance,
)
from features.schedule.models import UserPreferences, HotelStay, DeparturePoint

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

# 숙소 이동일 판단 기준 (km)
HOTEL_MOVE_THRESHOLD_KM = 30


# ─── 다중 숙소 헬퍼 ────────────────────────────────────────────

def _get_hotel_for_night(hotels: list[HotelStay], day_idx: int) -> dict | None:
    """
    day_idx(0-based)에 해당하는 밤을 보낼 숙소 반환.
    예) day_idx=0 → check_in_day=1 이고 check_out_day>1 인 숙소
    """
    day_num = day_idx + 1
    for h in hotels:
        if h.check_in_day <= day_num < h.check_out_day:
            return {"name": h.name, "lat": h.lat, "lng": h.lng, "address": h.address or ""}
    return None


def _is_hotel_move_day(day_idx: int, hotels: list[HotelStay]) -> bool:
    """전날 숙소와 오늘 밤 숙소가 다른 날 = 숙소 이동일"""
    if day_idx == 0:
        return False
    prev_hotel = _get_hotel_for_night(hotels, day_idx - 1)
    curr_hotel = _get_hotel_for_night(hotels, day_idx)
    if not prev_hotel or not curr_hotel:
        return False
    if prev_hotel["name"] == curr_hotel["name"]:
        return False
    return haversine_distance(prev_hotel, curr_hotel) >= HOTEL_MOVE_THRESHOLD_KM


def _max_places_for_day(day_idx: int, hotels: list[HotelStay], default: int) -> int:
    """숙소 이동일은 장소 수를 자동으로 줄임"""
    if _is_hotel_move_day(day_idx, hotels):
        return max(2, default // 2)
    return default


# ─── 다중 출발지 헬퍼 (신규) ──────────────────────────────────

def _resolve_departure_points(
    departure_points: list[DeparturePoint],
    departure_point: dict | None,
) -> list[DeparturePoint]:
    """
    구버전 단일 departure_point와 신버전 departure_points 통합.
    모델 validator에서 이미 변환되지만, service 직접 호출 시 대비.
    """
    if departure_points:
        return departure_points

    if departure_point:
        return [DeparturePoint(
            name=departure_point.get("name", "출발지"),
            lat=departure_point["lat"],
            lng=departure_point["lng"],
            address=departure_point.get("address", ""),
            day=1,
            is_return_point=True,
        )]
    return []


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

    # 1순위: 이 일차에 명시적으로 지정된 출발지
    for dp in departure_points:
        if dp.day == day_num:
            return {"name": dp.name, "lat": dp.lat,
                    "lng": dp.lng, "address": dp.address or "",
                    "category": "출발지"}

    # 2순위: 전날 밤 숙소
    if day_idx > 0:
        return _get_hotel_for_night(hotels, day_idx - 1)

    return None


def _get_return_point(departure_points: list[DeparturePoint]) -> dict | None:
    """
    마지막 날 복귀 기준점.
    is_return_point=True 인 출발지 중 day가 가장 작은 것(=원래 출발지)을 사용.
    없으면 day=1 출발지 사용.
    """
    return_candidates = [dp for dp in departure_points if dp.is_return_point]
    if return_candidates:
        dp = min(return_candidates, key=lambda x: x.day)
    elif departure_points:
        # 폴백: day=1 짜리
        day1 = [dp for dp in departure_points if dp.day == 1]
        dp = day1[0] if day1 else None
    else:
        return None

    if dp is None:
        return None
    return {"name": dp.name, "lat": dp.lat, "lng": dp.lng, "address": dp.address or ""}


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
                    p["meal_time"] = meal["type"]
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
        "is_hotel_move_day": False,
        "hotel_info": None,
        "departure_info": None,
    }


# ─── KMeans: 숙소 좌표 힌트 주입 ──────────────────────────────

def _build_kmeans(n_clusters: int, hotels: list[HotelStay], n_days: int) -> KMeans:
    """
    숙소가 있으면 숙소 좌표를 클러스터 초기 중심으로 사용.
    숙소 수 < n_clusters 이면 나머지는 랜덤 보완.
    """
    if not hotels or len(hotels) == 0:
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    centers: list[list[float]] = []
    for h in sorted(hotels, key=lambda x: x.check_in_day):
        nights = h.check_out_day - h.check_in_day
        # 연박 숙소는 비중 반영해 여러 번 추가 (최대 3회)
        for _ in range(min(nights, 3)):
            centers.append([h.lat, h.lng])

    # n_clusters 개수에 맞게 자르거나 랜덤 추가
    if len(centers) >= n_clusters:
        centers = centers[:n_clusters]
        return KMeans(n_clusters=n_clusters, init=np.array(centers), n_init=1)
    else:
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)


# ─── 메인 서비스 함수 ──────────────────────────────────────────

def generate_itinerary(
    places,
    n_days: int,
    transport_mode: str = "대중교통",
    hotel: dict = None,                      # deprecated
    hotels: list = None,                     # HotelStay 리스트
    departure_point: dict = None,            # deprecated
    departure_points: list = None,           # DeparturePoint 리스트 (신규)
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
    if hotels is None:
        hotels = []
    if departure_points is None:
        departure_points = []

    # ── 구버전 hotel 단일 Dict 호환 ──────────────────────────
    if hotel and not hotels:
        from features.schedule.models import HotelStay as _HS
        hotels = [_HS(
            name=hotel.get("name", "숙소"),
            lat=hotel["lat"], lng=hotel["lng"],
            address=hotel.get("address", ""),
            check_in_day=1, check_out_day=n_days,
        )]

    # ── 구버전 departure_point 단일 Dict 호환 ────────────────
    departure_points = _resolve_departure_points(departure_points, departure_point)

    # 마지막 날 복귀 기준점
    return_point = _get_return_point(departure_points)

    pace = user_preferences.pace
    prefs_dict = {"lunch_time": user_preferences.lunch_time,
                  "dinner_time": user_preferences.dinner_time}

    # 중복 체크
    dups = check_duplicate_places([
        {"name": p.name, "lat": p.lat, "lng": p.lng, "category": p.category}
        for p in places
    ])
    print(f"\n중복 체크: {str(len(dups)) + '개' if dups else '없음'}")

    # 숙소 정보 요약 출력
    if hotels:
        print(f"숙소 {len(hotels)}개:")
        for h in hotels:
            print(f"  - {h.name} ({h.check_in_day}일차 체크인 → {h.check_out_day}일차 체크아웃)")

    # 출발지 정보 요약 출력
    if departure_points:
        print(f"출발지 {len(departure_points)}개:")
        for dp in departure_points:
            return_mark = " [복귀 기준]" if dp.is_return_point else ""
            print(f"  - {dp.name} ({dp.day}일차{return_mark})")

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

    # KMeans 클러스터링 (숙소 좌표 힌트 포함)
    n_clusters = min(len(enriched), n_days)
    coords = [[p["lat"], p["lng"]] for p in enriched]
    scaled = StandardScaler().fit_transform(coords)
    kmeans = _build_kmeans(n_clusters, hotels, n_days)
    labels = kmeans.fit_predict(scaled)

    print("\n" + "=" * 60)
    print(f"[KMeans] n_clusters={n_clusters}, n_days={n_days}")
    print(f"{'장소':<20} {'lat':>10} {'lng':>10} {'cluster':>8}")
    print("-" * 60)
    for place, label in zip(enriched, labels):
        print(f"{place['name']:<20} {place['lat']:>10.6f} {place['lng']:>10.6f} {label:>8}")
    print("=" * 60 + "\n")

    pinned_names = {info["place"]["name"] for info in pinned_info.values()}
    clusters: dict = defaultdict(list)
    for place, label in zip(enriched, labels):
        if place["name"] not in pinned_names:
            clusters[label].append(place)

    # 각 클러스터의 중심 좌표 계산
    cluster_centers = {}
    for label, places in clusters.items():
        cluster_centers[label] = {
            "lat": sum(p["lat"] for p in places) / len(places),
            "lng": sum(p["lng"] for p in places) / len(places),
        }

    # 날짜별 기준 좌표 결정 (출발지 -> 숙소 -> 전날 숙소 순)
    day_anchors = []
    for day_idx in range(n_days):
        anchor = _get_departure_for_day(day_idx, departure_points, hotels)
        if anchor is None:
            anchor = _get_hotel_for_night(hotels, day_idx)
        if anchor is None and day_idx > 0:
            anchor = _get_hotel_for_night(hotels, day_idx - 1)
        day_anchors.append(anchor)

    # greedy로 클러스터 -> 날짜 매핑 (기준 좌표와 가장 가까운 클러스터 순)
    unassigned_clusters = list(clusters.keys())
    cluster_to_day = {}

    for day_idx in range(n_days):
        if not unassigned_clusters:
            break
        anchor = day_anchors[day_idx]
        if anchor:
            best = min(
                unassigned_clusters,
                key=lambda lbl: haversine_distance(anchor, cluster_centers[lbl])
            )
        else:
            best = max(unassigned_clusters, key=lambda lbl: len(clusters[lbl]))

        cluster_to_day[best] = day_idx
        unassigned_clusters.remove(best)

    # 매핑되지 못한 클러스터는 장소 수가 가장 적은 날에 추가
    day_assignments = [[] for _ in range(n_days)]
    for lbl in unassigned_clusters:
        day_idx = min(range(n_days), key=lambda i: len(day_assignments[i]))
        cluster_to_day[lbl] = day_idx

    print("[클러스터 -> 날짜 매핑]")
    for lbl, day_idx in cluster_to_day.items():
        center = cluster_centers[lbl]
        print(f"  Cluster {lbl} -> Day {day_idx+1} (중심: {center['lat']:.4f}, {center['lng']:.4f})")

    day_assignments = [[] for _ in range(n_days)]
    for lbl, day_idx in cluster_to_day.items():
        day_assignments[day_idx].extend(clusters[lbl])

    for day_idx in range(n_days):
        # 숙소 이동일이면 장소 수 제한
        max_p = _max_places_for_day(day_idx, hotels, len(day_assignments[day_idx]))
        day_assignments[day_idx] = day_assignments[day_idx][:max_p]

        day_assignments[day_idx] = _apply_pinned(
            day_assignments[day_idx], pinned_info, day_idx + 1
        )

    # ── 일별 타임라인 ──────────────────────────────────────────
    result: dict = {}

    for day_idx in range(n_days):
        day_key    = f"day_{day_idx + 1}"
        date_info  = dates_info[day_idx] if dates_info else None
        day_places = day_assignments[day_idx]

        # 오늘 밤 잘 숙소 & 아침 출발 위치
        tonight_hotel  = _get_hotel_for_night(hotels, day_idx)
        start_location = _get_departure_for_day(day_idx, departure_points, hotels)
        is_move_day    = _is_hotel_move_day(day_idx, hotels)

        # 이 날의 명시적 출발지 여부 확인 (타임라인 출력용)
        explicit_departure = next(
            (dp for dp in departure_points if dp.day == day_idx + 1), None
        )

        if not day_places:
            result[day_key] = _empty_day(day_idx, date_info, daily_start_time, daily_end_time)
            continue

        timeline = []
        total_travel, total_stay = 0, 0
        current_min   = start_h * 60 + start_m
        has_departure = False
        has_hotel_    = False
        sorted_places = []

        # ── 명시적 출발지가 있는 날: 출발지 도착 이벤트 삽입 ────
        if explicit_departure:
            dep_dict = {
                "name": explicit_departure.name,
                "lat":  explicit_departure.lat,
                "lng":  explicit_departure.lng,
                "address": explicit_departure.address or "",
            }
            stay_min = 120 if day_idx == 0 else 0  # 1일차는 대기 시간, 중간 출발지는 0
            timeline.append({
                "place": explicit_departure.name,
                "category": "출발지",
                "address": explicit_departure.address or "",
                "stay_minutes": stay_min,
                "time": minutes_to_time_str(current_min),
                "type": "arrival" if day_idx == 0 else "transfer_start",
                "pinned": False,
            })
            total_stay += stay_min
            current_min += stay_min
            has_departure = True

            sorted_places = nearest_neighbor_route(day_places, dep_dict)
            if sorted_places:
                t = calculate_travel_time(dep_dict, sorted_places[0], transport_mode)
                total_travel += t
                current_min  += int(t)

        # ── 출발지 없음: 숙소 or 이전 위치에서 출발 ────────────
        else:
            if start_location:
                sorted_places = nearest_neighbor_route(day_places, start_location)
                # 숙소 이동일: 이전 숙소 → 첫 장소 이동 시간 반영
                if is_move_day:
                    prev_hotel = _get_hotel_for_night(hotels, day_idx - 1)
                    if prev_hotel and sorted_places:
                        t = calculate_travel_time(prev_hotel, sorted_places[0], transport_mode)
                        total_travel += t
                        current_min  += int(t)
            else:
                sorted_places = nearest_neighbor_route(day_places, day_places[0])

        # ── 장소 방문 ───────────────────────────────────────────
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
            total_stay  += place["stay"]
            current_min += place["stay"]
            if i < len(sorted_places) - 1:
                t = calculate_travel_time(place, sorted_places[i+1], transport_mode)
                total_travel += t
                current_min  += int(t)

        # ── 마지막 날: 복귀 기준점으로 귀환 ────────────────────
        if day_idx == n_days - 1 and return_point:
            if sorted_places:
                t = calculate_travel_time(sorted_places[-1], return_point, transport_mode)
                total_travel += t
                current_min  += int(t)
            timeline.append({
                "place": return_point.get("name", "출발지"),
                "category": "출발지",
                "address": return_point.get("address", ""),
                "stay_minutes": 90,
                "time": minutes_to_time_str(current_min),
                "type": "departure",
                "pinned": False,
            })
            total_stay  += 90
            current_min += 90
            has_departure = True

        # ── 중간 날: 오늘 밤 숙소로 복귀 ──────────────────────
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
            has_hotel_ = True

        meal_times   = _find_meal_times(start_h*60 + start_m, current_min, prefs_dict)
        timeline     = _arrange_meals(timeline, meal_times)
        cat_warnings = check_category_sequence(timeline)
        is_over      = current_min > (end_h * 60 + end_m)
        total_min    = total_stay + int(total_travel)

        move_tag = " [숙소 이동일]" if is_move_day else ""
        dep_tag  = f" [출발지: {explicit_departure.name}]" if explicit_departure else ""
        print(f"Day {day_idx+1}{move_tag}{dep_tag}: {daily_start_time} ~ "
              f"{minutes_to_time_str(current_min)} ({total_min}분) "
              f"{'초과' if is_over else ''}")

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
            "is_hotel_move_day":     is_move_day,
            "hotel_info": {
                "tonight":     tonight_hotel,
                "start_from":  start_location,
            } if (tonight_hotel or start_location) else None,
            # ───────── 출발지 정보 (신규) ──────────────────────────
            "departure_info": {
                "name":    explicit_departure.name,
                "lat":     explicit_departure.lat,
                "lng":     explicit_departure.lng,
                "address": explicit_departure.address or "",
                "is_return_point": explicit_departure.is_return_point,
            } if explicit_departure else None,
            # ────────────────────────────────────────────────
            "meal_times":        meal_times,
            "category_warnings": cat_warnings,
        }

    return result