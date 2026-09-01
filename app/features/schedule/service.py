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
from features.schedule.odsay_api import CallCounter
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

# 장소 이름 키워드로 체류시간을 세분화 — "관광지" 한 카테고리 안에도
# 궁궐(오래 머묾)부터 전망대(잠깐 들름)까지 편차가 커서, 카테고리 하나로
# 뭉뚱그리면 하루 일정 과부하 판단(is_over_time, max_per_day)이 부정확해진다.
# 새 데이터/API 없이 이미 갖고 있는 장소명만으로 판단한다.
_STAY_KEYWORD_OVERRIDES: list[tuple[list[str], int]] = [
    (["궁", "고궁", "박물관", "미술관", "테마파크", "동물원", "수족관", "생태공원"], 120),
    (["전망대", "포토존", "다리", "야경", "분수", "정류장", "역"], 30),
    (["시장", "거리", "골목", "공원"], 60),
]


def _keyword_stay_override(name: str) -> int | None:
    for keywords, minutes in _STAY_KEYWORD_OVERRIDES:
        if any(kw in name for kw in keywords):
            return minutes
    return None

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
    keyword_override = _keyword_stay_override(place_dict.get("name", ""))
    base = keyword_override if keyword_override is not None else _BASE_STAY.get(place_dict.get("category", "관광지"), 60)
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
                {"name": p.name, "category": p.category, "is_landmark": p.is_landmark, "is_unique": p.is_unique},
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

    # ── 장소를 가장 가까운 앵커 그룹으로 먼저 묶는다 ─────────
    places_by_anchor_key: dict[tuple, list[dict]] = {k: [] for k in anchor_groups}
    for place in unassigned:
        best_anchor_key = min(
            anchor_groups.keys(),
            key=lambda k: haversine_distance(
                {"lat": k[0], "lng": k[1]}, place
            ) if k[0] is not None else float("inf"),
        )
        places_by_anchor_key[best_anchor_key].append(place)

    assignments: list[list[dict]] = [[] for _ in range(n_days)]

    for key, group_days in anchor_groups.items():
        group_places = places_by_anchor_key[key]
        if len(group_days) == 1:
            assignments[group_days[0]] = group_places
            continue

        # 여행 내내 숙소가 그대로라 여러 날이 같은 앵커를 공유하는 경우,
        # 앵커까지의 거리만으로는 어느 날에 넣을지 구분이 안 된다. 그대로
        # 두면 장소 개수만 맞춰 배정하다가(라운드로빈) 수원처럼 멀리 떨어진
        # 지역의 장소들이 하루에 묶이지 않고 여러 날에 흩어져서, 같은 지역을
        # 왕복하는 이동이 중복되는 문제가 생긴다. 장소들끼리의 지리적
        # 근접성으로 나눠서 같은 지역은 같은 날에 묶는다.
        sub_clusters = _geographic_cluster(group_places, len(group_days))
        for day_idx, cluster in zip(group_days, sub_clusters):
            assignments[day_idx] = cluster

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

_CAFE_CAP_BY_PACE: dict[str, int] = {"tight": 1, "normal": 2, "relaxed": 2}


def _cap_category_places(
    day_assignments: list[list],
    day_anchors: list[dict | None],
    category: str,
    cap: int,
    reason: str,
) -> tuple[list[list], list[dict]]:
    """
    하루 중 특정 카테고리 장소 개수가 cap을 넘으면 앵커(숙소/출발지)에서 가장
    먼 것부터 초과분으로 간주해 제외한다. _cap_meal_places/_cap_cafe_places가
    공유하는 공통 로직.
    """
    n_days = len(day_assignments)
    result = [list(d) for d in day_assignments]
    excluded: list[dict] = []

    for day_idx in range(n_days):
        matched = [p for p in result[day_idx] if p.get("category") == category]
        if len(matched) <= cap:
            continue

        anchor = day_anchors[day_idx]
        if anchor is not None:
            matched.sort(key=lambda p: haversine_distance(anchor, p))

        keep_names = {p["name"] for p in matched[:cap]}
        drop_names = {p["name"] for p in matched} - keep_names

        for name in drop_names:
            dropped = next(p for p in result[day_idx] if p["name"] == name)
            logger.info(
                "  [제외] '%s' Day %d — %s(%d개) 초과",
                name, day_idx + 1, category, cap
            )
            excluded.append({
                "name":     dropped["name"],
                "category": dropped["category"],
                "day":      day_idx + 1,
                "reason":   reason,
            })
        result[day_idx] = [p for p in result[day_idx] if p["name"] not in drop_names]

    return result, excluded


def _cap_meal_places(
    day_assignments: list[list],
    day_anchors: list[dict | None],
    meal_slot_count: int,
) -> tuple[list[list], list[dict]]:
    """
    하루 식사 시간대(점심/저녁) 개수를 넘는 "맛집"은 초과분으로 간주해 제외한다.

    지리적 배정(_assign_places_to_days)은 카테고리를 전혀 고려하지 않기 때문에,
    맛집이 특정 날에 3개 이상 몰리면 _arrange_meals_in_places()가 최대 2개
    (점심/저녁)만 식사 시간에 맞춰 옮기고 나머지는 그 자리에 남아 다른 맛집과
    나란히 배치되는 문제가 있었다. 여기서 미리 개수를 식사 슬롯 수만큼 잘라낸다.
    """
    return _cap_category_places(day_assignments, day_anchors, "맛집", meal_slot_count, "meal_slot_limit")


def _cap_cafe_places(
    day_assignments: list[list],
    day_anchors: list[dict | None],
    pace: str,
) -> tuple[list[list], list[dict]]:
    """
    카페는 맛집과 달리 고정된 식사 시간대가 없어 슬롯 수 대신 페이스 기반
    고정 상한(tight=1, normal/relaxed=2)을 쓴다.
    """
    cap = _CAFE_CAP_BY_PACE.get(pace, 2)
    return _cap_category_places(day_assignments, day_anchors, "카페", cap, "cafe_limit")


def _redistribute(
    day_assignments: list[list],
    day_anchors: list[dict | None],
    max_per_day: int,
    hotels: list,
    transport_mode: str,
) -> tuple[list[list], list[dict]]:
    n_days = len(day_assignments)
    result = [list(d) for d in day_assignments]
    excluded: list[dict] = []

    for day_idx in range(n_days):
        limit = _max_places_for_day(day_idx, hotels, max_per_day)
        if len(result[day_idx]) <= limit:
            continue

        candidates = result[day_idx]

        # ── 군집 밀도 기반 최적 조합 선택 ──────────────────────
        # 각 장소에 대해 "주변 장소들과의 평균 거리"를 계산해서
        # 서로 가장 밀집된 limit개 조합을 선택한다.
        #
        # 알고리즘:
        #   1. 모든 장소 쌍 거리 계산
        #   2. 각 장소의 "군집 점수" = 가장 가까운 (limit-1)개 장소까지의 평균 거리
        #      → 점수가 낮을수록 주변에 장소가 밀집되어 있음
        #   3. 군집 점수 낮은 순으로 limit개 선택
        #   4. 선택 결과에 "맛집"이 하나도 없으면 가장 군집 점수가 좋은 맛집으로
        #      가장 점수가 나쁜 항목 하나를 교체 (식사 장소가 통째로 잘리는 것 방지)

        n = len(candidates)

        # 장소 간 거리 행렬
        dist_matrix = [
            [haversine_distance(candidates[i], candidates[j]) for j in range(n)]
            for i in range(n)
        ]

        # 각 장소의 군집 점수: 자신 제외 가장 가까운 (limit-1)개의 평균 거리
        k = min(limit - 1, n - 1)
        cluster_scores = []
        for i in range(n):
            neighbors = sorted(dist_matrix[i][j] for j in range(n) if j != i)
            avg_dist = sum(neighbors[:k]) / k if k > 0 else 0
            cluster_scores.append((avg_dist, i))

        cluster_scores.sort()  # 평균 거리 오름차순 = 밀집된 장소 우선
        score_by_idx = {idx: score for score, idx in cluster_scores}

        selected_idx = set(idx for _, idx in cluster_scores[:limit])

        # ── 카테고리 균형 안전망: 식사 장소·카페가 전부 잘리지 않도록 ──
        selected_categories = {candidates[i]["category"] for i in selected_idx}
        for safety_category in ("맛집", "카페"):
            if safety_category in selected_categories:
                continue
            same_category_candidates = [
                i for i in range(n) if candidates[i]["category"] == safety_category
            ]
            if not same_category_candidates:
                continue
            best_idx = min(same_category_candidates, key=lambda i: score_by_idx[i])
            worst_selected_idx = max(selected_idx, key=lambda i: score_by_idx[i])
            selected_idx.discard(worst_selected_idx)
            selected_idx.add(best_idx)
            selected_categories.add(safety_category)

        dropped_idx = set(range(n)) - selected_idx

        result[day_idx] = [candidates[i] for i in range(n) if i in selected_idx]

        for i in dropped_idx:
            logger.info(
                "  [제외] '%s' Day %d — 동선 최적화 (군집 밀도 기반)",
                candidates[i]["name"], day_idx + 1
            )
            excluded.append({
                "name":     candidates[i]["name"],
                "category": candidates[i]["category"],
                "day":      day_idx + 1,
                "reason":   "capacity",
            })

    return result, excluded


# ─── 5. 핀 장소 통합 ──────────────────────────────────────────

def _check_pin_time_conflicts(
    timed_pins: list[dict],
    regular_places: list[dict],
    current_min: int,
    transport_mode: str,
) -> list[str]:
    warnings_list = []
    prev_time = current_min
    prev_place = None

    for pin in timed_pins:
        pin_min = parse_time_to_minutes(pin["time"])
        if prev_place is not None:
            travel = calculate_travel_time(prev_place, pin["place"], transport_mode)
            earliest_arrival = prev_time + int(travel["time"])
            if earliest_arrival > pin_min:
                warnings_list.append(
                    f"'{pin['place']['name']}' {pin['time']} 도착은 "
                    f"이전 일정 기준 최소 {minutes_to_time_str(earliest_arrival)} 이후 가능 "
                    f"(이동시간 추정 {int(travel['time'])}분 포함)"
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
        timed, regular, current_start_min, transport_mode
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
                       "window_start": lunch  - 45, "window_end": lunch  + 45})
    if 17*60+30 <= dinner <= 20*60 and start_min <= dinner <= end_min:
        result.append({"type": "dinner", "time": dinner,
                       "time_str": minutes_to_time_str(dinner),
                       "window_start": dinner - 45, "window_end": dinner + 45})
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

    used_meal_names: set[str] = set()  # 이미 배치된 맛집 추적

    for meal in meal_times:
        # 각 장소의 추정 도착 분 계산 (체류 + 이동시간 모두 포함)
        def _estimated_minute(idx: int) -> int:
            return current_start_min + sum(
                result[j].get("stay", avg_stay) + avg_travel for j in range(idx)
            )

        # 이미 window 안에 맛집이 있으면 스킵 (이미 배치된 것 제외)
        in_window = any(
            result[i].get("category") == "맛집"
            and result[i].get("name") not in used_meal_names
            and meal["window_start"] <= _estimated_minute(i) <= meal["window_end"]
            for i in range(len(result))
        )
        if in_window:
            # window 안에 있는 맛집을 used로 마킹
            for i in range(len(result)):
                if (result[i].get("category") == "맛집"
                        and result[i].get("name") not in used_meal_names
                        and meal["window_start"] <= _estimated_minute(i) <= meal["window_end"]):
                    used_meal_names.add(result[i]["name"])
                    break
            continue

        # 아직 배치 안 된 unpinned 맛집 후보 중 첫 번째 선택
        candidate_idx = next(
            (i for i, p in enumerate(result)
             if p.get("category") == "맛집"
             and not p.get("pinned")
             and p.get("name") not in used_meal_names),
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
        used_meal_names.add(candidate["name"])
        if candidate_idx < best_insert_idx:
            best_insert_idx -= 1
        result.insert(best_insert_idx, candidate)

        logger.debug(
            "[식사 배치] '%s' → 인덱스 %d (%s window)",
            candidate.get("name", "?"), best_insert_idx, meal["type"]
        )

    return result

def _route_distance(route: list[dict]) -> float:
    return sum(haversine_distance(route[i], route[i + 1]) for i in range(len(route) - 1))


def _diversify_categories(route: list[dict], max_run: int = 2) -> list[dict]:
    """
    같은 카테고리가 max_run개 넘게 연달아 나오면, 다른 카테고리 장소와
    자리를 바꿔서 완화한다. 거리 증가가 가장 적은 교환만 적용한다.

    거리 최적화(optimized_route)만 쓰면 우연히 카페 3곳이 지리적으로
    뭉쳐있을 때 "카페만 연속 3번 방문" 같은 부자연스러운 동선이 나올 수
    있어서, 순수 거리 최적화 뒤에 이 보정을 한 번 더 거친다.
    """
    route = list(route)
    n = len(route)
    guard = 0

    while guard < n * 2:
        guard += 1
        run_found = False

        for i in range(n - max_run):
            window = route[i:i + max_run + 1]
            categories = {p["category"] for p in window}
            if len(categories) != 1:
                continue

            run_found = True
            target = i + max_run
            run_category = route[target]["category"]

            best_j, best_dist = None, _route_distance(route)
            for j in range(n):
                if i <= j <= target or route[j]["category"] == run_category:
                    continue
                trial = list(route)
                trial[target], trial[j] = trial[j], trial[target]
                d = _route_distance(trial)
                if d < best_dist:
                    best_dist = d
                    best_j = j

            if best_j is not None:
                route[target], route[best_j] = route[best_j], route[target]
            break  # 하나 고쳤으면 처음부터 다시 스캔 (교환이 다른 구간에 영향을 줄 수 있음)

        if not run_found:
            break

    return route


def human_like_route(day_places, transport_mode: str = "대중교통"):
    """
    하루치 장소들의 방문 순서를 정한다.

    랜드마크가 있으면 그곳을 아침 첫 방문지로 고정하고, 나머지는 전부
    optimized_route()(최근접 이웃 + 2-opt)로 실제 거리 기준 최단 동선을 짠 뒤,
    같은 카테고리가 3번 넘게 연달아 나오면 _diversify_categories()로 완화한다.
    식사 시간대 배치는 이후 _arrange_meals_in_places()가 순서와 무관하게
    별도로 조정하므로 여기서는 식사 타이밍을 신경 쓰지 않는다.

    ⚠️ 예전에는 카테고리별로 그룹을 나눠 고정 패턴(관광지 2 → 맛집 1 → 카페 1 → ...)으로
       배치했는데, 그룹 내부 순서가 입력 순서를 그대로 따라가다 보니 위경도를 전혀
       고려하지 않아 동선이 지그재그로 튀는 문제가 있었다 (실측: 같은 6곳 기준 약 4.8배
       더 먼 거리). 이제는 거리 기반으로 순서를 정하고, 카테고리 쏠림만 별도로 보정한다.
    """
    if not day_places:
        return []

    landmarks = [p for p in day_places if p.get("is_landmark")]
    start = landmarks[0] if landmarks else day_places[0]
    remaining = [p for p in day_places if p is not start]

    route = [start] + optimized_route(remaining, start, transport_mode)
    return _diversify_categories(route)

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
    call_counter: CallCounter | None = None,
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
        current_min, transport_mode,
    )

    if not day_places:
        return _empty_day(day_idx, daily_start_time, daily_end_time, pin_warnings)

    timeline: list[dict] = []
    excluded_places: list[dict] = []
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

        sorted_places = human_like_route(day_places, transport_mode)
        first_travel_info = calculate_travel_time(dep_dict, sorted_places[0], transport_mode, call_counter) if sorted_places else {"time": 0, "payment": None, "transfer": None}
        first_travel = int(first_travel_info["time"])

        timeline.append({
            "place":           explicit_departure.name,
            "category":        "출발지",
            "address":         explicit_departure.address or "",
            "lat":             explicit_departure.lat,
            "lng":             explicit_departure.lng,
            "stay_minutes":    stay_min,
            "travel_minutes":  first_travel,
            "travel_payment":  first_travel_info["payment"],
            "travel_transfer": first_travel_info["transfer"],
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
        sorted_places = human_like_route(day_places, transport_mode)

        # 2일차 이후 숙소 출발 노드 추가
        if start_location and day_idx > 0:
            first_travel_info = calculate_travel_time(
                start_location, sorted_places[0], transport_mode, call_counter
            ) if sorted_places else {"time": 0, "payment": None, "transfer": None}
            first_travel = int(first_travel_info["time"])

            timeline.append({
                "place":           start_location["name"],
                "category":        "숙소",
                "address":         start_location.get("address", ""),
                "lat":             start_location.get("lat"),
                "lng":             start_location.get("lng"),
                "stay_minutes":    0,
                "travel_minutes":  first_travel,
                "travel_payment":  first_travel_info["payment"],
                "travel_transfer": first_travel_info["transfer"],
                "time":            minutes_to_time_str(current_min),
                "type":            "hotel_checkout",
                "pinned":          False,
            })
            total_travel += first_travel
            current_min  += first_travel
            has_departure = True

    # ── 식사 시간 배치 ────────────────────────────────────────
    # sorted_places(time 필드 없음) 상태에서 직접 호출.
    # 누적 체류시간 기반 추정으로 window를 판단하므로 time 필드 주입 불필요.
    meal_times = _find_meal_times(start_h * 60 + start_m, end_h * 60 + end_m, prefs_dict)
    avg_stay_for_meal = int(
        sum(p.get("stay", 75) for p in sorted_places) / len(sorted_places)
    ) if sorted_places else 75
    # avg_travel: 고정 추정값 사용 (숙소 출발 노드 포함 시 total_travel이 0이어서 나눗셈 오류 방지)
    avg_travel_for_meal = 20
    sorted_places = _arrange_meals_in_places(
        sorted_places, meal_times, current_min, avg_stay_for_meal, avg_travel_for_meal
    )

    # ── 장소 방문 ───────────────────────────────────────────────
    travel_minutes_list: list[int] = []

    used_lunch  = False
    used_dinner = False
    lunch_time  = next((m["time"] for m in meal_times if m["type"] == "lunch"),  None)
    dinner_time = next((m["time"] for m in meal_times if m["type"] == "dinner"), None)

    for i, place in enumerate(sorted_places):

        # 다음 장소 이동시간 미리 계산
        next_travel = 0
        next_travel_payment = None
        next_travel_transfer = None
        if i < len(sorted_places) - 1:
            next_travel_info = calculate_travel_time(place, sorted_places[i + 1], transport_mode, call_counter)
            next_travel = int(next_travel_info["time"])
            next_travel_payment = next_travel_info["payment"]
            next_travel_transfer = next_travel_info["transfer"]

        # 이전 장소 → 현재 장소 이동시간
        prev_travel = 0
        if i > 0:
            prev_travel = int(
                calculate_travel_time(sorted_places[i - 1], place, transport_mode, call_counter)["time"]
            )

        # "도착 시간" 기준 계산 (핵심)
        arrival_time = current_min + prev_travel

        # 점심 처리 — arrival이 lunch_time ±30분 이내일 때만 시간 스냅
        # 너무 이르면(arrival < lunch_time - 30) lunch_time까지 대기
        if (
            place["category"] == "맛집"
            and not used_lunch
            and lunch_time
        ):
            if arrival_time < lunch_time - 30:
                # 아직 점심 시간이 아님 → lunch_time까지 기다렸다가 먹기
                current_min = lunch_time
                used_lunch = True
            elif arrival_time <= lunch_time + 30:
                # 점심 window 안 → 스냅
                current_min = max(current_min, lunch_time)
                used_lunch = True
            # lunch_time + 30 초과면 점심 타이밍 놓친 것 → 스냅 없이 그냥 진행

        # 저녁 처리 — arrival이 dinner_time ±30분 이내일 때만 시간 스냅
        elif (
            place["category"] == "맛집"
            and used_lunch
            and not used_dinner
            and dinner_time
        ):
            if arrival_time < dinner_time - 30:
                current_min = dinner_time
                used_dinner = True
            elif arrival_time <= dinner_time + 30:
                current_min = max(current_min, dinner_time)
                used_dinner = True

        # 카페 시간 제한 (UX 개선) — 오전 카페는 일정에서 제외하고 사유를 기록
        if place["category"] == "카페" and current_min < 14 * 60:
            excluded_places.append({
                "name":     place["name"],
                "category": place["category"],
                "day":      day_idx + 1,
                "reason":   "morning_cafe",
            })
            continue

        # pinned 시간 우선
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
            "travel_payment":  next_travel_payment,
            "travel_transfer": next_travel_transfer,
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
            return_travel_info = calculate_travel_time(sorted_places[-1], return_point, transport_mode, call_counter)
            return_travel = int(return_travel_info["time"])
            total_travel += return_travel
            current_min  += return_travel
            # 마지막 방문 장소의 travel_minutes를 복귀 이동시간으로 소급 수정
            if timeline:
                timeline[-1]["travel_minutes"]  = return_travel
                timeline[-1]["travel_payment"]  = return_travel_info["payment"]
                timeline[-1]["travel_transfer"] = return_travel_info["transfer"]
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
            hotel_travel_info = calculate_travel_time(sorted_places[-1], tonight_hotel, transport_mode, call_counter)
            hotel_travel = int(hotel_travel_info["time"])
            total_travel += hotel_travel
            current_min  += hotel_travel
            # 마지막 방문 장소의 travel_minutes를 숙소 이동시간으로 소급 수정
            if timeline:
                timeline[-1]["travel_minutes"]  = hotel_travel
                timeline[-1]["travel_payment"]  = hotel_travel_info["payment"]
                timeline[-1]["travel_transfer"] = hotel_travel_info["transfer"]

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
        "excluded_places":   excluded_places,
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
        "excluded_places": [],
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

    # 지리적 배정은 카테고리를 고려하지 않으므로, 하루 식사 슬롯(점심/저녁) 수를
    # 넘는 맛집은 여기서 미리 제외한다 — 그대로 두면 식사 시간에 못 맞춘 여분의
    # 맛집이 다른 맛집 옆에 나란히 배치되는 문제가 생긴다.
    # "식사 시간 포함"을 꺼서 lunch/dinner가 사실상 비활성(23:59)인 경우나,
    # 하루가 너무 짧아 식사 window가 하나도 안 잡히는 경우엔 slot_count=0이
    # 되는데, 이때는 맛집을 아예 다 잘라내면 안 되므로 캡을 적용하지 않는다.
    meal_slot_count = len(_find_meal_times(
        start_h * 60 + start_m, end_h * 60 + end_m, prefs_dict
    ))
    if meal_slot_count > 0:
        day_assignments, meal_cap_excluded = _cap_meal_places(
            day_assignments, day_anchors, meal_slot_count
        )
    else:
        meal_cap_excluded = []

    # 카페도 같은 이유로 하루 개수를 페이스 기반으로 제한한다 (tight=1, normal/relaxed=2).
    day_assignments, cafe_cap_excluded = _cap_cafe_places(
        day_assignments, day_anchors, pace
    )

    avg_stay      = int(sum(p["stay"] for p in enriched) / len(enriched)) if enriched else 75
    available_min = (end_h * 60 + end_m) - (start_h * 60 + start_m)
    meal_buffer   = 60  # 점심·저녁 식사 시간 확보

    # pace에 따라 하루 최대 장소 수 조정
    # tight(효율적): 가능한 많이 / normal(균형): 80% / relaxed(여유): 60%
    _PACE_PLACE_RATIO = {"tight": 1.0, "normal": 0.8, "relaxed": 0.6}
    pace_ratio  = _PACE_PLACE_RATIO.get(pace, 0.8)
    max_per_day = max(1, int((available_min - meal_buffer) // (avg_stay + 35) * pace_ratio))
    logger.info(
        "[최적화] 하루 최대 장소 수: %d개 (avg_stay=%d분, 가용=%d분, pace=%s×%.1f)",
        max_per_day, avg_stay, available_min - meal_buffer, pace, pace_ratio
    )

    day_assignments, redistribute_excluded = _redistribute(
        day_assignments, day_anchors, max_per_day, hotels, transport_mode
    )

    result: dict = {}
    call_counter = CallCounter()  # 일정 생성 1회당 ODsay API 호출 상한 카운터
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
            call_counter     = call_counter,
        )
        # 재분배 단계(용량 초과·식사 슬롯 초과·카페 상한 초과)에서 이 날짜에 제외된 장소도 함께 보고
        day_data["excluded_places"] = day_data.get("excluded_places", []) + [
            e for e in redistribute_excluded + meal_cap_excluded + cafe_cap_excluded
            if e["day"] == day_idx + 1
        ]
        day_key = f"day_{day_idx + 1}"
        result[day_key] = {
            "date":     date_info["formatted"] if date_info else f"Day {day_idx + 1}",
            "date_raw": date_info["date"]      if date_info else None,
            **day_data,
        }

    result["duplicates"] = dups
    return result