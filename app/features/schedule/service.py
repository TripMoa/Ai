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
    haversine_distance, _travel_time_for_routing,
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

PACE_MULTIPLIER: dict[str, float] = {"tight": 0.85, "normal": 1.0, "relaxed": 1.3}

# 하루 가용 시간(식사 여유 60분 제외) 중 장소로 채우는 비율 — 하루 최대 장소 수 계산에 쓴다
_PACE_FILL_RATIO: dict[str, float] = {"tight": 0.85, "normal": 0.85, "relaxed": 0.75}

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

def _build_pinned_info(
    pinned_places, enriched: list[dict], n_days: int,
) -> tuple[dict, list[tuple[int, str]]]:
    """
    핀 정보를 place_index 기준으로 구성한다.

    반환: (pinned_info, notices) — notices는 (표시할 day, 안내 문구) 목록.
    - 핀의 day가 1..n_days를 벗어나면 그 날은 아예 그려지지 않아 장소가 조용히 사라지므로,
      핀을 해제하고 일반 장소로 배정되게 둔다(pinned_info에 넣지 않음).
    - 같은 장소가 여러 날에 핀돼 있으면 마지막 지정만 남는다(장소는 한 번만 방문).
    """
    pinned_info: dict = {}
    notices: list[tuple[int, str]] = []
    for pin in pinned_places:
        if not (0 <= pin.place_index < len(enriched)):
            continue
        name = enriched[pin.place_index]["name"]
        if not (1 <= pin.day <= n_days):
            notices.append((
                1,
                f"'{name}'의 고정 일차(DAY {pin.day})가 여행 일수({n_days}일)를 벗어나 "
                f"고정을 해제하고 일반 장소로 배정했어요",
            ))
            continue
        if pin.place_index in pinned_info:
            notices.append((
                pin.day,
                f"'{name}'이(가) 여러 날에 고정돼 있어 마지막 지정(DAY {pin.day})만 반영했어요",
            ))
        pinned_info[pin.place_index] = {
            "day":   pin.day,
            "time":  pin.time,
            "place": enriched[pin.place_index].copy(),
        }
    return pinned_info, notices


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


def _select_dense_indices(candidates: list, dist_matrix: list, limit: int) -> set:
    """
    후보 중 서로 가장 밀집된 limit개를 고른다 (군집 밀도 기반).

    알고리즘:
      1. 각 장소의 "군집 점수" = 가장 가까운 (limit-1)개 장소까지의 평균 거리
         → 점수가 낮을수록 주변에 장소가 밀집되어 있음
      2. 군집 점수 낮은 순으로 limit개 선택
      3. 선택 결과에 "맛집"/"카페"가 하나도 없으면 가장 군집 점수가 좋은 해당 카테고리로
         가장 점수가 나쁜 항목 하나를 교체 (식사 장소가 통째로 잘리는 것 방지)
    """
    n = len(candidates)
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

    return selected_idx


def _estimate_day_minutes(places: list, anchor: dict | None, transport_mode: str) -> float:
    """
    하루치 장소의 대략적인 소요 시간(분) = 체류 시간 합 + 이동 시간.

    계획 단계 추정용이라 ODsay 등 외부 API를 부르지 않는다(하버사인/캐시 기반).
    기준점(숙소·출발지)이 있으면 거기서 출발해 돌아오는 구간까지 포함한다.
    (기준점은 좌표만 쓴다 — "출발지" 카테고리를 그대로 넘기면 라우팅용 고정 시간이 붙어 과대 추정됨)
    """
    if not places:
        return 0.0

    stay = sum(p.get("stay", 60) for p in places)
    anchor_pt = {"lat": anchor["lat"], "lng": anchor["lng"]} if anchor else None

    if anchor_pt:
        start, rest = anchor_pt, places
    else:
        start, rest = places[0], places[1:]

    route = optimized_route(rest, start, transport_mode) if rest else []
    travel, prev = 0.0, start
    for p in route:
        travel += _travel_time_for_routing(prev, p, transport_mode)
        prev = p
    if anchor_pt and route:
        travel += _travel_time_for_routing(prev, anchor_pt, transport_mode)

    return stay + travel


def _redistribute(
    day_assignments: list[list],
    day_anchors: list[dict | None],
    day_budget_min: float,
    hotels: list,
    transport_mode: str,
    day_fixed: list[list] | None = None,
) -> tuple[list[list], list[dict]]:
    """
    하루 시간 예산(day_budget_min) 안에 들어오는 가장 많은 장소를 고른다.

    예전에는 "하루 최대 N곳"이라는 개수 상한으로 잘랐는데, 장소마다 체류 시간이 크게 달라
    (궁궐 2시간 vs 전망대 30분) 종료 시간을 넘기거나 반대로 한참 일찍 끝나는 일이 잦았다.
    이제는 장소별 실제 체류 시간 + 이동 시간의 추정 합이 예산 안에 드는 최대 개수를 고른다.
    숙소를 옮기는 날은 이동 부담이 커서 예산을 절반으로 줄이고 최소 2곳은 남긴다.

    day_fixed: 날짜별로 이미 그날에 고정된(핀) 장소. 빼거나 옮길 수 없지만 시간은 차지하므로
    예산 계산에 포함한다(안 그러면 핀이 있는 날은 핀 + 일반 장소가 예산 두 배로 몰려 종료 시간을 넘긴다).
    예산 때문에 빠진 장소는 다른 날에 여유가 있으면 그날로 옮기고, 어디에도 안 들어갈 때만 제외한다.
    (식사·카페는 날짜별 슬롯·개수 상한이 따로 있어 옮기지 않는다.)
    """
    n_days = len(day_assignments)
    result = [list(d) for d in day_assignments]
    fixed = day_fixed or [[] for _ in range(n_days)]
    excluded: list[dict] = []
    overflow: list[tuple[int, dict]] = []  # (원래 날짜 인덱스, 장소)

    budgets = [
        day_budget_min * (0.5 if _is_hotel_move_day(d, hotels) else 1.0)
        for d in range(n_days)
    ]

    for day_idx in range(n_days):
        candidates = result[day_idx]
        n = len(candidates)
        if n == 0:
            continue

        budget = budgets[day_idx]
        day_fixed_places = fixed[day_idx]
        anchor = day_anchors[day_idx]
        is_move_day = _is_hotel_move_day(day_idx, hotels)
        # 고정 장소가 이미 있으면 그날이 비지 않으므로 최소 유지 개수를 그만큼 줄인다
        min_keep = min(n, max(0, (2 if is_move_day else 1) - len(day_fixed_places)))

        # 전부 들어가면 그대로 둔다
        if _estimate_day_minutes(candidates + day_fixed_places, anchor, transport_mode) <= budget:
            continue

        dist_matrix = [
            [haversine_distance(candidates[i], candidates[j]) for j in range(n)]
            for i in range(n)
        ]

        def _dense(limit: int) -> set:
            return _select_dense_indices(candidates, dist_matrix, limit) if limit > 0 else set()

        # 예산 안에 드는 가장 큰 limit을 찾는다 (n-1부터 줄여가며 밀집 조합을 다시 고름)
        selected_idx = _dense(min_keep)
        for limit in range(n - 1, min_keep - 1, -1):
            trial = _dense(limit)
            trial_places = [candidates[i] for i in range(n) if i in trial]
            if _estimate_day_minutes(trial_places + day_fixed_places, anchor, transport_mode) <= budget:
                selected_idx = trial
                break

        result[day_idx] = [candidates[i] for i in range(n) if i in selected_idx]

        for i in sorted(set(range(n)) - selected_idx):
            overflow.append((day_idx, candidates[i]))

    for from_day, place in overflow:
        best_day, best_extra = None, None
        if place["category"] not in ("맛집", "카페"):
            for d in range(n_days):
                if d == from_day:
                    continue
                base = _estimate_day_minutes(result[d] + fixed[d], day_anchors[d], transport_mode)
                total = _estimate_day_minutes(result[d] + [place] + fixed[d], day_anchors[d], transport_mode)
                if total <= budgets[d] and (best_extra is None or total - base < best_extra):
                    best_day, best_extra = d, total - base
        if best_day is not None:
            logger.info("  [이동] '%s' Day %d → Day %d — 원래 날짜 시간 예산 초과", place["name"], from_day + 1, best_day + 1)
            result[best_day].append(place)
            continue
        logger.info(
            "  [제외] '%s' Day %d — 하루 시간 예산 초과 (군집 밀도 기반 선택)",
            place["name"], from_day + 1
        )
        excluded.append({
            "name":     place["name"],
            "category": place["category"],
            "day":      from_day + 1,
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
            # 사전 검사용 추정이라 ODsay를 부르지 않는다(호출 상한 카운터 밖에서 API를 쓰지 않도록)
            travel_min = int(_travel_time_for_routing(prev_place, pin["place"], transport_mode))
            earliest_arrival = prev_time + travel_min
            if earliest_arrival > pin_min:
                warnings_list.append(
                    f"'{pin['place']['name']}' {pin['time']} 도착은 "
                    f"이전 일정 기준 최소 {minutes_to_time_str(earliest_arrival)} 이후 가능 "
                    f"(이동시간 추정 {travel_min}분 포함)"
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

    # 방문 순서는 여기서 정하지 않는다. 핀 시각 사이의 빈 시간을 일반 장소로 채우는 일은
    # 시작 시각을 알아야 해서 _build_day_timeline이 _order_with_timed_pins로 처리한다.
    # (예전엔 여기서 순서를 짰지만 곧바로 human_like_route가 거리순으로 다시 섞어 핀 시간이 무시됐다)
    return regular + untimed_places + anchors, pin_warnings


def _order_with_timed_pins(
    day_places: list,
    start_min: int,
    transport_mode: str,
) -> list:
    """
    시간이 지정된 핀(pinned_time)이 있는 날의 방문 순서.

    핀을 시각순 기준점으로 두고, 각 핀 앞의 빈 시간에는 남은 일반 장소를 채운다.
    일반 장소의 기본 순서는 human_like_route(거리 기반)를 그대로 따르고, 다음 장소를 넣어도
    "체류 + 그 핀까지의 이동"이 핀 시각 안에 끝날 때만 채워 넣는다.
    핀 뒤에 남은 장소는 그 핀에서 가까운 순서로 이어 붙인다.

    이동시간은 ODsay를 부르지 않는 추정치(_travel_time_for_routing)만 쓴다.
    """
    timed = sorted(
        (p for p in day_places if p.get("pinned_time")),
        key=lambda p: parse_time_to_minutes(p["pinned_time"]),
    )
    free = [p for p in day_places if not p.get("pinned_time")]
    queue = human_like_route(free, transport_mode) if free else []

    def travel(a: dict | None, b: dict) -> float:
        return 0 if a is None else _travel_time_for_routing(a, b, transport_mode)

    result: list = []
    t = start_min
    prev: dict | None = None

    for pin in timed:
        pin_min = parse_time_to_minutes(pin["pinned_time"])

        while queue:
            cand = queue[0]
            finish = t + travel(prev, cand) + cand.get("stay", 60)
            if finish + _travel_time_for_routing(cand, pin, transport_mode) > pin_min:
                break
            result.append(queue.pop(0))
            t, prev = finish, cand

        result.append(pin)
        t = max(t + travel(prev, pin), pin_min) + pin.get("stay", 60)
        prev = pin

    if queue:
        result.extend(optimized_route(queue, prev, transport_mode))

    return result


# ─── 6. 하루 타임라인 생성 ────────────────────────────────────

def _find_meal_times(start_min: int, end_min: int, prefs: dict) -> list:
    lunch  = parse_time_to_minutes(prefs.get("lunch_time",  "12:00"))
    dinner = parse_time_to_minutes(prefs.get("dinner_time", "18:30"))
    result = []
    if 11*60+30 <= lunch  <= 14*60 and start_min <= lunch  <= end_min:
        result.append({"type": "lunch",  "time": lunch,
                       "time_str": minutes_to_time_str(lunch),
                       "window_start": lunch  - _MEAL_WINDOW_BEFORE_MIN,
                       "window_end":   lunch  + _MEAL_WINDOW_AFTER_MIN})
    if 17*60+30 <= dinner <= 20*60 and start_min <= dinner <= end_min:
        result.append({"type": "dinner", "time": dinner,
                       "time_str": minutes_to_time_str(dinner),
                       "window_start": dinner - _MEAL_WINDOW_BEFORE_MIN,
                       "window_end":   dinner + _MEAL_WINDOW_AFTER_MIN})
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

# 식사 시간대(창): 희망 식사 시각의 30분 전 ~ 90분 후. 맛집에 이 창 안에 도착하면 시각을 옮기지 않고 도착하는 대로 먹는다.
# 창보다 일찍 도착하면 창 시작까지 기다리되 최대 _MEAL_MAX_WAIT_MIN분까지만(그보다 이르면 식사로 치지 않는다).
_MEAL_WINDOW_BEFORE_MIN = 30
_MEAL_WINDOW_AFTER_MIN = 90
_MEAL_MAX_WAIT_MIN = 90
_MIDNIGHT_MIN = 24 * 60


def _clock_str(minutes: int) -> str:
    """시계 표기용 'HH:MM' — 자정을 넘긴 시각은 23:59로 맞춘다(초과 여부는 is_over_time이 알려준다)."""
    return minutes_to_time_str(min(minutes, _MIDNIGHT_MIN - 1))

# 카페 최소 방문 시각 — 많은 카페가 10~11시에 열어서(브런치 카페 포함) 그 전에는 문을 닫았을 수 있다.
# 영업시간 데이터가 없어 보수적으로 11시로 잡는다. 점심이 있는 날은 점심이 끝난 뒤(오후 휴식)로 더 늦춘다.
_CAFE_EARLIEST_MIN = 11 * 60
_LUNCH_DURATION_MIN = 60


def _arrange_cafes(
    places: list[dict],
    start_min: int,
    transport_mode: str,
    earliest_min: int = _CAFE_EARLIEST_MIN,
) -> list[dict]:
    """
    카페를 오후 휴식 자리로 옮긴다. 두 가지 경우에 그 뒤 장소 뒤로 미룬다.
      1. 추정 도착 시각이 earliest_min 이전인 카페 (점심 전·영업 전)
      2. 바로 앞 장소도 카페인 카페 (카페가 연달아 붙지 않게)

    예전에는 이른 카페를 일정에서 통째로 뺐지만, 지금은 순서만 바꾼다(시간이 안 되면 이후 단계에서
    시간 부족으로 빠진다). 사용자가 고정한 카페와, 뒤에 카페 아닌 장소가 없어 미룰 곳이 없는 카페는 그대로 둔다.
    start_min은 첫 장소에 도착하는 시각이다.
    """
    order = list(places)
    for _ in range(len(order) ** 2 + 1):
        t, prev, moved = start_min, None, False
        for i, p in enumerate(order):
            if prev is not None:
                t += int(_travel_time_for_routing(prev, p, transport_mode))
            if p.get("pinned_time"):
                t = max(t, parse_time_to_minutes(p["pinned_time"]))
            if (
                p["category"] == "카페" and not p.get("pinned")
                and (t < earliest_min or (prev is not None and prev["category"] == "카페"))
                and any(q["category"] != "카페" for q in order[i + 1:])
            ):
                j = next(k for k in range(i + 1, len(order)) if order[k]["category"] != "카페")
                order.insert(j, order.pop(i))  # pop 뒤 j번 원소는 j-1로 당겨지므로 그 바로 뒤에 들어간다
                moved = True
                break
            t += p.get("stay", 60)
            prev = p
        if not moved:
            break

    # 2단계: 위에서 못 푼 경우 — 뒤에 옮길 장소가 없어 카페가 연달아 붙어 있으면,
    # 그 카페를 앞쪽 (카페·식사가 아닌) 장소 앞으로 당긴다. 이때는 영업 시작 시각(11시)만 지킨다.
    def _arrivals(seq: list[dict]) -> list[int]:
        out, t, prev = [], start_min, None
        for q in seq:
            if prev is not None:
                t += int(_travel_time_for_routing(prev, q, transport_mode))
            if q.get("pinned_time"):
                t = max(t, parse_time_to_minutes(q["pinned_time"]))
            out.append(t)
            t += q.get("stay", 60)
            prev = q
        return out

    for _ in range(len(order)):
        fixed = False
        for i in range(1, len(order)):
            p = order[i]
            if p["category"] != "카페" or p.get("pinned") or order[i - 1]["category"] != "카페":
                continue
            for k in range(i - 2, -1, -1):
                if order[k]["category"] in ("카페", "맛집") or (k > 0 and order[k - 1]["category"] == "카페"):
                    continue
                cand = order[:i] + order[i + 1:]
                cand.insert(k, p)
                if _arrivals(cand)[k] >= _CAFE_EARLIEST_MIN:
                    order, fixed = cand, True
                    break
            if fixed:
                break
        if not fixed:
            break
    return order


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

    def _visit_order(route_start_min: int) -> list:
        # 시간 지정 핀이 있는 날은 핀 시각을 기준으로 앞뒤를 채우고, 없으면 기존 거리 기반 순서
        if any(p.get("pinned_time") for p in day_places):
            return _order_with_timed_pins(day_places, route_start_min, transport_mode)
        return human_like_route(day_places, transport_mode)

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

        sorted_places = _visit_order(current_min + stay_min)
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
        sorted_places = _visit_order(current_min)

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
    # 카페는 점심 뒤 오후 휴식으로 두고, 이른 카페·연달아 붙은 카페는 빼지 않고 뒤로 미룬다
    lunch_slot = next((m["time"] for m in meal_times if m["type"] == "lunch"), None)
    cafe_earliest = (
        max(_CAFE_EARLIEST_MIN, lunch_slot + _LUNCH_DURATION_MIN)
        if lunch_slot is not None and lunch_slot < 16 * 60 else _CAFE_EARLIEST_MIN
    )
    sorted_places = _arrange_cafes(sorted_places, current_min, transport_mode, cafe_earliest)

    # ── 장소 방문 ───────────────────────────────────────────────
    travel_minutes_list: list[int] = []

    used_lunch  = False
    used_dinner = False
    lunch_meal  = next((m for m in meal_times if m["type"] == "lunch"),  None)
    dinner_meal = next((m for m in meal_times if m["type"] == "dinner"), None)

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

        # 이 장소에 도착하는 시각 — 이전 장소에서의 이동시간은 앞 반복에서 이미 current_min에 더해져 있다
        arrival_time = current_min

        # 식사 시간 맞추기 — 맛집이 식사 시간대(창) 안에 도착하면 시각을 옮기지 않고 그대로 먹는다.
        # 창보다 조금 일찍 도착하면(최대 _MEAL_MAX_WAIT_MIN분) 창이 열릴 때까지 기다린다.
        # 아직 안 쓴 슬롯 중 창이 끝나지 않은 첫 슬롯을 고른다(점심을 건너뛴 날도 저녁은 맞춘다).
        # 시각이 고정된 핀은 핀 시간이 우선이라 건드리지 않는다.
        if place["category"] == "맛집" and not place.get("pinned_time"):
            if lunch_meal and not used_lunch and arrival_time <= lunch_meal["window_end"]:
                meal, slot = lunch_meal, "lunch"
            elif dinner_meal and not used_dinner and arrival_time <= dinner_meal["window_end"]:
                meal, slot = dinner_meal, "dinner"
            else:
                meal, slot = None, None
            if meal:
                if arrival_time >= meal["window_start"]:
                    is_meal = True
                elif meal["window_start"] - arrival_time <= _MEAL_MAX_WAIT_MIN:
                    current_min = meal["window_start"]  # 조금 이르면 창이 열릴 때까지 기다렸다 먹는다
                    is_meal = True
                else:
                    is_meal = False  # 너무 이르면 식사로 치지 않고 그냥 방문(슬롯은 남겨 둔다)
                if is_meal and slot == "lunch":
                    used_lunch = True
                elif is_meal and slot == "dinner":
                    used_dinner = True

        # pinned 시간 우선
        if place.get("pinned") and place.get("pinned_time"):
            pm = parse_time_to_minutes(place["pinned_time"])
            if pm > current_min:
                current_min = pm
            elif current_min - pm > 5 and not any(place["name"] in w for w in pin_warnings):
                # 앞 일정이 길어서 고정 시간보다 늦게 도착 — 조용히 넘어가지 않고 알려준다
                pin_warnings.append(
                    f"'{place['name']}' 고정 시간 {place['pinned_time']}보다 늦은 "
                    f"{minutes_to_time_str(current_min)}에 도착해요"
                )

        # 자정을 넘겨 시작하게 되는 장소는 시간표에 "25:10" 같은 시각이 생기므로 일정에서 뺀다
        if current_min >= _MIDNIGHT_MIN:
            excluded_places.append({
                "name":     place["name"],
                "category": place["category"],
                "day":      day_idx + 1,
                "reason":   "over_time",
            })
            if place.get("pinned"):
                pin_warnings.append(f"'{place['name']}' 고정 장소가 자정을 넘겨 일정에서 제외했어요")
            continue

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
            "pinned_time":     place.get("pinned_time"),
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
            "time":           _clock_str(current_min),
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
            "time":           _clock_str(current_min),
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
        "end_time":              _clock_str(current_min),
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

def _pick_over_time_victim(day_data: dict, end_min: int, timed_pin_end: int = 0) -> dict | None:
    """
    만들어진 하루 시간표가 종료 시간을 넘겼을 때 뺄 장소(타임라인 항목)를 고른다. 뺄 게 없으면 None.

    - 고정(핀) 장소는 절대 빼지 않는다. 시간을 지정한 핀 자체가 종료 시간 밖(timed_pin_end)이면
      장소를 빼도 해결되지 않으니, 종료 시간 이후에 시작하는 일반 장소만 빼고 나머지는 손대지 않는다
      (그 경우는 is_over_time과 경고로 남는다).
      시간 없이 일차만 고정한 핀은 앞의 장소를 빼면 그만큼 당겨지므로 일반 장소를 계속 뺀다.
    - 마지막에 방문하는 장소부터 뺀다. 다만 식사(맛집)는 하루 흐름에 중요하니 식사가 아닌 장소를 먼저 뺀다.
    - 하루에 최소 한 곳은 남긴다.
    """
    if not day_data.get("is_over_time"):
        return None

    visits = [p for p in day_data["places"] if p.get("type") == "visit"]

    removable = [p for p in visits if not p.get("pinned")]
    if timed_pin_end > end_min:
        # 시간 지정 핀 자체가 종료 시간 밖이면 앞 장소를 빼도 해결되지 않는다.
        # 다만 종료 시간 이후에 시작하는 일반 장소는 어차피 시간 밖이니 그것만 뺀다.
        removable = [p for p in removable if parse_time_to_minutes(p["time"]) >= end_min]
    if len(visits) <= 1 or not removable:
        return None

    non_meal = [p for p in removable if p.get("category") != "맛집"]
    return (non_meal or removable)[-1]


def _plan_days(
    places,
    n_days: int,
    transport_mode: str,
    hotels: list,
    departure_points: list,
    daily_start_time: str,
    daily_end_time: str,
    pinned_places: list,
    user_preferences: UserPreferences,
) -> dict:
    """
    장소를 날짜에 배정하고 하루 개수 상한을 적용하는 단계(타임라인 생성 이전).

    ODsay 같은 외부 API를 부르지 않는 순수 계산이라 일정 생성(generate_itinerary)과
    생성 전 예상(estimate_itinerary)이 같은 로직을 공유한다.
    상한 규칙을 바꿔도 예상이 어긋나지 않게 하려는 것.
    """
    pace       = user_preferences.pace
    prefs_dict = {
        "lunch_time":  user_preferences.lunch_time,
        "dinner_time": user_preferences.dinner_time,
    }

    enriched    = _enrich_places(places, pace)
    pinned_info, pin_notices = _build_pinned_info(pinned_places, enriched, n_days)

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

    # 사용자가 정한 하루 시작~종료 시간이 그날의 시간 예산이다.
    # pace의 영향은 두 군데(체류 시간 배율 PACE_MULTIPLIER, 아래 시간 채움 비율)에서 한 번씩만 반영한다.
    # 개수 상한 대신 "장소별 체류 + 이동 시간의 합이 예산 안에 드는 최대 개수"로 고른다(_redistribute).
    available_min = (end_h * 60 + end_m) - (start_h * 60 + start_m)
    fill_ratio    = _PACE_FILL_RATIO.get(pace, _PACE_FILL_RATIO["normal"])
    day_budget    = available_min * fill_ratio
    logger.info(
        "[최적화] 하루 시간 예산: %d분 (가용=%d분, pace=%s, 채움 %.2f)",
        int(day_budget), available_min, pace, fill_ratio
    )

    day_fixed = [
        [info["place"] for info in pinned_info.values() if info["day"] == d + 1]
        for d in range(n_days)
    ]
    day_assignments, redistribute_excluded = _redistribute(
        day_assignments, day_anchors, day_budget, hotels, transport_mode, day_fixed
    )
    max_per_day = max((len(d) for d in day_assignments), default=0)

    return {
        "prefs_dict":            prefs_dict,
        "pinned_info":           pinned_info,
        "pin_notices":           pin_notices,
        "day_assignments":       day_assignments,
        "meal_cap_excluded":     meal_cap_excluded,
        "cafe_cap_excluded":     cafe_cap_excluded,
        "redistribute_excluded": redistribute_excluded,
        "max_per_day":           max_per_day,
    }


def estimate_itinerary(
    places,
    n_days: int,
    transport_mode: str = "대중교통",
    hotels: list = None,
    departure_points: list = None,
    daily_start_time: str = "09:00",
    daily_end_time: str = "18:00",
    pinned_places: list = None,
    user_preferences: UserPreferences = None,
) -> dict:
    """
    일정 생성 전 예상 — 배정·개수 상한 단계까지만 돌려서 몇 곳이 들어갈지 알려준다.

    generate_itinerary와 같은 _plan_days를 쓰므로 하루 상한·식사/카페 제한이 실제 생성과 일치한다.
    (타임라인을 만든 뒤 종료 시간 초과로 추가로 빠지는 장소는 반영되지 않는다.)
    """
    if pinned_places    is None: pinned_places    = []
    if user_preferences is None: user_preferences = UserPreferences()
    if hotels           is None: hotels           = []
    if departure_points is None: departure_points = []

    plan = _plan_days(
        places, n_days, transport_mode, hotels, departure_points,
        daily_start_time, daily_end_time, pinned_places, user_preferences,
    )

    excluded = (
        plan["meal_cap_excluded"] + plan["cafe_cap_excluded"] + plan["redistribute_excluded"]
    )
    by_reason: dict[str, int] = defaultdict(int)
    for e in excluded:
        by_reason[e["reason"]] += 1

    return {
        "total_places":       len(places),
        "included":           len(places) - len(excluded),
        "excluded":           len(excluded),
        "max_per_day":        plan["max_per_day"],
        "n_days":             n_days,
        "excluded_by_reason": dict(by_reason),
    }


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

    logger.info(
        "일정 생성 시작: %d개 장소 / %d일 / pace=%s / 고정=%d개",
        len(places), n_days, user_preferences.pace, len(pinned_places),
    )

    dups = check_duplicate_places([
        {"name": p.name, "lat": p.lat, "lng": p.lng, "category": p.category}
        for p in places
    ])
    if dups:
        logger.warning("중복 장소 %d개 감지", len(dups))

    plan = _plan_days(
        places, n_days, transport_mode, hotels, departure_points,
        daily_start_time, daily_end_time, pinned_places, user_preferences,
    )
    prefs_dict            = plan["prefs_dict"]
    pinned_info           = plan["pinned_info"]
    pin_notices           = plan["pin_notices"]
    day_assignments       = plan["day_assignments"]
    meal_cap_excluded     = plan["meal_cap_excluded"]
    cafe_cap_excluded     = plan["cafe_cap_excluded"]
    redistribute_excluded = plan["redistribute_excluded"]

    dates_info   = calculate_dates(start_date, n_days) if start_date else []
    return_point = _get_return_point(departure_points)

    result: dict = {}
    call_counter = CallCounter()  # 일정 생성 1회당 ODsay API 호출 상한 카운터
    end_min = parse_time_to_minutes(daily_end_time)
    for day_idx in range(n_days):
        date_info = dates_info[day_idx] if dates_info else None
        day_places = day_assignments[day_idx]
        over_time_excluded: list[dict] = []
        timed_pin_end = max(
            (parse_time_to_minutes(info["time"]) + info["place"].get("stay", 60)
             for info in pinned_info.values() if info["day"] == day_idx + 1 and info.get("time")),
            default=0,
        )

        while True:
            day_data = _build_day_timeline(
                day_idx          = day_idx,
                day_places       = day_places,
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
            # 계획 단계는 추정이라 실제 시간표(실측 이동시간·식사 대기 포함)가 종료 시간을 넘을 수 있다.
            # 넘으면 마지막 장소부터 하나씩 빼고 다시 짠다 — 종료 시간을 실제로 지키기 위함.
            victim = _pick_over_time_victim(day_data, end_min, timed_pin_end)
            if victim is None:
                break
            remaining = [p for p in day_places if p["name"] != victim["place"]]
            if len(remaining) == len(day_places):
                break  # 목록에서 빠지지 않는 항목이면 더 시도해도 같은 결과 — 무한 반복 방지
            day_places = remaining
            over_time_excluded.append({
                "name":     victim["place"],
                "category": victim["category"],
                "day":      day_idx + 1,
                "reason":   "over_time",
            })
            logger.info("  [제외] '%s' Day %d — 종료 시간(%s) 초과", victim["place"], day_idx + 1, daily_end_time)

        # 재분배 단계(용량 초과·식사 슬롯 초과·카페 상한 초과)에서 이 날짜에 제외된 장소도 함께 보고
        day_data["excluded_places"] = day_data.get("excluded_places", []) + over_time_excluded + [
            e for e in redistribute_excluded + meal_cap_excluded + cafe_cap_excluded
            if e["day"] == day_idx + 1
        ]
        # 핀 구성 단계에서 생긴 안내(범위 밖 일차, 중복 고정 등)를 해당 날짜 경고에 합침
        day_data["pin_warnings"] = (day_data.get("pin_warnings") or []) + [
            msg for d, msg in pin_notices if d == day_idx + 1
        ]
        # 일반 장소를 다 덜어내도 종료 시간을 넘는다면 원인은 고정한 장소 — 사용자가 조정할 수 있게 알린다
        if day_data.get("is_over_time") and any(
            p.get("pinned") for p in day_data.get("places", []) if p.get("type") == "visit"
        ):
            day_data["pin_warnings"].append(
                f"고정한 장소 때문에 종료 시간({daily_end_time})을 넘겨요. "
                f"고정 시간이나 일차를 조정해 보세요"
            )
        day_key = f"day_{day_idx + 1}"
        result[day_key] = {
            "date":     date_info["formatted"] if date_info else f"Day {day_idx + 1}",
            "date_raw": date_info["date"]      if date_info else None,
            **day_data,
        }

    result["duplicates"] = dups
    return result