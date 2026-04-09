import re
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta


# ─── 좌표 / 거리 ───────────────────────────────────────────────

def convert_naver_coords_to_wgs84(mapx, mapy) -> tuple[float | None, float | None]:
    """네이버 검색 API 좌표 → WGS84 위경도"""
    try:
        lng = float(mapx) / 10_000_000.0
        lat = float(mapy) / 10_000_000.0
        if not (33 <= lat <= 43 and 124 <= lng <= 132):
            return None, None
        return lat, lng
    except Exception:
        return None, None


def haversine_distance(place1: dict, place2: dict) -> float:
    """두 장소 간 직선 거리 (km)"""
    lat1, lon1 = radians(place1["lat"]), radians(place1["lng"])
    lat2, lon2 = radians(place2["lat"]), radians(place2["lng"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))


def calculate_travel_time(place1: dict, place2: dict, mode: str = "대중교통") -> float:
    """
    교통수단별 이동 시간 추정 (분).

    직선 거리 기반이지만 도로 우회율(winding factor)과
    교통수단별 특성을 반영해 현실적으로 보정:
      - 도보:     도로 우회율 1.3, 속도 4 km/h, 5km 초과시 불가
      - 대중교통: 도로 우회율 1.4, 탑승 대기/환승 오버헤드 추가
                  단거리(<1km)는 도보 속도, 중거리는 20km/h, 장거리는 30km/h
      - 택시:     도로 우회율 1.3, 단거리 최소 10분, 이후 30~40km/h

    출발지 카테고리인 경우 공항/역 이동 시간으로 고정값 사용.
    """
    if "출발지" in place1.get("category", "") or "출발지" in place2.get("category", ""):
        return {"도보": 999, "대중교통": 70, "택시": 50}.get(mode, 70)

    d = haversine_distance(place1, place2)

    if mode == "도보":
        if d > 5:
            return 999
        road_d = d * 1.3
        return (road_d / 4) * 60

    elif mode == "대중교통":
        road_d = d * 1.4
        if d < 0.5:
            # 걸어가는 게 빠른 거리 - 실질 도보
            return (road_d / 4) * 60
        elif d < 2:
            # 버스/지하철 탑승 준비 + 환승 없음
            transit_time = (road_d / 20) * 60
            overhead = 10  # 대기
            return transit_time + overhead
        elif d < 10:
            transit_time = (road_d / 25) * 60
            overhead = 15  # 대기 + 환승 가능성
            return transit_time + overhead
        else:
            transit_time = (road_d / 30) * 60
            overhead = 20  # 환승 포함
            return transit_time + overhead

    elif mode == "택시":
        road_d = d * 1.3
        if d < 1:
            return 10  # 최소 요금 구간
        elif d < 5:
            return (road_d / 25) * 60 + 5
        elif d < 20:
            return (road_d / 35) * 60 + 5
        else:
            return (road_d / 40) * 60 + 5

    # 기본 fallback
    return 15 + (d / 20) * 60


# ─── 경로 최적화 ───────────────────────────────────────────────

def nearest_neighbor_route(places: list, start: dict) -> list:
    """Nearest Neighbor 경로 최적화"""
    if not places:
        return []
    route, current, unvisited = [], start, set(range(len(places)))
    while unvisited:
        idx = min(unvisited, key=lambda i: haversine_distance(current, places[i]))
        route.append(places[idx])
        current = places[idx]
        unvisited.remove(idx)
    return route


def two_opt_improve(route: list, mode: str = "대중교통", max_iter: int = 100) -> list:
    """
    2-opt 교환으로 Nearest Neighbor 결과를 개선.
    교환 후 총 이동 시간이 줄어들면 채택.
    장소가 3개 미만이면 그대로 반환.
    """
    if len(route) < 3:
        return route

    def total_time(r: list) -> float:
        return sum(
            calculate_travel_time(r[i], r[i + 1], mode)
            for i in range(len(r) - 1)
        )

    best = list(route)
    improved = True
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                before = (
                    calculate_travel_time(best[i], best[i + 1], mode)
                    + calculate_travel_time(best[j - 1], best[j], mode)
                    if j < len(best) else 0
                )
                after = (
                    calculate_travel_time(best[i], best[j - 1], mode)
                    + calculate_travel_time(best[i + 1], best[j], mode)
                    if j < len(best) else 0
                )
                if after < before - 0.5:  # 0.5분 이상 개선될 때만 교환
                    best[i + 1:j] = best[i + 1:j][::-1]
                    improved = True
    return best


def optimized_route(places: list, start: dict, mode: str = "대중교통") -> list:
    """Nearest Neighbor 후 2-opt 개선"""
    nn = nearest_neighbor_route(places, start)
    return two_opt_improve(nn, mode)


# ─── 시간 / 날짜 ───────────────────────────────────────────────

def parse_time_to_minutes(time_str: str) -> int:
    """'HH:MM' → 분"""
    try:
        h, m = map(int, time_str.split(":"))
        return h * 60 + m
    except Exception:
        return 0


def minutes_to_time_str(minutes: int) -> str:
    """분 → 'HH:MM'"""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def format_date_korean(date_str: str) -> str:
    """'YYYY-MM-DD' → 'YYYY년 M월 D일 (요일)'"""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        weekday = ["월", "화", "수", "목", "금", "토", "일"][d.weekday()]
        return f"{d.year}년 {d.month}월 {d.day}일 ({weekday})"
    except Exception:
        return date_str


def calculate_dates(start_date: str, n_days: int) -> list[dict]:
    """여행 기간 날짜 리스트"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        return [
            {
                "date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
                "formatted": format_date_korean((start + timedelta(days=i)).strftime("%Y-%m-%d")),
                "day_num": i + 1,
            }
            for i in range(n_days)
        ]
    except Exception:
        return []


# ─── 검색 쿼리 분류 ────────────────────────────────────────────

_EXCLUDE_KEYWORDS = [
    "맛집", "카페", "음식점", "레스토랑", "치킨", "피자",
    "호텔", "숙소", "병원", "약국", "편의점", "마트",
    "쇼핑", "백화점", "영화관", "노래방", "술집", "바",
    "찜질방", "pc방", "헬스장", "학원",
]
_ROAD_PATTERNS = [
    r"[가-힣]+시\s+[가-힣]+구\s+[가-힣]+로\s+\d+",
    r"[가-힣]+구\s+[가-힣]+로\s+\d+",
    r"[가-힣]+로\s+\d+번길\s+\d+",
    r"[가-힣]+로\s+\d+-\d+",
    r"[가-힣]+로\s+\d+",
]
_JIBUN_PATTERNS = [
    r"[가-힣]+시\s+[가-힣]+구\s+[가-힣]+동\s+\d+-\d+",
    r"[가-힣]+구\s+[가-힣]+동\s+\d+-\d+",
    r"[가-힣]+동\s+\d+-\d+",
    r"[가-힣]+동\s+\d+번지",
]

_FOOD_KW = [
    "음식점", "한식", "중식", "일식", "양식", "레스토랑", "식당",
    "치킨", "피자", "분식", "고기", "회", "초밥", "쌈밥", "찌개",
    "국밥", "칼국수", "돈까스", "파스타", "스테이크", "뷔페", "술집",
]
_CAFE_KW = ["카페", "커피", "디저트", "베이커리", "빵집", "케이크", "와플",
            "아이스크림", "티", "차", "브런치"]
_SHOP_KW = ["쇼핑", "백화점", "마트", "시장", "상가", "몰", "아울렛",
            "편의점", "상점", "매장", "부티크"]


def is_address_query(query: str) -> bool:
    """Geocoding API를 써야 할 주소인지 판단"""
    if any(kw in query for kw in _EXCLUDE_KEYWORDS):
        return False
    return any(re.search(p, query) for p in _ROAD_PATTERNS + _JIBUN_PATTERNS)


def categorize_place(naver_category: str) -> str:
    """네이버 카테고리 문자열 → 내부 카테고리"""
    c = naver_category.lower()
    if any(k in c for k in _CAFE_KW):  return "카페"
    if any(k in c for k in _FOOD_KW):  return "맛집"
    if any(k in c for k in _SHOP_KW):  return "쇼핑"
    return "관광지"


# ─── 일정 검증 ─────────────────────────────────────────────────

def _levenshtein_similarity(s1: str, s2: str) -> float:
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2: return 1.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2: return 0.0
    dp = [[0] * (l2 + 1) for _ in range(l1 + 1)]
    for i in range(l1 + 1): dp[i][0] = i
    for j in range(l2 + 1): dp[0][j] = j
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return 1 - dp[l1][l2] / max(l1, l2)


def check_duplicate_places(places: list) -> list:
    """장소 목록에서 중복 탐지"""
    duplicates = []
    for i, p1 in enumerate(places):
        for j, p2 in enumerate(places[i+1:], i+1):
            if p1["name"].strip().lower() == p2["name"].strip().lower():
                duplicates.append({"indices": [i, j], "reason": "same_name", "severity": "high",
                                   "place1": p1["name"], "place2": p2["name"]})
                continue
            dist = haversine_distance(p1, p2)
            sim  = _levenshtein_similarity(p1["name"], p2["name"])
            if dist < 0.01 and sim > 0.7:
                duplicates.append({"indices": [i, j], "reason": "similar_name_close", "severity": "high",
                                   "place1": p1["name"], "place2": p2["name"],
                                   "distance": round(dist*1000, 1), "similarity": round(sim*100)})
            elif dist < 0.0001:
                duplicates.append({"indices": [i, j], "reason": "same_location", "severity": "high",
                                   "place1": p1["name"], "place2": p2["name"]})
            elif dist < 0.01 and p1.get("category") == p2.get("category"):
                duplicates.append({"indices": [i, j], "reason": "nearby_same_category", "severity": "medium",
                                   "place1": p1["name"], "place2": p2["name"],
                                   "distance": round(dist*1000, 1), "category": p1.get("category")})
    return duplicates


def check_category_sequence(timeline: list) -> list:
    """동일 카테고리 3회 연속 경고"""
    warnings = []
    for i in range(len(timeline) - 2):
        cats = [timeline[i+k].get("category") for k in range(3)]
        if cats[0] == cats[1] == cats[2]:
            warnings.append({
                "start_index": i, "category": cats[0], "count": 3,
                "places": [timeline[i+k].get("place") for k in range(3)],
            })
    return warnings