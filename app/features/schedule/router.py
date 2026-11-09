import re
from datetime import datetime
from fastapi import APIRouter, HTTPException

from config import NAVER_CLIENT_ID, ODSAY_API_KEY
from features.schedule.naver_api import geocode_address, local_search
from features.schedule.utils import (
    haversine_distance, calculate_travel_time,
    is_address_query, categorize_place,
)
from features.schedule.models import ItineraryRequest, DistanceRequest, TransitRequest, RecomputeDayRequest
from features.schedule.recompute import recompute_day
from features.schedule.odsay_api import search_routes, _fail
from features.schedule.service import generate_itinerary, estimate_itinerary

router = APIRouter(prefix="/schedule", tags=["schedule"])


# ─── 장소 검색 ─────────────────────────────────────────────────

@router.get("/search")
async def search_place(query: str, display: int = 10):
    """
    스마트 검색
    - 도로명/지번 주소 → Geocoding API (유료)
    - 상호명/키워드    → 지역 검색 API (무료)
    """
    if not NAVER_CLIENT_ID:
        raise HTTPException(status_code=500, detail="네이버 API 키가 설정되지 않았습니다.")

    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="검색어를 입력하세요")

    display = max(1, min(display, 100))
    print(f"\n[schedule] 검색: '{query}'")

    if is_address_query(query):
        print("주소 판별 → Geocoding")
        result = await geocode_address(query)
        if result["success"] and result["places"]:
            return _build_response(result["places"], query, "geocoding", "geocoding (유료)")

        print("Geocoding 실패 → 지역 검색 폴백")
        result = await local_search(query, display)
        _raise_if_search_failed(result)
        _fill_category(result.get("places", []))
        return {
            **_build_response(result.get("places", []), query,
                              "local_search (fallback)", "local_search (무료)"),
            "message": "주소 검색 실패, 일반 검색 결과입니다",
        }

    print("일반 검색 → 지역 검색 API")
    result = await local_search(query, display)
    _raise_if_search_failed(result)
    _fill_category(result.get("places", []))
    return _build_response(
        result.get("places", []), query,
        "local_search", "local_search (무료)",
        total=result.get("total", 0),
    )


def _raise_if_search_failed(result: dict) -> None:
    """
    검색 API 호출 자체가 실패했으면(키 오류·한도 초과·네트워크 등) 빈 결과로 뭉개지 않고 502로 알린다.
    (예전에는 실패도 "결과 없음"으로 나가서 키 문제 같은 원인이 화면에서 안 보였다)
    자세한 사유는 서버 로그에만 남기고, 사용자에게는 일시적 문제라고 안내한다.
    """
    if result.get("success"):
        return
    print(f"[schedule] 검색 API 실패: {result.get('error')}", flush=True)
    raise HTTPException(
        status_code=502,
        detail="장소 검색 서비스에 문제가 있어 결과를 가져오지 못했어요. 잠시 후 다시 시도해주세요.",
    )


def _fill_category(places: list) -> None:
    for p in places:
        p["category"] = categorize_place(p.get("naver_category", ""))


def _build_response(places, query, method, api_used, total=None) -> dict:
    return {
        "success":  True,
        "total":    total if total is not None else len(places),
        "display":  len(places),
        "places":   places,
        "query":    query,
        "method":   method,
        "api_used": api_used,
    }


# ─── 일정 생성 ─────────────────────────────────────────────────

_TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$|^24:00$")


def _to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _validate_itinerary_request(request: ItineraryRequest) -> None:
    """일정 생성·예상이 공유하는 요청 검증 (n_days는 날짜 기준으로 보정)"""
    if not request.places:
        raise HTTPException(status_code=400, detail="장소를 추가하세요")

    # 시간이 비었거나 형식이 틀리면 엔진 안쪽에서 500으로 터지므로 여기서 사유와 함께 400으로 돌려준다
    for label, value in (("시작 시간", request.daily_start_time), ("종료 시간", request.daily_end_time)):
        if not value or not _TIME_PATTERN.match(value):
            raise HTTPException(status_code=400, detail=f"{label}을 올바르게 입력하세요 (예: 09:00)")
    if _to_minutes(request.daily_start_time) >= _to_minutes(request.daily_end_time):
        raise HTTPException(status_code=400, detail="종료 시간은 시작 시간보다 늦어야 합니다")

    if request.start_date and request.end_date:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end   = datetime.strptime(request.end_date,   "%Y-%m-%d")
        if end < start:
            raise HTTPException(status_code=400, detail="종료일이 시작일보다 빠릅니다")
        calculated = (end - start).days + 1
        if request.n_days != calculated:
            print(f"n_days 보정: {request.n_days} → {calculated}")
            request.n_days = calculated

    if len(request.places) < request.n_days:
        raise HTTPException(
            status_code=400,
            detail=f"{request.n_days}일 여행에는 최소 {request.n_days}개 이상의 장소가 필요합니다",
        )


# estimate / generate / calculate_distance는 안에서 동기 계산과 외부 API 호출(ODsay)을 하므로
# async def가 아니라 def로 둔다 — async def면 이벤트 루프를 통째로 붙잡아 그동안 다른 요청
# (장소 검색, /health 포함)이 전부 멈춘다. def로 두면 FastAPI가 스레드 풀에서 실행한다.
@router.post("/estimate")
def estimate(request: ItineraryRequest):
    """
    일정 생성 전 예상 — 몇 곳이 들어가고 몇 곳이 빠질지.
    배정·개수 상한 단계까지만 계산하므로 ODsay 등 외부 API를 부르지 않고 빠르다.
    """
    _validate_itinerary_request(request)

    result = estimate_itinerary(
        places           = request.places,
        n_days           = request.n_days,
        transport_mode   = request.transportation_mode,
        hotels           = request.hotels or [],
        departure_points = request.departure_points or [],
        daily_start_time = request.daily_start_time,
        daily_end_time   = request.daily_end_time,
        pinned_places    = request.pinned_places,
        user_preferences = request.user_preferences,
    )
    return {"success": True, **result}


@router.post("/generate")
def generate(request: ItineraryRequest):
    """일정 생성 (다중 숙소 & 다중 출발지 지원)"""
    _validate_itinerary_request(request)

    # models.py validator에서 이미 hotels/departure_points로 변환 완료
    itinerary = generate_itinerary(
        places           = request.places,
        n_days           = request.n_days,
        transport_mode   = request.transportation_mode,
        hotels           = request.hotels or [],
        departure_points = request.departure_points or [],
        start_date       = request.start_date,
        daily_start_time = request.daily_start_time,
        daily_end_time   = request.daily_end_time,
        pinned_places    = request.pinned_places,
        user_preferences = request.user_preferences,
    )
    import json
    print("\n===== 일정 생성 결과 =====")
    print(json.dumps(itinerary, indent=2, ensure_ascii=False))
    
    prefs = request.user_preferences
    dps   = request.departure_points or []
    duplicates = itinerary.pop("duplicates", [])

    return {
        "success":   True,
        "message":   "일정 생성 완료",
        "itinerary": itinerary,
        "warnings": {
            "duplicates":           duplicates,
            "has_duplicates":       bool(duplicates),
            "high_severity_count":   sum(1 for d in duplicates if d.get("severity") == "high"),
            "medium_severity_count": sum(1 for d in duplicates if d.get("severity") == "medium"),
        },
        "settings": {
            "n_days":              request.n_days,
            "start_date":          request.start_date,
            "end_date":            request.end_date,
            "transportation_mode": request.transportation_mode,
            "total_places":        len(request.places),
            "has_hotel":           bool(request.hotels),
            "hotels_count":        len(request.hotels or []),
            "hotels": [
                {
                    "name":          h.name,
                    "check_in_day":  h.check_in_day,
                    "check_out_day": h.check_out_day,
                    "nights":        h.check_out_day - h.check_in_day,
                }
                for h in (request.hotels or [])
            ],
            "has_departure_point":    bool(dps),
            "departure_points_count": len(dps),
            "departure_points": [
                {
                    "name":            dp.name,
                    "day":             dp.day,
                    "is_return_point": dp.is_return_point,
                    "lat":             dp.lat,
                    "lng":             dp.lng,
                }
                for dp in dps
            ],
            "daily_start_time":    request.daily_start_time,
            "daily_end_time":      request.daily_end_time,
            "pinned_places_count": len(request.pinned_places or []),
            "pace":        prefs.pace        if prefs else "normal",
            "lunch_time":  prefs.lunch_time  if prefs else "12:00",
            "dinner_time": prefs.dinner_time if prefs else "18:30",
        },
    }


# ─── 순서가 정해진 하루의 시각 재계산 (사용자가 순서·머무는 시간·고정 시각을 손봤을 때) ─────────────

@router.post("/recompute-day")
def recompute(request: RecomputeDayRequest):
    """
    노드 순서는 그대로 두고 시각만 다시 계산한다(이어 붙이기 + 고정 시각·식사 시간대·종료 시간 경고).
    외부 API를 부르지 않는 순수 계산이다.
    """
    for label, value in (("시작 시간", request.start_time), ("종료 시간", request.end_time)):
        if not value or not _TIME_PATTERN.match(value):
            raise HTTPException(status_code=400, detail=f"{label}을 올바르게 입력하세요 (예: 09:00)")
    if _to_minutes(request.start_time) >= _to_minutes(request.end_time):
        raise HTTPException(status_code=400, detail="종료 시간은 시작 시간보다 늦어야 합니다")
    for it in request.items:
        if it.pinned_time and not _TIME_PATTERN.match(it.pinned_time):
            raise HTTPException(status_code=400, detail=f"'{it.name}'의 고정 시각을 올바르게 입력하세요 (예: 12:30)")

    result = recompute_day(
        items=[it.model_dump() for it in request.items],
        start_time=request.start_time,
        end_time=request.end_time,
        transport_mode=request.transportation_mode,
        lunch_time=request.lunch_time,
        dinner_time=request.dinner_time,
    )
    return {"success": True, **result}


# ─── 구간 실시간 대중교통 경로 (사용자가 구간을 눌렀을 때) ────────────────────────

@router.post("/transit")
def transit(request: TransitRequest):
    """
    두 지점 사이의 실제 대중교통 경로(상위 3개, 단계별)를 ODsay에서 실시간으로 조회한다.
    결과는 저장하지 않고(캐시·DB·로그 없음) 그대로 돌려준다. 무료 한도(30건/일) 예산 안에서만 호출한다.
    실패해도(예산 소진, 경로 없음 등) 200으로 {"success": false, "reason", "message"}를 돌려줘서 화면이 안내를 띄운다.
    """
    if not ODSAY_API_KEY:
        return _fail("unavailable")
    return search_routes(
        request.start.lat, request.start.lng,
        request.end.lat, request.end.lng,
        ODSAY_API_KEY, top=3,
    )


# ─── 거리 계산 ─────────────────────────────────────────────────

@router.post("/calculate_distance")
def calculate_distance(request: DistanceRequest):
    try:
        travel = calculate_travel_time(request.place1, request.place2, request.mode)
        return {
            "success":             True,
            "distance_km":         round(haversine_distance(request.place1, request.place2), 2),
            "travel_time_minutes": round(travel["time"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))