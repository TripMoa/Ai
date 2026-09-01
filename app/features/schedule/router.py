from datetime import datetime
from fastapi import APIRouter, HTTPException

from config import NAVER_CLIENT_ID
from features.schedule.naver_api import geocode_address, local_search
from features.schedule.utils import (
    haversine_distance, calculate_travel_time,
    is_address_query, categorize_place,
)
from features.schedule.models import ItineraryRequest, DistanceRequest
from features.schedule.service import generate_itinerary

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
        _fill_category(result.get("places", []))
        return {
            **_build_response(result.get("places", []), query,
                              "local_search (fallback)", "local_search (무료)"),
            "message": "주소 검색 실패, 일반 검색 결과입니다",
        }

    print("일반 검색 → 지역 검색 API")
    result = await local_search(query, display)
    _fill_category(result.get("places", []))
    return _build_response(
        result.get("places", []), query,
        "local_search", "local_search (무료)",
        total=result.get("total", 0),
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

@router.post("/generate")
async def generate(request: ItineraryRequest):
    """일정 생성 (다중 숙소 & 다중 출발지 지원)"""
    if not request.places:
        raise HTTPException(status_code=400, detail="장소를 추가하세요")

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
            "dinner_time": prefs.dinner_time if prefs else "18:00",
        },
    }


# ─── 거리 계산 ─────────────────────────────────────────────────

@router.post("/calculate_distance")
async def calculate_distance(request: DistanceRequest):
    try:
        travel = calculate_travel_time(request.place1, request.place2, request.mode)
        return {
            "success":             True,
            "distance_km":         round(haversine_distance(request.place1, request.place2), 2),
            "travel_time_minutes": round(travel["time"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))