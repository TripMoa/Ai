from pydantic import BaseModel, model_validator, field_validator
from typing import Optional, Dict, List

# ─── 카테고리 정규화 맵 ───────────────────────────────────────
# 프론트에서 "관광"으로 넘어오면 "관광지"로 자동 변환
# 이 맵에 없는 값은 원본 유지 (하위 호환)
_CATEGORY_NORMALIZE: dict[str, str] = {
    "관광":  "관광지",
    "관광지": "관광지",
    "맛집":  "맛집",
    "카페":  "카페",
    "쇼핑":  "쇼핑",
    "숙소":  "숙소",
    "교통":  "출발지",   # 프론트 "교통" → 백엔드 "출발지" (공항/기차역 등 이동 거점)
    "출발지": "출발지",  # 이미 변환된 값도 통과
}


class PlaceInput(BaseModel):
    name: str
    lat: float
    lng: float
    category: str = "관광지"
    address: Optional[str] = ""
    is_landmark: bool = False
    is_unique: bool = False

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: str) -> str:
        """프론트 카테고리명("관광")을 백엔드 내부명("관광지")으로 자동 정규화."""
        normalized = _CATEGORY_NORMALIZE.get(str(v).strip(), v)
        return normalized


class PinnedPlace(BaseModel):
    place_index: int
    day: int
    time: Optional[str] = None


class UserPreferences(BaseModel):
    pace: str = "normal"        # tight | normal | relaxed
    lunch_time: str = "12:00"
    dinner_time: str = "18:00"


# ─── 다중 숙소 ────────────────────────────────────────────────

class HotelStay(BaseModel):
    """날짜별 숙소 정보"""
    name: str
    lat: float
    lng: float
    address: Optional[str] = ""
    check_in_day: int
    check_out_day: int

    @model_validator(mode="after")
    def validate_days(self):
        if self.check_out_day <= self.check_in_day:
            raise ValueError(
                f"check_out_day({self.check_out_day})는 "
                f"check_in_day({self.check_in_day})보다 커야 합니다"
            )
        return self


# ─── 다중 출발지 ──────────────────────────────────────────────

class DeparturePoint(BaseModel):
    """일차별 출발지 정보"""
    name: str
    lat: float
    lng: float
    address: Optional[str] = ""
    day: int = 1
    is_return_point: bool = False


class ItineraryRequest(BaseModel):
    places: List[PlaceInput]
    n_days: int = 3
    transportation_mode: str = "대중교통"
    hotels: Optional[List[HotelStay]] = []
    departure_points: Optional[List[DeparturePoint]] = []
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    daily_start_time: str = "09:00"
    daily_end_time: str = "18:00"
    pinned_places: Optional[List[PinnedPlace]] = []
    user_preferences: Optional[UserPreferences] = None

    @model_validator(mode="after")
    def deduplicate_departure_points(self):
        """같은 day의 출발지가 중복으로 들어오면 첫 번째만 유지"""
        if not self.departure_points:
            return self
        seen_days: set[int] = set()
        deduped = []
        for dp in self.departure_points:
            if dp.day not in seen_days:
                deduped.append(dp)
                seen_days.add(dp.day)
        self.departure_points = deduped
        return self


class DistanceRequest(BaseModel):
    place1: Dict
    place2: Dict
    mode: str = "대중교통"