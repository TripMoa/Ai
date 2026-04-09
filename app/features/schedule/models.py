from pydantic import BaseModel, model_validator
from typing import Optional, Dict, List


class PlaceInput(BaseModel):
    name: str
    lat: float
    lng: float
    category: str = "관광지"
    address: Optional[str] = ""
    is_landmark: bool = False
    is_unique: bool = False


class PinnedPlace(BaseModel):
    place_index: int
    day: int
    time: Optional[str] = None
    priority: str = "must"  # must | high | medium


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
    check_in_day: int   # 1-based: 몇 일차에 체크인
    check_out_day: int  # 체크아웃 일차 (이 날 아침에 이 숙소에서 출발)

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
    day: int = 1  # 이 출발지를 사용할 일차 (1-based)
    is_return_point: bool = False  # True면 마지막 날 복귀 기준점으로도 사용


class ItineraryRequest(BaseModel):
    places: List[PlaceInput]
    n_days: int = 3
    transportation_mode: str = "대중교통"

    # ──────────── 숙소: 하위 호환 유지 ───────────────────
    hotel: Optional[Dict] = None           # 기존: 단일 숙소 (deprecated)
    hotels: Optional[List[HotelStay]] = [] # 다중 숙소

    # ──────────── 출발지: 하위 호환 유지 ─────────────────
    departure_point: Optional[Dict] = None                  # 기존: 단일 출발지 (deprecated)
    departure_points: Optional[List[DeparturePoint]] = []   # 다중 출발지

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    daily_start_time: str = "09:00"
    daily_end_time: str = "18:00"
    pinned_places: Optional[List[PinnedPlace]] = []
    user_preferences: Optional[UserPreferences] = None

    @model_validator(mode="after")
    def migrate_single_hotel(self):
        if self.hotel and not self.hotels:
            self.hotels = [
                HotelStay(
                    name=self.hotel.get("name", "숙소"),
                    lat=self.hotel["lat"],
                    lng=self.hotel["lng"],
                    address=self.hotel.get("address", ""),
                    check_in_day=1,
                    check_out_day=self.n_days,
                )
            ]
        return self

    @model_validator(mode="after")
    def migrate_single_departure(self):
        if self.departure_point and not self.departure_points:
            self.departure_points = [
                DeparturePoint(
                    name=self.departure_point.get("name", "출발지"),
                    lat=self.departure_point["lat"],
                    lng=self.departure_point["lng"],
                    address=self.departure_point.get("address", ""),
                    day=1,
                    is_return_point=True,
                )
            ]
        return self

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