from pydantic import BaseModel
from typing import Optional, Dict, List


class PlaceInput(BaseModel):
    name: str
    lat: float
    lng: float
    category: str = "관광지"
    address: Optional[str] = ""
    open_time: Optional[str] = None
    close_time: Optional[str] = None
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


class ItineraryRequest(BaseModel):
    places: List[PlaceInput]
    n_days: int = 3
    transportation_mode: str = "대중교통"
    hotel: Optional[Dict] = None
    departure_point: Optional[Dict] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    daily_start_time: str = "09:00"
    daily_end_time: str = "18:00"
    pinned_places: Optional[List[PinnedPlace]] = []
    user_preferences: Optional[UserPreferences] = None


class DistanceRequest(BaseModel):
    place1: Dict
    place2: Dict
    mode: str = "대중교통"