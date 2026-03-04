import re
import requests
from config import (
    NAVER_CLIENT_ID, NAVER_CLIENT_SECRET,
    NAVER_MAP_CLIENT_ID, NAVER_MAP_CLIENT_SECRET,
)
from features.schedule.utils import convert_naver_coords_to_wgs84


def _strip_html(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


async def geocode_address(address: str) -> dict:
    """주소 → 좌표 (Geocoding API, 유료)"""
    if not NAVER_MAP_CLIENT_ID or not NAVER_MAP_CLIENT_SECRET:
        return {"success": False, "places": [], "error": "Geocoding API key not configured"}

    print(f"Geocoding API: '{address}'")
    try:
        res = requests.get(
            "https://maps.apigw.ntruss.com/map-geocode/v2/geocode",
            headers={
                "x-ncp-apigw-api-key-id": NAVER_MAP_CLIENT_ID,
                "x-ncp-apigw-api-key": NAVER_MAP_CLIENT_SECRET,
                "Accept": "application/json",
            },
            params={"query": address},
            timeout=5,
        )
        if res.status_code != 200:
            return {"success": False, "places": [], "error": res.text}

        data = res.json()
        if data.get("status") != "OK":
            return {"success": False, "places": [], "error": data.get("errorMessage", "Unknown")}

        addresses = data.get("addresses", [])
        if not addresses:
            return {"success": False, "places": [], "error": "No results"}

        places = [
            {
                "name": address,
                "address": addr.get("roadAddress") or addr.get("jibunAddress", ""),
                "lat": float(addr["y"]),
                "lng": float(addr["x"]),
                "category": "주소",
                "link": "", "description": "", "telephone": "",
                "naver_category": "주소",
            }
            for addr in addresses[:3]
        ]
        print(f"Geocoding 성공: {len(places)}개")
        return {"success": True, "places": places, "method": "geocoding"}

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"success": False, "places": [], "error": str(e)}


async def local_search(query: str, display: int = 10) -> dict:
    """상호명/키워드 검색 (지역 검색 API, 무료)"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {"success": False, "places": [], "error": "Search API key not configured"}

    print(f"지역 검색 API: '{query}'")
    try:
        res = requests.get(
            "https://openapi.naver.com/v1/search/local.json",
            headers={
                "X-Naver-Client-Id": NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
            },
            params={"query": query, "display": display, "sort": "random"},
            timeout=5,
        )
        if res.status_code != 200:
            return {"success": False, "places": [], "error": res.text}

        data = res.json()
        items = data.get("items", [])
        if not items:
            return {"success": True, "places": [], "method": "local_search",
                    "message": f"'{query}' 검색 결과가 없습니다."}

        places = []
        for item in items:
            try:
                lat, lng = convert_naver_coords_to_wgs84(
                    item.get("mapx", "0"), item.get("mapy", "0")
                )
                if lat is None:
                    continue
                naver_cat = item.get("category", "")
                places.append({
                    "name": _strip_html(item["title"]),
                    "address": item.get("roadAddress") or item.get("address", ""),
                    "lat": lat, "lng": lng,
                    "naver_category": naver_cat,
                    "link": item.get("link", ""),
                    "description": _strip_html(item.get("description", "")),
                    "telephone": item.get("telephone", ""),
                })
            except Exception as e:
                print(f"항목 처리 오류: {e}")

        print(f"지역 검색 성공: {len(places)}개")
        return {"success": True, "places": places, "method": "local_search",
                "total": data.get("total", 0)}

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"success": False, "places": [], "error": str(e)}