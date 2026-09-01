"""
odsay_api.py

ODsay 대중교통 길찾기 API 연동 모듈.

- 엔드포인트: GET https://api.odsay.com/v1/api/searchPubTransPathT
- 필수 파라미터: SX(출발경도), SY(출발위도), EX(도착경도), EY(도착위도), apiKey
- 응답에서 추출: totalTime(분), payment(원), transferCount(환승 횟수)

캐시 전략:
  - 파일 캐시 (odsay_cache.json): 서버 재시작 후에도 유지
  - 인메모리 캐시: 파일 캐시 위에 올라가는 1차 캐시 (파일 I/O 최소화)
  - 캐시 히트 시 ODsay API 호출 없음 → 1,000건/일 한도 보호

호출 상한:
  - get_transit_time()에 call_counter 파라미터 전달
  - 1회 일정 생성당 최대 MAX_CALLS_PER_REQUEST(20)건 초과 시 하버사인 폴백

apiKey에 '+' 등 특수문자가 포함된 경우 반드시 URL 인코딩 필요.
    requests 라이브러리의 params= 방식은 자동 인코딩하므로 안전합니다.
"""

import json
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── 설정 ─────────────────────────────────────────────────────

ODSAY_ENDPOINT        = "https://api.odsay.com/v1/api/searchPubTransPathT"
CACHE_FILE            = Path(__file__).parent / "odsay_cache.json"
MAX_CALLS_PER_REQUEST = 20   # 일정 생성 1회당 ODsay API 최대 호출 수


# ─── 캐시 초기화 ───────────────────────────────────────────────
# 인메모리 캐시: { "sx_sy_ex_ey": {"time": int, "payment": int, "transfer": int} }
_mem_cache: dict[str, dict] = {}


def _load_file_cache() -> None:
    """서버 시작 시 파일 캐시를 인메모리로 로드"""
    global _mem_cache
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _mem_cache = json.load(f)
            logger.info("ODsay 파일 캐시 로드: %d건", len(_mem_cache))
        except Exception as e:
            logger.warning("ODsay 캐시 파일 로드 실패 (초기화): %s", e)
            _mem_cache = {}
    else:
        _mem_cache = {}


def _save_file_cache() -> None:
    """인메모리 캐시를 파일에 저장"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_mem_cache, f, ensure_ascii=False)
    except Exception as e:
        logger.warning("ODsay 캐시 파일 저장 실패: %s", e)


# 모듈 로드 시 파일 캐시 자동 로드
_load_file_cache()


# ─── 캐시 키 ──────────────────────────────────────────────────

def _cache_key(sx: float, sy: float, ex: float, ey: float) -> str:
    """소수점 5자리로 반올림해 캐시 키 생성 (약 1m 정밀도)"""
    return f"{round(sx,5)}_{round(sy,5)}_{round(ex,5)}_{round(ey,5)}"


# ─── 호출 카운터 ───────────────────────────────────────────────

class CallCounter:
    """
    일정 생성 1회 범위 내 ODsay API 호출 수를 추적.

    사용법:
        counter = CallCounter()
        result = get_transit_time(..., call_counter=counter)
        # counter.count 로 현재까지 실제 API 호출 수 확인
    """
    def __init__(self, limit: int = MAX_CALLS_PER_REQUEST):
        self.count = 0
        self.limit = limit

    def exceeded(self) -> bool:
        return self.count >= self.limit

    def increment(self) -> None:
        self.count += 1


# ─── 캐시 전용 조회 (API 호출 없음) ─────────────────────────────

def get_cached_transit_time(start_lat: float, start_lng: float, end_lat: float, end_lng: float) -> dict | None:
    """
    캐시에 이미 있는 실측값만 반환 (네트워크 호출 전혀 안 함).

    경로 정렬(2-opt 등)에서 하버사인 직선거리 대신 "공짜로 이미 아는" 실측값을
    우선 활용하기 위한 용도 — 이전에 같은 구간을 조회한 적이 있으면 그 값을 쓴다.
    """
    key = _cache_key(start_lng, start_lat, end_lng, end_lat)
    return _mem_cache.get(key)


# ─── 메인 함수 ────────────────────────────────────────────────

def get_transit_time(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    api_key: str,
    call_counter: CallCounter | None = None,
) -> dict:
    """
    두 지점 간 대중교통 소요시간을 ODsay API로 조회.

    call_counter가 전달된 경우:
      - 상한(MAX_CALLS_PER_REQUEST) 초과 시 {"success": False, "error": "limit_exceeded"} 반환
      - 캐시 히트는 카운터 소모 없음 → 캐시된 구간은 상한과 무관하게 실측값 사용

    Returns:
        {
            "success":  bool,
            "time":     int,   # 소요시간 (분)
            "payment":  int,   # 요금 (원)
            "transfer": int,   # 환승 횟수
            "source":   str,   # "mem_cache" | "api"
            "error":    str,   # 실패 시 오류 메시지
        }
    """
    key = _cache_key(start_lng, start_lat, end_lng, end_lat)

    # ── 1. 인메모리 캐시 히트 (카운터 소모 없음) ──
    if key in _mem_cache:
        logger.debug("ODsay 캐시 히트: %s", key)
        return {**_mem_cache[key], "source": "mem_cache", "success": True}

    # ── 2. 호출 상한 체크 ──
    if call_counter is not None and call_counter.exceeded():
        logger.info(
            "ODsay 호출 상한 초과 (%d/%d건) → 하버사인 폴백",
            call_counter.count, call_counter.limit,
        )
        return {"success": False, "error": "limit_exceeded"}

    # ── 3. ODsay API 호출 ──
    try:
        res = requests.get(
            ODSAY_ENDPOINT,
            params={
                "SX":     start_lng,
                "SY":     start_lat,
                "EX":     end_lng,
                "EY":     end_lat,
                "apiKey": api_key,
            },
            headers={
                # URI 플랫폼 인증: ODsay가 Referer 헤더로 등록된 URI와 대조함
                # ODsay 콘솔에 등록한 URI(127.0.0.1:8000 / localhost:8000)와 일치해야 함
                "Referer": "http://localhost:8000",
            },
            timeout=5,
        )

        if call_counter is not None:
            call_counter.increment()

        if res.status_code != 200:
            logger.warning("ODsay HTTP 오류: %s", res.status_code)
            return {"success": False, "error": f"HTTP {res.status_code}"}

        data = res.json()

        if "error" in data:
            err = data["error"]
            # ODsay error 필드는 dict 또는 list 두 가지 형태로 올 수 있음
            if isinstance(err, list):
                err = err[0] if err else {}
            # ODsay는 'msg' 키를 사용 ('message' 아님)
            err_msg  = err.get("msg") or err.get("message", "Unknown ODsay error")
            err_code = str(err.get("code", "?"))

            if err_code == "-98":
                # 700m 이내: 도보 거리이므로 debug 레벨로만 기록
                logger.debug("ODsay -98 (700m 이내, 도보 거리): %s", err_msg)
                return {"success": False, "error": "too_close", "code": err_code}

            logger.warning("ODsay API 오류 [code=%s]: %s", err_code, err_msg)
            return {"success": False, "error": err_msg, "code": err_code}

        result_info = data.get("result", {})
        path_list   = result_info.get("path", [])

        if not path_list:
            logger.info("ODsay 경로 없음: (%s,%s)→(%s,%s)", start_lat, start_lng, end_lat, end_lng)
            return {"success": False, "error": "경로 없음"}

        best = path_list[0]
        info = best.get("info", {})

        total_time = int(info.get("totalTime",     0))
        payment    = int(info.get("payment",       0))
        transfer   = int(info.get("transferCount", 0))

        cached_value = {
            "time":     total_time,
            "payment":  payment,
            "transfer": transfer,
        }

        # ── 4. 캐시 저장 (인메모리 + 파일) ──
        _mem_cache[key] = cached_value
        _save_file_cache()

        print(
            f"ODsay 조회 성공: {total_time}분 / {payment}원 / 환승 {transfer}회 | "
            f"캐시 {len(_mem_cache)}건 | 이번 요청 API 호출 {call_counter.count if call_counter else -1}건"
        )
        return {**cached_value, "source": "api", "success": True}

    except requests.Timeout:
        logger.warning("ODsay 타임아웃")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        logger.warning("ODsay 예외: %s", e)
        return {"success": False, "error": str(e)}


# ─── 유틸 ─────────────────────────────────────────────────────

def cache_stats() -> dict:
    """현재 캐시 상태 반환 (디버깅용)"""
    return {
        "mem_cache_count": len(_mem_cache),
        "cache_file":      str(CACHE_FILE),
        "file_exists":     CACHE_FILE.exists(),
    }