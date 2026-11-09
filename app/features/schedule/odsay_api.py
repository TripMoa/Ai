"""
odsay_api.py

ODsay 대중교통 길찾기 API 연동 모듈.

- 엔드포인트: GET https://api.odsay.com/v1/api/searchPubTransPathT
- 필수 파라미터: SX(출발경도), SY(출발위도), EX(도착경도), EY(도착위도), apiKey
- 무료(Basic) 플랜은 하루 30건 (https://lab.odsay.com/doc/totalPolicy)

이 모듈은 두 가지 일을 한다.
  1. search_routes(): 사용자가 일정 화면에서 구간을 눌렀을 때 "실시간 조회"해서 경로 후보(상위 N개)와
     단계별 경로를 그대로 돌려준다. (기본 사용처)
  2. get_transit_time(): 일정 생성 중 구간 소요시간 조회. ODSAY_ON_GENERATE=true일 때만 쓰인다(기본 꺼짐).

저장 정책:
  - ODsay 결과 데이터는 저장하지 않는다 — 캐시(파일·메모리)도, DB도 쓰지 않고 응답으로만 돌려준다.
    (약관 제4조 4.5.10: 결과 데이터의 무단 복제·저장·가공·배포 금지, 제7조 7.3: 결과 데이터의 권리는 ODsay에 있음)
  - 하루 몇 번 불렀는지(숫자)만 odsay_usage.json에 기억한다. 응답 내용은 로그에도 남기지 않는다.
  - 화면에는 "powered by www.ODsay.com" 표기가 필요하다 (프론트에서 처리).

호출 제한:
  - 하루 총량: ODSAY_DAILY_LIMIT(무료 Basic은 30건/일)의 90%까지만 쓴다. 다 차면 그날은 호출하지 않는다.
  - 일정 생성 1회당 최대 MAX_CALLS_PER_REQUEST건 (get_transit_time()의 call_counter)
  - 연속 실패가 이어지면 잠시(쿨다운) 호출을 쉰다 (장애·키 오류·한도 초과 때 헛호출 방지)

apiKey는 서버에만 두고 브라우저로 내보내지 않는다(약관 6.3). 특수문자가 있어도 requests의 params=가 자동 인코딩한다.
"""

import json
import logging
import os
import threading
import time
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config as _config
from config import ODSAY_REFERER

logger = logging.getLogger(__name__)

# ─── 설정 ─────────────────────────────────────────────────────

ODSAY_ENDPOINT        = "https://api.odsay.com/v1/api/searchPubTransPathT"
# 하루 한도 — 무료(Basic)는 30건/일. 설정은 config(.env의 ODSAY_DAILY_LIMIT)에서 온다.
# (테스트에서 config를 대체해도 동작하도록 getattr 기본값을 둔다)
ODSAY_DAILY_LIMIT     = int(getattr(_config, "ODSAY_DAILY_LIMIT", 30))
DAILY_BUDGET_RATIO    = 0.9   # 한도의 90%까지만 사용 — 서버 재배포로 카운터가 초기화되는 경우 등에 대비한 여유
DAILY_BUDGET          = int(ODSAY_DAILY_LIMIT * DAILY_BUDGET_RATIO)
MAX_CALLS_PER_REQUEST = int(getattr(_config, "ODSAY_MAX_CALLS_PER_REQUEST", 10))  # 일정 생성 1회당 최대 호출 수
USAGE_FILE            = Path(__file__).parent / "odsay_usage.json"
COOLDOWN_SECONDS      = 600   # 연속 실패로 포기한 뒤 ODsay를 쉬는 시간
# ODsay가 "정상적으로 답했지만 경로가 없다"고 알려주는 코드 — 실패(장애·키 오류)로 세면 안 된다
# (https://lab.odsay.com/guide/guide 길찾기 API: 3 출발 정류장 없음, 4 도착 정류장 없음, 5 둘 다 없음,
#  6 서비스 지역 아님, -99 검색 결과 없음. -98은 700m 이내로 따로 처리)
_NO_ROUTE_CODES       = {"3", "4", "5", "6", "-99"}
_KST                  = timezone(timedelta(hours=9))


# ─── 하루 총량 예산 / 쿨다운 ────────────────────────────────────
# 요청 하나의 상한(CallCounter)과 별개로, 서버 전체가 하루에 부른 횟수를 센다.
# 실패한 시도도 센다(ODsay가 어떻게 집계하는지 알 수 없어 보수적으로).

_usage_lock = threading.Lock()
_usage: dict = {"date": "", "count": 0}
_exhaustion_logged_for = ""
_cooldown_until = 0.0  # time.monotonic() 기준
_streak_lock = threading.Lock()
_streak_failures = 0   # search_routes(사용자 조회)의 연속 실패 수 — 요청 간에 이어진다


def _today() -> str:
    return datetime.now(_KST).strftime("%Y-%m-%d")


def _load_usage() -> None:
    """서버 시작 시 오늘 사용량을 파일에서 복구 (날짜가 다르면 0부터)"""
    global _usage
    try:
        with open(USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == _today():
            _usage = {"date": data["date"], "count": int(data.get("count", 0))}
            return
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("ODsay 사용량 파일 로드 실패 (0부터 시작): %s", e)
    _usage = {"date": _today(), "count": 0}


def _save_usage() -> None:
    try:
        tmp_path = USAGE_FILE.with_name(USAGE_FILE.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_usage, f)
        os.replace(tmp_path, USAGE_FILE)
    except Exception as e:
        logger.warning("ODsay 사용량 파일 저장 실패: %s", e)


_load_usage()


def _reserve_call() -> bool:
    """오늘 예산이 남았으면 1건을 예약(사용 처리)하고 True. 예산이 다 찼으면 False."""
    global _exhaustion_logged_for
    with _usage_lock:
        today = _today()
        if _usage["date"] != today:  # 날짜(한국 시간)가 바뀌면 0부터
            _usage["date"], _usage["count"] = today, 0
        if _usage["count"] >= DAILY_BUDGET:
            if _exhaustion_logged_for != today:
                logger.warning(
                    "ODsay 오늘 예산 소진 (%d/%d건, 하루 한도 %d건) → 오늘은 호출하지 않음",
                    _usage["count"], DAILY_BUDGET, ODSAY_DAILY_LIMIT,
                )
                _exhaustion_logged_for = today
            return False
        _usage["count"] += 1
        _save_usage()
        return True


def usage_stats() -> dict:
    """오늘 ODsay 사용량 (디버깅·로그용)"""
    with _usage_lock:
        return {
            "date":          _usage["date"],
            "used":          _usage["count"],
            "budget":        DAILY_BUDGET,
            "daily_limit":   ODSAY_DAILY_LIMIT,
            "cooldown_left": max(0, int(_cooldown_until - time.monotonic())),
        }


def _start_cooldown() -> None:
    global _cooldown_until
    _cooldown_until = time.monotonic() + COOLDOWN_SECONDS
    logger.warning("ODsay 연속 실패 → %d초 동안 호출을 쉬어요", COOLDOWN_SECONDS)


def _note_user_lookup(ok: bool) -> None:
    """search_routes의 연속 실패를 요청 간에 이어서 세고, 3번 이어지면 쿨다운을 건다"""
    global _streak_failures
    with _streak_lock:
        if ok:
            _streak_failures = 0
            return
        _streak_failures += 1
        if _streak_failures >= CallCounter.MAX_CONSECUTIVE_FAILURES:
            _streak_failures = 0
            _start_cooldown()


def _record_failure(call_counter: "CallCounter | None") -> None:
    """실패를 요청 카운터에 기록하고, 연속 실패 한도에 닿으면 서버 전체 쿨다운을 건다"""
    if call_counter is None:
        return
    was_giving_up = call_counter.exceeded()
    call_counter.record_failure()
    if call_counter.consecutive_failures >= call_counter.MAX_CONSECUTIVE_FAILURES and not was_giving_up:
        _start_cooldown()


# ─── 호출 카운터 ───────────────────────────────────────────────

class CallCounter:
    """
    일정 생성 1회 범위 내 ODsay API 호출 수를 추적.

    - count: 호출 "시도" 수 (타임아웃·연결 실패도 포함 — 응답을 받은 것만 세면 ODsay가 죽었을 때
      상한에 걸리지 않고 구간마다 타임아웃을 기다리게 된다)
    - consecutive_failures: 연속 실패 수. MAX_CONSECUTIVE_FAILURES에 닿으면 이번 요청은 ODsay를
      포기하고 나머지는 직선거리 추정으로 진행한다 (ODsay 장애·키 오류일 때 요청이 길어지는 것 방지)
    """
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, limit: int = MAX_CALLS_PER_REQUEST):
        self.count = 0
        self.limit = limit
        self.consecutive_failures = 0

    def exceeded(self) -> bool:
        return self.count >= self.limit or self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES

    def increment(self) -> None:
        self.count += 1

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0


# ─── 일정 생성용: 구간 소요시간 (ODSAY_ON_GENERATE=true일 때만) ──────────

def get_transit_time(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    api_key: str,
    call_counter: CallCounter | None = None,
) -> dict:
    """
    두 지점 간 대중교통 소요시간을 ODsay API로 조회. 결과는 저장하지 않는다.

    Returns:
        {"success": bool, "time": int(분), "payment": int(원), "transfer": int(환승), "error": str}
        실패·예산 소진 시 {"success": False, "error": ...} — 호출자가 하버사인 추정으로 폴백한다.
    """
    # ── 1. 호출 상한 체크 (쿨다운 / 요청 상한·연속 실패 / 하루 총량) ──
    if time.monotonic() < _cooldown_until:
        logger.debug("ODsay 쿨다운 중 → 하버사인 폴백")
        return {"success": False, "error": "limit_exceeded"}
    if call_counter is not None and call_counter.exceeded():
        logger.info(
            "ODsay 호출 중단 (시도 %d/%d건, 연속 실패 %d회) → 하버사인 폴백",
            call_counter.count, call_counter.limit, call_counter.consecutive_failures,
        )
        return {"success": False, "error": "limit_exceeded"}
    if not _reserve_call():  # 하루 총량 예산 (calculate_distance처럼 카운터 없이 부르는 경우도 포함)
        return {"success": False, "error": "limit_exceeded"}

    # ── 2. ODsay API 호출 ──
    # 시도 자체를 센다 — 타임아웃·연결 오류로 예외가 나도 카운트되도록 요청 전에 올린다
    if call_counter is not None:
        call_counter.increment()
    try:
        res = requests.get(
            ODSAY_ENDPOINT,
            params={"SX": start_lng, "SY": start_lat, "EX": end_lng, "EY": end_lat, "apiKey": api_key},
            headers={
                # URI 플랫폼 인증: ODsay가 Referer 헤더로 등록된 URI와 대조함
                # ODsay 콘솔에 등록한 URI와 일치해야 함 — .env의 ODSAY_REFERER로 설정
                "Referer": ODSAY_REFERER,
            },
            timeout=5,
        )

        if res.status_code != 200:
            logger.warning("ODsay HTTP 오류: %s", res.status_code)
            _record_failure(call_counter)
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
                # 700m 이내: 도보 거리이므로 debug 레벨로만 기록 (정상 응답이라 실패로 세지 않음)
                logger.debug("ODsay -98 (700m 이내, 도보 거리): %s", err_msg)
                if call_counter is not None:
                    call_counter.record_success()
                return {"success": False, "error": "too_close", "code": err_code}

            if err_code in _NO_ROUTE_CODES:
                # 정류장이 없거나 서비스 지역이 아닌 구간 — 정상 응답이므로 연속 실패로 세지 않는다
                logger.info("ODsay 경로 없음 [code=%s]: %s", err_code, err_msg)
                if call_counter is not None:
                    call_counter.record_success()
                return {"success": False, "error": err_msg, "code": err_code}

            logger.warning("ODsay API 오류 [code=%s]: %s", err_code, err_msg)
            _record_failure(call_counter)  # 키·Referer 오류·한도 초과·입력 오류(-8/-9)처럼 계속 실패할 가능성이 큰 오류
            return {"success": False, "error": err_msg, "code": err_code}

        path_list = data.get("result", {}).get("path", [])
        if not path_list:
            if call_counter is not None:
                call_counter.record_success()  # 정상 응답(경로만 없음)이라 실패로 세지 않음
            return {"success": False, "error": "경로 없음"}

        info = path_list[0].get("info", {})
        if call_counter is not None:
            call_counter.record_success()
        return {
            "success":  True,
            "time":     int(info.get("totalTime", 0)),
            "payment":  int(info.get("payment", 0)),
            # 응답에는 transferCount가 없고 버스·지하철 탑승 횟수만 온다 → 환승 = 탑승 횟수 - 1
            "transfer": max(0, int(info.get("busTransitCount", 0)) + int(info.get("subwayTransitCount", 0)) - 1),
        }

    except requests.Timeout:
        logger.warning("ODsay 타임아웃")
        _record_failure(call_counter)
        return {"success": False, "error": "timeout"}
    except Exception as e:
        logger.warning("ODsay 예외: %s", e)
        _record_failure(call_counter)
        return {"success": False, "error": str(e)}


# ─── 사용자 조회용: 경로 후보 + 단계별 경로 (실시간, 저장 안 함) ─────────────

_TRAFFIC_TYPE = {1: "subway", 2: "bus", 3: "walk"}

_USER_MESSAGES = {
    "budget":      "오늘 실시간 조회 한도를 다 썼어요. 내일 다시 시도하거나 지도 앱에서 확인해주세요.",
    "cooldown":    "실시간 조회가 잠시 원활하지 않아요. 잠시 후 다시 시도하거나 지도 앱에서 확인해주세요.",
    "too_close":   "두 장소가 가까워서(700m 이내) 걸어서 이동하는 걸 추천해요.",
    "no_route":    "이 구간은 대중교통 경로를 찾지 못했어요.",
    "error":       "실시간 경로를 가져오지 못했어요. 잠시 후 다시 시도해주세요.",
    "unavailable": "실시간 경로 조회를 사용할 수 없어요.",
}


def _fail(reason: str) -> dict:
    return {"success": False, "reason": reason, "message": _USER_MESSAGES[reason]}


def _parse_step(sp: dict) -> dict | None:
    kind = _TRAFFIC_TYPE.get(sp.get("trafficType"))
    if kind is None:
        return None
    minutes = int(sp.get("sectionTime", 0) or 0)
    distance = int(sp.get("distance", 0) or 0)
    step: dict = {"type": kind, "minutes": minutes, "distance_m": distance}
    if kind == "walk":
        return step if (minutes > 0 or distance > 0) else None  # 0분·0m 환승 도보는 생략

    lanes = sp.get("lane") or [{}]
    names = [str(l.get("name") or l.get("busNo") or "").strip() for l in lanes]
    step["line"] = " / ".join(n for n in dict.fromkeys(names) if n)
    step["start"] = sp.get("startName", "")
    step["end"] = sp.get("endName", "")
    step["stations"] = int(sp.get("stationCount", 0) or 0)
    interval = sp.get("intervalTime")
    if interval:
        step["interval_min"] = int(interval)
    if kind == "subway":
        if sp.get("way"):
            step["direction"] = sp["way"]
        if sp.get("door"):
            step["door"] = sp["door"]
        if sp.get("startExitNo"):
            step["exit_no"] = str(sp["startExitNo"])
    return step


def search_routes(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    api_key: str,
    top: int = 3,
) -> dict:
    """
    사용자가 구간을 눌렀을 때 실시간으로 조회한다. 소요 시간이 짧은 상위 top개 경로와 단계별 경로를 돌려준다.

    결과는 저장하지 않고(캐시·DB·로그 없음) 호출자에게만 돌려준다. 하루 예산·쿨다운 규칙은 get_transit_time과 같다.

    Returns:
        성공: {"success": True, "search_type": "intra"|"intercity", "routes": [ {...}, ... ]}
        실패: {"success": False, "reason": "budget"|"cooldown"|"too_close"|"no_route"|"error", "message": str}
    """
    if time.monotonic() < _cooldown_until:
        return _fail("cooldown")
    if not _reserve_call():
        return _fail("budget")

    try:
        res = requests.get(
            ODSAY_ENDPOINT,
            params={"SX": start_lng, "SY": start_lat, "EX": end_lng, "EY": end_lat, "apiKey": api_key},
            headers={"Referer": ODSAY_REFERER},
            timeout=8,
        )
    except Exception as e:  # 타임아웃·연결 오류
        logger.warning("ODsay 조회 실패: %s", type(e).__name__)
        _note_user_lookup(False)
        return _fail("error")

    if res.status_code != 200:
        logger.warning("ODsay HTTP 오류: %s", res.status_code)
        _note_user_lookup(False)
        return _fail("error")

    try:
        data = res.json()
    except ValueError:
        _note_user_lookup(False)
        return _fail("error")

    if "error" in data:
        err = data["error"]
        if isinstance(err, list):
            err = err[0] if err else {}
        err_code = str(err.get("code", "?"))
        if err_code == "-98":
            _note_user_lookup(True)
            return _fail("too_close")
        if err_code in _NO_ROUTE_CODES:
            _note_user_lookup(True)
            return _fail("no_route")
        logger.warning("ODsay API 오류 [code=%s]", err_code)  # 사유 문구는 남기지 않는다(코드만)
        _note_user_lookup(False)
        return _fail("error")

    result = data.get("result") or {}
    paths = result.get("path") or []
    if not paths:
        _note_user_lookup(True)
        return _fail("no_route")

    _note_user_lookup(True)
    paths = sorted(paths, key=lambda p: int((p.get("info") or {}).get("totalTime", 10**9)))[: max(1, top)]

    routes = []
    for p in paths:
        info = p.get("info") or {}
        rides = int(info.get("busTransitCount", 0) or 0) + int(info.get("subwayTransitCount", 0) or 0)
        steps = [s for s in (_parse_step(sp) for sp in (p.get("subPath") or [])) if s]
        routes.append({
            "path_type":     p.get("pathType"),          # 1 지하철, 2 버스, 3 버스+지하철
            "total_time":    int(info.get("totalTime", 0) or 0),
            "payment":       int(info.get("payment", 0) or 0),
            "transfers":     max(0, rides - 1),           # 환승 = 탑승 횟수 - 1
            # info.totalWalkTime은 -1(값 없음)로 오는 경우가 있어 단계별 도보 시간을 직접 합산한다
            "walk_minutes":  sum(st["minutes"] for st in steps if st["type"] == "walk"),
            "walk_meters":   max(0, int(info.get("totalWalk", 0) or 0)),
            "station_count": int(info.get("totalStationCount", 0) or 0),
            "first_station": info.get("firstStartStation", ""),
            "last_station":  info.get("lastEndStation", ""),
            "steps":         steps,
        })

    # searchType: 0 도시 내, 그 외는 도시 간(도시 간은 터미널 사이 구간만 제공 — 안내 문구를 화면에서 붙인다)
    return {
        "success":     True,
        "search_type": "intra" if int(result.get("searchType", 0) or 0) == 0 else "intercity",
        "routes":      routes,
    }
