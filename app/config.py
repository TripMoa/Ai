import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

BASE_DIR = Path(__file__).parent

dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)

NAVER_CLIENT_ID: str | None = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET: str | None = os.getenv("NAVER_CLIENT_SECRET")
NAVER_MAP_CLIENT_ID: str | None = os.getenv("NAVER_MAP_CLIENT_ID")
NAVER_MAP_CLIENT_SECRET: str | None = os.getenv("NAVER_MAP_CLIENT_SECRET")
ODSAY_API_KEY: str | None = os.getenv("ODSAY_API_KEY")

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# 일정 생성 중에도 ODsay로 구간 소요시간을 조회할지 (기본 꺼짐).
# 무료(Basic)는 하루 30건이라 생성마다 부를 수 없고, 결과를 저장(캐시)할 수도 없다.
# 생성은 직선거리 기반 추정으로 하고, ODsay는 사용자가 구간을 눌렀을 때만 실시간 조회한다.
# 유료 플랜 등으로 여유가 생기면 .env에서 true로 켠다.
ODSAY_ON_GENERATE: bool = os.getenv("ODSAY_ON_GENERATE", "false").strip().lower() in ("1", "true", "yes")

# ODsay 하루 호출 한도 — 무료(Basic)는 30건/일 (https://lab.odsay.com/doc/totalPolicy).
# 유료 플랜이면 ODsay 콘솔의 "제한 호출수"에 맞춰 .env에서 올린다.
ODSAY_DAILY_LIMIT: int = _int_env("ODSAY_DAILY_LIMIT", 30)
# 일정 생성 1회에 쓸 수 있는 ODsay 호출 상한 (하루 한도가 남아 있어도 한 요청이 다 쓰지 않게)
ODSAY_MAX_CALLS_PER_REQUEST: int = _int_env("ODSAY_MAX_CALLS_PER_REQUEST", 10)

# ODsay 콘솔에 등록한 URI와 일치해야 하는 Referer (배포 도메인에 맞게 .env에서 설정)
ODSAY_REFERER: str = os.getenv("ODSAY_REFERER", "http://localhost:8000")

# CORS 허용 origin (콤마로 구분). 미설정 시 로컬 개발용 기본값만 허용
CORS_ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    if origin.strip()
]

CLOVA_API_KEY: str | None = os.getenv("CLOVA_API_KEY")
CLOVA_API_URL: str | None = os.getenv("CLOVA_API_URL")

BADWORD_API_KEY: str | None = os.getenv("BADWORD_API_KEY")
BADWORD_API_URL: str | None = os.getenv("BADWORD_API_URL")
