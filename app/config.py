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

print("\n" + "=" * 60)
print("환경변수 로딩 완료")
print(f"   NAVER_CLIENT_ID:         {NAVER_CLIENT_ID[:10] + '...' if NAVER_CLIENT_ID else '없음'}")
print(f"   NAVER_CLIENT_SECRET:     {'있음' if NAVER_CLIENT_SECRET else '없음'}")
print(f"   NAVER_MAP_CLIENT_ID:     {NAVER_MAP_CLIENT_ID[:10] + '...' if NAVER_MAP_CLIENT_ID else '없음'}")
print(f"   NAVER_MAP_CLIENT_SECRET: {'있음' if NAVER_MAP_CLIENT_SECRET else '없음'}")
print("=" * 60 + "\n")