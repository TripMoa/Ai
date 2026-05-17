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

CLOVA_API_KEY: str | None = os.getenv("CLOVA_API_KEY")
CLOVA_API_URL: str | None = os.getenv("CLOVA_API_URL")