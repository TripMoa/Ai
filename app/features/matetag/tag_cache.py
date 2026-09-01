import httpx
from functools import lru_cache
import os

@lru_cache(maxsize=1)
def get_tag_cache() -> dict[str, set[str]]:
    response = httpx.get(f"{os.getenv('SPRING_BOOT_URL')}/api/mate/tags")
    response.raise_for_status()

    result = {"style": set(), "vibe": set()}
    for tag in response.json():
        category = tag["category"].lower()
        result[category].add(tag["name"])

    return result

tag_cache = get_tag_cache()