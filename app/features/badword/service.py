import json
import httpx
import uuid
from config import BADWORD_API_KEY, BADWORD_API_URL
from features.badword.models import BadwordRequest, BadwordResponse

def check_badword(request: BadwordRequest) -> BadwordResponse:
    try:
        prompt = _build_prompt(request.text)
        raw_content = _call_clova(prompt)
        return _parse_llm_response(raw_content)
    except Exception as e:
        print(f"[badword] LLM 분석 실패: {e}")
        return BadwordResponse(isBlocked=False, reason="")

def _build_prompt(text: str) -> str:
    return f"""
너는 여행 커뮤니티 서비스의 금칙어 필터링 도우미야.
다음 텍스트에 욕설, 비하, 혐오 표현 등 부적절한 내용이 있는지 판단해줘.

규칙:
1. 응답은 반드시 JSON 객체 하나만 반환해.
2. 설명 문장, 마크다운, 코드블록 없이 JSON만 반환해.

입력 텍스트:
{text}

반드시 아래 JSON 형식으로만 반환해.
{{
  "isBlocked": true,
  "reason": "욕설 포함"
}}
""".strip()

def _call_clova(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {BADWORD_API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "messages": [
            {"role": "system", "content": "너는 금칙어 필터링 도우미다. JSON만 반환한다."},
            {"role": "user", "content": prompt},
        ],
        "topP": 0.8,
        "topK": 0,
        "maxTokens": 256,
        "temperature": 0.2,
        "repeatPenalty": 1.1,
    }
    with httpx.Client() as client:
        response = client.post(BADWORD_API_URL, headers=headers, json=body, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data["result"]["message"]["content"]

def _parse_llm_response(raw_content: str) -> BadwordResponse:
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    return BadwordResponse(
        isBlocked=data.get("isBlocked", False),
        reason=data.get("reason", "")
    )