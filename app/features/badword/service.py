import json
import re
import httpx
import uuid
from config import BADWORD_API_KEY, BADWORD_API_URL
from features.badword.models import BadwordRequest, BadwordResponse


def check_badword(request: BadwordRequest) -> BadwordResponse:
    # 1차: 명백히 의미 없는 반복/나열 패턴은 LLM 호출 없이 바로 통과
    if _is_meaningless_spam(request.text):
        return BadwordResponse(isBlocked=False, reason="의미 없는 반복 패턴")

    # 2차: 애매하거나 문맥 판단이 필요한 텍스트만 LLM에 위임
    try:
        prompt = _build_prompt(request.text)
        raw_content = _call_clova(prompt)
        return _parse_llm_response(raw_content)
    except Exception as e:
        print(f"[badword] LLM 분석 실패: {e}")
        return BadwordResponse(isBlocked=False, reason="")


def _is_meaningless_spam(text: str) -> bool:
    """
    LLM한테 가기 전에 걸러낼 '명백히 무해한' 패턴.
    - 규칙은 최대한 좁고 명확하게 잡는다 (오탐 방지를 위해 실제 욕설 반복 변형까지
      함께 통과시키지 않도록 주의).
    """
    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    # (1) 같은 문자가 5회 이상 연속 반복 (예: ㅋㅋㅋㅋㅋ, ㅎㅎㅎㅎㅎ, ㅠㅠㅠㅠㅠ)
    #     자음/모음/일부 감탄 문자 반복에 한정해 오탐 방지 (임의 문자 전체 대상 X)
    if re.search(r'([ㄱ-ㅎㅏ-ㅣ.!?~])\1{4,}', stripped):
        # 단, 반복 문자를 제외한 나머지가 실제 단어를 포함할 수 있으니
        # 텍스트 전체가 반복+자모/기호로만 구성된 경우에만 안전하게 통과 처리
        pass

    # (2) 텍스트 전체가 자음/모음/공백/일부 기호로만 구성된 경우
    #     (예: "ㄴㄹㅇㄴㄹㅇ", "ㅋㅋㅋㅋㅋㅋ", "ㅁㅊㄴㄴ" 같은 순수 낱자모 나열)
    if re.fullmatch(r'[ㄱ-ㅎㅏ-ㅣ\s]{2,}', stripped):
        return True

    return False


def _build_prompt(text: str) -> str:
    return f"""
너는 여행 커뮤니티 서비스의 금칙어 필터링 도우미야.
다음 텍스트에 실제 욕설, 비하, 혐오 표현이 있는지 판단해줘.

주의사항:
- 의미 없는 키보드 매싱, 자음/모음 나열, 감탄사 반복(ㅋㅋㅋ, ㅎㅎㅎ 등)은 욕설이 아니야.
- 실제 사전에 존재하는 비속어 단어이거나, 명백한 욕설/비하 표현일 때만 true로 판단해.
- 애매하면 false로 판단해 (과잉 차단 금지).
- reason에는 어떤 단어나 표현 때문에 그렇게 판단했는지 구체적으로 적어.

예시:
- "카카카카ㅋㅋㅋㅋㅋ" → isBlocked: false (의미 없는 웃음 표현)
- "ㄴㄹㅇㄴㄹㅇ" → isBlocked: false (의미 없는 자음 나열)
- "씨발 진짜" → isBlocked: true (욕설)
- "개새끼야" → isBlocked: true (욕설)

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
        "temperature": 0.0,  # 판별 작업은 재현성이 중요하므로 0으로 낮춤
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