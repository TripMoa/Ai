"""
service.py

OCR 후처리 데이터를 네이버 CLOVA LLM으로 자동 분류

흐름:
  1. analyze_receipt
     - OCR 데이터 분석 진입점
     - 정보 부족 시 fallback 반환
     - LLM 호출 및 응답 처리

  2. _is_insufficient_ocr_data
     - 상호명/메뉴명 존재 여부 확인

  3. _build_prompt
     - OCR 데이터를 LLM 프롬프트로 변환

  4. _call_clova
     - 네이버 CLOVA API 호출

  5. _parse_llm_response
     - LLM JSON 응답 파싱 및 검증

  6. _fallback
     - LLM 실패 시 기본 응답 반환

  7. _fallback_item_memo
     - 기본 itemMemo 생성
"""

import json
import requests
import uuid

from config import CLOVA_API_KEY, CLOVA_API_URL
from features.ocr.models import OcrAnalyzeRequest, OcrAnalyzeResponse


VALID_CATEGORIES = {"FOOD", "TRANS", "STAY", "SHOP", "TICKET", "ETC"}


def analyze_receipt(request: OcrAnalyzeRequest) -> OcrAnalyzeResponse:
    """
    OCR 자동채움 LLM 분석 진입점
    - 핵심 정보가 없으면 LLM 호출 없이 fallback
    - 실패해도 Spring 흐름이 끊기지 않도록 fallback 반환
    """
    try:
        if _is_insufficient_ocr_data(request):
            return _fallback(request)

        prompt = _build_prompt(request)
        raw_content = _call_clova(prompt)
        return _parse_llm_response(raw_content, request)

    except Exception as e:
        print(f"[ocr] 네이버 LLM 분석 실패: {e}")
        return _fallback(request)
    
    
def _is_insufficient_ocr_data(request: OcrAnalyzeRequest) -> bool:
    """
    상호명/메뉴명이 없으면 LLM이 추측할 가능성이 높으므로
    LLM 호출하지 않고 fallback 처리
    """
    has_store = bool(request.storeName and request.storeName.strip())
    has_menu = bool(request.menuName and request.menuName.strip())

    return not has_store and not has_menu


def _build_prompt(request: OcrAnalyzeRequest) -> str:
    """
    OCR 후처리 데이터를 LLM이 이해할 수 있는 프롬프트로 변환
    """
    return f"""
너는 여행 경비 관리 서비스의 영수증 자동 분류 도우미야.

다음 OCR 후처리 데이터를 보고 사용자가 저장할 지출 항목명(itemMemo)과 카테고리(category)를 정해줘.

카테고리는 반드시 아래 enum 중 하나만 사용해.

- FOOD: 식비, 음식점, 카페, 음료, 간식
- TRANS: 교통, 택시, 버스, 지하철, 기차, 주유, 주차
- STAY: 숙소, 호텔, 게스트하우스, 펜션
- SHOP: 쇼핑, 편의점, 마트, 기념품, 물품 구매
- TICKET: 티켓, 관광지 입장권, 전시, 공연, 액티비티
- ETC: 기타, 분류가 애매한 지출

규칙:
1. 응답은 반드시 JSON 객체 하나만 반환해.
2. category는 반드시 FOOD, TRANS, STAY, SHOP, TICKET, ETC 중 하나여야 해.
3. itemMemo는 사용자가 이해하기 쉬운 한국어로 짧게 만들어.
4. 결제방식, 날짜, 금액은 category 판단 참고용으로만 사용해.
5. 확실하지 않으면 ETC로 반환해.
6. 설명 문장, 마크다운, 코드블록 없이 JSON만 반환해.

입력 데이터:
상호명: {request.storeName or ""}
메뉴명: {request.menuName or ""}
결제방식: {request.paymentMethod or ""}
결제날짜: {request.dateTime or ""}
총금액: {request.totalAmount if request.totalAmount is not None else ""}

반드시 아래 JSON 형식으로만 반환해.
itemMemo에는 입력 데이터에 없는 예시 문구를 만들지 마.

{{
  "itemMemo": "...",
  "category": "ETC"
}}
""".strip()


def _call_clova(prompt: str) -> str:
    if not CLOVA_API_KEY or not CLOVA_API_URL:
        raise RuntimeError("CLOVA 설정이 없습니다.")

    headers = {
        "Authorization": f"Bearer {CLOVA_API_KEY}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = {
        "messages": [
            {
                "role": "system",
                "content": "너는 영수증 OCR 데이터를 여행 경비 항목으로 분류하는 도우미다. JSON만 반환한다.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "topP": 0.8,
        "topK": 0,
        "maxTokens": 256,
        "temperature": 0.2,
        "repeatPenalty": 1.1,
    }

    response = requests.post(
        CLOVA_API_URL,
        headers=headers,
        json=body,
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    return data["result"]["message"]["content"]


def _parse_llm_response(
    raw_content: str,
    request: OcrAnalyzeRequest,
) -> OcrAnalyzeResponse:
    """
    LLM 응답 문자열을 JSON으로 파싱하고,
    category가 허용된 enum인지 검증
    """
    cleaned = raw_content.strip()

    # 혹시 LLM이 ```json 코드블록으로 감싸서 주는 경우 제거
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    data = json.loads(cleaned)

    raw_item_memo = (data.get("itemMemo") or "").strip()
    raw_category = (data.get("category") or "ETC").strip().upper()

    invalid_item_memos = {
        "",
        "...",
        "예시 항목",
        "스터디 24시간 이용권",
    }

    if raw_item_memo in invalid_item_memos:
        item_memo = _fallback_item_memo(request)
    else:
        item_memo = raw_item_memo

    if raw_category not in VALID_CATEGORIES:
        category = "ETC"
    else:
        category = raw_category

    return OcrAnalyzeResponse(
        itemMemo=item_memo,
        category=category,
    )


def _fallback(request: OcrAnalyzeRequest) -> OcrAnalyzeResponse:
    """
    LLM 호출 실패/응답 파싱 실패 시 기본 자동채움 반환
    """
    return OcrAnalyzeResponse(
        itemMemo=_fallback_item_memo(request),
        category="ETC",
    )


def _fallback_item_memo(request: OcrAnalyzeRequest) -> str:
    """
    LLM 실패/응답 이상 시 itemMemo 기본값 생성
    """
    if request.menuName and request.menuName.strip():
        return request.menuName.strip()

    if request.storeName and request.storeName.strip():
        return f"{request.storeName.strip()} 지출"

    return "기타 항목"