# langchain mate 태그 추출 로직
from dotenv import load_dotenv
from langchain_naver import ChatClovaX
from .tag_prompt import tag_prompt
from pydantic import BaseModel, Field
from .tag_cache import get_tag_cache
from typing import List

load_dotenv()

class TagResult(BaseModel):
    style_tags: List[str] = Field(
        description="여행 방식 태그 (STYLE 카테고리에서 1~2개)"
    )
    vibe_tags: List[str] = Field(
        description="동행 분위기 태그 (VIBE 카테고리에서 0~1개)"
    )

class TagExtractRequest(BaseModel):
    post_id: int
    content: str
    destination: str

def extract_tags(request: TagExtractRequest) -> TagResult:
    llm = ChatClovaX(model="HCX-007", max_tokens=1024, thinking={"effort" : "none"})
    structured_llm = llm.with_structured_output(TagResult, method="json_schema")

    tag_cache = get_tag_cache()

    prompt = tag_prompt.invoke({
        "style_tags": ", ".join(tag_cache["style"]),
        "vibe_tags": ", ".join(tag_cache["vibe"]),
        "content": request.content,
        "destination": request.destination
    })

    result = structured_llm.invoke(prompt)
    return result
