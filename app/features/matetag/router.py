# tag api 엔드포인트
from fastapi import APIRouter, HTTPException
from .service import extract_tags, TagExtractRequest

router = APIRouter(prefix='/mate', tags=["tags"])

@router.post("/extract")
def tag_extract(request: TagExtractRequest):
    try:
        result = extract_tags(request)
        return {
            "post_id": request.post_id,
            "style_tags": result.style_tags,
            "vibe_tags": result.vibe_tags
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))