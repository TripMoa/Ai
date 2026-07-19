from fastapi import APIRouter, HTTPException
from features.badword.models import BadwordRequest, BadwordResponse
from features.badword.service import check_badword

router = APIRouter(prefix="/badword", tags=["badword"])

@router.post("/check", response_model=BadwordResponse)
async def check(request: BadwordRequest):
    try:
        return check_badword(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))