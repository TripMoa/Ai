from fastapi import APIRouter, HTTPException
from features.ocr.models import OcrAnalyzeRequest, OcrAnalyzeResponse
from features.ocr.service import analyze_receipt

router = APIRouter(prefix="/ocr", tags=["ocr"])


@router.post("/analyze", response_model=OcrAnalyzeResponse)
async def analyze(request: OcrAnalyzeRequest):
    try:
        return analyze_receipt(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))