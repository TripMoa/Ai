from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from features.ocr.router import router as ocr_router
from features.badword.router import router as badword_router
import uvicorn

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import CORS_ALLOWED_ORIGINS
from features.schedule.router import router as schedule_router

from features.matetag.router import router as tag_router

app = FastAPI(title="Travel AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(schedule_router)
app.include_router(ocr_router)
app.include_router(tag_router)
app.include_router(badword_router)


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)