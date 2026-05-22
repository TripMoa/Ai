from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from features.ocr.router import router as ocr_router
import uvicorn

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import NAVER_MAP_CLIENT_ID
from features.schedule.router import router as schedule_router

from features.matetag.router import router as tag_router

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Travel AI", version="1.0.0")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(schedule_router)
app.include_router(ocr_router)
app.include_router(tag_router)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "naver_map_key": NAVER_MAP_CLIENT_ID or "",
    })


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)