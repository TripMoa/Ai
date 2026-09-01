from pydantic import BaseModel

class BadwordRequest(BaseModel):
    text: str

class BadwordResponse(BaseModel):
    isBlocked: bool
    reason: str