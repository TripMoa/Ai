from pydantic import BaseModel
from typing import Literal, Optional


ExpenseCategory = Literal["FOOD", "TRANS", "STAY", "SHOP", "TICKET", "ETC"]


class OcrAnalyzeRequest(BaseModel):
    storeName: Optional[str] = None
    menuName: Optional[str] = None
    paymentMethod: Optional[str] = None
    dateTime: Optional[str] = None
    totalAmount: Optional[int] = None


class OcrAnalyzeResponse(BaseModel):
    itemMemo: str
    category: ExpenseCategory