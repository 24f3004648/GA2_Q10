import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from extractor import extract

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str = Field(max_length=20000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class InvoiceFields(BaseModel):
    vendor: str = Field(min_length=1)
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


@app.post("/extract", response_model=InvoiceFields)
def extract_invoice(req: ExtractRequest):
    try:
        fields = extract(req.text)
    except Exception:
        fields = None
    if fields is None:
        # Garbage in -> a clean 422, never a 500.
        raise HTTPException(status_code=422, detail="could not extract invoice fields from text")
    return InvoiceFields(**fields)


@app.get("/")
def root():
    return {"status": "ok"}
