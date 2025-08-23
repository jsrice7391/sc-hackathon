from pydantic import BaseModel, Field
from datetime import date

from .database import SessionLocal

class LaunchCreate(BaseModel):
    rocket_name: str = Field(..., example="Falcon 9")
    launch_date: date = Field(..., example="2025-12-01")


class LaunchRead(BaseModel):
    id: int
    rocket_name: str
    launch_date: date

    class Config:
        orm_mode = True

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
