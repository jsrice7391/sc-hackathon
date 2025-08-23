from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List
from datetime import date

from . import models
from .database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

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

@app.post("/schedule", response_model=LaunchRead)
def schedule_launch(launch: LaunchCreate, db: Session = Depends(get_db)):
    db_launch = models.Launch(rocket_name=launch.rocket_name, launch_date=launch.launch_date)
    db.add(db_launch)
    db.commit()
    db.refresh(db_launch)
    return db_launch

@app.get("/launches", response_model=List[LaunchRead])
def get_launches(db: Session = Depends(get_db)):
    return db.query(models.Launch).all()