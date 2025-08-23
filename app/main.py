import json
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List

from . import models
from app.models.launch import LaunchCreate, LaunchRead, get_db

app = FastAPI()

@app.post("/schedule", response_model=LaunchRead)
def schedule_launch(launch: LaunchCreate, db: Session = Depends(get_db)):
    db_launch = models.Launch(rocket_name=launch.rocket_name, launch_date=launch.launch_date)
    db.add(db_launch)
    db.commit()
    db.refresh(db_launch)
    return db_launch


@app.get("/launches")
def get_launches(db: Session = Depends(get_db)):
    return json.dumps({
        "status": 200
    })