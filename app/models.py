from sqlalchemy import Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Launch(Base):
    __tablename__ = "launches"
    id = Column(Integer, primary_key=True, index=True)
    rocket_name = Column(String, nullable=False)
    launch_date = Column(Date, nullable=False)
