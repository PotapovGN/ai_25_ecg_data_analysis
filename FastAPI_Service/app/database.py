from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

engine = create_engine("sqlite:///history.db")
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class RequestHistory(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)
    status = Column(String)
    request_metadata  = Column(Text)

Base.metadata.create_all(engine)