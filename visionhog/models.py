from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import Optional

# SQLAlchemy setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./visionhog.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Model
class StreamDB(Base):
    __tablename__ = "streams"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String)
    team = Column(String)

# Pydantic Models
class StreamBase(BaseModel):
    prompt: str
    team: str

class StreamCreate(StreamBase):
    pass

class Stream(StreamBase):
    id: int

    class Config:
        from_attributes = True

# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()