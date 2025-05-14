from sqlalchemy import Column, Integer, String, create_engine, Boolean, DateTime, Float, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

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
    emit_events = Column(Boolean, default=False)

    # Relationship to stream chunks
    chunks = relationship("StreamChunk", back_populates="stream")

class StreamChunk(Base):
    __tablename__ = "stream_chunks"

    id = Column(Integer, primary_key=True, index=True)
    stream_id = Column(Integer, ForeignKey("streams.id"), nullable=False)
    team_id = Column(String, nullable=False)
    s3_video_key = Column(String, nullable=False)
    s3_analysis_key = Column(String, nullable=True)
    clip_name = Column(String, nullable=False)
    processed_at = Column(DateTime, nullable=False)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    analysis_json = Column(String, nullable=True)

    # Relationship to stream
    stream = relationship("StreamDB", back_populates="chunks")

# Pydantic Models
class StreamBase(BaseModel):
    prompt: str
    team: str
    emit_events: bool = False

class StreamCreate(StreamBase):
    pass

class Stream(StreamBase):
    id: int

    class Config:
        from_attributes = True

class StreamChunkBase(BaseModel):
    stream_id: int
    team_id: str
    s3_video_key: str
    s3_analysis_key: Optional[str] = None
    clip_name: str
    processed_at: datetime
    processing_time: float
    analysis_json: Optional[str] = None

class StreamChunkCreate(StreamChunkBase):
    pass

class StreamChunkResponse(StreamChunkBase):
    id: int
    created_at: datetime
    s3_video_url: Optional[str] = None  # Full S3 URL for the video
    s3_analysis_url: Optional[str] = None  # Full S3 URL for the analysis
    analysis_text: Optional[str] = None  # The actual analysis text content

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