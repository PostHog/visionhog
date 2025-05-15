import ffmpeg
import os
import time
import datetime
import threading
import queue
import shutil
import boto3
from botocore.config import Config
import json
import asyncio
import logging
import logging.handlers
from pathlib import Path
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
import uvicorn
from pydantic import BaseModel
from typing import Set, Dict, List, Optional
from sqlalchemy.orm import Session, selectinload, joinedload
from posthog import Posthog
from alembic.config import Config as AlembicConfig
from alembic import command
from dotenv import load_dotenv

# Create a queue for logging
log_queue = queue.Queue()

# Custom logging handler that puts records in the queue
class QueueHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_queue.put(msg)
        except Exception:
            self.handleError(record)

# Configure logging
logger = logging.getLogger('visionhog')
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create queue handler
queue_handler = QueueHandler()
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

def log_worker():
    """Worker thread that processes log messages from the queue"""
    while not exit_flag.is_set():
        try:
            # Get log message from queue with timeout to check exit flag
            try:
                msg = log_queue.get(timeout=1.0)
                print(msg)  # Print to stdout
                log_queue.task_done()
            except queue.Empty:
                continue
        except Exception as e:
            print(f"Error in log worker: {e}")

# Load environment variables from .env file
load_dotenv()

from .models import StreamDB, Stream, StreamCreate, StreamChunk, StreamChunkResponse, get_db, SessionLocal



# Configuration
STREAM_URL = os.getenv("STREAM_URL", "http://127.0.0.1:8080/live/show.flv")  # HTTP FLV stream endpoint
OUTPUT_DIR = Path("video_clips")
PROCESSED_DIR = Path("processed_clips")  # For clips that have been analyzed
MAX_CLIPS_TO_KEEP = 100  # Maximum number of clips to store
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
POSTHOG_ENV_KEY = os.getenv("POSTHOG_ENV_KEY")
POSTHOG_HOST = os.getenv("POSTHOG_HOST", "http://host.docker.internal:8010")
CHUNK_DURATION = 10  # Duration of each clip in seconds

# Storage configuration
USE_MINIO = os.getenv("USE_MINIO", "true").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "posthog-vision")  # Configurable S3 bucket name
S3_PREFIX = os.getenv("S3_PREFIX", "teams/2")  # Configurable S3 prefix

# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Initialize PostHog client
posthog = Posthog(POSTHOG_ENV_KEY, host='http://localhost:8010')

# Initialize S3/MinIO client
if USE_MINIO:
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        verify=MINIO_SECURE
    )
else:
    s3_client = boto3.client('s3')

# Create FastAPI app
app = FastAPI(title="VisionHog API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="visionhog/static"), name="static")

# SSE connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[asyncio.Queue] = set()

    async def connect(self):
        queue = asyncio.Queue()
        self.active_connections.add(queue)
        return queue

    def disconnect(self, queue: asyncio.Queue):
        self.active_connections.remove(queue)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.put(message)

manager = ConnectionManager()

# Pydantic model for prompt updates
class PromptUpdate(BaseModel):
    prompt: str
    emit_events: Optional[bool] = None

def get_team_prompt(db: Session, team_id: str) -> str:
    """Get the prompt for a specific team from the database"""
    stream = db.query(StreamDB).filter(StreamDB.team == team_id).first()
    if stream is None:
        # Create default stream for team if it doesn't exist
        default_prompt = """
Analyze this video of a retail environment and identify key customer events and interactions.
For each event, provide a description and its approximate timestamp in the video.
Return the output as a valid JSON array of objects that follows PostHog's event schema. Each object must have the following fields:

- "event": String - The specific customer action (e.g., "CustomerEntered", "ProductInteraction", "CustomerExited")
- "properties": Object containing:
  - "timestamp": "HH:MM:SS" (String format for hours, minutes, seconds)
  - "distinct_id": String - A unique identifier for the customer (e.g., "customer_1", "customer_2")
  - "description": String - A concise description of what the customer did
  - "location": String - Area within the retail space
  - "duration_seconds": Number - Approximate duration of the activity in seconds
  - "interaction_type": String - Type of activity (e.g., "entry_exit", "product_engagement", "service_usage", "staff_interaction")

  Example of expected JSON output:
[
  {
    "event": "CustomerEntered",
    "properties": {
      "timestamp": "00:00:15",
      "distinct_id": "customer_1",
      "description": "Customer enters through the main entrance",
      "location": "entrance",
      "duration_seconds": 5,
      "interaction_type": "entry_exit"
    }
  },
  {
    "event": "ProductInteraction",
    "properties": {
      "timestamp": "00:01:20",
      "distinct_id": "customer_1",
      "description": "Customer examines product on display",
      "location": "main_floor",
      "duration_seconds": 25,
      "interaction_type": "product_engagement"
    }
  },
  {
    "event": "StaffInteraction",
    "properties": {
      "timestamp": "00:02:15",
      "distinct_id": "customer_1",
      "description": "Customer speaks with staff member",
      "location": "help_desk",
      "duration_seconds": 45,
      "interaction_type": "staff_interaction"
    }
  },
  {
    "event": "CustomerExited",
    "properties": {
      "timestamp": "00:05:30",
      "distinct_id": "customer_1",
      "description": "Customer leaves through the main exit",
      "location": "exit",
      "duration_seconds": 8,
      "interaction_type": "entry_exit"
    }
  }
]

Ensure the output is only the JSON array and nothing else.
If no specific events are identifiable, return an empty array [].

Common event types to look for:
- CustomerEntered: When a customer enters the retail space
- BrowsingBehavior: When a customer is looking around without specific engagement
- ProductInteraction: When a customer engages with products or displays
- ServiceUsage: When a customer uses a service offered in the space
- StaffInteraction: When a customer interacts with staff
- PurchaseActivity: When a customer makes or attempts a purchase
- CustomerExited: When a customer leaves the retail space

Use the same distinct_id to track the same customer throughout multiple events.
"""
        stream = StreamDB(team=team_id, prompt=default_prompt)
        db.add(stream)
        db.commit()
        db.refresh(stream)
    return stream.prompt

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Landing page with links to available endpoints"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VisionHog API</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                margin-bottom: 30px;
            }
            .endpoints {
                display: grid;
                gap: 20px;
            }
            .endpoint {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 6px;
                border: 1px solid #e9ecef;
            }
            .endpoint h2 {
                margin: 0 0 10px 0;
                color: #007bff;
            }
            .endpoint p {
                margin: 0 0 15px 0;
                color: #666;
            }
            .endpoint a {
                display: inline-block;
                padding: 8px 16px;
                background: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                transition: background-color 0.2s;
            }
            .endpoint a:hover {
                background: #0056b3;
            }
            .method {
                font-family: monospace;
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 0.9em;
                color: #495057;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>VisionHog API</h1>
            <div class="endpoints">
                <div class="endpoint">
                    <h2>SSE Test Page</h2>
                    <p>Interactive test page for Server-Sent Events (SSE) streaming of video analysis results.</p>
                    <a href="/static/sse-test.html">Open SSE Test Page</a>
                </div>
                <div class="endpoint">
                    <h2>API Documentation</h2>
                    <p>Interactive API documentation with Swagger UI.</p>
                    <a href="/docs">View API Docs</a>
                </div>
                <div class="endpoint">
                    <h2>Current Prompt</h2>
                    <p>View the current prompt used for video analysis.</p>
                    <a href="/prompt">View Prompt</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/prompt")
async def get_prompt(db: Session = Depends(get_db)):
    """Get the current prompt used for video analysis"""
    return {"prompt": get_team_prompt(db, "2")}

@app.post("/prompt")
async def update_prompt(prompt_update: PromptUpdate, db: Session = Depends(get_db)):
    """Update the prompt used for video analysis and optionally update emit_events flag"""
    stream = db.query(StreamDB).filter(StreamDB.team == "2").first()
    if stream is None:
        stream = StreamDB(
            team="2",
            prompt=prompt_update.prompt,
            emit_events=prompt_update.emit_events if prompt_update.emit_events is not None else False
        )
        db.add(stream)
    else:
        stream.prompt = prompt_update.prompt
        if prompt_update.emit_events is not None:
            stream.emit_events = prompt_update.emit_events
    db.commit()
    db.refresh(stream)
    return {
        "message": "Prompt updated successfully",
        "prompt": stream.prompt,
        "emit_events": stream.emit_events
    }

@app.get("/stream-analysis")
async def stream_analysis():
    """Stream analysis results using Server-Sent Events"""
    async def event_generator():
        queue = await manager.connect()
        try:
            while True:
                data = await queue.get()
                yield {
                    "event": "analysis",
                    "data": data
                }
        except Exception as e:
            print(f"Error in SSE stream: {e}")
        finally:
            manager.disconnect(queue)

    return EventSourceResponse(event_generator())

@app.get("/streams", response_model=List[Stream])
async def get_streams(db: Session = Depends(get_db)):
    """Get all streams"""
    return db.query(StreamDB).all()

@app.post("/streams", response_model=Stream)
async def create_stream(stream: StreamCreate, db: Session = Depends(get_db)):
    """Create a new stream"""
    db_stream = StreamDB(**stream.model_dump())
    db.add(db_stream)
    db.commit()
    db.refresh(db_stream)
    return db_stream

@app.get("/streams/{stream_id}", response_model=Stream)
async def get_stream(stream_id: int, db: Session = Depends(get_db)):
    """Get a specific stream by ID"""
    stream = db.query(StreamDB).filter(StreamDB.id == stream_id).first()
    if stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    return stream

def get_s3_url(key: str) -> str:
    """Generate the appropriate S3/MinIO URL for a given key"""
    if USE_MINIO:
        return f"{MINIO_ENDPOINT}/{S3_BUCKET}/{key}"
    else:
        return f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"

@app.get("/teams/{team_id}/chunks", response_model=List[StreamChunkResponse])
async def list_team_chunks(
    team_id: str,
    limit: Optional[int] = Query(100, ge=1, le=1000),
    offset: Optional[int] = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    List chunks for a specific team, ordered by processed_at time (newest first).

    Parameters:
    - team_id: The team identifier
    - limit: Maximum number of chunks to return (default: 100, max: 1000)
    - offset: Number of chunks to skip (for pagination)
    """
    chunks = (
        db.query(StreamChunk)
        .options(joinedload(StreamChunk.stream).load_only(StreamDB.id))
        .filter(StreamChunk.team_id == team_id)
        .order_by(StreamChunk.processed_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    # Enhance chunks with additional information
    enhanced_chunks = []
    for chunk in chunks:
        # Generate S3/MinIO URLs
        s3_video_url = get_s3_url(chunk.s3_video_key)
        s3_analysis_url = get_s3_url(chunk.s3_analysis_key) if chunk.s3_analysis_key else None

        # Create enhanced chunk response
        enhanced_chunk = StreamChunkResponse(
            id=chunk.id,
            stream_id=chunk.stream_id,
            team_id=chunk.team_id,
            s3_video_key=chunk.s3_video_key,
            s3_analysis_key=chunk.s3_analysis_key,
            clip_name=chunk.clip_name,
            processed_at=chunk.processed_at,
            processing_time=chunk.processing_time,
            created_at=chunk.created_at,
            stream=chunk.stream,  # Include the full stream information
            s3_video_url=s3_video_url,
            s3_analysis_url=s3_analysis_url,
            analysis_text=chunk.analysis_json  # Use the analysis_json from the database
        )
        enhanced_chunks.append(enhanced_chunk)

    return enhanced_chunks

@app.put("/streams/{stream_id}", response_model=Stream)
async def update_stream(stream_id: int, stream: StreamCreate, db: Session = Depends(get_db)):
    """Update a stream"""
    db_stream = db.query(StreamDB).filter(StreamDB.id == stream_id).first()
    if db_stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")

    for key, value in stream.model_dump().items():
        setattr(db_stream, key, value)

    db.commit()
    db.refresh(db_stream)
    return db_stream

@app.delete("/streams/{stream_id}")
async def delete_stream(stream_id: int, db: Session = Depends(get_db)):
    """Delete a stream"""
    db_stream = db.query(StreamDB).filter(StreamDB.id == stream_id).first()
    if db_stream is None:
        raise HTTPException(status_code=404, detail="Stream not found")

    db.delete(db_stream)
    db.commit()
    return {"message": "Stream deleted successfully"}

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Set up Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# Create a queue for processing clips
clip_queue = queue.Queue()
# Create a flag for signaling threads to exit
exit_flag = threading.Event()

def analyze_with_gemini(video_path, db: Session):
    """Send video to Gemini for analysis and return results"""
    try:
        # Read the video as bytes
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        # Get the stream configuration for team 2
        stream = db.query(StreamDB).filter(StreamDB.team == "2").first()
        if stream is None:
            raise Exception("No stream configuration found for team 2")

        # Send to Gemini for analysis using new API format
        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ),
                    types.Part(text=stream.prompt)
                ]
            )
        )

        # Sanitize the response by removing markdown code block formatting
        text = response.text
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json prefix
        elif text.startswith("```"):
            text = text[3:]  # Remove ``` prefix
        if text.endswith("```"):
            text = text[:-3]  # Remove ``` suffix

        # Strip any leading/trailing whitespace
        text = text.strip()

        return text
    except Exception as e:
        print(f"Error analyzing with Gemini: {e}")
        return f"Analysis failed: {str(e)}"

def process_clip_worker():
    """Worker function to process clips in the queue"""
    logger.info("Clip processing worker started")

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while not exit_flag.is_set() or not clip_queue.empty():
        try:
            # Get clip from queue with timeout to check exit flag periodically
            try:
                clip_path = clip_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            logger.info(f"Processing clip: {clip_path}")

            # Create a new database session for this thread
            db = SessionLocal()
            try:
                # Get stream configuration
                stream = db.query(StreamDB).filter(StreamDB.team == "2").first()
                if stream is None:
                    logger.warning("No stream configuration found for team 2")
                    continue

                # Analyze with Gemini
                start_time = time.time()
                analysis = analyze_with_gemini(clip_path, db)
                processing_time = time.time() - start_time

                # Create analysis result object
                analysis_result = {
                    "clip_name": clip_path.name,
                    "processed_at": datetime.datetime.now().isoformat(),
                    "processing_time": processing_time,
                    "analysis": analysis
                }

                # Broadcast analysis result to SSE clients using the thread's event loop
                loop.run_until_complete(manager.broadcast(json.dumps(analysis_result)))

                # Save results
                results_path = clip_path.with_suffix('.txt')
                with open(results_path, 'w') as f:
                    f.write(f"Analysis of {clip_path.name}:\n")
                    f.write(f"Processed at: {datetime.datetime.now().isoformat()}\n")
                    f.write(f"Processing time: {processing_time:.2f} seconds\n\n")
                    f.write(analysis)

                logger.info(f"Analysis saved to {results_path}")

                # If emit_events is enabled, send events to PostHog
                if stream.emit_events:
                    try:
                        # Parse the analysis result as JSON
                        events = json.loads(analysis)

                        # Send each event to PostHog
                        for event in events:
                            # Add additional properties
                            event['properties'].update({
                                'team_id': stream.team,
                                'video_clip': clip_path.name,
                                'processed_at': datetime.datetime.now().isoformat(),
                                'processing_time': processing_time
                            })

                            # Send to PostHog
                            posthog.capture(
                                distinct_id=event['properties']['distinct_id'],
                                event=event['event'],
                                properties=event['properties']
                            )

                        logger.info(f"Successfully emitted {len(events)} events to PostHog")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse analysis result as JSON: {e}")
                    except Exception as e:
                        logger.error(f"Error emitting events to PostHog: {e}")

                # Move to processed directory
                dest_path = PROCESSED_DIR / clip_path.name
                shutil.move(str(clip_path), str(dest_path))

                # Also move the analysis file
                if results_path.exists():
                    shutil.move(str(results_path), str(PROCESSED_DIR / results_path.name))

                # Upload to S3 and create database record
                try:
                    # Upload video file
                    s3_video_key = f"{S3_PREFIX}/{clip_path.name}"
                    s3_client.upload_file(str(dest_path), S3_BUCKET, s3_video_key)
                    logger.info(f"Uploaded video to s3://{S3_BUCKET}/{s3_video_key}")

                    # Upload analysis file
                    s3_analysis_key = f"{S3_PREFIX}/{results_path.name}"
                    s3_client.upload_file(str(PROCESSED_DIR / results_path.name), S3_BUCKET, s3_analysis_key)
                    logger.info(f"Uploaded analysis to s3://{S3_BUCKET}/{s3_analysis_key}")

                    # Remove files from processed_clips after successful upload
                    try:
                        dest_path.unlink()
                        (PROCESSED_DIR / results_path.name).unlink()
                        logger.info(f"Removed processed files from {PROCESSED_DIR}")
                    except Exception as e:
                        logger.error(f"Error removing processed files: {e}")

                    # Create stream chunk record
                    stream_chunk = StreamChunk(
                        stream_id=stream.id,
                        team_id=stream.team,
                        s3_video_key=s3_video_key,
                        s3_analysis_key=s3_analysis_key,
                        clip_name=clip_path.name,
                        processed_at=datetime.datetime.now(),
                        processing_time=processing_time,
                        analysis_json=analysis  # Save the analysis JSON
                    )
                    db.add(stream_chunk)
                    db.commit()
                    logger.info(f"Created stream chunk record with ID: {stream_chunk.id}")

                except Exception as e:
                    logger.error(f"Error uploading to S3 or creating database record: {e}")
                    db.rollback()

                clip_queue.task_done()
            finally:
                # Always close the database session
                db.close()

        except Exception as e:
            logger.error(f"Error in processing worker: {e}")

    logger.info("Clip processing worker stopped")

def cleanup_old_clips():
    """Remove excess clips to manage disk space"""
    try:
        # Get all clips in both directories
        all_clips = list(OUTPUT_DIR.glob("clip_*.mp4")) + list(PROCESSED_DIR.glob("clip_*.mp4"))

        # Sort by creation time (oldest first)
        all_clips.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest clips if we have too many
        if len(all_clips) > MAX_CLIPS_TO_KEEP:
            clips_to_remove = all_clips[:-MAX_CLIPS_TO_KEEP]
            for clip in clips_to_remove:
                print(f"Removing old clip: {clip}")
                clip.unlink(missing_ok=True)

                # Also remove associated text file if it exists
                text_file = clip.with_suffix('.txt')
                if text_file.exists():
                    text_file.unlink()
    except Exception as e:
        print(f"Error during cleanup: {e}")

def capture_stream_chunks():
    """Capture the RTMP stream in chunks"""
    chunk_count = 0
    last_cleanup = time.time()

    try:
        while not exit_flag.is_set():
            # Generate unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"clip_{timestamp}.mp4"

            logger.info(f"Capturing chunk {chunk_count + 1} to {output_path}...")

            # Use FFmpeg to capture a segment of the stream
            try:
                # Determine stream type from URL
                is_hls = STREAM_URL.endswith('.m3u8')

                # Configure input based on stream type
                if is_hls:
                    # HLS stream configuration
                    stream = ffmpeg.input(
                        STREAM_URL,
                        t=CHUNK_DURATION,
                        f='hls',
                        re=None,
                        timeout=5000000,
                        headers='User-Agent: Mozilla/5.0\r\nAccept: */*\r\nConnection: keep-alive\r\n'
                    )
                else:
                    # HTTP FLV stream configuration
                    stream = ffmpeg.input(
                        STREAM_URL,
                        t=CHUNK_DURATION,
                        f='flv',
                        re=None,
                        timeout=5000000,
                        headers='User-Agent: Mozilla/5.0\r\nAccept: */*\r\nConnection: keep-alive\r\n'
                    )

                # Configure output with more specific options
                stream = ffmpeg.output(
                    stream,
                    str(output_path),
                    c="copy",
                    f="mp4",
                    movflags="faststart"
                )

                # Start capturing with verbose output for debugging
                start_time = time.time()
                try:
                    ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=False)
                except ffmpeg.Error as e:
                    logger.error(f"FFmpeg stderr output: {e.stderr.decode() if e.stderr else 'No stderr output'}")
                    raise

                # Calculate actual capture time
                elapsed = time.time() - start_time
                logger.info(f"Chunk captured in {elapsed:.2f} seconds")

                # Check if file was created and has content
                if output_path.exists() and output_path.stat().st_size > 0:
                    logger.info(f"Successfully saved chunk to {output_path} ({output_path.stat().st_size} bytes)")
                    chunk_count += 1

                    # Add to processing queue
                    clip_queue.put(output_path)

                    # Perform cleanup every 10 chunks or once per hour
                    if chunk_count % 10 == 0 or time.time() - last_cleanup > 3600:
                        cleanup_old_clips()
                        last_cleanup = time.time()
                else:
                    logger.warning(f"Failed to capture chunk or empty file created")

                # Calculate time adjustment to maintain precise intervals
                time_adjustment = max(0, CHUNK_DURATION - elapsed)
                if time_adjustment > 0:
                    logger.info(f"Waiting {time_adjustment:.2f} seconds to align with {CHUNK_DURATION}-second intervals...")

                    # Use small sleep increments to check exit flag
                    end_wait = time.time() + time_adjustment
                    while time.time() < end_wait and not exit_flag.is_set():
                        time.sleep(0.1)

            except ffmpeg.Error as e:
                logger.error(f"FFmpeg error: {e}")
                # Short pause before retry
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error during capture: {e}")
                # Short pause before retry
                time.sleep(1)

            # Check exit flag
            if exit_flag.is_set():
                break

    except KeyboardInterrupt:
        logger.info("Capturing stopped by user")
    finally:
        logger.info("Capture process ending, waiting for queue to empty...")
        # Wait for queue to be processed
        clip_queue.join()

def bootstrap_default_stream():
    """Ensure default stream configuration exists for team 2"""
    db = SessionLocal()
    try:
        # Check if default stream exists
        stream = db.query(StreamDB).filter(StreamDB.team == "2").first()
        if stream is None:
            logger.info("Creating default stream configuration for team 2...")
            default_prompt = """
Analyze this video of a retail environment and identify key customer events and interactions.
For each event, provide a description and its approximate timestamp in the video.
Return the output as a valid JSON array of objects that follows PostHog's event schema. Each object must have the following fields:

- "event": String - The specific customer action (e.g., "CustomerEntered", "ProductInteraction", "CustomerExited")
- "timestamp": "HH:MM:SS" (String format for hours, minutes, seconds)
- "properties": Object containing:
  - "distinct_id": String - A unique identifier for the customer (e.g., "customer_1", "customer_2")
  - "description": String - A concise description of what the customer did
  - "location": String - Area within the retail space
  - "duration_seconds": Number - Approximate duration of the activity in seconds
  - "interaction_type": String - Type of activity (e.g., "entry_exit", "product_engagement", "service_usage", "staff_interaction")

  Example of expected JSON output:
[
  {
    "event": "CustomerEntered",
    "timestamp": "00:00:15",
    "properties": {
      "distinct_id": "customer_1",
      "description": "Customer enters through the main entrance",
      "location": "entrance",
      "duration_seconds": 5,
      "interaction_type": "entry_exit"
    }
  },
  {
    "event": "ProductInteraction",
    "timestamp": "00:01:20",
    "properties": {
      "distinct_id": "customer_1",
      "description": "Customer examines product on display",
      "location": "main_floor",
      "duration_seconds": 25,
      "interaction_type": "product_engagement"
    }
  },
  {
    "event": "StaffInteraction",
    "timestamp": "00:02:15",
    "properties": {
      "distinct_id": "customer_1",
      "description": "Customer speaks with staff member",
      "location": "help_desk",
      "duration_seconds": 45,
      "interaction_type": "staff_interaction"
    }
  },
  {
    "event": "CustomerExited",
    "timestamp": "00:05:30",
    "properties": {
      "distinct_id": "customer_1",
      "description": "Customer leaves through the main exit",
      "location": "exit",
      "duration_seconds": 8,
      "interaction_type": "entry_exit"
    }
  }
]

Ensure the output is only the JSON array and nothing else.
If no specific events are identifiable, return an empty array [].

Common event types to look for:
- CustomerEntered: When a customer enters the retail space
- BrowsingBehavior: When a customer is looking around without specific engagement
- ProductInteraction: When a customer engages with products or displays
- ServiceUsage: When a customer uses a service offered in the space
- StaffInteraction: When a customer interacts with staff
- PurchaseActivity: When a customer makes or attempts a purchase
- CustomerExited: When a customer leaves the retail space

Use the same distinct_id to track the same customer throughout multiple events.
"""
            # Create default stream
            default_stream = StreamDB(
                team="2",
                prompt=default_prompt,
                emit_events=False
            )
            db.add(default_stream)
            db.commit()
            logger.info("Default stream configuration created successfully")
        else:
            logger.info("Default stream configuration already exists")
    except Exception as e:
        logger.error(f"Error during bootstrap: {e}")
        db.rollback()
    finally:
        db.close()

def run_migrations():
    """Run database migrations"""
    try:
        # Get the path to the migrations directory (one level up from the main.py file)
        migrations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations")

        # Create Alembic configuration
        alembic_cfg = AlembicConfig(os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini"))

        # Run the migration
        try:
            # Run the migration
            command.upgrade(alembic_cfg, "head")
            logger.info("Database migrations completed successfully")
        except Exception as e:
            # Check if this is a "table already exists" error
            if "table streams already exists" in str(e):
                logger.info("Database tables already exist, skipping migrations")
            else:
                # Re-raise if it's a different error
                raise
    except Exception as e:
        logger.error(f"Error running migrations: {e}")
        logger.info("ignoring")

def ensure_bucket_exists():
    """Ensure that the S3/MinIO bucket exists, creating it if necessary"""
    logger.info(f"Ensuring bucket {S3_BUCKET} exists...")
    try:
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET)
            logger.info(f"Bucket {S3_BUCKET} already exists")
        except s3_client.exceptions.ClientError as e:
            # If a 404 error, then the bucket does not exist
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                logger.info(f"Bucket {S3_BUCKET} does not exist, creating it...")
                if USE_MINIO:
                    # For MinIO, just create the bucket
                    s3_client.create_bucket(Bucket=S3_BUCKET)
                else:
                    # For AWS S3, create with region configuration
                    location = {'LocationConstraint': s3_client.meta.region_name}
                    s3_client.create_bucket(
                        Bucket=S3_BUCKET,
                        CreateBucketConfiguration=location
                    )
                logger.info(f"Bucket {S3_BUCKET} created successfully")
            else:
                # If it's another error, raise it
                raise
    except Exception as e:
        logger.error(f"Error ensuring bucket exists: {e}")
        logger.info("Will attempt to use the bucket anyway")

def run_fastapi():
    """Run the FastAPI server"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8069,
        reload=False
    )

def main():
    logger.info(f"Starting VisionHog service...")
    logger.info(f"Stream URL: {STREAM_URL}")
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")
    logger.info(f"Processed directory: {PROCESSED_DIR.absolute()}")
    logger.info(f"Storage: {'MinIO' if USE_MINIO else 'S3'}")
    if USE_MINIO:
        logger.info(f"MinIO endpoint: {MINIO_ENDPOINT}")

    try:
        # Start logging thread
        log_thread = threading.Thread(target=log_worker)
        log_thread.daemon = True
        log_thread.start()

        # Run database migrations
        run_migrations()

        # Bootstrap default stream configuration
        bootstrap_default_stream()

        # Ensure S3/MinIO bucket exists
        ensure_bucket_exists()

        # Start processing thread
        processing_thread = threading.Thread(target=process_clip_worker)
        processing_thread.daemon = True
        processing_thread.start()

        # Start capturing chunks in a separate thread
        capture_thread = threading.Thread(target=capture_stream_chunks)
        capture_thread.daemon = True
        capture_thread.start()

        # Start FastAPI server in a separate thread
        api_thread = threading.Thread(target=run_fastapi)
        api_thread.daemon = True
        api_thread.start()

        # Keep the main thread alive and handle keyboard interrupt
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        # Signal threads to exit
        exit_flag.set()

        # Wait for processing thread to finish
        if 'processing_thread' in locals():
            processing_thread.join(timeout=10)
        if 'capture_thread' in locals():
            capture_thread.join(timeout=10)
        if 'api_thread' in locals():
            api_thread.join(timeout=10)
        if 'log_thread' in locals():
            log_thread.join(timeout=10)

        # Clear out video_clips directory during shutdown
        logger.info("Cleaning up video_clips directory...")
        for file in OUTPUT_DIR.glob("*"):
            try:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            except Exception as e:
                logger.error(f"Error removing {file}: {e}")

        logger.info("Program terminated")

if __name__ == "__main__":
    main()