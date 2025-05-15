# VisionHog RTMP Server

A Python application that serves as an RTMP server for video streaming with a FastAPI web interface to view all active streams. The application uses Gemini AI to analyze video content and can generate events for PostHog.

## Features

- RTMP server for receiving video streams
- FastAPI web interface to list all active streams
- View any stream in the browser
- Video analysis with Google's Gemini AI
- S3 storage integration for video clips and analysis results
- PostHog event tracking integration
- Automatic database migrations using Alembic

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/visionhog.git
cd visionhog

# Install dependencies with UV
uv pip install -e .

# Create a .env file from the example
cp .env.example .env
# Edit the .env file with your actual API keys and configuration
```

## Environment Variables

Copy the `.env.example` file to `.env` and configure the following variables:

```
# Required
GEMINI_API_KEY=your_gemini_api_key_here    # Google Gemini API key for video analysis

# Optional
POSTHOG_ENV_KEY=your_posthog_api_key_here  # PostHog API key for event tracking
S3_BUCKET=your_bucket_name                 # S3 bucket for storing videos (default: posthog-vision)
S3_PREFIX=your_prefix                      # S3 prefix for organizing files (default: teams/2)

# AWS credentials (required if using S3)
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1
```

## Database Setup

The application uses SQLite with Alembic for database migrations. Migrations are automatically run when the application starts. If you need to run migrations manually:

```bash
# Run migrations manually
cd visionhog
alembic upgrade head

# Create a new migration
alembic revision --autogenerate -m "description of changes"
```

## Usage

```bash
# Start the server
python -m visionhog.main
```

The RTMP server will listen on `rtmp://localhost:1935/live` and the web interface will be available at `http://localhost:8059`.

### Streaming to the server

Using FFmpeg:
```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -f flv rtmp://localhost:1935/live/stream_name
```

Or configure your streaming software (OBS, etc.) to stream to `rtmp://localhost:1935/live/stream_name`.

## API Endpoints

- `GET /`: Landing page with links to available endpoints
- `GET /prompt`: Get the current prompt used for video analysis
- `POST /prompt`: Update the prompt used for video analysis
- `GET /stream-analysis`: Stream analysis results using Server-Sent Events
- `GET /streams`: Get all streams
- `POST /streams`: Create a new stream
- `GET /streams/{stream_id}`: Get a specific stream by ID
- `GET /teams/{team_id}/chunks`: List chunks for a specific team
- `PUT /streams/{stream_id}`: Update a stream
- `DELETE /streams/{stream_id}`: Delete a stream

## License

MIT