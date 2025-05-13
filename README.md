# VisionHog RTMP Server

A Python application that serves as an RTMP server for video streaming with a FastAPI web interface to view all active streams.

## Features

- RTMP server for receiving video streams
- FastAPI web interface to list all active streams
- View any stream in the browser

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/visionhog.git
cd visionhog

# Install dependencies with UV
uv pip install -e .
```

## Usage

```bash
# Start the server
python -m visionhog.main
```

The RTMP server will listen on `rtmp://localhost:1935/live` and the web interface will be available at `http://localhost:8000`.

### Streaming to the server

Using FFmpeg:
```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a aac -f flv rtmp://localhost:1935/live/stream_name
```

Or configure your streaming software (OBS, etc.) to stream to `rtmp://localhost:1935/live/stream_name`.

## License

MIT