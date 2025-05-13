#!/usr/bin/env python3

import asyncio
import threading
import uvicorn
import time
import logging
from .rtmp_server.server import RTMPServer
from .api.app import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_demo_streams(rtmp_server, delay=3):
    """Add demo streams for testing"""
    logger.info(f"Waiting {delay} seconds before adding demo streams...")
    time.sleep(delay)
    
    # Add a few example streams for demo purposes
    rtmp_server.add_stream("demo1", "rtmp://example.com/live/demo1")
    rtmp_server.add_stream("demo2", "rtmp://example.com/live/demo2")
    rtmp_server.add_stream("test_stream", "rtmp://example.com/live/test_stream")
    
    logger.info("Added 3 demo streams. In a real application, these would be actual RTMP streams.")
    logger.info("To stream to this server with FFmpeg: ffmpeg -i video.mp4 -c:v libx264 -c:a aac -f flv rtmp://localhost:1935/live/mystream")
    logger.info("To stream with OBS: Set Server to rtmp://localhost:1935/live and Stream Key to your_stream_name")

def main():
    # Create and start RTMP server in a separate thread
    rtmp_server = RTMPServer()
    rtmp_thread = threading.Thread(target=rtmp_server.start)
    rtmp_thread.daemon = True
    rtmp_thread.start()
    
    # Add some demo streams in a separate thread
    demo_thread = threading.Thread(target=add_demo_streams, args=(rtmp_server,))
    demo_thread.daemon = True
    demo_thread.start()
    
    # Create FastAPI app with reference to the RTMP server
    app = create_app(rtmp_server)
    
    # Start the FastAPI web interface
    logger.info("Starting web interface on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()