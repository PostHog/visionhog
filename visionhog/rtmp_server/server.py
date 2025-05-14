import os
import time
import subprocess
import threading
import logging
from typing import Dict, Optional, List
from pathlib import Path
import shutil
import json
import socket

logger = logging.getLogger(__name__)

class RTMPServer:
    """
    A simple RTMP server that tracks streams and converts them to HLS.
    """
    def __init__(self, host="0.0.0.0", port=1935, hls_dir="./hls"):
        self.host = host
        self.port = port
        self.hls_dir = Path(hls_dir)
        self._active_streams = {}  # name -> stream info
        self._processes = {}  # name -> process
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._running = False

        # Create HLS directory if it doesn't exist
        os.makedirs(self.hls_dir, exist_ok=True)

    def _detect_active_rtmp_streams(self):
        """
        Simulates RTMP stream detection by checking if FFmpeg can connect to them.
        In a real implementation, you would need an actual RTMP server.
        """
        # This is a simplified implementation for demonstration purposes
        # In reality, you'd likely be using a separate RTMP server like nginx-rtmp

        return []  # Return empty list for demo as we'll manually add streams

    def _monitor_streams(self):
        """Thread function that continuously monitors for RTMP streams"""
        logger.info("Stream monitor started")

        while self._running:
            try:
                # This is where we'd detect new RTMP streams coming in
                # For demonstration, we'll mock this functionality

                # In a real implementation, you'd detect new streams and add them
                # to self._active_streams, then start HLS conversion

                # Sleep to avoid high CPU usage
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in stream monitor: {e}")

    def start(self):
        """Start the RTMP server"""
        logger.info(f"Starting RTMP server on rtmp://{self.host}:{self.port}/live")

        # In a real implementation, we'd start an actual RTMP server here
        # For demonstration, we'll just set up the stream monitoring thread

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_streams)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        # For demonstration, check if port 1935 is available
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((self.host, self.port))
            sock.close()
            logger.info(f"RTMP port {self.port} is available")
        except socket.error:
            logger.warning(f"RTMP port {self.port} may already be in use - this is just a demo")

    def add_stream(self, stream_name, rtmp_url=None):
        """
        Manually add a stream for testing purposes.
        In a real implementation, this would be called automatically when
        the RTMP server detects a new stream.
        """
        if rtmp_url is None:
            rtmp_url = f"rtmp://{self.host}:{self.port}/live/{stream_name}"

        logger.info(f"Adding stream: {stream_name}")

        # Get HLS URL
        hls_url = f"/hls/{stream_name}/index.m3u8"

        # Add to active streams
        with self._lock:
            self._active_streams[stream_name] = {
                'name': stream_name,
                'path': f"live/{stream_name}",
                'rtmp_url': rtmp_url,
                'hls_url': hls_url,
                'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'active': True
            }

        # Start HLS conversion
        self.start_hls_conversion(stream_name, rtmp_url)

        return self._active_streams[stream_name]

    def remove_stream(self, stream_name):
        """
        Remove a stream and stop its HLS conversion.
        In a real implementation, this would be called automatically when
        the RTMP server detects a stream has ended.
        """
        logger.info(f"Removing stream: {stream_name}")

        # Stop HLS conversion
        self.stop_hls_conversion(stream_name)

        # Remove from active streams
        with self._lock:
            if stream_name in self._active_streams:
                del self._active_streams[stream_name]

        # Clean up HLS files with a delay (to allow clients to finish viewing)
        threading.Timer(30.0, lambda: self.cleanup_hls_stream(stream_name)).start()

    def start_hls_conversion(self, stream_name, rtmp_url):
        """
        Start HLS conversion for a stream using FFmpeg.
        """
        # Create stream directory
        stream_dir = self.hls_dir / stream_name
        os.makedirs(stream_dir, exist_ok=True)

        # Start FFmpeg to convert from RTMP to HLS
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", rtmp_url,
            "-c:v", "copy",
            "-c:a", "aac",
            "-hls_time", "2",
            "-hls_list_size", "10",
            "-hls_flags", "delete_segments",
            "-hls_segment_type", "mpegts",
            "-hls_segment_filename", f"{stream_dir}/%d.ts",
            "-f", "hls",
            f"{stream_dir}/index.m3u8"
        ]

        logger.info(f"Starting FFmpeg with command: {' '.join(ffmpeg_cmd)}")

        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Store the process separately from the stream info
        with self._lock:
            self._processes[stream_name] = process

    def stop_hls_conversion(self, stream_name):
        """Stop HLS conversion for a stream"""
        logger.info(f"Stopping HLS conversion for stream: {stream_name}")

        with self._lock:
            if stream_name in self._processes:
                process = self._processes[stream_name]
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                del self._processes[stream_name]

    def cleanup_hls_stream(self, stream_name):
        """Clean up HLS files for a stream"""
        stream_dir = self.hls_dir / stream_name
        if os.path.exists(stream_dir):
            logger.info(f"Cleaning up HLS files for stream: {stream_name}")
            shutil.rmtree(stream_dir)

    def get_active_streams(self) -> List[Dict]:
        """Get list of all active streams"""
        with self._lock:
            # Create a copy to avoid race conditions
            return [stream.copy() for stream in self._active_streams.values()]

    def get_stream(self, stream_name: str) -> Optional[Dict]:
        """Get information about a specific stream"""
        with self._lock:
            return self._active_streams.get(stream_name, None)

    def shutdown(self):
        """Shutdown the server and cleanup resources"""
        logger.info("Shutting down RTMP server")

        # Stop all HLS conversions
        with self._lock:
            for stream_name in list(self._active_streams.keys()):
                self.stop_hls_conversion(stream_name)

        # Stop the monitor thread
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)

        # Clean up all HLS files
        if os.path.exists(self.hls_dir):
            shutil.rmtree(self.hls_dir)
            logger.info("Cleaned up all HLS files")