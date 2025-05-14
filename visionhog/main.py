import ffmpeg
import os
import time
import datetime
import threading
import queue
import shutil
from pathlib import Path
from google import genai
from google.genai import types

# Configuration
RTMP_URL = "http://localhost:8080/live/show.flv"  # HTTP FLV stream endpoint
OUTPUT_DIR = Path("video_clips")
PROCESSED_DIR = Path("processed_clips")  # For clips that have been analyzed
MAX_CLIPS_TO_KEEP = 100  # Maximum number of clips to store
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHUNK_DURATION = 10  # Duration of each clip in seconds

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Set up Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# Create a queue for processing clips
clip_queue = queue.Queue()
# Create a flag for signaling threads to exit
exit_flag = threading.Event()

def analyze_with_gemini(video_path):
    """Send video to Gemini for analysis and return results"""
    try:
        # Read the video as bytes
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        # Send to Gemini for analysis using new API format
        response = client.models.generate_content(
            model='models/gemini-2.0-flash',
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ),
                    types.Part(text='Analyze this video clip and describe what\'s happening.')
                ]
            )
        )

        return response.text
    except Exception as e:
        print(f"Error analyzing with Gemini: {e}")
        return f"Analysis failed: {str(e)}"

def process_clip_worker():
    """Worker function to process clips in the queue"""
    print("Clip processing worker started")

    while not exit_flag.is_set() or not clip_queue.empty():
        try:
            # Get clip from queue with timeout to check exit flag periodically
            try:
                clip_path = clip_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            print(f"Processing clip: {clip_path}")

            # Analyze with Gemini
            start_time = time.time()
            analysis = analyze_with_gemini(clip_path)
            processing_time = time.time() - start_time

            # Save results
            results_path = clip_path.with_suffix('.txt')
            with open(results_path, 'w') as f:
                f.write(f"Analysis of {clip_path.name}:\n")
                f.write(f"Processed at: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Processing time: {processing_time:.2f} seconds\n\n")
                f.write(analysis)

            print(f"Analysis saved to {results_path}")

            # Move to processed directory
            dest_path = PROCESSED_DIR / clip_path.name
            shutil.move(str(clip_path), str(dest_path))

            # Also move the analysis file
            if results_path.exists():
                shutil.move(str(results_path), str(PROCESSED_DIR / results_path.name))

            clip_queue.task_done()

        except Exception as e:
            print(f"Error in processing worker: {e}")

    print("Clip processing worker stopped")

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

            print(f"Capturing chunk {chunk_count + 1} to {output_path}...")

            # Use FFmpeg to capture a segment of the stream
            try:
                # Configure input for HTTP FLV stream with appropriate options
                stream = ffmpeg.input(
                    RTMP_URL,
                    t=CHUNK_DURATION,
                    f='flv',
                    re=None,
                    timeout=5000000,  # Increase timeout
                    headers='User-Agent: Mozilla/5.0\r\nAccept: */*\r\nConnection: keep-alive\r\n'  # Add headers
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
                    print(f"FFmpeg stderr output: {e.stderr.decode() if e.stderr else 'No stderr output'}")
                    raise

                # Calculate actual capture time
                elapsed = time.time() - start_time
                print(f"Chunk captured in {elapsed:.2f} seconds")

                # Check if file was created and has content
                if output_path.exists() and output_path.stat().st_size > 0:
                    print(f"Successfully saved chunk to {output_path} ({output_path.stat().st_size} bytes)")
                    chunk_count += 1

                    # Add to processing queue
                    clip_queue.put(output_path)

                    # Perform cleanup every 10 chunks or once per hour
                    if chunk_count % 10 == 0 or time.time() - last_cleanup > 3600:
                        cleanup_old_clips()
                        last_cleanup = time.time()
                else:
                    print(f"Warning: Failed to capture chunk or empty file created")

                # Calculate time adjustment to maintain precise intervals
                time_adjustment = max(0, CHUNK_DURATION - elapsed)
                if time_adjustment > 0:
                    print(f"Waiting {time_adjustment:.2f} seconds to align with {CHUNK_DURATION}-second intervals...")

                    # Use small sleep increments to check exit flag
                    end_wait = time.time() + time_adjustment
                    while time.time() < end_wait and not exit_flag.is_set():
                        time.sleep(0.1)

            except ffmpeg.Error as e:
                print(f"FFmpeg error: {e}")
                # Short pause before retry
                time.sleep(1)

            except Exception as e:
                print(f"Error during capture: {e}")
                # Short pause before retry
                time.sleep(1)

            # Check exit flag
            if exit_flag.is_set():
                break

    except KeyboardInterrupt:
        print("Capturing stopped by user")
    finally:
        print("Capture process ending, waiting for queue to empty...")
        # Wait for queue to be processed
        clip_queue.join()

def main():
    print(f"Starting to capture {CHUNK_DURATION}-second chunks from {RTMP_URL}")
    print(f"Saving clips to {OUTPUT_DIR.absolute()}")
    print(f"Processed clips will be moved to {PROCESSED_DIR.absolute()}")
    print("Press Ctrl+C to stop capturing")

    try:
        # Start processing thread
        processing_thread = threading.Thread(target=process_clip_worker)
        processing_thread.daemon = True
        processing_thread.start()

        # Start capturing chunks
        capture_stream_chunks()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Signal threads to exit
        exit_flag.set()

        # Wait for processing thread to finish
        if 'processing_thread' in locals():
            processing_thread.join(timeout=10)

        print("Program terminated")

if __name__ == "__main__":
    main()