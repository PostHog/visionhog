#!/bin/bash

# Configuration
SOURCE_URL="https://wzmedia.dot.ca.gov/D3/5_JCT50_SAC5_NB.stream/playlist.m3u8"
RTMP_URL="rtmp://localhost:1935/live/show"
MAX_RETRIES=0  # 0 means infinite retries
RETRY_DELAY=1  # seconds to wait between retries
LOG_FILE="stream_log.txt"

echo "Starting stream monitor script at $(date)" | tee -a "$LOG_FILE"

retry_count=0
while [ $MAX_RETRIES -eq 0 ] || [ $retry_count -lt $MAX_RETRIES ]; do
    echo "$(date) - Starting FFmpeg stream (attempt $((retry_count+1)))" | tee -a "$LOG_FILE"

    # Run FFmpeg with all your parameters and capture output
    output=$(ffmpeg -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 \
        -i "$SOURCE_URL" \
        -c:v libx264 -preset veryfast -b:v 500k -maxrate 500k -bufsize 1000k \
        -s 640x360 -r 15 -c:a aac -b:a 64k \
        -f flv "$RTMP_URL" 2>&1)

    # Save the output to log file
    echo "$output" | tee -a "$LOG_FILE"

    exit_code=$?

    # Check for common error patterns in the output
    if echo "$output" | grep -q "Error\|Failed\|Broken pipe\|Conversion failed"; then
        echo "$(date) - FFmpeg failed with errors in output, restarting in $RETRY_DELAY seconds..." | tee -a "$LOG_FILE"
        sleep $RETRY_DELAY
        retry_count=$((retry_count+1))
        continue
    fi

    if [ $exit_code -eq 0 ]; then
        echo "$(date) - FFmpeg exited normally with code $exit_code" | tee -a "$LOG_FILE"
        break
    else
        echo "$(date) - FFmpeg exited with code $exit_code, restarting in $RETRY_DELAY seconds..." | tee -a "$LOG_FILE"
        sleep $RETRY_DELAY
        retry_count=$((retry_count+1))
    fi
done

if [ $MAX_RETRIES -ne 0 ] && [ $retry_count -ge $MAX_RETRIES ]; then
    echo "$(date) - Maximum retry attempts ($MAX_RETRIES) reached, giving up." | tee -a "$LOG_FILE"
fi