FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create directories for video storage
RUN mkdir -p video_clips processed_clips

# Run the application
CMD ["python", "-m", "visionhog.main"]