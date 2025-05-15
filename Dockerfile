FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire application first
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p video_clips processed_clips

# Expose the port
EXPOSE 8069

# Run the application
CMD ["python", "-m", "visionhog.main"]