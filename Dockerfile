FROM python:3.9-slim

# Install system dependencies
# ffmpeg for video processing
# libgl1-mesa-glx and libglib2.0-0 for opencv
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for cache optimization
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories for mounting
RUN mkdir -p input output weights

# Default command matches instructions, but expects volume mounts
ENTRYPOINT ["python3", "pipeline.py"]
CMD ["--input_dir", "/app/input", "--output_dir", "/app/output"]
