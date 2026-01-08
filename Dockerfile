FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Disable Python output buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first for cache optimization
COPY requirements.txt .

# Install non-PyTorch dependencies first
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories for mounting
RUN mkdir -p input output src/models/weights

ENTRYPOINT ["python3", "entrypoint.py"]
CMD ["--input_dir", "/app/input", "--output_dir", "/app/output"]
