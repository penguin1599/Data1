# Video Processing Pipeline

This repository contains a robust data processing pipeline designed to create high-quality audio-visual datasets from raw video collections. It is specifically optimized for generating talking-head datasets with consistent framing, synchronization, and quality.

## Features

- **Automated Pipeline**: End-to-end processing from raw video to organized clips.
- **Robustness**: Handles broken files and standardizes video/audio formats (25 fps, 16kHz).
- **Scene Detection**: Automatically splits videos at scene changes to prevent cutting mid-shot.
- **Smart Segmentation**: Generates training-ready clips (5-10 seconds).
- **Face Processing**: Uses **InsightFace** for high-accuracy face detection, alignment, and cropping (256x256).
- **Sync Validation**: Incorporates **SyncNet** logic to verify and correct audio-visual synchronization.
- **Containerized**: Fully Dockerized for consistent execution across environments.

## Project Structure

```
├── Dockerfile              # Docker environment definition
├── pipeline.py             # Main entry point script
├── requirements.txt        # Python dependencies
├── src/                    # Processing modules
│   ├── cleaner.py          # File integrity checks
│   ├── face_processor.py   # Face detection & cropping (InsightFace)
│   ├── models/             # ML model weights
│   ├── quality_filter.py   # Visual quality assessment
│   ├── resampler.py        # FFmpeg standardization
│   ├── scene_detector.py   # Scene change detection
│   ├── segmenter.py        # Video segmentation logic
│   ├── speaker_organizer.py# Clustering/organizing output
│   └── sync_filter.py      # AV sync verification
├── weights/                # Directory for model weights
├── input/                  # Raw input videos (mounted)
└── output/                 # Processed results (mounted)
```

## Quick Start (Docker)

The easiest way to run the pipeline is using Docker.

### 1. Build the Image
```bash
docker build -t dataprocess .
# or use the split build task in VS Code
```

### 2. Run the Pipeline
Mount your local input and output directories:

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/weights:/app/src/models/weights \
  --gpus all \
  dataprocess \
  --input_dir /app/input \
  --output_dir /app/output
```

> **Note**: Requires NVIDIA Container Toolkit for GPU acceleration (`--gpus all`). If running on CPU, omit the flag (performance will be lower).

## Local Development

### Prerequisites
- Python 3.9+
- FFmpeg installed and in PATH
- CUDA 11.x+ (recommended)

### Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
python pipeline.py --input_dir ./input --output_dir ./output
```

## IDE Support
This project includes a `.vscode/tasks.json` configuration for running Docker builds directly from VS Code.
