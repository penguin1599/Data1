#!/usr/bin/env python3
"""Entrypoint script - detects GPU and installs correct PyTorch version."""

import subprocess
import sys
import os
import urllib.request


def run_cmd(cmd):
    """Run a command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def download_file(url, dest_path):
    """Download a file with progress."""
    print(f"    Downloading to {dest_path}...")

    # Use gdown for Google Drive URLs
    if "drive.google.com" in url:
        try:
            import gdown
        except ImportError:
            print("    Installing gdown for Google Drive downloads...")
            subprocess.run(["pip", "install", "-q", "gdown"])
            import gdown

        try:
            gdown.download(url, dest_path, quiet=False)
            return os.path.exists(dest_path)
        except Exception as e:
            print(f"\n    Download failed: {e}")
            return False

    # Regular download with progress
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r    Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n    Download failed: {e}")
        return False


def download_weights():
    """Download model weights if not present."""
    weights_dir = "/app/src/models/weights"
    os.makedirs(weights_dir, exist_ok=True)

    weights = {
        "syncnet_v2.model": "https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model",
        "hyperiqa.model": "https://drive.google.com/uc?export=download&id=1OOUmnbvpGea0LIGpIWEbOyxfWx6UCiiE"
    }

    print("Checking model weights...")

    for filename, url in weights.items():
        filepath = os.path.join(weights_dir, filename)
        if os.path.exists(filepath):
            print(f"  - {filename}: OK")
        else:
            print(f"  - {filename}: Downloading...")
            if download_file(url, filepath):
                print(f"    Done!")
            else:
                print(f"    WARNING: Failed to download {filename}")


def check_pytorch():
    """Check if PyTorch with CUDA is already working."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch {torch.__version__} with CUDA {torch.version.cuda} already installed!")
            return True
    except ImportError:
        pass
    return False


def get_gpu_info():
    """Get GPU compute capability using nvidia-smi."""
    code, out, err = run_cmd("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
    if code != 0:
        print("ERROR: nvidia-smi not found. Make sure you're running with --gpus all")
        sys.exit(1)

    line = out.split('\n')[0]
    name, compute_cap = line.split(', ')
    major, minor = map(int, compute_cap.split('.'))

    print(f"GPU: {name}")
    print(f"Compute Capability: {compute_cap}")

    return name, major, minor


def install_pytorch(major, minor):
    """Install appropriate PyTorch version based on GPU architecture."""
    print()
    print("=" * 50)

    if major >= 10:
        print("Detected RTX 50 series (Blackwell architecture)")
        print("Installing PyTorch nightly with CUDA 12.8...")
        url = "https://download.pytorch.org/whl/nightly/cu128"
    elif major == 8 and minor >= 9:
        print("Detected RTX 40 series (Ada Lovelace architecture)")
        print("Installing PyTorch stable with CUDA 12.4...")
        url = "https://download.pytorch.org/whl/cu124"
    elif major == 8:
        print("Detected RTX 30 series (Ampere architecture)")
        print("Installing PyTorch stable with CUDA 12.1...")
        url = "https://download.pytorch.org/whl/cu121"
    else:
        print(f"Detected older GPU (compute capability {major}.{minor})")
        print("Installing PyTorch stable with CUDA 11.8...")
        url = "https://download.pytorch.org/whl/cu118"

    print("=" * 50)
    print()

    # Use --pre for nightly builds
    if major >= 10:
        cmd = ["pip", "install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", url]
    else:
        cmd = ["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", url]

    # Run pip with live output (shows download progress)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Installation failed!")
        sys.exit(1)

    # Verify installation
    print()
    print("Verifying PyTorch installation...")
    subprocess.run([sys.executable, "-c", """
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"""])
    print()
    print("PyTorch installation complete!")
    print("=" * 50)


def main():
    print("=" * 50)
    print("GPU Detection and PyTorch Setup")
    print("=" * 50)

    if not check_pytorch():
        name, major, minor = get_gpu_info()
        install_pytorch(major, minor)

    # Download model weights if needed
    print()
    download_weights()

    # Run the pipeline
    print()
    print("Starting pipeline...")
    print()

    args = ["python3", "pipeline.py"] + sys.argv[1:]
    os.execvp("python3", args)


if __name__ == "__main__":
    main()
