import os
import shutil
import cv2
import ffmpeg

def is_video_valid(file_path):
    """
    Checks if a video file is valid by attempting to open it with OpenCV
    and probing it with FFmpeg.
    """
    if not os.path.exists(file_path):
        return False

    # Check with OpenCV
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False
    cap.release()

    # Check with FFmpeg probe
    try:
        ffmpeg.probe(file_path)
    except ffmpeg.Error:
        return False

    return True

def clean_file(file_path):
    """
    Removes the file if it is not valid.
    Returns True if file was valid (kept), False if invalid (removed).
    """
    if is_video_valid(file_path):
        return True
    
    broken_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'broken_files')
    os.makedirs(broken_dir, exist_ok=True)
    
    print(f"Moving broken file to: {broken_dir}")
    try:
        file_name = os.path.basename(file_path)
        shutil.move(file_path, os.path.join(broken_dir, file_name))
    except OSError as e:
        print(f"Error moving {file_path}: {e}")
    return False
