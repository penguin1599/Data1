import ffmpeg
import logging

def resample_video(input_path, output_path, target_fps=25, target_sr=16000):
    """
    Resample video to target FPS and audio sample rate using FFmpeg.
    
    Args:
        input_path (str): Path to input video file.
        output_path (str): Path to save resampled video.
        target_fps (int): Target frames per second.
        target_sr (int): Target audio sample rate (Hz).
        
    Returns:
        bool: True if successful, False otherwise.
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Construct FFmpeg stream
        # -r sets frame rate
        # -ar sets audio sample rate
        # -ac 1 sets audio to mono (common for speech processing)
        # -y overwrites output (handled by overwrite_output() method)
        
        job = (
            ffmpeg
            .input(input_path)
            .output(output_path, r=target_fps, ar=target_sr, ac=1, loglevel="error")
            .overwrite_output()
        )
        
        job.run(capture_stdout=True, capture_stderr=True)
        return True
        
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode('utf8') if e.stderr else "Unknown FFmpeg error"
        logger.error(f"FFmpeg error processing {input_path}:\n{error_msg}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error resampling {input_path}: {str(e)}")
        return False
