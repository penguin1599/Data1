import os
import ffmpeg

def split_video(video_path, output_dir, scenes, min_duration=5.0, max_duration=10.0):
    """
    Splits the video into segments based on scenes.
    Sub-segments large scenes into chunks of max_duration.
    
    Args:
        video_path: Path to source video.
        output_dir: Directory to save segments.
        scenes: List of (start_sec, end_sec).
        min_duration: Minimum duration to keep a clip.
        max_duration: Maximum duration of a clip.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    generated_files = []

    for i, (start, end) in enumerate(scenes):
        duration = end - start
        
        # If scene is too short, skip
        if duration < min_duration:
            continue
        
        current_start = start
        chunk_idx = 0
        
        # Sub-divide the scene if it's too long
        while current_start < end:
            # Determine chunk end
            chunk_end = min(current_start + max_duration, end)
            chunk_dur = chunk_end - current_start
            
            # If the remainder is too small (e.g. < 2s), we might drop it or append to previous?
            # For strictness, if it's less than min_duration, we just skip this tail unless
            # it's the ONLY chunk (which is covered by initial check).
            # But prompt says "Split each video into 5-10 second segments."
            # A simple greedy approach: take 10s chunks. If last chunk < 5s, maybe ignore it.
            
            if chunk_dur < min_duration and chunk_idx > 0:
                # If it's a tail piece shorter than min custom, skip it
                break
            
            # Construct output filename
            out_name = f"{base_name}_scene{i}_chunk{chunk_idx}.mp4"
            out_path = os.path.join(output_dir, out_name)
            
            try:
                (
                    ffmpeg
                    .input(video_path, ss=current_start, t=chunk_dur)
                    .output(out_path, vcodec='libx264', crf=0, preset='slow', audio_bitrate='192k', loglevel="error")
                    .run(overwrite_output=True)
                )
                generated_files.append(out_path)
            except ffmpeg.Error as e:
                print(f"Error splitting {video_path} at {current_start}-{chunk_end}: {e}")
            
            current_start += chunk_dur
            chunk_idx += 1
            
    return generated_files
