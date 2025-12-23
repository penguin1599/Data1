import os
import argparse
import glob
import yaml
import logging
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import modules
from src.cleaner import clean_file
from src.resampler import resample_video
from src.scene_detector import find_scenes
from src.segmenter import split_video
from src.face_processor import FaceProcessor
from src.sync_filter import SyncFilter
from src.quality_filter import QualityFilter
from src.speaker_organizer import SpeakerOrganizer

# Global dictionary to hold models in worker processes
models = {}

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'pipeline.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def init_worker(config):
    """
    Initialize models for each worker process.
    This prevents re-initializing large models for every video and allows distinct CUDA contexts?
    Note: CUDA with multiprocessing can be tricky. 'spawn' start method is usually required.
    But for simplicity in this script, we'll try standard initialization.
    If using CPU (defaults often), this is fine.
    """
    global models
    print(f"Initializing worker models (Process {os.getpid()})...")
    
    # Initialize models
    # Device selection logic could be improved (e.g., round-robin info from main?)
    # For now, let FaceProcessor detect.
    models['face_processor'] = FaceProcessor(output_size=config['pipeline']['face_size'])
    models['sync_filter'] = SyncFilter() # Device auto-detect usually 'cpu' unless specified
    models['quality_filter'] = QualityFilter()
    models['speaker_organizer'] = SpeakerOrganizer() # This might need to be shared or done at end? 
    # SpeakerOrganizer clusters embeddings. Doing this per process means we just collect embeddings 
    # and "organize" later? 
    # Actually, SpeakerOrganizer.add_clip just appends to internal list? 
    # No, we'd need to return the (embedding, path) tuples to the main process for clustering!
    # The original pipeline added to organizer, then called .organize() at the end.
    # In parallel, we should return the organizer data.

def preprocess_video(video_path, steps_dirs, config):
    """
    Phase 1: Pre-process video into segments.
    Returns a list of segment paths.
    """
    logger = logging.getLogger(__name__)
    base_name = os.path.basename(video_path)
    
    # Step 1: Broken File Check
    if not clean_file(video_path):
        logger.warning(f"File broken/invalid: {video_path}")
        return []

    # Step 2: Resample
    resampled_path = os.path.join(steps_dirs['resampled'], base_name)
    if not os.path.exists(resampled_path):
        if not resample_video(video_path, resampled_path, 
                              target_fps=config['pipeline']['target_fps'], 
                              target_sr=config['pipeline']['target_sample_rate']):
            logger.error(f"Resampling failed: {video_path}")
            return []
    
    # Step 3: Scene Detect
    import ffmpeg
    scenes = find_scenes(resampled_path)
    if not scenes:
        logger.info(f"No scenes found in {base_name}, assuming single scene.")
        try:
            probe = ffmpeg.probe(resampled_path)
            duration = float(probe['format']['duration'])
            scenes = [(0, duration)]
        except Exception as e:
            logger.error(f"Failed to probe {resampled_path}: {e}")
            return []

    # Step 4: Segmentation
    # This writes files to steps_dirs['segmented']
    video_segments = split_video(resampled_path, steps_dirs['segmented'], scenes, 
                                 min_duration=config['pipeline']['min_duration'], 
                                 max_duration=config['pipeline']['max_duration'])
    
    return video_segments

def process_segment(segment_path, steps_dirs, config):
    """
    Phase 2: Process a single segment (Face -> Sync -> Quality).
    Returns (final_path, embedding) or None.
    """
    global models
    logger = logging.getLogger(__name__)
    seg_name = os.path.basename(segment_path)
    final_path = os.path.join(steps_dirs['final'], seg_name)
    
    # Resume Check: if final output AND embedding exist
    npy_path = os.path.splitext(final_path)[0] + '.npy'
    
    if os.path.exists(final_path) and os.path.exists(npy_path):
        logger.info(f"Skipping existing final clip (embedding found): {final_path}")
        try:
            embedding = np.load(npy_path)
            return (final_path, embedding)
        except Exception as e:
            logger.warning(f"Failed to load embedding for {final_path}, re-processing: {e}")
            # fall through
    
    # Step 5: Face Processing
    face_out_path = os.path.join(steps_dirs['faces'], seg_name)
    embedding = models['face_processor'].process_video(segment_path, face_out_path)
    
    if embedding is None:
        return None
    
    # Step 6: Sync Filter
    is_synced = models['sync_filter'].process(face_out_path, final_path, 
                                             min_conf=config['pipeline']['sync_threshold'])
    if not is_synced:
        if os.path.exists(final_path):
            os.remove(final_path)
        return None
    
    # Step 7: Quality Filter
    if not models['quality_filter'].process(final_path, min_score=config['pipeline']['quality_threshold']):
        if os.path.exists(final_path):
            os.remove(final_path)
        return None
        
    # SAVE EMBEDDING
    try:
        np.save(npy_path, embedding)
    except Exception as e:
        logger.error(f"Failed to save embedding to {npy_path}: {e}")

    return (final_path, embedding)

def main(args):
    # Load config
    config = load_config(args.config) if args.config else load_config('config.yaml')
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Setup directories
    steps_dirs = {
        'resampled': os.path.join(output_dir, 'step1_resampled'),
        'segmented': os.path.join(output_dir, 'step2_segmented'),
        'faces': os.path.join(output_dir, 'step3_faces'),
        'final': os.path.join(output_dir, 'final_set')
    }
    
    for d in steps_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Setup Logging
    logger = setup_logging(output_dir)
    logger.info("Pipeline started.")
    
    # Gather files
    video_files = glob.glob(os.path.join(input_dir, '*.mp4')) + \
                  glob.glob(os.path.join(input_dir, '*.avi')) + \
                  glob.glob(os.path.join(input_dir, '*.mov')) + \
                  glob.glob(os.path.join(input_dir, '*.mkv'))
                  
    logger.info(f"Found {len(video_files)} videos in {input_dir}")

    # Initialize Organizer (Main Process)
    speaker_organizer = SpeakerOrganizer() # We'll feed it results from workers

    # Parallel Processing
    max_workers = args.workers if args.workers else max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Starting processing with {max_workers} workers.")
    
    from functools import partial
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(config,)) as executor:
        
        # --- PHASE 1: PRE-PROCESSING (Video -> Segments) ---
        logger.info("PHASE 1: Pre-processing videos to generate segments...")
        preprocess_func = partial(preprocess_video, steps_dirs=steps_dirs, config=config)
        
        # This is usually fast (IO bound mostly), but let's run it.
        # Note: preprocess_video returns a LIST of paths.
        all_segments = []
        
        # We can run this in parallel too if multiple input videos
        futures_prep = [executor.submit(preprocess_func, v) for v in video_files]
        
        for future in tqdm(as_completed(futures_prep), total=len(video_files), desc="Pre-processing Videos"):
            try:
                segs = future.result()
                all_segments.extend(segs)
            except Exception as e:
                logger.error(f"Pre-processing failed for a video: {e}")

        logger.info(f"PHASE 1 Complete. Found {len(all_segments)} total segments to process.")

        # --- PHASE 2: PROCESSING SEGMENTS (Segment -> Final) ---
        if not all_segments:
            logger.info("No segments to process. Exiting.")
            return

        logger.info(f"PHASE 2: Processing {len(all_segments)} segments in parallel...")
        
        process_func = partial(process_segment, steps_dirs=steps_dirs, config=config)
        
        # Submit all segments
        futures_proc = [executor.submit(process_func, seg) for seg in all_segments]
        
        for future in tqdm(as_completed(futures_proc), total=len(all_segments), desc="Processing Segments"):
            try:
                result = future.result()
                if result:
                    path, emb = result
                    speaker_organizer.add_clip(path, emb)
            except Exception as e:
                logger.error(f"Segment processing failed: {e}")

    logger.info("Pipeline processing complete. Organizing by speaker...")
    speaker_organizer.organize(steps_dirs['final'])
    logger.info("Organization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument('--input_dir', type=str, required=True, help="Input directory containing raw videos")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for processed data")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to config file")
    parser.add_argument('--workers', type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()
    main(args)
