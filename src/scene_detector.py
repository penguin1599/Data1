from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def find_scenes(video_path, threshold=27.0):
    """
    Detects scenes in a video using PySceneDetect.
    Returns a list of (start_time_seconds, end_time_seconds).
    """
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # Add ContentDetector algorithm (constructor takes threshold as arg).
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    # Base timestamp usually 0
    video_manager.set_downscale_factor()
    video_manager.start()
    
    # Perform scene detection on video_manager.
    scene_manager.detect_scenes(frame_source=video_manager)
    
    # Obtain list of detected scenes.
    scene_list = scene_manager.get_scene_list(base_timecode=video_manager.get_base_timecode())
    
    scenes_in_seconds = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        scenes_in_seconds.append((start_time, end_time))
        
    return scenes_in_seconds
