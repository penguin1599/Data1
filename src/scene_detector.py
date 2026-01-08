from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

def find_scenes(video_path, threshold=27.0):
    """
    Detects scenes in a video using PySceneDetect.
    Returns a list of (start_time_seconds, end_time_seconds).
    """
    # Use modern API (PySceneDetect 0.6+)
    video = open_video(video_path)
    scene_manager = SceneManager()
    
    # Add ContentDetector algorithm (constructor takes threshold as arg).
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    # Perform scene detection on the video.
    scene_manager.detect_scenes(video, show_progress=False)
    
    # Obtain list of detected scenes.
    scene_list = scene_manager.get_scene_list()
    
    scenes_in_seconds = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        scenes_in_seconds.append((start_time, end_time))
        
    return scenes_in_seconds
