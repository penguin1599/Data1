import os
import torch
import cv2
import numpy as np
import logging
from .models.hyperiqa import HyperIQA
from torchvision import transforms

class QualityFilter:
    def __init__(self, device=None, weights_path='weights/hyperiqa.model'):
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.model = HyperIQA()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load weights
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(device)
                self.model.eval()
                self.logger.info(f"HyperIQA weights loaded successfully from {weights_path}")
                self.ready = True
            except Exception as e:
                self.logger.error(f"Failed to load HyperIQA weights: {e}")
                raise RuntimeError(f"HyperIQA failed to initialize: {e}")
        else:
            raise FileNotFoundError(f"HyperIQA weights not found at {weights_path}")

    def evaluate(self, video_path):
        """
        Calculates average quality score for the video.
        Returns float score (0-100 MOS scale).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.warning(f"Could not open video: {video_path}")
            return 0.0
        
        scores = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 10th frame for efficiency
            if frame_idx % 10 == 0:
                try:
                    # Preprocess
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        score = self.model(input_tensor).item()
                        # HyperIQA outputs in 0-100 MOS range
                        scores.append(score)
                except Exception as e:
                    self.logger.warning(f"Error evaluating frame {frame_idx}: {e}")
            
            frame_idx += 1
            
        cap.release()
        
        if not scores:
            return 0.0
            
        return np.mean(scores)

    def process(self, video_path, min_score=40.0):
        """
        Returns True if video quality >= min_score, False otherwise.
        """
        score = self.evaluate(video_path)
        if score < min_score:
            self.logger.info(f"Dropping {video_path}: Quality score {score:.2f} < {min_score}")
            return False
        self.logger.info(f"Passed {video_path}: Quality score {score:.2f} >= {min_score}")
        return True
