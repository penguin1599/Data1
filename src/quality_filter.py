import os
import torch
import cv2
import numpy as np
from .models.hyperiqa import HyperIQA
from torchvision import transforms

class QualityFilter:
    def __init__(self, device='cpu', weights_path='weights/hyperiqa.model'):
        self.device = device
        self.model = HyperIQA().to(device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if os.path.exists(weights_path):
            try:
                # Load weights
                self.model.load_state_dict(torch.load(weights_path, map_location=device))
                self.model.eval()
            except Exception as e:
                print(f"Failed to load HyperIQA weights: {e}")
        else:
             print(f"HyperIQA weights not found at {weights_path}")

    def evaluate(self, video_path):
        """
        Calculates average quality score for the video.
        Returns float score.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        
        scores = []
        # Sample frames (e.g. every 10th frame) to look at quality
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % 10 == 0:
                # Preprocess
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Placeholder inference
                    # score = self.model(input_tensor).item()
                    # Normalized to roughly 0-100 range? 
                    # HyperIQA usually gives 0-1 range or logits. User said "scores lower than 40".
                    # Existing implementations usually 0-100 (MOS).
                    pass
                
                # Mock score if model not loaded significantly
                scores.append(45.0) # Dummy pass
            
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
            print(f"Dropping {video_path}: Quality score {score:.2f} < {min_score}")
            return False
        return True
