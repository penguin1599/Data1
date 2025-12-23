import cv2
import numpy as np
import insightface
import ffmpeg
from insightface.app import FaceAnalysis

class FaceProcessor:
    def __init__(self, output_size=256, device='cuda'):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.output_size = output_size

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Could not open {input_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup FFmpeg process for lossless output
        # Map audio from original file (input_path) and video from pipe
        audio_in = ffmpeg.input(input_path).audio
        video_in = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{self.output_size}x{self.output_size}', r=fps)
        
        process = (
            ffmpeg
            .output(video_in, audio_in, output_path, vcodec='libx264', preset='slow', crf=0, pix_fmt='yuv420p', acodec='copy')
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        frame_count = 0
        success_count = 0
        embeddings = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                faces = self.app.get(frame)
                
                if len(faces) == 0:
                    # Fill with black if no face found to keep sync
                    blank = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
                    process.stdin.write(blank.tobytes())
                    continue

                largest_face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)[0]
                
                # Store embedding
                if largest_face.embedding is not None:
                     embeddings.append(largest_face.embedding)

                kps = largest_face.kps
                
                src1 = np.array([
                    [38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.9658, 92.2041]
                ], dtype=np.float32)
                
                scale = self.output_size / 112.0
                src_target = src1 * scale
                
                M, _ = cv2.estimateAffinePartial2D(kps, src_target, method=cv2.LMEDS)
                
                if M is None:
                    blank = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
                    process.stdin.write(blank.tobytes())
                    continue

                warped = cv2.warpAffine(frame, M, (self.output_size, self.output_size), borderValue=0.0)
                process.stdin.write(warped.tobytes())
                success_count += 1
                
        finally:
            cap.release()
            process.stdin.close()
            process.wait()
        
        if success_count == 0 and frame_count > 0:
            return None
        
        # Return average embedding if available
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            return avg_embedding
        return None

