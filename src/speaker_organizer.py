import os
import shutil
import numpy as np

class SpeakerOrganizer:
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.clusters = [] # List of {'embedding': avg_emb, 'files': [paths]}

    def compute_similarity(self, emb1, emb2):
        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(emb1, emb2) / (norm1 * norm2)

    def add_clip(self, file_path, embedding):
        if embedding is None:
            # Cannot cluster without embedding, put in "unknown"?
            self.add_to_cluster(file_path, -1) # -1 ID for unknown
            return

        # Find best matching cluster
        best_idx = -1
        best_sim = -1.0
        
        for i, cluster in enumerate(self.clusters):
            sim = self.compute_similarity(embedding, cluster['embedding'])
            if sim > self.threshold and sim > best_sim:
                best_sim = sim
                best_idx = i
        
        if best_idx != -1:
            # Update cluster
            cluster = self.clusters[best_idx]
            cluster['files'].append(file_path)
            # Update average embedding
            n = len(cluster['files'])
            # Moving average: new_avg = (old_sum + new_emb) / n = (old_avg * (n-1) + new_emb) / n
            cluster['embedding'] = (cluster['embedding'] * (n - 1) + embedding) / n
        else:
            # Create new cluster
            self.clusters.append({
                'embedding': embedding,
                'files': [file_path]
            })

    def add_to_cluster(self, file_path, cluster_id):
        # Handle special cases if needed, but for now just skip logic for unknown
        pass

    def organize(self, output_dir):
        """
        Moves files into speaker_X folders.
        """
        print(f"Organizing {len(self.clusters)} identified speakers...")
        
        for i, cluster in enumerate(self.clusters):
            speaker_dir = os.path.join(output_dir, f"speaker_{i+1}")
            os.makedirs(speaker_dir, exist_ok=True)
            
            for file_path in cluster['files']:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    target = os.path.join(speaker_dir, file_name)
                    shutil.move(file_path, target)
                else:
                    print(f"Warning: File {file_path} not found during organization")
                    
        # Optional: Handle unknowns if I stored them separately
