
import torch
import os

weights_path = 'src/models/weights/hyperiqa.model'
if not os.path.exists(weights_path):
    print(f"Weights not found at {weights_path}")
    exit(1)

try:
    checkpoint = torch.load(weights_path, map_location='cpu')
    print("Keys found in checkpoint:")
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        keys = list(checkpoint['state_dict'].keys())
    elif isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
    else:
        print("Checkpoint is not a dict")
        keys = []
        
    with open('hyperiqa_keys.txt', 'w') as f:
        for k in keys: # Print ALL keys
            f.write(k + '\n')
    print("Keys written to hyperiqa_keys.txt")
except Exception as e:
    print(f"Error loading: {e}")
