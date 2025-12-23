import torch
import torch.nn as nn

class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()
        # Simplified SyncNet architecture for reference.
        # Ensure actual weights match this structure.
        
        # Audio Encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )

        # Video Encoder (Lip reading)
        self.face_encoder = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), # Note: Architecture varies by implementation
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )
        
        self.fc_audio = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 512))
        self.fc_face = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 512))

    def forward(self, audio, face):
        # Audio: [B, 1, 13, 20] (MFCC) -> [B, 512, 1, 1] -> flat
        # Face: [B, 15, H/2, W/2] (Video 5 frames stacked channels 3*5=15 grayscale? or 5*3 RGB?)
        # Usually SyncNet uses 5 grayscale frames.
        
        a = self.audio_encoder(audio)
        a = a.view(a.size(0), -1)
        a = self.fc_audio(a)

        v = self.face_encoder(face)
        v = v.view(v.size(0), -1)
        v = self.fc_face(v)
        
        return a, v
