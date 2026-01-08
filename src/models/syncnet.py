import torch
import torch.nn as nn

class SyncNet(nn.Module):
    """
    SyncNet model for audio-visual sync detection.
    Architecture matches the SyncNet_v2 checkpoint exactly.
    Based on: https://github.com/joonson/syncnet_python
    
    The layer indices must match the checkpoint exactly:
    - Audio: 0-10 (first 3 conv blocks), 11-16 (last 2 conv blocks), 17-20 (final block)
    - No extra MaxPool layers between conv blocks 3-4-5
    """
    def __init__(self):
        super(SyncNet, self).__init__()
        
        # Audio Encoder - layer indices MUST match checkpoint exactly
        # Checkpoint has: 0,1,2,3, 4,5,6,7, 8,9,10, 11,12,13, 14,15,16,17, 18,19,20
        self.netcnnaud = nn.Sequential(
            # Block 1: indices 0-3
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),      # 0
            nn.BatchNorm2d(64),                                                         # 1
            nn.ReLU(inplace=True),                                                      # 2
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),                           # 3
            
            # Block 2: indices 4-7
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),    # 4
            nn.BatchNorm2d(192),                                                        # 5
            nn.ReLU(inplace=True),                                                      # 6
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),                           # 7
            
            # Block 3: indices 8-10 (NO MaxPool after this!)
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),   # 8
            nn.BatchNorm2d(384),                                                        # 9
            nn.ReLU(inplace=True),                                                      # 10
            
            # Block 4: indices 11-13 (NO MaxPool after this!)
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),   # 11
            nn.BatchNorm2d(256),                                                        # 12
            nn.ReLU(inplace=True),                                                      # 13
            
            # Block 5: indices 14-17
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),   # 14
            nn.BatchNorm2d(256),                                                        # 15
            nn.ReLU(inplace=True),                                                      # 16
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),                           # 17
            
            # Block 6 (final): indices 18-20
            nn.Conv2d(256, 512, kernel_size=(5, 4), stride=(1, 1), padding=(0, 0)),   # 18
            nn.BatchNorm2d(512),                                                        # 19
            nn.ReLU(inplace=True),                                                      # 20
        )

        # Video/Lip Encoder - uses 3 input channels (RGB)
        self.netcnnlip = nn.Sequential(
            # Block 1: indices 0-3
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            # Block 2: indices 4-7
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Block 3: indices 8-10
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Block 4: indices 11-13
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Block 5: indices 14-17
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            
            # Block 6 (final): indices 18-20
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        # FC layers for audio
        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
        )
        
        # FC layers for lip/video
        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
        )

    def forward_aud(self, x):
        x = self.netcnnaud(x)
        x = x.view(x.size(0), -1)
        x = self.netfcaud(x)
        return x

    def forward_lip(self, x):
        x = self.netcnnlip(x)
        x = x.view(x.size(0), -1)
        x = self.netfclip(x)
        return x

    def forward(self, audio, lip):
        a = self.forward_aud(audio)
        v = self.forward_lip(lip)
        return a, v
