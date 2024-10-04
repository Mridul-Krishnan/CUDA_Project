import torch
import torch.nn as nn
import torch.nn.functional as F

class EgoMotionModel(nn.Module):
    def __init__(self, MotionEncoder):
        super(EgoMotionModel, self).__init__()
        
        self.encoder = MotionEncoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1), # Shape: [B, 64, H, W]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 6, kernel_size=1)  # Output: 6D pose (3 for rotation, 3 for translation)

        )
        
        
    def forward(self, images, targets):
        motion_features = self.encoder(images, targets)
        # Process the motion features through convolutional layers
        # Predict the 6D pose (global average pooling to get 6D output)
        pose = self.decoder(motion_features)  # Shape: [B, 6, H, W]

        pose = pose.mean([2, 3])  # Global average pooling over H, W to get shape [B, 6]
        
        return pose  # Shape: [B, 6] (6D vector for rotation and translation)
