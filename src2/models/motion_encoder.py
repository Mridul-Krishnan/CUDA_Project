import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionEncoder(nn.Module):
    def __init__(self, resnet, input_channels=1024):
        super(MotionEncoder, self).__init__()
        self.imageEncoder = resnet
        self.conv1 = nn.Conv2d(input_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, images, targets):

        h_t = self.imageEncoder(images)
        h_t1 = self.imageEncoder(targets)
        # Concatenate feature maps from two consecutive frames along the channel dimension
        combined_features = torch.cat([h_t, h_t1], dim=1)  # Shape: [B, 1024, H, W]
        
        # Pass through convolutional layers with batch normalization and ReLU activation
        x = F.relu(self.bn1(self.conv1(combined_features)))
        x = F.relu(self.bn2(self.conv2(x)))
        motion_features = F.relu(self.bn3(self.conv3(x)))
        
        return motion_features  # Shape: [B, 128, H, W]
