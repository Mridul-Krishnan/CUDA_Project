import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlowModel(nn.Module):
    def __init__(self, MotionEncoder, input_channels=128, output_size=(256, 512)):
        super(OpticalFlowModel, self).__init__()
        self.output_size = output_size
        self.encoder = MotionEncoder
        # Define convolutional layers for refining the optical flow
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),  # conv1
            nn.ReLU(),                                                  # Activation after conv1
            nn.Conv2d(256, 128, kernel_size=3, padding=1),           # conv2
            nn.ReLU(),                                                  # Activation after conv2
            nn.Conv2d(128, 96, kernel_size=3, padding=1),            # conv3
            nn.ReLU(),                                                  # Activation after conv3
            nn.Conv2d(96, 64, kernel_size=3, padding=1),             # conv4
            nn.ReLU(),                                                  # Activation after conv4
            nn.Conv2d(64, 32, kernel_size=3, padding=1),              # conv5 
            # Final layer to predict optical flow (2 channels: horizontal and vertical flow)
            nn.Conv2d(32, 2, kernel_size=3, padding=1)  # Output is a 2-channel flow map 
        )
        

        # Upsampling layers to refine optical flow at different stages
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, images, targets):
        motion_features = self.encoder(images, targets)
        # Process the input features through a series of convolutional layers
        # Predict the optical flow (output has 2 channels: one for horizontal and one for vertical flow)
        flow = self.decoder(motion_features)
        
        # upsample the predicted flow to match the original input resolution
        flow_upsampled = F.interpolate(flow, size=self.output_size, mode='bilinear', align_corners=True)

        return flow_upsampled  
