import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.nn.init as init

class Upconv(nn.Module):
    """Upsampling block with upsample + convolution"""
    def __init__(self, in_channels, out_channels):
        super(Upconv, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsampling
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),  # Convolution
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.upconv(x)

class MonoDepth2Decoder(nn.Module):
    def __init__(self, num_ch_enc, scales=[0, 1, 2, 3], num_output_channels=1):
        super(MonoDepth2Decoder, self).__init__()

        # Initialize the decoder layers for each scale
        self.num_output_channels = num_output_channels
        self.scales = scales
        self.num_ch_enc = num_ch_enc

        # Number of channels for each layer in the decoder
        self.num_ch_dec = [16, 32, 64, 128, 256]

        # Upsampling layers (MonoDepth2 style)
        self.upconv_4 = Upconv(self.num_ch_enc[-1], self.num_ch_dec[4])  # Upsample from final encoder layer
        self.upconv_3 = Upconv(self.num_ch_dec[4], self.num_ch_dec[3])
        self.upconv_2 = Upconv(self.num_ch_dec[3], self.num_ch_dec[2])
        self.upconv_1 = Upconv(self.num_ch_dec[2], self.num_ch_dec[1])
        self.upconv_0 = Upconv(self.num_ch_dec[1], self.num_ch_dec[0])

        # Output layer for predicting the depth map
        self.dispconv_3 = nn.Conv2d(self.num_ch_dec[3], self.num_output_channels, kernel_size=3, stride=1, padding=1)
        self.dispconv_2 = nn.Conv2d(self.num_ch_dec[2], self.num_output_channels, kernel_size=3, stride=1, padding=1)
        self.dispconv_1 = nn.Conv2d(self.num_ch_dec[1], self.num_output_channels, kernel_size=3, stride=1, padding=1)
        self.dispconv_0 = nn.Conv2d(self.num_ch_dec[0], self.num_output_channels, kernel_size=3, stride=1, padding=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        """
        input_features: List of feature maps from the encoder, from lowest to highest resolution.
        """
        x = input_features[-1]  # Start from the deepest feature map
        x = self.upconv_4(x)

        x = self.upconv_3(x)
        disp3 = self.dispconv_3(x)

        x = self.upconv_2(x)
        disp2 = self.dispconv_2(x)

        x = self.upconv_1(x)
        disp1 = self.dispconv_1(x)

        x = self.upconv_0(x)
        disp0 = self.dispconv_0(x)

        disp3 = self.sigmoid(disp3)
        disp2 = self.sigmoid(disp2)
        disp1 = self.sigmoid(disp1)
        disp0 = self.sigmoid(disp0)

        return [disp0, disp1, disp2, disp3]  # Return depth predictions at multiple scales

class DepthEstimationModel(nn.Module):
    def __init__(self, resnet):
        super(DepthEstimationModel, self).__init__()

        # Load pretrained ResNet18 model
        #resnet = models.resnet18(pretrained=True)
        
        # Use all layers except the fully connected layer
        self.encoder = resnet  # Exclude avgpool and FC layer

        # Define the number of channels for each block in the encoder (ResNet18 specific)
        self.num_ch_enc = [64, 64, 128, 256, 512]

        # Initialize MonoDepth2 decoder with ResNet encoder channels
        self.decoder = MonoDepth2Decoder(self.num_ch_enc)
        self.initialize_weights_he(self.decoder)

    def initialize_weights_he(self, model):
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: Input image tensor of shape (B, C, H, W)
        Returns:
            List of depth predictions at multiple scales
        """
        # Encoder
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            features.append(x)

        # Decoder (MonoDepth2 style)
        depth_outputs = self.decoder(features)

        return depth_outputs  # Return depth predictions at multiple scales