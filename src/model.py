# src/model.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    2-Layer Residual Block with an identity skip connection: Output = ReLU(F(x) + x)
    Solves the vanishing gradient problem by creating an unobstructed highway for backprop.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Layer A: 3x3 Conv maintaining shape
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Layer B: 3x3 Conv maintaining shape
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # Store input for the skip highway
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Add the original input directly
        return self.relu(out)


class MiniResNet(nn.Module):
    """
    7-layer Mini-ResNet (strictly within the <= 10 layer limit).
    Uses Global Average Pooling to avoid dense parameter explosion.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Layer 1: Stem feature extractor (3 -> 32, stays 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Downsampling layer (32 -> 64, shrinks 32x32 to 16x16)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Layers 3 & 4: Residual block (refines features at 16x16, channels stay 64)
        self.res_block = ResidualBlock(channels=64)
        
        # Layer 5: High-level downsampling (64 -> 128, shrinks 16x16 to 8x8)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Layer 6: Global Average Pooling (averages 8x8 spatial grid to 1x1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Layer 7: Final classification projection (128 -> 10 classes)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.downsample1(x)
        x = self.res_block(x)
        x = self.downsample2(x)
        x = self.gap(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    print("Testing MiniResNet architecture...")
    model = MiniResNet(num_classes=10)
    
    # Pass a dummy batch of 4 images through the network
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Expected: [4, 10])")
    print(f"Total Trainable Parameters: {total_params:,}")
    print("Architecture verified successfully.")