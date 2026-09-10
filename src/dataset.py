# src/dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Image Statistics for CIFAR-10 (Pre-calculated across all 50,000 images)
# These represent the average brightness and spread of Red, Green, and Blue channels.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_transforms():
    """
    Defines the transformation pipeline applied to every image before it enters the GPU.
    """
    # Pipeline for Training Images: includes Augmentation
    train_transform = transforms.Compose([
        # Augmentation 1: Randomly flip the image left-to-right (50% probability)
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Augmentation 2: Add 4 pixels of padding around image, then randomly crop 32x32 back out
        transforms.RandomCrop(32, padding=4),
        
        # Convert raw pixel values (0 to 255) into PyTorch Tensors (0.0 to 1.0)
        transforms.ToTensor(),
        
        # Normalize: (pixel - mean) / std so numbers center around 0.0 with a spread of 1.0
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    # Pipeline for Validation/Test Images: NO Augmentation!
    # During testing, we evaluate raw, unaltered images.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    return train_transform, test_transform


def get_dataloaders(data_dir="./data", batch_size=64, num_workers=2):
    """
    Downloads CIFAR-10 (if not already present) and wraps it in streaming DataLoaders.
    """
    train_transform, test_transform = get_transforms()

    # Download and load training set
    train_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )

    # Download and load testing/validation set
    test_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )

    # Wrap in DataLoaders for batching and multi-thread streaming
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,          # Shuffle so batches are diverse every epoch
        num_workers=num_workers, 
        pin_memory=True        # Fast memory transfer from RAM to GPU
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,         # No need to shuffle test data
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check: Verify shapes and loading
    print("Testing DataLoader pipeline...")
    train_loader, test_loader = get_dataloaders(batch_size=4)
    images, labels = next(iter(train_loader))
    
    print(f"Loaded Batch Shape: {images.shape}")
    print(f"Batch Labels: {labels}")
    print("Dataset pipeline is working properly.")