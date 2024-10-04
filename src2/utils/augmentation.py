from torchvision import transforms

def get_augmentations():
    """
    Returns the augmentation pipeline for the training dataset.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((256, 512)),  # Resize the images to a fixed size
        # transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        transforms.ToTensor(),  # Convert PIL image to Tensor
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize with ImageNet stats
    ])
    return train_transforms

def get_augmentations_normal():
    """
    Returns the augmentation pipeline for the training dataset.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((256, 512)),  # Resize the images to a fixed size
        transforms.ToTensor(),  # Convert PIL image to Tensor
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize with ImageNet stats
    ])
    return train_transforms