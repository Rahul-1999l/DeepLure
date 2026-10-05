import torchvision.transforms as transforms

def get_train_transforms(
    use_color_aug=True,
    crop_scale=(0.8, 1.0),
    rotation_deg=15,
    jitter_brightness=0.3,
    jitter_contrast=0.3,
    jitter_saturation=0.3,
    jitter_hue=0.15,
    grayscale_prob=0.25
):
    """
    Color-invariant training data augmentations.
    Optionally disables color jitter and grayscale when use_color_aug=False.
    """
    transform_list = [
        transforms.RandomResizedCrop(224, scale=crop_scale),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=rotation_deg)
    ]
    
    if use_color_aug:
        transform_list.extend([
            transforms.ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
                hue=jitter_hue
            ),
            transforms.RandomGrayscale(p=grayscale_prob)
        ])
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)

def get_eval_transforms():
    """
    Deterministic evaluation transforms for validation, query, and gallery images.
    No stochastic augmentations.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
