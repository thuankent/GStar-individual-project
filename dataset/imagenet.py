import torch
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def prepare_raw_data(args):
    """
    Prepare dataset with minimal transforms (only PIL to tensor, no resize/normalize)
    This allows model-specific transforms to be applied later
    """
    print("[Info] Preparing Raw Image Dataset")
    # Only convert PIL to tensor, no resizing or normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
    print(f"[Info] Dataset size: {len(dataset)}")
    print(f"[Info] Max images: {args.max_images}")
    if args.max_images:
        if args.max_images > len(dataset):
            print(f"[Warning] max_images {args.max_images} is greater than dataset size {len(dataset)}. Using full dataset.")
            args.max_images = len(dataset)
        else:       
            subset_indices = random.sample(range(len(dataset)), args.max_images)
            dataset = torch.utils.data.Subset(dataset, indices=subset_indices)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    return dataloader

def get_model_transform(image_size):
    """ Get model-specific transform for the given image size. Then, normalize the image """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet mean and std
                             std=[0.229, 0.224, 0.225])
    ])