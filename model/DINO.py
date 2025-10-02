import os
import torch
import tqdm
import torch.nn as nn
from dataset.imagenet import get_model_transform

def get_dino_model(model_name):
    """ Loads the DINO model from torch.hub, modifies it for feature extraction
    Params:
        model_name: The name of the DINO model variant
    Return: The modified DINO model
    """
    print(f"[Info] Loading DINO model: {model_name}")
    # Load DINOv2 model and its weights from torch.hub
    encoder = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=True)
    encoder.head = nn.Identity() # Remove the classification head
    encoder.eval() # Switch the model to eval mode
    # Freeze all of the parameters
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder

@torch.no_grad()
def get_dino_representation(x, model):
    """ Extracts DINO representations from raw images
   Params:
        x: The tensor storing the images input
        model: The DINO model
    Return: the class tokens representation of the DINO model
    """
    # Forward pass! Note that we only need the cls token and the model is handling that
    cls_token = model.forward(x) # Shape: [B, D]
    return cls_token

def extract_features(args, dataloader, model_name, image_size, save_features=None, checkpoint_path=None, save_path=None):
    """ Main function to extract features from the DINO model
    Params:
        args: Arguments of the parser from the main function
        dataloader: The dataloader of the dataset
        model_name: DINO model
        image_size: The preprocessed image size for the input
        save_features: The option to indicate whether we store the features or not for later usage
        checkpoint_path: Provide if the features are stored elsewhere
        save_path: The option to indicate the path to store the features 
    Return: The feature extraction from the DINO model
    """
    # Check for the path of the saved features, if the file exits, we use that
    # features without the need to rerun the extraction again!
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path)

    # Load DINO model 
    model = get_dino_model(model_name).to(args.device)
    # Prepare to store features
    all_cls_tokens = []

    # Extract features for all of the data
    print("[Info] Starting Feature Extraction")
    for _, (images, _) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract the batch of images
        images = images.to(args.device)
        img_transform = get_model_transform(image_size)
        images = img_transform(images)
        # Extract the features vector of the DINO model
        cls_token = get_dino_representation(images, model) # Shapes: [B, D]
        # Append the current batch to the overall list
        all_cls_tokens.append(cls_token.cpu())
        
    # Concatenate all cls features
    all_cls_tokens = torch.cat(all_cls_tokens, dim=0) # Shape: [Total, D]
    print(f"[Info] Extracted Class Tokens shape: {all_cls_tokens.shape}")

    # Optionally save features
    if save_features:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(all_cls_tokens, save_path)
        print(f"[Info] Features saved to {save_path} in PyTorch format")

    print("[Info] Feature Extraction Pipeline Completed Successfully")
    return all_cls_tokens