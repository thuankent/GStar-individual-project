import os
import torch
import tqdm 
from dataset.imagenet import get_model_transform
from util.download_SiT import find_model
from util.models_SiT import SiT_XL_2
from diffusers.models import AutoencoderKL

def get_SiT_model(model_name, vae_name, image_size):
    """ Loads the SiT model from torch, modifies it for feature extraction
    Params:
        model_name: The name of the SiT model variant
        vae_name: The name of the corresponding VAE model
        image_size: The size of the input images
    Returns: The SiT and VAE models
    """
    print(f"[Info] Loading SiT model: {model_name}")
    # Load model SiT, its state dict, and make it as the eval mode
    latent_size = image_size // 8
    model = SiT_XL_2(input_size=latent_size)
    state_dict = find_model(f"SiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model.eval()

    # Load VAE, its state dict, and make it as the eval mode
    vae = AutoencoderKL.from_pretrained(vae_name)
    vae.eval()
    return model, vae

def interpolant(t):
    """ Implement linear interpolant as proposed by the SiT paper
    Param:
        t: The current timestep
    Return: the corresponding alpha_t, sigma_t, d_alpha_t, and d_sigma_t values
    """
    alpha_t = 1 - t
    sigma_t = t
    d_alpha_t = -1
    d_sigma_t = 1
    return alpha_t, sigma_t, d_alpha_t, d_sigma_t

@torch.no_grad()
def get_SiT_representation(args, x, model, vae, depth):
    """ Extracts SiT representations from raw images 
    Params: 
        args: Arguments of the parser from the main function
        x: The tensor storing the images input
        model: The SiT model
        vae: The corresponding VAE model
        depth: The layer where we extract the SiT features
    Return: the features extaction from the SiT model
    """
    # Prepare the input for feedforward pass to extract SiT features (t and y)
    B = x.shape[0] # Extract the batch size dimension
    t = torch.full((B,), 0.25, device=args.device)
    t = t.reshape((B, 1, 1, 1))
    y = torch.full((B,), 1000, dtype = torch.long, device=args.device)

    # Prepare the input images: including injecting noises and pass through the VAE
    noises = torch.randn_like(x)
    alpha_t, sigma_t, _, _ = interpolant(t)
    x_noised = alpha_t * x + sigma_t * noises
    z = vae.encode(x_noised).latent_dist.sample() * 0.18215

    # Extract the features from the SiT model
    features = model.forward_feats(z, t.flatten(), y, depth)
    # Take the average over the patch axis to align at the later stage
    features = torch.mean(features, dim=1)
    return features

def extract_features(args, dataloader, model_name, vae_name, image_size, depth, save_features=None, checkpoint_path=None, save_path=None):
    """ Main function to extract features from the SiT model
    Params:
        args: Arguments of the parser from the main function
        dataloader: The dataloader of the dataset
        model_name: SiT model
        vae_name: VAE model
        image_size: The preprocessed image size for the input
        depth: The layer where we extract the SiT features
        save_features: The option to indicate whether we store the features or not for later usage
        checkpoint_path: Provide if the features are stored elsewhere
        save_path: The option to indicate the path to store the features 
    Return: The feature extraction from the SiT model
    """
    # Check for the path of the saved features, if the file exits, we use that
    # features without the need to rerun the extraction again!
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path)

    # Load SiT model 
    model, vae = get_SiT_model(model_name, vae_name, image_size)
    model = model.to(args.device)
    vae = vae.to(args.device)
    # Prepare to store features
    all_cls_tokens = []

    # Extract features for all of the data
    print("[Info] Starting Feature Extraction")
    for _, (images, _) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract the batch of images
        images = images.to(args.device)
        img_transform = get_model_transform(image_size)
        images = img_transform(images)
        # Extract the features vector of the DINOv2 model
        cls_token = get_SiT_representation(args, images, model, vae, depth) # Shapes: [B, D]
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