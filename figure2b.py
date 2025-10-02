import os
import torch
import argparse
import numpy as np
import model.DINOv2 as DINOv2
import model.DINO as DINO
import model.MAE as MAE
import model.SiT as SiT

from dataset.imagenet import prepare_raw_data
from util.metrics import AlignmentMetrics # Import the module to compute CKNNA features alignment 
                                          # from the plantonic representation repo

def main():
    """ Main driver function. Use this module to extract features from model A and B. Then, calculate the CKNNA score """
    # Dealing with parser for arguments
    parser = argparse.ArgumentParser(description="Features alignment calculation pipeline")
    # Arguments of the settings in common: model name, batch size, device usage, seed, and the path for the evaluation dataset
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader (default: 256)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the image dataset (e.g., ImageNet)")

    parser.add_argument("--model_A", type=str, default="dinov2_vitg14", help="The name of model A specifically")
    parser.add_argument("--save_features_A", type=str, default=None, help="Path to save extracted features A (optional)")
    parser.add_argument("--features_checkpoint_A", type=str, default='cc', help="Path to load saved features A (optional)")
    parser.add_argument("--image_size_A", type=int, default=224, help="Input image resolution to feed to model A (default: 224)")

    parser.add_argument("--model_B", type=str, default="SiT_XL-2", help="The name of model B specifically")
    parser.add_argument("--save_features_B", type=str, default=None, help="Path to save extracted features B (optional)")
    parser.add_argument("--features_checkpoint_B", type=str, default='cc', help="Path to load saved features B (optional)")
    parser.add_argument("--image_size_B", type=int, default=256, help="Input image resolution to feed to model B (default: 224)")

    parser.add_argument("--depth", type=int, default=20, help="The depth layer to extract diffusion SiT model")
    parser.add_argument("--max_images", type=int, default=10000, help="Maximum number of images to process (default: 10000)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers (default: 4)")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Check device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # To specify which GPU to use
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Prepare the dataset to extract features (i.e. ImageNet)
    dataloader = prepare_raw_data(args)
    # Default VAE model for extracting latents of the SiT model
    vae_model = 'stabilityai/sd-vae-ft-ema'

    # Handling the features extraction of the first model (model A)
    if "dinov2" in args.model_A:
        feature_A = DINOv2.extract_features(args, dataloader, args.model_A,
                                            args.image_size_A,
                                            args.save_features_A,
                                            checkpoint_path=args.features_checkpoint_A,
                                            save_path=args.save_features_A)
    elif "mae" in args.model_A:
        feature_A = MAE.extract_features(args, dataloader, args.model_A,
                                         args.image_size_A,
                                         args.save_features_A,
                                         checkpoint_path=args.features_checkpoint_A,
                                         save_path=args.save_features_A)
    elif "dino" in args.model_A:
        feature_A = DINO.extract_features(args, dataloader, args.model_A,
                                          args.image_size_A,
                                          args.save_features_A,
                                          checkpoint_path=args.features_checkpoint_A,
                                          save_path=args.save_features_A)
    elif "SiT" in args.model_A:
        feature_A = SiT.extract_features(args, dataloader, args.model_A, vae_model,
                                         args.image_size_A,
                                         args.depth,
                                         args.save_features_A,
                                         checkpoint_path=args.features_checkpoint_A,
                                         save_path=args.save_features_A)
    else:
        raise ValueError("Currently only support DINO, MAE, and SiT models family")

    # Handling the features extraction of the first model (model B)
    if "dinov2" in args.model_B:
        feature_B = DINOv2.extract_features(args, dataloader, args.model_B,
                                             args.image_size_B,
                                             args.save_features_B,
                                             checkpoint_path=args.features_checkpoint_B,
                                             save_path=args.save_features_B)
    elif "mae" in args.model_B:
        feature_B = MAE.extract_features(args, dataloader, args.model_B,
                                            args.image_size_B,
                                            args.save_features_B,
                                            checkpoint_path=args.features_checkpoint_B,
                                            save_path=args.save_features_B)
    elif "dino" in args.model_B:
        feature_B = DINO.extract_features(args, dataloader, args.model_B,
                                          args.image_size_B,
                                          args.save_features_B,
                                          checkpoint_path=args.features_checkpoint_B,
                                          save_path=args.save_features_B)
    elif "SiT" in args.model_B:
        feature_B = SiT.extract_features(args, dataloader, args.model_B, vae_model,
                                         args.image_size_B,
                                         args.depth,
                                         args.save_features_B,
                                         checkpoint_path=args.features_checkpoint_B,
                                         save_path=args.save_features_B)
    else:
        raise ValueError("Currently only support DINO, MAE, and SiT models family")

    # Normalize the features
    feature_A = torch.nn.functional.normalize(feature_A, dim=1) 
    feature_B = torch.nn.functional.normalize(feature_B, dim=1)

    # Measure CKNNA score from the Platonic Representation Hypothesis paper
    score_platonic = AlignmentMetrics.cknna(feature_A, feature_B, topk=10, unbiased=True)
    print(f"The CKNNA score between {args.model_A} and {args.model_B} is {score_platonic}")

if __name__ == "__main__":
    main()



