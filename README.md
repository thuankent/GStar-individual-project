# SiT and DINOv2-g alignments
This project measures the features alignment between DINOv2-g model with diffusion transformers SiT model using CKNNA on ImageNet dataset

## Requirements
- Python >= 3.10  
- CUDA >= 11.8
- PyTorch

## Dataset download
Currently, we provide experiments for the validation set of the ImageNet dataset. You can place the data where you want and can specifiy it via `--data-path` arguments in the main script `figure2b.sh`.
Link to download the validation set of the ImageNet dataset
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
```

## Running
```bash
sh figure2b.sh
```
Please make sure to change the path to your dataset and the models that you want to experiment. Max images is the number of images using to measure the alignments. The validation set of the ImageNet dataset consists of 50000 images.
