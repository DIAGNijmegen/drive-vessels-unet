# Retinal vessel segmentation on DRIVE: Training and Deployment using Docker

This repository will detail how you can train and deploy a U-Net for retinal vessel segmentation on the DRIVE dataset using Docker.

💡 To use the pre-trained model and wrap it in a Docker container, follow the instructions provided in this [blog](https://grand-challenge.org/blogs/create-an-algorithm/).

## Requirements (for training)

* monai
* SimpleITK
* numpy
* scipy
* skimage
* torch  
* torchvision

## Training

For training the algorithm, first download the DRIVE dataset and place the files under `data/`.

Start training your algorithm by executing
```bash
python train.py
```

## Pre-trained weights

Pre-trained weights are available in this repository under the name `best_metric_model_segmentation2d_dict.pth`.

## Inference

Run `inference.py` to take a test image and plot the prediction along with the input image.
