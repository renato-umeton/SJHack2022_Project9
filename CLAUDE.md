# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a biohackathon project (SJHack2022 Project 9) focused on cell image segmentation and clustering using deep learning approaches. The project combines:

1. CellPose v2 for cell segmentation with ground truth validation
2. Convolutional autoencoders for feature learning and dimensionality reduction
3. K-means clustering on learned representations

## Environment Setup

This project requires a Cellpose conda environment. See: https://cellpose.readthedocs.io/en/latest/index.html

Key dependencies:
- PyTorch (with CUDA support for GPU acceleration)
- CellPose v2
- scikit-learn (for metrics and clustering)
- scikit-image
- albumentations (for image augmentation)
- t-SNE for visualization

## Running CellPose Segmentation

The `runCellPoseV2wGT.py` script runs CellPose v2 segmentation with ground truth validation.

### Basic Usage

```bash
python runCellPoseV2wGT.py <input_directory> <output_directory>
```

### With Custom Parameters

```bash
# Using a pretrained model
python runCellPoseV2wGT.py <input_dir> <output_dir> \
  -FT 1 -PT -1 \
  -PTM /path/to/pretrained_model.pt

# Adjusting thresholds
python runCellPoseV2wGT.py <input_dir> <output_dir> \
  -FT 0.4 -PT 0.0 -D 30
```

### Key Parameters

- `-PTM`: Path to pretrained CellPose model (default: None)
- `-M`: CellPose model type (default: "TN1")
- `-D`: Diameter for cell detection (default: 30)
- `-FT`: Flow threshold (default: 0.4)
- `-PT`: Probability threshold (default: 0.0)
- `-IT`: Image type regex (default: "*.tif")
- `-C`: Channels (default: [0,0])

### Ground Truth Requirements

- Ground truth masks must be in the same directory as input images
- Naming convention: `<image_name>_Mask.<extension>`
- Ground truth uses Class 1 values = 0
- CellPose generated segmentation uses Class 1 values != 0

### Output

1. CellPose segmentation plots with flow (PNG)
2. F1 score metrics (tab-delimited text file: `Metrics.txt`)
3. Raw segmentation masks (PNG)

## Jupyter Notebooks

### ae_conv_mnist.ipynb

Convolutional autoencoder trained on MNIST dataset. Simple architecture used for testing the autoencoder approach.

### ae_conv_cells.ipynb

Advanced convolutional autoencoder for cell image analysis:

**Architecture:**
- Encoder: Conv layers with batch normalization and max pooling
- Bottleneck: MLP reduces to 256-dimensional latent space
- Decoder: Transposed convolutions to reconstruct 128x128x3 images

**Training:**
- Uses albumentations for data augmentation (flip, rotate, brightness/contrast)
- Z-score normalization per channel
- MSE loss for reconstruction
- Adam optimizer

**Analysis Pipeline:**
1. Extract 256-dimensional latent representations
2. Apply t-SNE for visualization
3. K-means clustering (default: 12 clusters)
4. Save cluster labels and visualizations

**Key Functions:**
- `z_score()` / `z_score_arr()`: Normalize images
- `un_z_score()`: Denormalize for visualization
- `DatasetTrain`: Custom dataset with transforms
- `fashion_scatter()`: t-SNE visualization with cluster labels

## Code Architecture

### CellPose Script (runCellPoseV2wGT.py)

**Main Pipeline:**
1. `parseArguments()`: CLI argument parsing
2. `sampleWalk()`: Find input images (excludes `_Mask` files)
3. `buildParameters()`: Construct CellPose parameter dict
4. `runCellPose2()`: Execute segmentation on all samples
5. `getMetrics()`: Calculate F1 scores vs ground truth
6. `createPlots()`: Generate visualization outputs

**Metric Calculation:**
- Ground truth: pixels == 0 are Class 1
- Predictions: pixels != 0 are Class 1
- Uses sklearn's `f1_score` on flattened pixel arrays

### Autoencoder Architecture

**Conv Layer Modules:**
- `ConvLayerNorm`: Conv2d + BatchNorm + LeakyReLU
- `ConvLayer`: Conv2d + LeakyReLU
- `ConvTransposeLayerNorm`: ConvTranspose2d + BatchNorm + LeakyReLU
- `ConvTransposeLayer`: ConvTranspose2d + LeakyReLU

**Full Autoencoder (ae_conv_cells.ipynb):**
- `Encoder`: Convolutional feature extraction to (batch, 32, 14, 14)
- `MLP_down`: Flatten and reduce to 256-D latent space
- `MLP_up`: Expand from 256-D back to (batch, 32, 14, 14)
- `Decoder`: Transposed convolutions to reconstruct input size

The bottleneck MLP enables dimensionality reduction while the convolutional layers capture spatial features.

## Development Notes

### GPU Usage

Both notebooks and CellPose script are designed for GPU acceleration:
- Check `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
- CellPose models initialized with `gpu=True`

### Image Conventions

- CellPose works with various formats (controlled by `-IT` parameter)
- Autoencoder expects 3-channel RGB images
- Default cell crop size: 128x128x3
- Channel-wise z-score normalization is critical for training

### Clustering Output

K-means results are saved to:
- Cluster labels: `.npy` file
- t-SNE visualization: PNG
- Sample grid by cluster: PNG showing representative images per cluster
