# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Team:** Silicon Squad  
**Members:** Gopika B, Gobika V, Brindha P  
**Problem Statement:** AI-Based Restoration of Degraded Images for Semiconductor Inspection (KLA)

## Overview

This project restores degraded semiconductor wafer inspection images (speckle noise, Gaussian noise, and reduced spatial resolution) back to clean, high-resolution images using a deep learning model. The model takes a 128x128 degraded grayscale image as input and outputs a restored 256x256 clean image.

## Model Architecture

A U-Net style convolutional neural network with residual connections:
- 3-level encoder-decoder architecture with skip connections
- Residual blocks (Conv + BatchNorm + ReLU with skip connections) for stable training
- Final upsampling layer to convert 128x128 input to 256x256 output (2x super-resolution)
- Combined loss function: L1 loss + SSIM loss + Sobel edge loss (for sharp edge preservation)

## Results

| Metric | Score |
|--------|-------|
| SSIM   | ~0.77 |
| PSNR   | ~28.5 dB |

## Trained Model Weights

Due to GitHub's 25MB file size limit, the trained model weights are hosted on Google Drive:

**Download:** https://drive.google.com/file/d/123t2-2aei8XhJhjLSYWjENyuj5z57h2r/view?usp=sharing

Download `restoration_model_v4.pt` and place it in the repository root before running `evaluate.py`.

## Restored Test Outputs

Due to file size limits, the restored test outputs are hosted on Google Drive:

**Download:** https://drive.google.com/file/d/1c2t5UituB5rzrR7vDC1d-2h9SNQCV41v/view?usp=sharing

This ZIP contains the model's restored outputs (`.npy` files) for all 400 images in the provided test set.

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd SiliconSquad_KLA_PS01
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run inference on test images
```bash
python evaluate.py --input_dir <path_to_test_noisy_images> --output_dir <path_to_save_restored_images> --model_path restoration_model_v4.pt
```

This will:
- Load the trained model
- Read all `.npy` degraded test images from `input_dir`
- Restore each image
- Save restored `.npy` images to `output_dir`

### 4. Training (to reproduce from scratch)
See `train.py` or `training_notebook.ipynb` for the full training pipeline, including data loading, model definition, and training loop.

## Files in this Repository

| File | Description |
|------|--------------|
| `README.md` | This file |
| `evaluate.py` | Standalone inference/evaluation script |
| `train.py` | Training script to reproduce the model from scratch |
| `restoration_model_v4.pt` | Final trained model weights |
| `restored_test_outputs/` | Model outputs on the provided test set |
| `requirements.txt` | Python dependencies |

## Technology Stack

- Python, PyTorch
- NumPy
- scikit-image (for SSIM/PSNR evaluation)
- pytorch-msssim (for SSIM loss during training)
- Trained on Google Colab (T4 GPU)
