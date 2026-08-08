"""
Evaluation script for AI-Based Restoration of Degraded Images
Team: Silicon Squad
Usage: python evaluate.py --input_dir <path_to_test_images> --output_dir <path_to_save_outputs> --model_path restoration_model_v4.pt
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn


# ==== Model Architecture (must match train.py) ====
class ConvBlockV2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class RestorationNetV4(nn.Module):
    """
    U-Net style architecture with residual connections.
    Input: (B, 1, 128, 128) degraded grayscale image
    Output: (B, 1, 256, 256) restored clean image (2x super-resolution + denoising)
    """
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlockV2(1, 96)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlockV2(96, 192)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlockV2(192, 384)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlockV2(384, 768)

        self.up3 = nn.ConvTranspose2d(768, 384, 2, stride=2)
        self.dec3 = ConvBlockV2(768, 384)
        self.up2 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.dec2 = ConvBlockV2(384, 192)
        self.up1 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec1 = ConvBlockV2(192, 96)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], 1))
        out = self.final_up(d1)
        return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate restoration model on test images")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing degraded .npy test images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory where restored .npy images will be saved")
    parser.add_argument("--model_path", type=str, default="restoration_model_v4.pt", help="Path to trained model weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RestorationNetV4().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    input_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".npy")])
    print(f"Found {len(input_files)} test images.")

    with torch.no_grad():
        for fname in input_files:
            noisy = np.load(os.path.join(args.input_dir, fname)).astype(np.float32)
            noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

            output = model(noisy_tensor)
            restored = output.squeeze().cpu().numpy()

            save_path = os.path.join(args.output_dir, fname)
            np.save(save_path, restored)

    print(f"Inference complete. Restored images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()