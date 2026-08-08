"""
Training script for AI-Based Restoration of Degraded Images
Team: Silicon Squad
Reproduces the full training pipeline from scratch.
Usage: python train.py --data_dir <path_to_train_zip_extracted_folder> --epochs 80
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim as ssim_loss_fn


# ==== Dataset ====
class WaferDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.files = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        noisy = np.load(os.path.join(self.noisy_dir, fname)).astype(np.float32)
        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        return noisy, gt


# ==== Model Architecture ====
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


# ==== Sobel Edge Loss (for sharp edge preservation) ====
def sobel_edges(img):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    ex = torch.nn.functional.conv2d(img, sobel_x, padding=1)
    ey = torch.nn.functional.conv2d(img, sobel_y, padding=1)
    return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)


class CombinedLoss(nn.Module):
    """Combined L1 + SSIM + Edge loss."""
    def __init__(self, alpha=0.6, beta=0.25, gamma=0.15):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, output, target):
        l1_loss = self.l1(output, target)
        s_loss = 1 - ssim_loss_fn(output, target, data_range=1.0, size_average=True)
        edge_loss = self.l1(sobel_edges(output), sobel_edges(target))
        return self.alpha * l1_loss + self.beta * s_loss + self.gamma * edge_loss


def main():
    parser = argparse.ArgumentParser(description="Train the restoration model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to extracted train folder containing NoisyLR/ and GT/ subfolders")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="restoration_model_v4.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    noisy_dir = os.path.join(args.data_dir, "NoisyLR")
    gt_dir = os.path.join(args.data_dir, "GT")

    dataset = WaferDataset(noisy_dir, gt_dir)
    print("Total samples:", len(dataset))

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = RestorationNetV4().to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                output = model(noisy)
                loss = criterion(output, gt)
                val_running_loss += loss.item() * noisy.size(0)
        val_loss = val_running_loss / len(val_ds)
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

    print("Training complete!")
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()