#!/usr/bin/env python3
"""
GPU-Optimized Multi-Input Image Segmentation Model Training Script
This version is optimized for better GPU utilization and Linux compatibility.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import random
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DualInputDataset(Dataset):
    """Dataset class for dual input images with dual labels"""
    
    def __init__(self, data_dir, image_files, transform=None, is_train=True):
        self.data_dir = data_dir
        self.image_files = image_files
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load dual input images
        im1_path = os.path.join(self.data_dir, 'im1', img_name)
        im2_path = os.path.join(self.data_dir, 'im2', img_name)
        
        im1 = Image.open(im1_path).convert('RGB')
        im2 = Image.open(im2_path).convert('RGB')
        
        # Load dual labels
        label1_path = os.path.join(self.data_dir, 'label1', img_name)
        label2_path = os.path.join(self.data_dir, 'label2', img_name)
        
        label1 = Image.open(label1_path).convert('L')  # Convert to grayscale
        label2 = Image.open(label2_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            im1 = self.transform(im1)
            im2 = self.transform(im2)
            # Apply same resize to labels
            label_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            label1 = label_transform(label1)
            label2 = label_transform(label2)
        
        # Normalize labels to 0, 1, 2 classes
        label1 = (label1 * 255).long()
        label2 = (label2 * 255).long()
        
        # Create proper class mapping: 0->0, 128->1, 255->2
        label1_new = torch.zeros_like(label1)
        label2_new = torch.zeros_like(label2)
        
        label1_new[label1 == 128] = 1
        label1_new[label1 == 255] = 2
        label2_new[label2 == 128] = 1
        label2_new[label2 == 255] = 2
        
        label1 = label1_new
        label2 = label2_new
        
        return {
            'im1': im1,
            'im2': im2,
            'label1': label1.squeeze(0),
            'label2': label2.squeeze(0),
            'filename': img_name
        }

class DualInputUNet(nn.Module):
    """GPU-Optimized U-Net model with dual inputs and dual outputs"""
    
    def __init__(self, in_channels=6, out_channels=3):
        super(DualInputUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Upsampling layers
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        # Decoder for output 1
        self.dec4_1 = self.conv_block(1024, 512)
        self.dec3_1 = self.conv_block(512, 256)
        self.dec2_1 = self.conv_block(256, 128)
        self.dec1_1 = self.conv_block(128, 64)
        self.final1 = nn.Conv2d(64, out_channels, 1)
        
        # Decoder for output 2
        self.dec4_2 = self.conv_block(1024, 512)
        self.dec3_2 = self.conv_block(512, 256)
        self.dec2_2 = self.conv_block(256, 128)
        self.dec1_2 = self.conv_block(128, 64)
        self.final2 = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x1, x2):
        # Concatenate dual inputs
        x = torch.cat([x1, x2], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder 1
        d4_1 = self.upconv4(b)
        d4_1 = torch.cat([d4_1, e4], dim=1)
        d4_1 = self.dec4_1(d4_1)
        
        d3_1 = self.upconv3(d4_1)
        d3_1 = torch.cat([d3_1, e3], dim=1)
        d3_1 = self.dec3_1(d3_1)
        
        d2_1 = self.upconv2(d3_1)
        d2_1 = torch.cat([d2_1, e2], dim=1)
        d2_1 = self.dec2_1(d2_1)
        
        d1_1 = self.upconv1(d2_1)
        d1_1 = torch.cat([d1_1, e1], dim=1)
        d1_1 = self.dec1_1(d1_1)
        
        out1 = self.final1(d1_1)
        
        # Decoder 2 (shared encoder, separate decoder)
        d4_2 = self.upconv4(b)
        d4_2 = torch.cat([d4_2, e4], dim=1)
        d4_2 = self.dec4_2(d4_2)
        
        d3_2 = self.upconv3(d4_2)
        d3_2 = torch.cat([d3_2, e3], dim=1)
        d3_2 = self.dec3_2(d3_2)
        
        d2_2 = self.upconv2(d3_2)
        d2_2 = torch.cat([d2_2, e2], dim=1)
        d2_2 = self.dec2_2(d2_2)
        
        d1_2 = self.upconv1(d2_2)
        d1_2 = torch.cat([d1_2, e1], dim=1)
        d1_2 = self.dec1_2(d1_2)
        
        out2 = self.final2(d1_2)
        
        return out1, out2

def calculate_iou(pred, target, num_classes=3):
    """Calculate IoU for each class"""
    ious = []
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = 1.0  # Perfect score if both are empty
        else:
            iou = intersection / union
        ious.append(iou)
    
    return np.mean(ious)

def get_optimal_batch_size(device):
    """Determine optimal batch size based on available memory"""
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_memory >= 8:
            return 8  # Larger batch size for high-end GPUs
        elif gpu_memory >= 4:
            return 4  # Medium batch size
        else:
            return 2  # Conservative for lower-end GPUs
    else:
        return 2  # Conservative for CPU

def get_optimal_num_workers():
    """Get optimal number of workers based on system"""
    import multiprocessing
    return min(4, multiprocessing.cpu_count())

def train_model():
    """Main training function with GPU optimizations"""
    
    # Enhanced device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("🐌 Using CPU (consider using GPU for faster training)")
    
    # Optimal batch size and workers
    batch_size = get_optimal_batch_size(device)
    num_workers = get_optimal_num_workers()
    
    print(f"📦 Batch size: {batch_size}")
    print(f"👥 Num workers: {num_workers}")
    
    # Data transforms with GPU-friendly settings
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get image files and limit to 20 for training
    train_files = sorted(os.listdir('train/im1'))[:20]
    test_files = sorted(os.listdir('test/im1'))[:10]
    
    print(f"🏋️ Training on {len(train_files)} images")
    print(f"🧪 Testing on {len(test_files)} images")
    
    # Create datasets
    train_dataset = DualInputDataset('train', train_files, transform=transform, is_train=True)
    test_dataset = DualInputDataset('test', test_files, transform=transform, is_train=False)
    
    # Create data loaders with optimal settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Initialize model
    model = DualInputUNet(in_channels=6, out_channels=3).to(device)
    
    # GPU-optimized settings
    if device.type == 'cuda':
        model = torch.compile(model) if hasattr(torch, 'compile') else model  # PyTorch 2.0+
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # AdamW for better generalization
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)  # Better scheduler
    
    # Training loop
    num_epochs = 20
    train_losses = []
    
    print("🚀 Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            im1 = batch['im1'].to(device, non_blocking=True)
            im2 = batch['im2'].to(device, non_blocking=True)
            label1 = batch['label1'].to(device, non_blocking=True)
            label2 = batch['label2'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            out1, out2 = model(im1, im2)
            
            # Calculate loss
            loss1 = criterion(out1, label1)
            loss2 = criterion(out2, label2)
            total_loss = loss1 + loss2
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_iou1 = 0.0
            val_iou2 = 0.0
            
            with torch.no_grad():
                for batch in test_loader:
                    im1 = batch['im1'].to(device, non_blocking=True)
                    im2 = batch['im2'].to(device, non_blocking=True)
                    label1 = batch['label1'].to(device, non_blocking=True)
                    label2 = batch['label2'].to(device, non_blocking=True)
                    
                    out1, out2 = model(im1, im2)
                    
                    loss1 = criterion(out1, label1)
                    loss2 = criterion(out2, label2)
                    val_loss += (loss1 + loss2).item()
                    
                    # Calculate IoU
                    pred1 = torch.argmax(out1, dim=1)
                    pred2 = torch.argmax(out2, dim=1)
                    
                    val_iou1 += calculate_iou(pred1, label1)
                    val_iou2 += calculate_iou(pred2, label2)
            
            val_loss /= len(test_loader)
            val_iou1 /= len(test_loader)
            val_iou2 /= len(test_loader)
            
            print(f'✅ Validation - Loss: {val_loss:.4f}, IoU1: {val_iou1:.4f}, IoU2: {val_iou2:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'dual_input_unet_gpu_optimized.pth')
    print("💾 Model saved as 'dual_input_unet_gpu_optimized.pth'")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss (GPU Optimized)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss_gpu_optimized.png')
    plt.show()
    
    print("🎉 Training completed successfully!")
    print(f"📊 Final training loss: {train_losses[-1]:.4f}")

if __name__ == "__main__":
    # Check if required directories exist
    if not os.path.exists('train') or not os.path.exists('test'):
        print("❌ Error: 'train' and 'test' directories not found!")
        exit(1)
    
    required_dirs = ['train/im1', 'train/im2', 'train/label1', 'train/label2',
                     'test/im1', 'test/im2', 'test/label1', 'test/label2']
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Error: Directory '{dir_path}' not found!")
            exit(1)
    
    print("✅ Dataset structure verified. Starting training...")
    train_model() 