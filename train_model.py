#!/usr/bin/env python3
"""
Multi-Input Image Segmentation Model Training Script
This script trains a U-Net model on dual input images with dual segmentation masks.
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
    """U-Net model with dual inputs and dual outputs"""
    
    def __init__(self, in_channels=6, out_channels=3):  # 6 channels (3+3), 3 classes per output
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
        self.dec4_1 = self.conv_block(1024, 512)  # 512 + 512 from skip connection
        self.dec3_1 = self.conv_block(512, 256)   # 256 + 256 from skip connection
        self.dec2_1 = self.conv_block(256, 128)   # 128 + 128 from skip connection
        self.dec1_1 = self.conv_block(128, 64)    # 64 + 64 from skip connection
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

def train_model():
    """Main training function"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize for faster training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get image files and limit to 20 for training
    train_files = sorted(os.listdir('train/im1'))[:20]
    test_files = sorted(os.listdir('test/im1'))[:10]  # Use 10 for testing
    
    print(f"Training on {len(train_files)} images")
    print(f"Testing on {len(test_files)} images")
    
    # Create datasets
    train_dataset = DualInputDataset('train', train_files, transform=transform, is_train=True)
    test_dataset = DualInputDataset('test', test_files, transform=transform, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Initialize model
    model = DualInputUNet(in_channels=6, out_channels=3).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    train_losses = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            im1 = batch['im1'].to(device)
            im2 = batch['im2'].to(device)
            label1 = batch['label1'].to(device)
            label2 = batch['label2'].to(device)
            
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
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
        
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
                    im1 = batch['im1'].to(device)
                    im2 = batch['im2'].to(device)
                    label1 = batch['label1'].to(device)
                    label2 = batch['label2'].to(device)
                    
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
            
            print(f'Validation - Loss: {val_loss:.4f}, IoU1: {val_iou1:.4f}, IoU2: {val_iou2:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'dual_input_unet.pth')
    print("Model saved as 'dual_input_unet.pth'")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    # Final evaluation
    print("\nFinal Evaluation:")
    model.eval()
    test_iou1 = 0.0
    test_iou2 = 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            im1 = batch['im1'].to(device)
            im2 = batch['im2'].to(device)
            label1 = batch['label1'].to(device)
            label2 = batch['label2'].to(device)
            
            out1, out2 = model(im1, im2)
            pred1 = torch.argmax(out1, dim=1)
            pred2 = torch.argmax(out2, dim=1)
            
            iou1 = calculate_iou(pred1, label1)
            iou2 = calculate_iou(pred2, label2)
            
            test_iou1 += iou1
            test_iou2 += iou2
            
            print(f"Test batch {i+1}: IoU1={iou1:.4f}, IoU2={iou2:.4f}")
            
            # Save a few sample predictions
            if i < 3:
                # Convert tensors to numpy for visualization
                img1_np = im1[0].cpu().numpy().transpose(1, 2, 0)
                img1_np = (img1_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img1_np = np.clip(img1_np, 0, 1)
                
                pred1_np = pred1[0].cpu().numpy()
                label1_np = label1[0].cpu().numpy()
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img1_np)
                axes[0].set_title('Input Image 1')
                axes[0].axis('off')
                
                axes[1].imshow(label1_np, cmap='viridis')
                axes[1].set_title('Ground Truth Label 1')
                axes[1].axis('off')
                
                axes[2].imshow(pred1_np, cmap='viridis')
                axes[2].set_title('Predicted Label 1')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'prediction_sample_{i+1}.png')
                plt.show()
    
    test_iou1 /= len(test_loader)
    test_iou2 /= len(test_loader)
    
    print(f"\nFinal Test Results:")
    print(f"Average IoU for Label 1: {test_iou1:.4f}")
    print(f"Average IoU for Label 2: {test_iou2:.4f}")
    print(f"Overall Average IoU: {(test_iou1 + test_iou2) / 2:.4f}")

if __name__ == "__main__":
    # Check if required directories exist
    if not os.path.exists('train') or not os.path.exists('test'):
        print("Error: 'train' and 'test' directories not found!")
        exit(1)
    
    required_dirs = ['train/im1', 'train/im2', 'train/label1', 'train/label2',
                     'test/im1', 'test/im2', 'test/label1', 'test/label2']
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Error: Directory '{dir_path}' not found!")
            exit(1)
    
    print("Dataset structure verified. Starting training...")
    train_model() 